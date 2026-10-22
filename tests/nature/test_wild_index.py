"""Sylva-wild index: build, resume, load and hybrid search (fake embedder, no model download)."""

import hashlib
import json

import numpy as np
import pytest

from app.services.nature.text import tokenize
from app.services.wild import index as wild
from scripts.nature.wild import build_index


class FakeEmbedder:
    """Hashed bag of words: deterministic, and similar texts get similar vectors."""

    model = "fake-bow"
    dim = 64

    def __init__(self):
        self.calls = 0

    def __call__(self, texts, kind):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        return wild.normalise_rows(out)


GROUPS = {
    "shark-trust": {"id": "shark-trust", "name": "Shark Trust", "donate_url": "https://example.org/shark"},
}

RECORDS = [
    {"id": "r1", "title": "Shark skin riblets reduce drag on ship hulls", "kind": "paper", "source": "openalex",
     "url": "https://doi.org/10.1/r1", "doi": "10.1/r1", "year": 2020, "primary_sector": "aerospace-transport",
     "organism": {"name": "shark", "gbif_key": 1, "rank": "ORDER"}, "group_ids": ["shark-trust", "gone"],
     "functions": ["reduce-drag"], "sort_reason": "dropped", "licence": "CC-BY-4.0"},
    {"id": "r2", "title": "Termite mound ventilation for passive building cooling", "kind": "paper",
     "source": "openalex", "url": "https://doi.org/10.1/r2", "year": 2019, "primary_sector": "buildings-architecture",
     "summary": "Termite mounds keep a steady temperature by convection.", "summary_licence": "CC-BY"},
    {"id": "r3", "title": "Lotus leaf self-cleaning superhydrophobic coating", "kind": "paper", "source": "openalex",
     "url": "https://doi.org/10.1/r3", "year": 2018, "primary_sector": "chemicals", "sector_tags": ["buildings-architecture"]},
    {"id": "r4", "title": "Passive cooling paint inspired by desert ants", "kind": "paper", "source": "openalex",
     "url": "https://doi.org/10.1/r4", "year": 2021, "primary_sector": "construction-materials"},
    {"id": "r5", "title": "Gecko adhesive gripper for robots", "kind": "project", "source": "cordis",
     "url": "https://cordis.europa.eu/project/id/5", "year": 2017, "primary_sector": "machinery-robotics"},
]


def write_catalogue(path, records=RECORDS):
    path.write_text(json.dumps({"version": "t", "generated_on": "2026-10-11", "count": len(records),
                                "records": records}), encoding="utf-8")


@pytest.fixture
def built(tmp_path):
    cat = tmp_path / "catalogue.json"
    write_catalogue(cat)
    out = tmp_path / "index"
    emb = FakeEmbedder()
    assert build_index.main(["--catalogue", str(cat), "--out", str(out), "--part-size", "2"], embedder=emb) == 0
    return out, emb


def test_build_writes_all_files_and_meta(built):
    out, _ = built
    for f in ("meta.json", "vectors.npy", "bm25.npz", "vocab.json", "docs.json.gz"):
        assert (out / f).exists()
    assert not (out / "parts").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["count"] == 5 and meta["dim"] == 64 and meta["model"] == "fake-bow"
    assert meta["catalogue_generated_on"] == "2026-10-11"
    assert np.load(out / "vectors.npy").dtype == np.int8


def test_docs_keep_citation_fields_only(built):
    out, emb = built
    idx = wild.WildIndex(out, embedder=emb, groups=GROUPS)
    d = idx.docs[0]
    assert d["url"] == "https://doi.org/10.1/r1" and d["doi"] == "10.1/r1"
    assert "sort_reason" not in d and "licence" not in d


def test_exact_terms_rank_first_and_groups_resolve(built):
    out, emb = built
    idx = wild.WildIndex(out, embedder=emb, groups=GROUPS)
    hits = idx.search("reduce drag on a ship hull", k=3)
    assert hits[0]["id"] == "r1"
    assert hits[0]["score"] == 1.0
    assert hits[0]["ranks"]["bm25"] == 1
    assert hits[0]["groups"] == [GROUPS["shark-trust"]]  # unknown group id "gone" dropped


def test_sector_is_a_soft_boost_not_a_filter(built):
    out, emb = built
    idx = wild.WildIndex(out, embedder=emb, groups=GROUPS)
    plain = [h["id"] for h in idx.search("passive cooling", k=5)]
    boosted = [h["id"] for h in idx.search("passive cooling", k=5, sector="construction-materials")]
    assert set(plain) == set(boosted)
    assert boosted[0] == "r4"
    assert any(h["primary_sector"] != "construction-materials"
               for h in idx.search("passive cooling", k=5, sector="construction-materials"))


def test_empty_and_unknown_queries(built):
    out, emb = built
    idx = wild.WildIndex(out, embedder=emb, groups=GROUPS)
    assert idx.search("   ") == []
    # no BM25 term matches; dense still ranks every record, so results come back
    assert len(idx.search("zzzz qqqq", k=2)) == 2


def test_model_mismatch_and_count_mismatch_raise(built):
    out, emb = built
    other = FakeEmbedder()
    other.model = "other"
    with pytest.raises(ValueError, match="built with"):
        wild.WildIndex(out, embedder=other, groups=GROUPS)
    meta = json.loads((out / "meta.json").read_text())
    meta["count"] = 99
    (out / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="record count"):
        wild.WildIndex(out, embedder=emb, groups=GROUPS)


def test_resume_skips_finished_parts(tmp_path):
    texts = [f"text number {i}" for i in range(5)]
    parts = tmp_path / "parts"
    emb = FakeEmbedder()
    first = build_index.embed_in_parts(texts, parts, "h1", 2, emb)
    assert emb.calls == 3
    (parts / "part-0002.npy").unlink()
    again = build_index.embed_in_parts(texts, parts, "h1", 2, emb)
    assert emb.calls == 4  # only the missing part was embedded
    np.testing.assert_array_equal(first, again)
    build_index.embed_in_parts(texts, parts, "h2", 2, emb)  # other text: all parts redone
    assert emb.calls == 7


def test_quantise_keeps_ranking():
    rng = np.random.default_rng(0)
    m = wild.normalise_rows(rng.normal(size=(200, 32)).astype(np.float32))
    q8, scale = wild.quantise(m)
    v = m[3]
    assert np.argmax(m @ v) == np.argmax((q8 @ v) / scale) == 3


def test_bm25_matches_reference_formula():
    texts = ["shark skin drag", "shark shark fin", "lotus leaf"]
    mat, terms = wild.build_bm25(texts)
    col = terms.index("shark")
    toks = [tokenize(t) for t in texts]
    avgdl = sum(map(len, toks)) / 3
    idf = np.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    expect = [idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(t) / avgdl)) if tf else 0.0
              for t in toks for tf in [t.count("shark")]]
    np.testing.assert_allclose(mat[:, col].toarray().ravel(), expect, rtol=1e-5)


def test_missing_catalogue_returns_2(tmp_path):
    assert build_index.main(["--catalogue", str(tmp_path / "nope.json"), "--out", str(tmp_path / "o")],
                            embedder=FakeEmbedder()) == 2


def test_bm25_loads_without_scipy(built, monkeypatch):
    """The served path reads bm25.npz with numpy alone (no scipy on Vercel)."""
    from scipy import sparse

    out, _ = built
    ref = sparse.load_npz(out / "bm25.npz").tocsc()
    monkeypatch.setitem(__import__("sys").modules, "scipy", None)
    cols = wild.Bm25Columns(out / "bm25.npz")
    assert cols.shape == ref.shape
    pick = [0, 3, ref.shape[1] - 1]
    assert np.allclose(cols.column_sum(pick), np.asarray(ref[:, pick].sum(axis=1)).ravel())


def write_bert_tokenizer(path, vocab):
    path.write_text(json.dumps({
        "normalizer": {"type": "BertNormalizer", "clean_text": True, "handle_chinese_chars": True,
                       "strip_accents": None, "lowercase": True},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {"type": "TemplateProcessing", "special_tokens": {}},
        "model": {"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                  "max_input_chars_per_word": 20, "vocab": vocab},
    }), encoding="utf-8")


def test_wordpiece_normalises_splits_and_matches_longest_piece(tmp_path):
    from app.services.wild.wordpiece import WordPiece

    vocab = {"[PAD]": 0, "[UNK]": 1, "ship": 2, "hull": 3, "cafe": 4, "-": 5, "##s": 6, "re": 7, "##duce": 8}
    write_bert_tokenizer(tmp_path / "t.json", vocab)
    wp = WordPiece(tmp_path / "t.json")
    assert wp.encode("Ships-HULL") == [2, 6, 5, 3]
    assert wp.encode("Café" + chr(9) + "REDUCE" + chr(0)) == [4, 7, 8]  # accents stripped, control char dropped
    assert wp.encode("shipx " + "a" * 30) == [1, 1]  # no full match; over-long word
    assert wp.encode("ship ship ship", max_tokens=2) == [2, 2]


def test_static_embedder_is_mean_of_rows_without_unk(tmp_path):
    vocab = {"[PAD]": 0, "[UNK]": 1, "ship": 2, "hull": 3, "drag": 4}
    d = tmp_path / "query_model"
    d.mkdir()
    write_bert_tokenizer(d / "tokenizer.json", vocab)
    table = np.array([[0, 0], [9, 9], [1, 0], [0, 1], [1, 1]], dtype=np.int8)
    np.save(d / "embeddings.npy", table)
    np.save(d / "scales.npy", np.array([1, 1, 2, 2, 1], dtype=np.float32))
    (d / "model.json").write_text(json.dumps({"model": "m", "dim": 2, "unk_id": 1, "max_tokens": 8}))
    emb = wild.StaticEmbedder(d)
    got = emb(["ship hull zzz", "zzz", ""], "query")
    assert np.allclose(got[0], [2 ** -0.5, 2 ** -0.5])  # (2,0)+(0,2), the unknown word skipped
    assert np.allclose(got[1], 0) and np.allclose(got[2], 0)
    assert isinstance(wild.default_embedder("m", tmp_path), wild.StaticEmbedder)
