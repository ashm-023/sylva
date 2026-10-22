"""
Sylva-wild: searchable index over every record of the open nature catalogue.

Hybrid retrieval, no API key needed:
- BM25 over title, organism, summary, functions and design area, stored as a
  sparse column matrix of precomputed term weights (scipy), so 99k records fit
  in a few tens of MB;
- dense vectors from a small local static embedding model (fastembed,
  minishlab/potion-retrieval-32M: a token lookup plus mean pooling, thousands
  of records a second on CPU, so the query side needs no transformer), stored
  as int8 with one global scale. BAAI/bge-small-en-v1.5 is the stronger
  alternative (--model) but ran at 5-35 records/s on the build laptop's CPU;
- reciprocal-rank fusion of the two ranked lists (k=60), as in
  app/services/nature/search.py.

The index is built once by scripts/nature/wild/build_index.py into
data/nature/wild/index/ (committed, so the Vercel function ships it).

Serving needs only numpy: BM25 is read as raw CSC arrays, and
queries are embedded by StaticEmbedder from the potion table exported into
index/query_model/ (scripts/nature/wild/export_query_model.py). scipy and
fastembed are build-time only (requirements-wild.txt).
"""

import gzip
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Literal, Optional, Protocol

import numpy as np

from app.services.nature.search import rrf
from app.services.nature.text import tokenize

if TYPE_CHECKING:
    from scipy import sparse

ROOT = Path(__file__).resolve().parents[3]
INDEX_DIR = ROOT / "data" / "nature" / "wild" / "index"
GROUPS_PATH = ROOT / "data" / "nature" / "catalogue" / "groups.json"
MODEL_CACHE = ROOT / "data" / "nature" / "wild" / ".model_cache"
QUERY_MODEL_DIR = "query_model"  # inside the index dir

DEFAULT_MODEL = "minishlab/potion-retrieval-32M"
BM25_K1, BM25_B = 1.5, 0.75
DENSE_SUMMARY_CHARS = 1000  # keep the head of the abstract (transformer models read <= 512 tokens)
DEPTH = 200  # depth of each ranked list entering the fusion
DENSE_CHUNK = 8192  # rows per block in dense_scores (~16 MB float32 temp at dim 512)
SECTOR_BOOST = {"primary": 1.15, "tag": 1.05}  # soft: sector labels are ~63% precise

# Fields kept per record in docs.json.gz (what a result needs to cite and link).
DOC_FIELDS = (
    "id", "title", "url", "doi", "year", "kind", "source", "primary_sector", "sector_tags",
    "area", "functions", "organism", "habitat_id", "group_ids", "summary", "summary_licence",
)


class Embedder(Protocol):
    model: str
    dim: int

    def __call__(self, texts: list[str], kind: Literal["document", "query"]) -> np.ndarray: ...


class FastEmbedder:
    """Local ONNX embedder. Documents are embedded as-is; queries get the model's
    retrieval instruction (fastembed query_embed). Output is L2-normalised float32."""

    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: Path = MODEL_CACHE,
                 batch_size: int = 256, threads: Optional[int] = None):
        from fastembed import TextEmbedding

        self.model = model
        try:
            # Cached copy first: the online check alone costs ~30 s per start.
            self._m = TextEmbedding(model_name=model, cache_dir=str(cache_dir), threads=threads,
                                    local_files_only=True)
        except Exception:  # not downloaded yet
            self._m = TextEmbedding(model_name=model, cache_dir=str(cache_dir), threads=threads)
        self.batch_size = batch_size
        self.dim = int(np.asarray(next(iter(self._m.embed(["probe"])))).shape[0])

    def __call__(self, texts: list[str], kind: Literal["document", "query"]) -> np.ndarray:
        if kind == "query":
            vecs = list(self._m.query_embed(texts))
        else:
            vecs = list(self._m.embed(texts, batch_size=self.batch_size))
        return normalise_rows(np.asarray(vecs, dtype=np.float32).reshape(len(texts), -1))


class StaticEmbedder:
    """potion without onnxruntime: mean of the tokens' embedding rows ([UNK]
    skipped), L2-normalised, exactly what the ONNX graph computes. Reads the files
    written by scripts/nature/wild/export_query_model.py."""

    def __init__(self, model_dir: Path):
        from app.services.wild.wordpiece import WordPiece

        cfg = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
        self.model = cfg["model"]
        self.dim = int(cfg["dim"])
        self.unk_id = int(cfg["unk_id"])
        self.max_tokens = int(cfg["max_tokens"])
        self.table = np.load(model_dir / "embeddings.npy")
        self.scales = np.load(model_dir / "scales.npy")
        self.tok = WordPiece(model_dir / "tokenizer.json")

    def __call__(self, texts: list[str], kind: Literal["document", "query"]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            ids = np.asarray([t for t in self.tok.encode(text, self.max_tokens) if t != self.unk_id],
                             dtype=np.int64)
            if ids.size:
                out[i] = (self.table[ids].astype(np.float32) * self.scales[ids, None]).sum(axis=0)
        return normalise_rows(out)


def default_embedder(model: str, index_dir: Path) -> "Embedder":
    """The exported numpy model when the index carries it, else fastembed (local builds)."""
    model_dir = index_dir / QUERY_MODEL_DIR
    if (model_dir / "model.json").exists():
        return StaticEmbedder(model_dir)
    return FastEmbedder(model)


def normalise_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0, 1.0, norms)


# ---------------------------------------------------------------- documents

def _words(ids: Iterable[str]) -> str:
    return " ".join(i.replace("-", " ") for i in ids or [])


def doc_fields(record: dict) -> dict:
    return {k: record[k] for k in DOC_FIELDS if record.get(k) not in (None, [], "")}


def organism_name(doc: dict) -> str:
    """Catalogue organisms are {name, gbif_key, rank}."""
    return ((doc.get("organism") or {}).get("name") or "").strip()


def dense_text(doc: dict) -> str:
    """What the embedding model reads: title, organism and the head of the abstract."""
    parts = [doc["title"], organism_name(doc), (doc.get("summary") or "")[:DENSE_SUMMARY_CHARS]]
    return ". ".join(p.strip() for p in parts if p and p.strip())


def sparse_text(doc: dict) -> str:
    """What BM25 reads: the title twice (titles are the only text on 85% of records),
    organism, full abstract, function and design-area words."""
    parts = [doc["title"], doc["title"], organism_name(doc), doc.get("summary") or "",
             _words(doc.get("functions")), (doc.get("area") or "").replace("-", " ")]
    return " ".join(p for p in parts if p)


def text_hash(texts: list[str], model: str) -> str:
    h = hashlib.sha256(model.encode("utf-8"))
    for t in texts:
        h.update(b"\x00" + t.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------- BM25

def build_bm25(texts: list[str], k1: float = BM25_K1, b: float = BM25_B) -> tuple["sparse.csc_matrix", list[str]]:
    """Return (n_docs x n_terms CSC matrix of BM25 weights, vocabulary in column order)."""
    from scipy import sparse

    vocab: dict[str, int] = {}
    rows, cols, tfs, lens = [], [], [], []
    for i, text in enumerate(texts):
        toks = tokenize(text)
        lens.append(len(toks))
        counts: dict[int, int] = {}
        for t in toks:
            j = vocab.setdefault(t, len(vocab))
            counts[j] = counts.get(j, 0) + 1
        rows += [i] * len(counts)
        cols += counts.keys()
        tfs += counts.values()
    n = len(texts)
    rows_a = np.asarray(rows, dtype=np.int32)
    cols_a = np.asarray(cols, dtype=np.int32)
    tf = np.asarray(tfs, dtype=np.float32)
    dl = np.asarray(lens, dtype=np.float32)
    avgdl = float(dl.mean()) if n and dl.sum() else 1.0
    df = np.bincount(cols_a, minlength=len(vocab)).astype(np.float32)
    idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
    norm = k1 * (1.0 - b + b * dl[rows_a] / avgdl)
    w = idf[cols_a] * tf * (k1 + 1.0) / (tf + norm)
    mat = sparse.csc_matrix((w.astype(np.float32), (rows_a, cols_a)), shape=(n, len(vocab)))
    terms = [""] * len(vocab)
    for t, j in vocab.items():
        terms[j] = t
    return mat, terms


# ---------------------------------------------------------------- vectors

def quantise(mat: np.ndarray) -> tuple[np.ndarray, float]:
    """int8 with one global scale. Rows are unit vectors, so components are < 1."""
    peak = float(np.abs(mat).max()) or 1.0
    scale = 127.0 / peak
    return np.clip(np.rint(mat * scale), -127, 127).astype(np.int8), scale


# ---------------------------------------------------------------- write / load

def write_index(out: Path, docs: list[dict], vectors: np.ndarray, embedder_model: str,
                hash_: str, extra_meta: Optional[dict] = None) -> dict:
    from scipy import sparse

    out.mkdir(parents=True, exist_ok=True)
    if vectors.shape[0] != len(docs):
        raise ValueError(f"{vectors.shape[0]} vectors for {len(docs)} documents")
    q, scale = quantise(normalise_rows(vectors.astype(np.float32)))
    np.save(out / "vectors.npy", q)
    bm, terms = build_bm25([sparse_text(d) for d in docs])
    sparse.save_npz(out / "bm25.npz", bm, compressed=True)
    (out / "vocab.json").write_text(json.dumps(terms, ensure_ascii=False), encoding="utf-8")
    with gzip.open(out / "docs.json.gz", "wt", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, separators=(",", ":"))
    meta = {
        "model": embedder_model,
        "dim": int(vectors.shape[1]),
        "count": len(docs),
        "scale": scale,
        "text_hash": hash_,
        "bm25": {"k1": BM25_K1, "b": BM25_B, "terms": len(terms)},
        **(extra_meta or {}),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_groups(path: Path = GROUPS_PATH) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {g["id"]: {"id": g["id"], "name": g["name"], "donate_url": g["donate_url"]}
            for g in data["groups"]}


class Bm25Columns:
    """The CSC arrays of bm25.npz (scipy's save_npz format) read with numpy alone:
    a query only ever sums a few columns, so scipy is not needed to serve."""

    def __init__(self, path: Path):
        with np.load(path) as z:
            fmt = z["format"].item()
            if (fmt.decode() if isinstance(fmt, bytes) else fmt) != "csc":
                raise ValueError(f"{path} is not a CSC matrix; rebuild the index")
            self.data, self.indices, self.indptr = z["data"], z["indices"], z["indptr"]
            self.shape = tuple(int(n) for n in z["shape"])

    def column_sum(self, cols: list[int]) -> np.ndarray:
        out = np.zeros(self.shape[0], dtype=np.float32)
        for j in cols:
            lo, hi = self.indptr[j], self.indptr[j + 1]
            out[self.indices[lo:hi]] += self.data[lo:hi]  # rows are unique within a column
        return out


class WildIndex:
    def __init__(self, index_dir: Path = INDEX_DIR, embedder: Optional[Embedder] = None,
                 groups: Optional[dict[str, dict]] = None,
                 embedder_factory: Optional[Callable[[str], Embedder]] = None):
        self.meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        self.vectors = np.load(index_dir / "vectors.npy")
        self.bm25 = Bm25Columns(index_dir / "bm25.npz")
        terms = json.loads((index_dir / "vocab.json").read_text(encoding="utf-8"))
        self.vocab = {t: j for j, t in enumerate(terms)}
        with gzip.open(index_dir / "docs.json.gz", "rt", encoding="utf-8") as f:
            self.docs: list[dict] = json.load(f)
        if not (len(self.docs) == self.vectors.shape[0] == self.bm25.shape[0] == self.meta["count"]):
            raise ValueError("index files disagree on the record count; rebuild the index")
        if embedder is None:
            embedder = (embedder_factory(self.meta["model"]) if embedder_factory
                        else default_embedder(self.meta["model"], index_dir))
        self.embedder = embedder
        if self.embedder.model != self.meta["model"]:
            raise ValueError(f"index built with {self.meta['model']}, embedder is {self.embedder.model}")
        self.groups = groups if groups is not None else load_groups()

    def __len__(self) -> int:
        return len(self.docs)

    def bm25_scores(self, query: str) -> np.ndarray:
        cols = sorted({self.vocab[t] for t in tokenize(query) if t in self.vocab})
        if not cols:
            return np.zeros(len(self.docs), dtype=np.float32)
        return self.bm25.column_sum(cols)

    def dense_scores(self, query: str) -> np.ndarray:
        # Score in row blocks: a whole-matrix `int8 @ float32` would upcast all ~99k x dim
        # vectors to a temporary float32 copy (about 200 MB at dim 512) on every query.
        q = np.asarray(self.embedder([query], "query")[0], dtype=np.float32)
        n = self.vectors.shape[0]
        out = np.empty(n, dtype=np.float32)
        for start in range(0, n, DENSE_CHUNK):
            out[start:start + DENSE_CHUNK] = self.vectors[start:start + DENSE_CHUNK] @ q
        return out / self.meta["scale"]

    @staticmethod
    def _top(scores: np.ndarray, depth: int, positive_only: bool) -> list[int]:
        pos = np.flatnonzero(scores > 0) if positive_only else np.arange(scores.size)
        if pos.size == 0:
            return []
        k = min(depth, pos.size)
        part = pos[np.argpartition(-scores[pos], k - 1)[:k]]
        return part[np.lexsort((part, -scores[part]))].tolist()

    def search(self, query: str, k: int = 10, sector: Optional[str] = None,
               depth: int = DEPTH) -> list[dict]:
        """Top-k records for a free-text query. `sector` is a soft boost, never a filter."""
        query = (query or "").strip()
        if not query:
            return []
        sparse_rank = self._top(self.bm25_scores(query), depth, positive_only=True)
        dense_rank = self._top(self.dense_scores(query), depth, positive_only=False)
        fused = rrf([sparse_rank, dense_rank])
        if sector:
            for i in fused:
                d = self.docs[i]
                if d.get("primary_sector") == sector:
                    fused[i] *= SECTOR_BOOST["primary"]
                elif sector in (d.get("sector_tags") or []):
                    fused[i] *= SECTOR_BOOST["tag"]
        order = sorted(fused, key=lambda i: (-fused[i], self.docs[i]["id"]))[:k]
        if not order:
            return []
        top = fused[order[0]]
        s_pos = {i: r + 1 for r, i in enumerate(sparse_rank)}
        d_pos = {i: r + 1 for r, i in enumerate(dense_rank)}
        hits = []
        for i in order:
            doc = self.docs[i]
            hits.append({
                **doc,
                "score": round(fused[i] / top, 4),
                "ranks": {"bm25": s_pos.get(i), "dense": d_pos.get(i)},
                "groups": [self.groups[g] for g in doc.get("group_ids", []) if g in self.groups],
            })
        return hits
