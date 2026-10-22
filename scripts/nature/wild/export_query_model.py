"""
Export the potion query model into the Sylva-wild index as plain numpy files.

    python -m scripts.nature.wild.export_query_model

potion-retrieval-32M is a static model: a query vector is the mean of its tokens'
embedding rows, L2-normalised. So the server needs only the tokenizer and the
embedding table, not fastembed or onnxruntime (too big for the Vercel function).
This reads the table from the fastembed ONNX file already in the local model cache
(needs the `onnx` package, build time only) and writes, into data/nature/wild/index/:

    query_model/tokenizer.json   the model's tokenizer (1.4 MB)
    query_model/embeddings.npy   int8 table, vocab x dim (32 MB)
    query_model/scales.npy       float32 per-row scale (row = int8 * scale)
    query_model/model.json       model name, vocab size, dim, [UNK] id

Then it checks the plain-Python tokenizer (app/services/wild/wordpiece.py) against
`tokenizers` on every catalogue title and abstract, and the numpy embedder against
fastembed on sample queries.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

from app.services.wild.index import (
    DEFAULT_MODEL, INDEX_DIR, MODEL_CACHE, QUERY_MODEL_DIR, FastEmbedder, StaticEmbedder,
)
from app.services.wild.wordpiece import WordPiece

SAMPLE_QUERIES = [
    "reduce drag on a ship hull",
    "anti-icing coating for wind turbine blades",
    "passive cooling for a building in a hot dry climate",
    "water harvesting from fog in the desert",
    "self-healing concrete",
    "adhesive that works underwater",
    "Restoring mangroves to protect a coastline from storm surge",
    "filter microplastics out of wastewater",
]


def onnx_path(model: str, cache: Path) -> Path:
    repo = "models--" + model.replace("/", "--").lower() + "-onnx"
    found = sorted((cache / repo / "snapshots").glob("*/model.onnx"))
    if not found:
        raise SystemExit(f"{model} is not in {cache}; run build_index once to download it")
    return found[-1]


def quantise_rows(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    peak = np.abs(table).max(axis=1)
    scales = np.where(peak == 0, 1.0, peak / 127.0).astype(np.float32)
    q = np.clip(np.rint(table / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--index", type=Path, default=INDEX_DIR)
    args = ap.parse_args(argv)

    import onnx
    from onnx import numpy_helper

    src = onnx_path(args.model, MODEL_CACHE)
    graph = onnx.load(str(src)).graph
    table = next(numpy_helper.to_array(t) for t in graph.initializer
                 if t.name == "embeddings.weight").astype(np.float32)
    # The graph leaves the [UNK] token out of the mean (Equal(input_ids, unk)); padding is masked.
    unk_name = next(n.input[1] for n in graph.node if n.op_type == "Equal" and n.input[0] == "input_ids")
    unk_id = next(int(numpy_helper.to_array(t)) for t in graph.initializer if t.name == unk_name)

    out = args.index / QUERY_MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    q, scales = quantise_rows(table)
    np.save(out / "embeddings.npy", q)
    np.save(out / "scales.npy", scales)
    (out / "tokenizer.json").write_bytes((src.parent / "tokenizer.json").read_bytes())
    (out / "model.json").write_text(json.dumps({
        "model": args.model, "vocab": int(table.shape[0]), "dim": int(table.shape[1]),
        "unk_id": unk_id, "max_tokens": 512,
    }, indent=2), encoding="utf-8")

    from tokenizers import Tokenizer  # comes with fastembed

    ref_tok, my_tok = Tokenizer.from_file(str(out / "tokenizer.json")), WordPiece(out / "tokenizer.json")
    with gzip.open(args.index / "docs.json.gz", "rt", encoding="utf-8") as f:
        texts = [t for d in json.load(f) for t in (d["title"], d.get("summary")) if t]
    bad = sum(ref_tok.encode(t).ids != my_tok.encode(t) for t in texts)
    print(f"WordPiece against tokenizers on {len(texts)} catalogue texts: {bad} differ")
    if bad:
        return 1

    ref = FastEmbedder(args.model)(SAMPLE_QUERIES, "query")
    got = StaticEmbedder(out)(SAMPLE_QUERIES, "query")
    cos = (ref * got).sum(axis=1)
    print(f"wrote {out}: {table.shape[0]} x {table.shape[1]}, [UNK] id {unk_id}")
    print(f"cosine to fastembed on {len(cos)} queries: min {cos.min():.5f}, mean {cos.mean():.5f}")
    if cos.min() < 0.999:
        print("numpy embedder does not match fastembed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
