# Sylva-wild search index

A local, key-free search index over every record of the Open Nature Catalogue
(`data/nature/catalogue/`), for the Sylva-wild page: a user describes their sector and
what they want to build, and gets matching studies with links and, where the record
names a habitat or flagship organism, donation links to groups that protect it.

This build covers **99,309 of 99,309 records** (catalogue of 2026-10-11, built 2026-10-14).

## How it works

- **Keyword search (BM25)** over title (counted twice), organism, abstract, function
  and design-area words. Stored as a sparse matrix of precomputed weights
  (`bm25.npz`, 83,380 terms).
- **Meaning search (dense vectors)** from `minishlab/potion-retrieval-32M`, a static
  embedding model run locally with fastembed (no API key). It reads the title, the
  organism and the first 1,000 characters of the abstract. Vectors are stored as int8
  with one global scale (`vectors.npy`, 512 dimensions).
- **Fusion:** reciprocal-rank fusion (k=60) of the top 200 of each list, as in the
  nature library search (`app/services/nature/search.py`).
- **Sector** is a soft boost (x1.15 for the primary sector, x1.05 for a tagged one),
  never a filter, because the catalogue's sector labels are about 63% precise.

`BAAI/bge-small-en-v1.5` is the stronger model and can be chosen with `--model`, but
on the build laptop's CPU it ran at 5 to 35 records a second (about 5 hours for the
catalogue); potion ran at about 1,600 a second (2 minutes).

## Build and query

```
pip install -r requirements-wild.txt
python -m scripts.nature.wild.build_index            # needs data/nature/catalogue/build/catalogue.json
python -m scripts.nature.wild.export_query_model     # numpy copy of the query model for the server
python -m scripts.nature.wild.query "reduce drag on a ship hull" --sector aerospace-transport
```

The build writes vectors in parts of 5,000 records, so an interrupted run resumes.
`--limit N` builds a trial index of the first N records. The model downloads once into
`data/nature/wild/.model_cache/` (gitignored).

**The built index is committed** (about 108 MB) so the Vercel function ships it; rebuild
and commit it again when the catalogue changes. Serving needs only numpy
(`requirements.txt`): BM25 is read as raw CSC arrays, and a query is embedded by a
plain-Python WordPiece tokenizer (`app/services/wild/wordpiece.py`) plus the exported
potion table, which is all the static model does. scipy, fastembed and onnx are
build-time only. `export_query_model` checks the tokenizer against `tokenizers` on every
catalogue title and abstract (0 of 114,628 differed) and the embedder against fastembed
(cosine 0.99996 or better). The top 10 results matched the fastembed ones on all 8 test
queries; on 3 of them two neighbours swapped places (int8 rounding).

On Vercel each cold start loads the index again (about 1 s); `/wild/ask` waits for it
rather than answering 503. Dependencies plus bundled data come to about 200 MB of the
250 MB function limit.

| File | Size | Content |
|---|---|---|
| `index/vectors.npy` | 48.5 MB | int8 dense vectors, one row per record |
| `index/bm25.npz` | 10.8 MB | BM25 weights, records x terms |
| `index/vocab.json` | 1.0 MB | the terms, in column order |
| `index/docs.json.gz` | 14.2 MB | per record: id, title, url, doi, year, kind, source, sectors, area, functions, organism, habitat, group ids, summary and its licence |
| `index/meta.json` | | model, dimensions, count, scale, text hash, catalogue date, build date |
| `index/query_model/embeddings.npy` | 32.3 MB | potion token table, int8, 63,091 x 512 |
| `index/query_model/scales.npy` | 0.3 MB | per-row scale of that table |
| `index/query_model/tokenizer.json` | 1.4 MB | the model's WordPiece tokenizer |
| `index/query_model/model.json` | | model name, dimensions, [UNK] id (left out of the mean), token limit |

## Measured on this build

- Query time: about 0.05 s per phrase on the laptop, after loading. Loading takes about 1 s
  with the exported query model (it took about 15 s through fastembed and onnxruntime).
- int8 against float32 vectors: 98.7% of the dense top 20 is the same (10,000-record sample,
  12 queries; worst query 95%).
- 12 test problems, one per sector: most of the top 5 results were on topic by eye. A
  short, specific query works best ("anti-icing coating for wind turbine blades" finds
  the icing project first; "coating that stops ice forming on wind turbine blades" drifts
  to wind-turbine papers in general), so the page should turn the user's text into a
  short problem statement before searching.

## Known limits

- Only 23% of the top results in the test carried an abstract; the rest are a title and
  facts, so an answer built on them must stay close to the title.
- Only 22% of the top results had a donation group, and the groups are matched to the
  record's habitat by rules, so some look loose (for example soil groups for an ant study).
- English model: Spanish or Portuguese text should be translated to English first.
- The owner skim of the catalogue's sector labels is still pending.
