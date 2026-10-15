"""Regression tests for the 2026-10-15 review fixes in Sylva-wild."""

import asyncio
from types import SimpleNamespace

import numpy as np

from app.routers import wild as wild_router
from app.services import rag_service
from app.services.wild import index as wild_index
from app.services.wild.limits import RateLimiter, cache_key

from tests.wild.test_rag import BODY, client  # noqa: F401  (fixture)


def _req(xff=None, host="127.0.0.1"):
    headers = {"x-forwarded-for": xff} if xff is not None else {}
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=host))


def test_client_identity_is_the_last_forwarded_hop():
    # The caller controls everything before the proxy's own entry.
    assert wild_router._client(_req("6.6.6.6, 203.0.113.9")) == "203.0.113.9"
    assert wild_router._client(_req("203.0.113.9")) == "203.0.113.9"
    assert wild_router._client(_req("")) == "127.0.0.1"
    assert wild_router._client(_req(None, host="10.1.1.1")) == "10.1.1.1"


def _fake_gemini(calls):
    async def call(api_key, prompt):
        calls.append(prompt)
        if "search phrases" in prompt:
            return {"queries": ["ship hulls"]}, "gemini-test"
        return {"summary": "ok", "solutions": [{"title": "Riblets", "refs": ["R1"]}]}, "gemini-test"

    return call


def test_spoofed_forwarded_for_does_not_reset_the_ai_limit(client, monkeypatch):  # noqa: F811
    calls = []
    monkeypatch.setattr(rag_service, "_api_key", lambda: "AIzaFAKE")
    monkeypatch.setattr(rag_service, "_call_gemini", _fake_gemini(calls))
    monkeypatch.setattr(wild_router, "limiter", RateLimiter(limit=1))
    models = []
    for i in range(3):
        r = client.post(
            "/api/v1/wild/ask",
            json={**BODY, "problem": f"ship hull drag question number {i}"},
            headers={"x-forwarded-for": f"198.51.100.{i}, 203.0.113.9"},
        )
        models.append(r.json()["model"])
    assert models == ["gemini-test", "search only", "search only"]


def test_slow_model_falls_back_to_search_only_and_is_not_cached(client, monkeypatch):  # noqa: F811
    async def slow(api_key, prompt):
        await asyncio.sleep(5)
        return {"queries": ["ship"]}, "gemini-test"

    monkeypatch.setattr(rag_service, "_api_key", lambda: "AIzaFAKE")
    monkeypatch.setattr(rag_service, "_call_gemini", slow)
    monkeypatch.setattr(wild_router, "ASK_TIMEOUT_S", 0.05)
    r = client.post("/api/v1/wild/ask", json=BODY)
    assert r.status_code == 200
    d = r.json()
    assert d["model"] == "search only" and "too long" in d["summary"] and d["records"]
    assert wild_router.cache.get(cache_key(BODY["intent"], BODY["sector"], BODY["problem"])) is None


class _Embed:
    model = "fake"

    def __init__(self, q):
        self.q = q

    def __call__(self, texts, kind):
        return [self.q]


def test_dense_scores_in_blocks_match_the_full_product(monkeypatch):
    rng = np.random.default_rng(7)
    vectors = rng.integers(-127, 128, size=(53, 16), dtype=np.int8)
    q = rng.standard_normal(16).astype(np.float32)
    idx = wild_index.WildIndex.__new__(wild_index.WildIndex)
    idx.vectors, idx.meta, idx.embedder = vectors, {"scale": 127.0}, _Embed(q)
    monkeypatch.setattr(wild_index, "DENSE_CHUNK", 10)  # forces 6 blocks, the last one partial
    got = idx.dense_scores("anything")
    want = (vectors.astype(np.float32) @ q) / 127.0
    assert got.dtype == np.float32 and got.shape == (53,)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-4)
