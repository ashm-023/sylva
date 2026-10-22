"""Sylva-wild RAG, guard, limits and /wild router (fake index, mocked Gemini)."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.models.wild import WildAskRequest
from app.routers import wild as wild_router
from app.services import rag_service
from app.services.wild import rag
from app.services.wild.limits import AnswerCache, RateLimiter, cache_key
from app.services.wild.store import holder

GROUP = {"id": "shark-trust", "name": "Shark Trust", "donate_url": "https://example.org/donate"}


def rec(i, title, sector="aerospace-transport", summary=None, organism=None, groups=()):
    d = {"id": f"oa-W{i}", "title": title, "url": f"https://doi.org/10.1/{i}", "doi": f"10.1/{i}",
         "year": 2020, "kind": "paper", "source": "openalex", "primary_sector": sector,
         "functions": ["reduce-drag"], "groups": list(groups)}
    if summary:
        d["summary"], d["summary_licence"] = summary, "CC-BY"
    if organism:
        d["organism"] = {"name": organism, "gbif_key": 1, "rank": "ORDER"}
    return d


RECORDS = [
    rec(1, "Shark skin riblets reduce drag on ship hulls", organism="shark", groups=[GROUP],
        summary="Riblets modelled on shark skin cut friction drag by 5% in tank tests."),
    rec(2, "Salvinia air layer hull coating"),
    rec(3, "Mussel-inspired antifouling surface", sector="chemicals"),
    rec(4, "Termite mound passive cooling", sector="buildings-architecture"),
]


class FakeIndex:
    """search() returns records whose title shares a word with the query (deterministic)."""

    def __init__(self, records=RECORDS):
        self.records = records
        self.calls = []
        self.meta = {"built_on": "2026-10-14"}

    def __len__(self):
        return len(self.records)

    def search(self, query, k=10, sector=None):
        self.calls.append((query, sector))
        words = set(query.lower().split())
        hits = [r for r in self.records if words & set(r["title"].lower().split())]
        return hits[:k]


def run(coro):
    return asyncio.run(coro)


def llm_returning(*answers):
    """Fake _call_gemini: returns the given dicts in order (Exception instances are raised)."""
    seq = list(answers)
    prompts = []

    async def call(api_key, prompt):
        prompts.append(prompt)
        out = seq.pop(0)
        if isinstance(out, Exception):
            raise out
        return out, "gemini-test"

    call.prompts = prompts
    return call


REQ = WildAskRequest(intent="fix", sector="aerospace-transport", problem="Our ship hulls lose fuel to drag")
LABELS = {"aerospace-transport": "Aerospace and transport", "chemicals": "Chemicals"}


# ── guard ──────────────────────────────────────────────────────────────────────


def test_guard_keeps_only_given_refs_and_drops_uncited_solutions():
    raw = {
        "summary": "Shark skin [R1, R9] helps; see R7 and https://evil.example/x",
        "solutions": [
            {"title": "Riblets", "nature_strategy": "shark skin", "why_it_fits": "less drag [R1]",
             "caveat": "", "refs": ["R1", "r2", "R1", "R99", "oa-W1"]},
            {"title": "Made up", "nature_strategy": "x", "why_it_fits": "y", "refs": ["R12"]},
            {"title": "", "refs": ["R1"]},
            "not a dict",
        ],
        "next_steps": ["Read R2", 5, "Visit www.example.com"],
    }
    out = rag.guard(raw, n=3)
    assert [s["title"] for s in out["solutions"]] == ["Riblets"]
    assert out["solutions"][0]["refs"] == ["R1", "R2"]
    assert "R9" not in out["summary"] and "R7" not in out["summary"] and "http" not in out["summary"]
    assert "[R1]" in out["summary"]
    assert out["next_steps"] == ["Read R2", "Visit"]
    assert {"R9", "R7", "R99", "oa-W1", "R12"} <= set(out["dropped_refs"])


def test_guard_handles_non_dict_and_caps_solutions():
    assert rag.guard(["x"], 3)["solutions"] == []
    many = {"solutions": [{"title": f"S{i}", "refs": ["R1"]} for i in range(9)]}
    assert len(rag.guard(many, 3)["solutions"]) == rag.MAX_SOLUTIONS


def test_clean_queries():
    assert rag.clean_queries({"queries": ["drag reduction ship hull", "Drag reduction ship hull",
                                          "  antifouling   coating ", 3, "x https://a.b", "four"]},
                             "raw") == ["drag reduction ship hull", "antifouling coating", "x"]
    assert rag.clean_queries({"queries": []}, "the raw problem") == ["the raw problem"]
    assert rag.clean_queries("nonsense", "p") == ["p"]


# ── retrieval ──────────────────────────────────────────────────────────────────


def test_retrieve_fuses_phrases_and_passes_sector():
    idx = FakeIndex()
    hits = rag.retrieve(idx, ["ship hull coating", "antifouling surface"], "aerospace-transport")
    ids = [h["id"] for h in hits]
    assert set(ids) == {"oa-W1", "oa-W2", "oa-W3"}
    assert all(sector == "aerospace-transport" for _, sector in idx.calls)


def test_records_carry_links_and_groups_from_the_data():
    out = rag.record_hits(RECORDS[:2])
    assert out[0]["ref"] == "R1" and out[1]["ref"] == "R2"
    assert out[0]["url"] == "https://doi.org/10.1/1" and out[0]["groups"] == [GROUP]
    assert out[0]["organism"] == "shark" and out[0]["has_summary"] is True
    assert out[1]["has_summary"] is False


def test_prompt_marks_title_only_records():
    text = rag.records_for_prompt(RECORDS[:2])
    assert '"abstract": "Riblets' in text and '"note": "title only"' in text
    assert "doi" not in text and "http" not in text  # the model never sees links


# ── ask ────────────────────────────────────────────────────────────────────────


def test_ask_without_key_is_search_only():
    out = run(rag.ask(REQ, FakeIndex(), None, llm_returning(), LABELS))
    assert out["model"] == "search only"
    assert out["queries"] == [REQ.problem]
    assert out["solutions"][0]["refs"] == ["R1"]
    assert out["records"][0]["id"] == "oa-W1"
    assert "no AI key" in out["summary"]


def test_ask_with_key_uses_plan_then_answer_and_guard():
    call = llm_returning(
        {"queries": ["ship hulls drag", "antifouling surface"]},
        {"summary": "Two routes [R1].", "solutions": [
            {"title": "Riblets", "nature_strategy": "shark skin", "why_it_fits": "cuts drag",
             "caveat": "lab only", "refs": ["R1"]},
            {"title": "Invented", "nature_strategy": "x", "why_it_fits": "y", "refs": ["R40"]}],
         "next_steps": ["Pilot on one hull"]},
    )
    out = run(rag.ask(REQ, FakeIndex(), "AIzaFAKE", call, LABELS))
    assert out["model"] == "gemini-test"
    assert out["queries"] == ["ship hulls drag", "antifouling surface"]
    assert [s["title"] for s in out["solutions"]] == ["Riblets"]
    assert "Aerospace and transport" in call.prompts[0] and "fix a problem" in call.prompts[0]
    assert "R1" in call.prompts[1] and "AIzaFAKE" not in "".join(call.prompts)


def test_ask_falls_back_when_answer_fails_or_cites_nothing():
    failing = llm_returning({"queries": ["ship hulls"]}, RuntimeError("boom AIzaSECRETSECRETSECRETSECRET"))
    out = run(rag.ask(REQ, FakeIndex(), "AIzaFAKE", failing, LABELS))
    assert out["model"] == "search only" and "AI step failed" in out["summary"]
    uncited = llm_returning({"queries": ["ship hulls"]}, {"summary": "hm", "solutions": []})
    out = run(rag.ask(REQ, FakeIndex(), "AIzaFAKE", uncited, LABELS))
    assert out["model"] == "search only" and "cited no study" in out["summary"]


def test_ask_plan_failure_uses_raw_problem():
    call = llm_returning(RuntimeError("plan down"), {"solutions": [{"title": "T", "refs": ["R1"]}]})
    out = run(rag.ask(REQ, FakeIndex(), "AIzaFAKE", call, LABELS))
    assert out["queries"] == [REQ.problem] and out["model"] == "gemini-test"


def test_ask_with_no_hits():
    req = WildAskRequest(intent="build", sector=None, problem="zzzz qqqq wwww")
    out = run(rag.ask(req, FakeIndex(), None, llm_returning(), LABELS))
    assert out["records"] == [] and out["solutions"] == []
    assert "No study" in out["summary"]


def test_rate_limited_ask_skips_llm():
    call = llm_returning()
    out = run(rag.ask(REQ, FakeIndex(), "AIzaFAKE", call, LABELS, use_llm=False,
                      no_llm_reason="AI limit reached"))
    assert call.prompts == [] and "AI limit reached" in out["summary"]


# ── limits ─────────────────────────────────────────────────────────────────────


def test_cache_expires_and_evicts():
    now = [0.0]
    c = AnswerCache(ttl_s=10, max_items=2, clock=lambda: now[0])
    c.put(("a",), {"v": 1})
    assert c.get(("a",)) == {"v": 1}
    now[0] = 11
    assert c.get(("a",)) is None
    c.put(("a",), {}), c.put(("b",), {}), c.put(("c",), {})
    assert c.get(("a",)) is None and c.get(("c",)) == {}
    assert cache_key("fix", None, "  Ship  HULLS ") == cache_key("fix", "", "ship hulls")


def test_rate_limiter_window():
    now = [0.0]
    lim = RateLimiter(limit=2, window_s=60, clock=lambda: now[0])
    assert lim.allow("ip") and lim.allow("ip") and not lim.allow("ip")
    assert lim.allow("other")
    now[0] = 61
    assert lim.allow("ip")


# ── router ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch):
    from app.main import app

    wild_router.cache.clear()
    wild_router.limiter.clear()
    old = (holder._index, holder._status, holder._thread)
    holder.set(FakeIndex())
    monkeypatch.setattr(rag_service, "_api_key", lambda: None)
    yield TestClient(app)
    holder._index, holder._status, holder._thread = old


BODY = {"intent": "fix", "sector": "aerospace-transport", "problem": "Our ship hulls lose fuel to drag"}


def test_meta_lists_sectors_without_catch_all(client):
    m = client.get("/api/v1/wild/meta").json()
    ids = [s["id"] for s in m["sectors"]]
    assert "other" not in ids and "aerospace-transport" in ids and len(ids) == 15
    assert m["index_status"] == "ready" and m["record_count"] == 4 and m["llm_available"] is False
    assert "Gemini" in m["notice"] and "CC BY" in m["attribution"]


def test_ask_endpoint_and_validation(client):
    r = client.post("/api/v1/wild/ask", json=BODY)
    assert r.status_code == 200
    d = r.json()
    assert d["records"][0]["groups"] == [GROUP] and d["attribution"]
    assert client.post("/api/v1/wild/ask", json={**BODY, "sector": "nope"}).status_code == 422
    assert client.post("/api/v1/wild/ask", json={**BODY, "problem": "short"}).status_code == 422
    assert client.post("/api/v1/wild/ask", json={**BODY, "intent": "buy"}).status_code == 422
    assert client.post("/api/v1/wild/ask", json={**BODY, "sector": ""}).status_code == 200


def test_ask_503_while_index_not_ready(client):
    holder._index, holder._status = None, "missing"
    r = client.post("/api/v1/wild/ask", json=BODY)
    assert r.status_code == 503 and "not built" in r.json()["detail"]


def test_llm_answers_are_cached_and_rate_limited(client, monkeypatch):
    calls = []

    async def fake_call(api_key, prompt):
        calls.append(prompt)
        if "search phrases" in prompt:
            return {"queries": ["ship hulls"]}, "gemini-test"
        return {"summary": "ok", "solutions": [{"title": "Riblets", "refs": ["R1"]}]}, "gemini-test"

    monkeypatch.setattr(rag_service, "_api_key", lambda: "AIzaFAKE")
    monkeypatch.setattr(rag_service, "_call_gemini", fake_call)
    first = client.post("/api/v1/wild/ask", json=BODY).json()
    again = client.post("/api/v1/wild/ask", json={**BODY, "problem": "our ship HULLS lose fuel to drag"}).json()
    assert first["model"] == again["model"] == "gemini-test" and len(calls) == 2  # second was cached

    monkeypatch.setattr(wild_router, "limiter", RateLimiter(limit=0))
    limited = client.post("/api/v1/wild/ask", json={**BODY, "problem": "a different hull question"}).json()
    assert limited["model"] == "search only" and "limit" in limited["summary"] and len(calls) == 2
    assert "AIzaFAKE" not in str(limited)


def test_ask_waits_for_an_index_still_loading(client):
    import threading

    gate = threading.Event()

    def load():
        gate.wait(5)
        holder._index, holder._status = FakeIndex(), "ready"

    holder._index, holder._status = None, "loading"
    holder._thread = threading.Thread(target=load, daemon=True)
    holder._thread.start()
    threading.Timer(0.2, gate.set).start()
    assert client.post("/api/v1/wild/ask", json=BODY).status_code == 200
