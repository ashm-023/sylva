"""
Sylva - /wild router: onboarding (intent, industry, problem) in, cited
nature-inspired solutions over the whole open catalogue out.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from app.models.wild import WildAskRequest, WildAskResponse, WildMeta
from app.services import rag_service
from app.services.wild import rag
from app.services.wild.limits import AnswerCache, RateLimiter, cache_key
from app.services.wild.store import attribution, holder, sector_options

router = APIRouter(prefix="/wild")

cache = AnswerCache()
limiter = RateLimiter()

# Whole-request budget for the model path: under Vercel's 60 s function limit and the
# page's 90 s client timeout. Past it the answer falls back to search only.
ASK_TIMEOUT_S = 40

_UNAVAILABLE = {
    "loading": "The search index is loading. Try again in a few seconds.",
    "missing": "The search index is not built on this server (python -m scripts.nature.wild.build_index).",
    "error": "The search index failed to load on this server.",
}


def _client(request: Request) -> str:
    """Rate-limit identity. The *last* X-Forwarded-For entry is the address our own proxy
    (Vercel, Render) saw; earlier entries are whatever the caller sent and can be faked."""
    hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
    if hops:
        return hops[-1]
    return request.client.host if request.client else "unknown"


@router.get("/meta", response_model=WildMeta)
def meta() -> dict:
    status = holder.status
    index = holder.index
    return {
        "sectors": sector_options(),
        "record_count": len(index) if index else 0,
        "index_status": status,
        "index_built_on": index.meta.get("built_on") if index else None,
        "llm_available": bool(rag_service._api_key()),
        "notice": rag.NOTICE,
        "attribution": attribution(),
    }


@router.post("/ask", response_model=WildAskResponse)
async def ask(req: WildAskRequest, request: Request) -> dict:
    labels = {s["id"]: s["label_en"] for s in sector_options()}
    if req.sector is not None and req.sector not in labels:
        raise HTTPException(status_code=422, detail=f"Unknown industry: {req.sector}")
    index = holder.index
    if index is None:
        status = holder.status
        raise HTTPException(status_code=503, detail=_UNAVAILABLE.get(status, _UNAVAILABLE["error"]))

    key = cache_key(req.intent, req.sector, req.problem)
    cached = cache.get(key)
    if cached is not None:
        return cached

    api_key = rag_service._api_key()
    use_llm = bool(api_key) and limiter.allow(_client(request))
    reason = "no AI key on this server" if not api_key else "AI limit reached for now; search results only"
    try:
        result = await asyncio.wait_for(
            rag.ask(req, index, api_key, rag_service._call_gemini, labels,
                    use_llm=use_llm, no_llm_reason=reason),
            timeout=ASK_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        # The model was too slow; answer from search alone rather than leave the visitor waiting.
        result = await rag.ask(req, index, api_key, rag_service._call_gemini, labels,
                               use_llm=False, no_llm_reason="AI took too long; search results only")
    result["attribution"] = attribution()
    if result["model"] != "search only":  # never pin a degraded answer for an hour
        cache.put(key, result)
    return result
