"""
Sylva — RAG farm plan generation

Pipeline:
  1. matched species + farm profile in
  2. keyword retrieve agroforestry knowledge chunks
  3. call Gemini (REST) with structured JSON output
  4. fall back to a deterministic plan if no key / LLM fails

Uses the Generative Language REST API directly so we are not tied to a
specific google-genai / google.generativeai SDK version.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

import httpx

LOG = logging.getLogger("sylva.rag")

# Tried in order. As of 2026-10-22 gemini-2.5-flash answers 404 to new keys, and the
# full Flash models often answer 503 (overloaded) while the Lite ones reply in ~2 s.
GEMINI_MODELS = [
    os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]
# Per model attempt, so one slow model leaves time for the next inside a caller's budget.
GEMINI_TIMEOUT_S = 15.0

KNOWLEDGE_CORPUS: list[dict] = [
    {
        "id": "alley-cropping",
        "tags": ["spacing", "layout", "cropland", "alley"],
        "text": (
            "Alley cropping places rows of trees with wide alleys between them for "
            "annual crops. Tree rows are typically spaced 12-20 m apart, oriented "
            "east-west to reduce shading of the crop alleys."
        ),
    },
    {
        "id": "n-fixing-benefit",
        "tags": ["nitrogen", "soil", "legume", "fertiliser"],
        "text": (
            "Nitrogen-fixing trees (legumes) form root nodules with Rhizobium, "
            "adding roughly 50-200 kg N/ha/year. Interplanting them with cereals can "
            "cut fertiliser need by 30-50% once established."
        ),
    },
    {
        "id": "degraded-restoration",
        "tags": ["degraded", "restoration", "erosion", "pioneer"],
        "text": (
            "On degraded land, start with hardy pioneer species that tolerate poor "
            "soil and drought. Nitrogen-fixing, drought-tolerant species rebuild soil "
            "biology fastest and create microclimates for later plantings."
        ),
    },
    {
        "id": "establishment-phase",
        "tags": ["timeline", "cashflow", "establishment", "finance"],
        "text": (
            "Years 0-3 are establishment: trees are planted, growth is slow, little "
            "tree income while crops continue. Years 3-7 bring first diversification "
            "income. Years 7+ compound as inputs drop and tree outputs mature."
        ),
    },
    {
        "id": "mediterranean-species",
        "tags": ["mediterranean", "spain", "portugal", "italy", "greece", "drought"],
        "text": (
            "For Mediterranean climates with dry summers, olive, carob, almond, fig "
            "and stone pine are proven agroforestry species. They tolerate alkaline "
            "soils and summer drought; irrigate mainly during establishment."
        ),
    },
    {
        "id": "windbreak",
        "tags": ["shelter", "wind", "layout", "erosion"],
        "text": (
            "Shelterbelts on field edges reduce wind speed for 10-15x tree height "
            "downwind, cutting erosion and crop water stress. Use dense, wind-firm "
            "species on the windward edge."
        ),
    },
    {
        "id": "soil-ph-matching",
        "tags": ["ph", "soil", "matching"],
        "text": (
            "Match species to soil pH. Many Mediterranean trees prefer pH 6.5-8.5. "
            "Acid-loving species struggle above pH 7.5. Always confirm topsoil pH "
            "before selecting stock."
        ),
    },
    {
        "id": "intercrop-spacing",
        "tags": ["spacing", "competition", "water"],
        "text": (
            "Leave a 1-2 m cultivated buffer between tree row and crop to limit root "
            "competition for water — wider on dry sites."
        ),
    },
    {
        "id": "eu-eco-schemes",
        "tags": ["subsidy", "eu", "cap", "finance", "spain", "france", "germany"],
        "text": (
            "EU CAP eco-schemes reward climate/environment practices such as diverse "
            "rotations, cover crops and biodiversity grassland. Pair agroforestry "
            "layouts with documented soil monitoring so applications stay bankable."
        ),
    },
    {
        "id": "sensor-verification",
        "tags": ["sensor", "compliance", "hardware", "monitoring"],
        "text": (
            "Continuous soil moisture, pH, EC and temperature logs from field nodes "
            "provide verifiable practice evidence for subsidy renewals and corporate "
            "regenerative contracts — self-reporting alone is increasingly rejected."
        ),
    },
]


def retrieve(farm_profile: dict, species_names: list[str], k: int = 5) -> list[dict]:
    """Score knowledge chunks by keyword overlap with farm context + species."""
    terms: set[str] = set()

    soil = (farm_profile.get("soil") or {}).get("topsoil") or {}
    if soil.get("texture_class"):
        terms.update(re.findall(r"[a-z]+", soil["texture_class"].lower()))
    if soil.get("ph") is not None:
        terms.update(["ph", "soil", "matching"])

    ndvi = farm_profile.get("ndvi") or {}
    score = ndvi.get("health_score")
    if score is not None and score <= 0.3:
        terms.update(["degraded", "restoration", "pioneer", "erosion"])

    country = (farm_profile.get("country") or "").lower()
    if country:
        terms.add(country)
    if country in ("spain", "portugal", "italy", "greece", "france"):
        terms.update(["mediterranean", "drought", "eu", "cap", "subsidy"])
    if country in ("germany", "netherlands", "poland", "ireland"):
        terms.update(["eu", "cap", "subsidy"])
    if country in ("canada", "united states", "usa", "us"):
        terms.update(["finance", "subsidy"])

    terms.update(["spacing", "layout", "timeline", "nitrogen", "cashflow", "monitoring"])

    for name in species_names:
        for token in re.findall(r"[a-z]+", (name or "").lower()):
            if len(token) > 3:
                terms.add(token)

    scored: list[tuple[int, dict]] = []
    for chunk in KNOWLEDGE_CORPUS:
        hit = len(terms.intersection(chunk["tags"]))
        # soft boost if any query term appears in the body text
        body = chunk["text"].lower()
        hit += sum(1 for t in terms if len(t) > 4 and t in body) // 3
        if hit > 0:
            scored.append((hit, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


PLAN_PROMPT = """You are an expert agroforestry advisor writing a farm transition plan.

FARM PROFILE:
{profile}

TOP MATCHED SPECIES (already filtered to this farm's soil and climate):
{species}

RELEVANT AGROFORESTRY KNOWLEDGE:
{context}

Write a concise, practical agroforestry transition plan for this specific farm.
Structure it as JSON with these keys:
- "summary": 2-3 sentence overview of the opportunity for this farm
- "recommended_layout": which agroforestry design fits (alley cropping, silvopasture, shelterbelt, etc.) and why
- "priority_species": list of 3-5 objects {{"species": "...", "reason": "..."}} drawn ONLY from the matched list
- "phased_plan": object with "years_0_3", "years_3_7", "years_7_plus"
- "soil_notes": what the soil data means for this plan
- "next_actions": list of 3 concrete next steps the farmer can take this season

Base everything on the actual farm data and matched species above.
Do not invent species that are not in the matched list.
Output ONLY valid JSON.
"""


@dataclass
class RAGResult:
    plan: dict
    retrieved_chunks: list[str]
    model: str


def _api_key() -> str | None:
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENAI_API_KEY")
    )


async def generate_farm_plan(
    farm_profile: dict,
    matched_species: list[dict],
    top_species: int = 8,
) -> RAGResult:
    species_subset = matched_species[:top_species]
    species_names = [s.get("species", "") for s in species_subset]
    chunks = retrieve(farm_profile, species_names)
    context_text = "\n".join(f"- {c['text']}" for c in chunks) or "- (no corpus hits)"

    api_key = _api_key()
    if not api_key:
        LOG.warning("GEMINI_API_KEY not set — returning fallback plan")
        return RAGResult(
            plan=_fallback_plan(farm_profile, species_subset, chunks),
            retrieved_chunks=[c["id"] for c in chunks],
            model="fallback (no API key)",
        )

    prompt = PLAN_PROMPT.format(
        profile=json.dumps(_slim_profile(farm_profile), indent=2),
        species=json.dumps(
            [
                {
                    "species": s.get("species"),
                    "uses": s.get("uses"),
                    "nitrogen_fixer": s.get("nitrogen_fixer"),
                    "drought_tolerance": s.get("drought_tolerance"),
                    "score": s.get("total_score"),
                }
                for s in species_subset
            ],
            indent=2,
        ),
        context=context_text,
    )

    try:
        plan, model_used = await _call_gemini(api_key, prompt)
        return RAGResult(
            plan=plan,
            retrieved_chunks=[c["id"] for c in chunks],
            model=model_used,
        )
    except Exception as e:
        LOG.warning("LLM generation failed (%s) — returning fallback", e)
        return RAGResult(
            plan=_fallback_plan(farm_profile, species_subset, chunks),
            retrieved_chunks=[c["id"] for c in chunks],
            model=f"fallback (LLM error: {e})",
        )


_KEY_RE = re.compile(r"AIza[0-9A-Za-z_-]{20,}|AQ\.[0-9A-Za-z_.-]{20,}")


def _scrub(text: str, api_key: str) -> str:
    """Remove the API key (and anything shaped like a Google key) from error text."""
    if api_key:
        text = text.replace(api_key, "***")
    return _KEY_RE.sub("***", text)


async def _call_gemini(api_key: str, prompt: str) -> tuple[dict, str]:
    """Try Gemini models in order; return (parsed_json, model_name).

    The key travels in the x-goog-api-key header, never in the URL, and is
    scrubbed from every error message this raises.
    """
    # de-dupe while preserving order
    models: list[str] = []
    for m in GEMINI_MODELS:
        if m and m not in models:
            models.append(m)

    errors: list[str] = []
    async with httpx.AsyncClient(timeout=GEMINI_TIMEOUT_S) as client:
        for model in models:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent"
            )
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "responseMimeType": "application/json",
                },
            }
            try:
                resp = await client.post(url, headers={"x-goog-api-key": api_key}, json=payload)
                if resp.status_code >= 400:
                    errors.append(f"{model}: HTTP {resp.status_code} {_scrub(resp.text, api_key)[:200]}")
                    continue
                data = resp.json()
                text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                )
                if not text:
                    errors.append(f"{model}: empty response")
                    continue
                plan = _parse_json(text)
                return plan, model
            except Exception as e:
                errors.append(f"{model}: {_scrub(str(e), api_key)}")
                continue

    raise RuntimeError("; ".join(errors) or "all Gemini models failed")


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _slim_profile(profile: dict) -> dict:
    soil = (profile.get("soil") or {}).get("topsoil") or {}
    ndvi = profile.get("ndvi") or {}
    topo = profile.get("topography") or {}
    elev = (topo.get("elevation") or {}) if isinstance(topo, dict) else {}
    return {
        "country": profile.get("country"),
        "lat": profile.get("lat"),
        "lon": profile.get("lon"),
        "soil_ph": soil.get("ph"),
        "soil_texture": soil.get("texture_class"),
        "organic_carbon_g_kg": soil.get("organic_carbon_g_kg"),
        "ndvi_health_score": ndvi.get("health_score"),
        "ndvi_label": ndvi.get("health_label"),
        "elevation_mean_m": elev.get("mean_m"),
    }


def _fallback_plan(profile: dict, species: list[dict], chunks: list[dict]) -> dict:
    soil = (profile.get("soil") or {}).get("topsoil") or {}
    ndvi = profile.get("ndvi") or {}
    degraded = ndvi.get("health_score") is not None and ndvi["health_score"] <= 0.3
    layout = (
        "restoration planting with pioneer species"
        if degraded
        else "alley cropping"
    )
    return {
        "summary": (
            f"This {profile.get('country') or 'farm'} site has pH {soil.get('ph')} "
            f"{soil.get('texture_class') or ''} soil. "
            + (
                "Vegetation health is low, so the priority is soil restoration. "
                if degraded
                else "Conditions support diversified agroforestry. "
            )
            + f"{len(species)} species match this farm's profile."
        ),
        "recommended_layout": layout,
        "priority_species": [
            {
                "species": s.get("species"),
                "reason": (
                    "nitrogen-fixing, builds soil"
                    if s.get("nitrogen_fixer")
                    else f"suited to this soil; uses: {', '.join((s.get('uses') or [])[:2])}"
                ),
            }
            for s in species[:5]
        ],
        "phased_plan": {
            "years_0_3": "Establish trees, minimal tree income, crops continue as normal.",
            "years_3_7": "First diversification income from fruit/pods/thinnings; reduce fertiliser.",
            "years_7_plus": "Compounding returns as input costs fall and tree outputs mature.",
        },
        "soil_notes": (
            f"Topsoil pH {soil.get('ph')}, texture {soil.get('texture_class')}, "
            f"organic carbon {soil.get('organic_carbon_g_kg')} g/kg."
        ),
        "next_actions": [
            "Confirm topsoil pH with a lab test before ordering nursery stock.",
            "Plant 1-2 nitrogen-fixing rows on the most degraded strip first.",
            "Install the soil node and log moisture/EC through the first dry season.",
        ],
        "_note": "Generated without LLM. Set GEMINI_API_KEY for a richer plan.",
        "_retrieved": [c.get("id") for c in chunks],
    }
