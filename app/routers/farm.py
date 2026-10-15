"""
Sylva — /farm router
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

from app.models.farm import FarmProfile, FarmProfileRequest
from app.services import gbif_service, ndvi_service, soil_service, topo_service
from app.utils.geometry import bbox_from_radius

LOG = logging.getLogger("sylva.router.farm")
router = APIRouter()

# Whole-request ceiling. Per-source caps keep one hung API from freezing the UI.
PROFILE_TIMEOUT_S = 45
SOIL_TIMEOUT_S = 10
GBIF_TIMEOUT_S = 18
TOPO_TIMEOUT_S = 15
NDVI_TIMEOUT_S = 5


async def _gather_named(tasks: dict[str, asyncio.Future]) -> dict[str, object]:
    """Run named coroutines concurrently; exceptions become values."""
    keys = list(tasks.keys())
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    return dict(zip(keys, results))


async def _build_profile(req: FarmProfileRequest) -> FarmProfile:
    bbox = bbox_from_radius(req.lat, req.lon, req.radius_km)
    errors: dict[str, str] = {}
    warnings: list[str] = []

    profile = FarmProfile(
        lat=req.lat,
        lon=req.lon,
        radius_km=req.radius_km,
        country=req.country,
        bbox=bbox,
    )

    tasks: dict[str, object] = {
        "soil": asyncio.wait_for(
            soil_service.fetch_soil(
                req.lat, req.lon, timeout=SOIL_TIMEOUT_S, country=req.country
            ),
            timeout=SOIL_TIMEOUT_S + 3,
        ),
        "species": asyncio.wait_for(
            gbif_service.fetch_species(bbox, timeout=GBIF_TIMEOUT_S),
            timeout=GBIF_TIMEOUT_S + 2,
        ),
        "ndvi": asyncio.wait_for(
            ndvi_service.fetch_ndvi(bbox),
            timeout=NDVI_TIMEOUT_S,
        ),
        # Open-Meteo — no API key required
        "topography": asyncio.wait_for(
            topo_service.fetch_topography(bbox, timeout=TOPO_TIMEOUT_S),
            timeout=TOPO_TIMEOUT_S + 2,
        ),
    }

    results = await _gather_named(tasks)

    for name, result in results.items():
        if isinstance(result, Exception):
            msg = (
                f"timed out after deadline"
                if isinstance(result, asyncio.TimeoutError)
                else str(result)
            )
            LOG.warning("%s failed: %s", name, msg)
            errors[name] = msg
            continue
        if name == "soil":
            profile.soil = result
            if (result.raw_properties or {}).get("estimated"):
                reason = result.raw_properties.get("reason") or "SoilGrids unavailable"
                warnings.append(f"Soil values are a regional estimate, not measured ({reason}).")
        elif name == "ndvi":
            profile.ndvi = result
        elif name == "species":
            profile.observed_species = result
        elif name == "topography":
            profile.topography = result

    profile.errors = errors
    profile.warnings = warnings
    return profile


@router.post("/farm/profile", response_model=FarmProfile)
async def get_farm_profile(req: FarmProfileRequest):
    """
    Full farm profile: soil + topography + NDVI vegetation health + observed
    species. Each source runs concurrently and independently — partial results
    are returned if a source fails.
    """
    try:
        return await asyncio.wait_for(_build_profile(req), timeout=PROFILE_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"farm profile timed out after {PROFILE_TIMEOUT_S}s",
        )


@router.get("/farm/profile", response_model=FarmProfile)
async def get_farm_profile_get(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(default=5.0, ge=0.5, le=100.0),
    country: str | None = Query(default=None),
):
    """GET convenience wrapper — same behaviour as POST."""
    return await get_farm_profile(
        FarmProfileRequest(lat=lat, lon=lon, radius_km=radius_km, country=country)
    )
