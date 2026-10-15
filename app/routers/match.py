"""
Sylva — /farm/match + /species/* router
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.models.farm import (
    FarmProfile,
    FarmProfileRequest,
    NDVIProfile,
    SoilProfile,
    TopsoilSummary,
)
from app.models.match import FarmMatchRequest, FarmMatchResponse, SpeciesMatchScore
from app.routers.farm import PROFILE_TIMEOUT_S, _build_profile
from app.services.species_match_service import SpeciesMatchService

LOG = logging.getLogger("sylva.router.match")
router = APIRouter()
match_service = SpeciesMatchService()


@router.post("/farm/match", response_model=FarmMatchResponse)
async def get_farm_match(req: FarmMatchRequest):
    """Combined farm profile + species matching."""
    try:
        profile_req = FarmProfileRequest(
            lat=req.lat,
            lon=req.lon,
            radius_km=req.radius_km,
            country=req.country,
        )
        farm_profile = await asyncio.wait_for(
            _build_profile(profile_req), timeout=PROFILE_TIMEOUT_S
        )

        if farm_profile.soil is None or farm_profile.soil.topsoil.ph is None:
            raise HTTPException(
                status_code=422,
                detail="No soil pH for this location — cannot match species.",
            )

        matches = match_service.match_species(
            farm=farm_profile,
            requested_uses=req.requested_uses,
            top_n=req.top_n,
        )
        return FarmMatchResponse(farm_profile=farm_profile, matches=matches)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504, detail=f"farm profile timed out after {PROFILE_TIMEOUT_S}s"
        )


@router.get("/species/search", response_model=list[SpeciesMatchScore])
async def search_species(
    soil_ph: Optional[float] = None,
    ndvi_health: Optional[float] = None,
    soil_texture: Optional[str] = None,
    uses: Optional[str] = None,
    top_n: int = Query(default=10, ge=1, le=50),
):
    """Direct species search bypassing farm profiling."""
    soil_profile = None
    if soil_ph is not None or soil_texture is not None:
        soil_profile = SoilProfile(
            source="mock",
            topsoil=TopsoilSummary(
                ph=soil_ph,
                texture_class=soil_texture,
                organic_carbon_g_kg=None,
                nitrogen_g_kg=None,
                clay_pct=None,
                sand_pct=None,
                silt_pct=None,
                bulk_density_kg_dm3=None,
                cec_cmol_kg=None,
            ),
        )

    ndvi_profile = None
    if ndvi_health is not None:
        ndvi_profile = NDVIProfile(
            source="mock",
            n_observations=1,
            mean_ndvi=ndvi_health,
            min_ndvi=0.0,
            max_ndvi=1.0,
            health_score=max(0.0, min(1.0, ndvi_health)),
            health_label="mock",
            outliers_removed=0,
        )

    mock_profile = FarmProfile(
        lat=0.0,
        lon=0.0,
        radius_km=1.0,
        country=None,
        bbox={"south": 0, "north": 0, "west": 0, "east": 0},
        soil=soil_profile,
        ndvi=ndvi_profile,
    )

    requested_uses = [u.strip() for u in uses.split(",") if u.strip()] if uses else []
    try:
        return match_service.match_species(
            farm=mock_profile,
            requested_uses=requested_uses,
            top_n=top_n,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/species/reload")
async def reload_species_db():
    """Reload the species database from disk."""
    try:
        n = match_service.reload_db()
        return {"status": "success", "species_count": n}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
