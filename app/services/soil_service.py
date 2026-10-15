"""
Sylva — soil profile service

1) SoilGrids (ISRIC) — best when reachable
2) Regional estimate — always available so matching never hard-fails on cloud hosts

SoilGrids often times out or returns nulls for urban centroids (e.g. Córdoba city).
We probe nearby farmland offsets, then fall back to a labeled estimate.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from app.models.farm import SoilProfile, TopsoilSummary
from app.utils.soil import usda_texture_class

LOG = logging.getLogger("sylva.soil")

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
# Keep the payload small — full multi-depth queries are what hang in production.
PROPERTIES = ["phh2o", "soc", "nitrogen", "bdod", "clay", "sand", "silt", "cec"]
DEPTHS = ["0-5cm"]

# Wider nudges so city-centre geocodes still hit mapped farmland (~1–12 km).
OFFSETS = [
    (0.0, 0.0),
    (0.02, 0.0),
    (-0.02, 0.0),
    (0.0, 0.02),
    (0.0, -0.02),
    (0.05, 0.0),
    (-0.05, 0.0),
    (0.0, 0.05),
    (0.08, 0.05),
    (-0.08, -0.05),
    (0.1, 0.0),
    (-0.1, 0.0),
]

# Stop probing offsets when less than this is left of the soil budget.
MIN_REQUEST_S = 1.0

HEADERS = {
    "User-Agent": "SylvaAgroforestry/0.2 (farm-profile; https://github.com/ashm-023/sylva)",
    "Accept": "application/json",
}


async def _query(
    client: httpx.AsyncClient, lat: float, lon: float, timeout: Optional[httpx.Timeout] = None
) -> dict:
    params = [("lon", lon), ("lat", lat), ("value", "mean")]
    params += [("property", p) for p in PROPERTIES]
    params += [("depth", d) for d in DEPTHS]
    kwargs = {"timeout": timeout} if timeout is not None else {}
    resp = await client.get(SOILGRIDS_URL, params=params, headers=HEADERS, **kwargs)
    resp.raise_for_status()
    return resp.json()


def _parse(data: dict) -> tuple[dict, bool]:
    parsed: dict[str, dict] = {}
    any_value = False
    for layer in data.get("properties", {}).get("layers", []):
        name = layer["name"]
        unit = layer.get("unit_measure", {})
        d_factor = unit.get("d_factor", 1) or 1
        depths_out = {}
        for d in layer.get("depths", []):
            mean = d.get("values", {}).get("mean")
            if mean is not None:
                any_value = True
                depths_out[d["label"]] = mean / d_factor
            else:
                depths_out[d["label"]] = None
        parsed[name] = {"units": unit.get("target_units"), "depths": depths_out}
    return parsed, any_value


def _topsoil_from_parsed(parsed: dict) -> TopsoilSummary:
    def top(prop: str):
        return parsed.get(prop, {}).get("depths", {}).get("0-5cm")

    ph, sand, silt, clay = top("phh2o"), top("sand"), top("silt"), top("clay")
    return TopsoilSummary(
        ph=ph,
        organic_carbon_g_kg=top("soc"),
        nitrogen_g_kg=top("nitrogen"),
        clay_pct=clay,
        sand_pct=sand,
        silt_pct=silt,
        bulk_density_kg_dm3=top("bdod"),
        cec_cmol_kg=top("cec"),
        texture_class=(
            usda_texture_class(sand, silt, clay)
            if None not in (sand, silt, clay)
            else None
        ),
    )


class SoilNoDataError(RuntimeError):
    """SoilGrids answered for every probed cell, but none had data (water, urban, unmapped)."""


def estimate_soil(
    lat: float, lon: float, country: Optional[str] = None, reason: str = "SoilGrids unavailable"
) -> SoilProfile:
    """
    Coarse regional defaults so the product still works when SoilGrids is down.
    Clearly labeled — never pretend this is a lab measurement.
    """
    c = (country or "").strip().lower()

    # Mediterranean basin
    if c in {"spain", "portugal", "italy", "greece", "france", "morocco", "tunisia", "algeria"} or (
        30 <= lat <= 46 and -10 <= lon <= 40
    ):
        top = TopsoilSummary(
            ph=7.6,
            organic_carbon_g_kg=12.0,
            nitrogen_g_kg=1.2,
            clay_pct=28.0,
            sand_pct=35.0,
            silt_pct=37.0,
            bulk_density_kg_dm3=1.35,
            cec_cmol_kg=18.0,
            texture_class="clay loam",
        )
        label = "estimated Mediterranean defaults"
    # Humid temperate / northern Europe & Canada
    elif c in {"canada", "germany", "poland", "ireland", "united kingdom", "uk"} or lat > 48:
        top = TopsoilSummary(
            ph=6.3,
            organic_carbon_g_kg=22.0,
            nitrogen_g_kg=2.0,
            clay_pct=22.0,
            sand_pct=40.0,
            silt_pct=38.0,
            bulk_density_kg_dm3=1.25,
            cec_cmol_kg=16.0,
            texture_class="loam",
        )
        label = "estimated temperate defaults"
    # East Africa highland / tropical savanna band
    elif -5 <= lat <= 5 or c in {"kenya", "uganda", "tanzania", "ethiopia"}:
        top = TopsoilSummary(
            ph=6.0,
            organic_carbon_g_kg=14.0,
            nitrogen_g_kg=1.4,
            clay_pct=30.0,
            sand_pct=45.0,
            silt_pct=25.0,
            bulk_density_kg_dm3=1.3,
            cec_cmol_kg=14.0,
            texture_class="sandy clay loam",
        )
        label = "estimated tropical highland defaults"
    else:
        top = TopsoilSummary(
            ph=6.5,
            organic_carbon_g_kg=15.0,
            nitrogen_g_kg=1.5,
            clay_pct=25.0,
            sand_pct=40.0,
            silt_pct=35.0,
            bulk_density_kg_dm3=1.3,
            cec_cmol_kg=15.0,
            texture_class="loam",
        )
        label = "estimated global defaults"

    return SoilProfile(
        source=f"Regional estimate ({label}) — SoilGrids unavailable",
        topsoil=top,
        raw_properties={
            "estimated": True,
            "reason": reason,
            "lat": lat,
            "lon": lon,
            "country": country,
        },
    )


async def fetch_soil(
    lat: float,
    lon: float,
    timeout: int = 12,
    country: Optional[str] = None,
    allow_estimate: bool = True,
) -> SoilProfile:
    """Soil for a point. `timeout` is the budget for the whole call, not per request.

    Nearby offsets are probed until the budget runs out, so a slow but responding
    SoilGrids still leaves time for the regional estimate inside the caller's deadline.
    The estimate is used only when SoilGrids could not be asked everything (errors,
    timeouts, budget spent). If every offset answered with empty cells the point is
    most likely water or unmapped, and SoilNoDataError is raised instead of inventing
    soil for it.
    """
    LOG.info("SoilGrids: querying (%.4f, %.4f)", lat, lon)
    last_err: Exception | None = None
    deadline = time.monotonic() + timeout
    answered_empty = 0

    try:
        async with httpx.AsyncClient() as client:
            for dlat, dlon in OFFSETS:
                remaining = deadline - time.monotonic()
                if remaining < MIN_REQUEST_S:
                    last_err = last_err or TimeoutError(f"soil budget of {timeout}s spent")
                    break
                q_lat, q_lon = lat + dlat, lon + dlon
                limits = httpx.Timeout(remaining, connect=min(5.0, remaining))
                try:
                    data = await _query(client, q_lat, q_lon, timeout=limits)
                except Exception as e:
                    last_err = e
                    LOG.warning("SoilGrids offset (%.3f,%.3f) failed: %s", dlat, dlon, e)
                    # Don't burn the whole budget on a dead endpoint
                    if isinstance(e, (httpx.TimeoutException, httpx.ConnectError)):
                        break
                    continue

                parsed, any_value = _parse(data)
                if not any_value:
                    answered_empty += 1
                    continue

                topsoil = _topsoil_from_parsed(parsed)
                if topsoil.ph is None:
                    answered_empty += 1
                    continue

                src = "SoilGrids v2.0 (ISRIC)"
                if (q_lat, q_lon) != (lat, lon):
                    src += f" nearby ({q_lat:.4f}, {q_lon:.4f})"
                LOG.info("SoilGrids: ok — pH=%s, texture=%s", topsoil.ph, topsoil.texture_class)
                return SoilProfile(source=src, topsoil=topsoil, raw_properties=parsed)
    except Exception as e:
        last_err = e
        LOG.warning("SoilGrids failed: %s", e)

    no_data_msg = (
        "SoilGrids returned no data for this location (likely water, urban, "
        "or an unmapped cell). Try a coordinate on cultivated land."
    )
    if answered_empty == len(OFFSETS):
        raise SoilNoDataError(no_data_msg)

    if allow_estimate:
        reason = f"SoilGrids unavailable ({last_err or 'no usable cell in time'})"
        LOG.warning("%s — using regional estimate", reason)
        return estimate_soil(lat, lon, country, reason=reason)

    raise RuntimeError(no_data_msg)
