"""Regression tests for the 2026-10-15 review fixes: soil budget and no-data handling,
estimate warnings, sensor read access and bounds, and graded texture matching."""

import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.models.farm import FarmProfileRequest
from app.routers import farm, sensors
from app.services import soil_service
from app.services.species_match_service import SpeciesMatchService

EMPTY = {"properties": {"layers": []}}
GOOD = {
    "properties": {
        "layers": [
            {"name": "phh2o", "unit_measure": {"d_factor": 10}, "depths": [{"label": "0-5cm", "values": {"mean": 65}}]},
        ]
    }
}


def run(coro):
    return asyncio.run(coro)


# ── soil ──────────────────────────────────────────────────────────────────────


def test_every_offset_empty_means_no_data_not_an_estimate(monkeypatch):
    async def empty(client, lat, lon, timeout=None):
        return EMPTY

    monkeypatch.setattr(soil_service, "_query", empty)
    with pytest.raises(soil_service.SoilNoDataError):
        run(soil_service.fetch_soil(38.0, 5.0, timeout=5))


def test_unreachable_soilgrids_gives_a_labelled_estimate(monkeypatch):
    async def down(client, lat, lon, timeout=None):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(soil_service, "_query", down)
    soil = run(soil_service.fetch_soil(37.9, -4.7, timeout=5, country="Spain"))
    assert soil.raw_properties["estimated"] is True and "down" in soil.raw_properties["reason"]


def test_slow_soilgrids_stops_at_the_budget_and_estimates(monkeypatch):
    async def slow_empty(client, lat, lon, timeout=None):
        await asyncio.sleep(0.3)
        return EMPTY

    monkeypatch.setattr(soil_service, "_query", slow_empty)
    monkeypatch.setattr(soil_service, "MIN_REQUEST_S", 0.2)
    t0 = time.monotonic()
    soil = run(soil_service.fetch_soil(37.9, -4.7, timeout=1))
    assert time.monotonic() - t0 < 1.5  # well inside the caller's SOIL_TIMEOUT_S + 3
    assert soil.raw_properties["estimated"] is True


def test_first_good_cell_wins(monkeypatch):
    async def good(client, lat, lon, timeout=None):
        return GOOD

    monkeypatch.setattr(soil_service, "_query", good)
    soil = run(soil_service.fetch_soil(37.9, -4.7, timeout=5))
    assert soil.source.startswith("SoilGrids") and soil.topsoil.ph == 6.5


def test_profile_warns_when_soil_is_estimated(monkeypatch):
    async def estimated(lat, lon, timeout=12, country=None, allow_estimate=True):
        return soil_service.estimate_soil(lat, lon, country, reason="SoilGrids unavailable (test)")

    async def nothing(*a, **k):
        return None

    monkeypatch.setattr(soil_service, "fetch_soil", estimated)
    for mod, fn in ((farm.gbif_service, "fetch_species"), (farm.ndvi_service, "fetch_ndvi"),
                    (farm.topo_service, "fetch_topography")):
        monkeypatch.setattr(mod, fn, nothing)
    profile = run(farm._build_profile(FarmProfileRequest(lat=37.9, lon=-4.7, country="Spain")))
    assert profile.soil is not None
    assert any("regional estimate" in w and "(test)" in w for w in profile.warnings)


# ── sensors ───────────────────────────────────────────────────────────────────


@pytest.fixture
def api(monkeypatch):
    from app.main import app

    monkeypatch.delenv("SENSOR_INGEST_TOKEN", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("SYLVA_REQUIRE_AUTH", raising=False)
    sensors._STORE.clear()
    yield TestClient(app)
    sensors._STORE.clear()


def _batch(device, n=1):
    return {"device_id": device, "samples": [{"ts_unix": i, "lat": 37.9, "lon": -4.7} for i in range(n)]}


def test_latest_bounds_n(api):
    api.post("/api/v1/sensors/ingest", json=_batch("node-01", 3))
    assert api.get("/api/v1/sensors/node-01/latest?n=0").status_code == 422
    assert api.get("/api/v1/sensors/node-01/latest?n=501").status_code == 422
    assert len(api.get("/api/v1/sensors/node-01/latest?n=2").json()["samples"]) == 2


def test_latest_needs_the_token_when_one_is_set(api, monkeypatch):
    monkeypatch.setenv("SENSOR_INGEST_TOKEN", "s3cret")
    ok = api.post("/api/v1/sensors/ingest", json=_batch("node-01"), headers={"X-Device-Token": "s3cret"})
    assert ok.status_code == 200
    assert api.get("/api/v1/sensors/node-01/latest").status_code == 401
    r = api.get("/api/v1/sensors/node-01/latest", headers={"X-Device-Token": "s3cret"})
    assert r.status_code == 200 and r.json()["samples"][0]["lat"] == 37.9


def test_store_drops_the_least_recently_updated_device(api, monkeypatch):
    monkeypatch.setattr(sensors, "MAX_DEVICES", 3)
    for d in ("node-a", "node-b", "node-c"):
        api.post("/api/v1/sensors/ingest", json=_batch(d))
    api.post("/api/v1/sensors/ingest", json=_batch("node-a"))  # a is now the freshest
    api.post("/api/v1/sensors/ingest", json=_batch("node-d"))
    assert list(sensors._STORE) == ["node-c", "node-a", "node-d"]


# ── texture ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "farm_tex, pref, score",
    [
        ("sandy clay loam", "sandy clay loam", 15.0),
        ("sandy loam", "sandy loams", 15.0),
        ("loam", "loam", 15.0),
        ("silt loam", "loam", 10.0),  # same dominant texture
        ("sandy clay loam", "silt loam", 10.0),
        ("sandy clay loam", "sand", 5.0),  # only a modifier shared
        ("sandy clay loam", "clay", 5.0),
        ("clay", "sand", 0.0),
        ("loam", "well-drained", 0.0),
    ],
)
def test_texture_overlap_is_graded(farm_tex, pref, score):
    assert SpeciesMatchService._texture_overlap(farm_tex, pref) == score


def test_texture_score_takes_the_best_preference():
    svc = SpeciesMatchService()
    assert svc._texture_score({"soil_texture_preference": ["clay", "silt loam"]}, "sandy clay loam") == 10.0
    assert svc._texture_score({"soil_texture_preference": []}, "loam") == 0.0
    assert svc._texture_score({"soil_texture_preference": ["loam"]}, None) == 0.0
