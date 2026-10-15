"""
Sylva — sensor ingest stub for ESP32 soil nodes.

Accepts offline-buffered batches from field hardware. Persistence is
in-memory for the prototype; swap for Postgres/Timescale later.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

LOG = logging.getLogger("sylva.router.sensors")
router = APIRouter()

# Prototype store: device_id → list of samples, least recently updated device first.
_STORE: dict[str, list[dict[str, Any]]] = {}
MAX_SAMPLES_PER_DEVICE = 5000
MAX_DEVICES = 200  # memory bound; the least recently updated device is dropped past this
MAX_LATEST = 500


def _check_token(provided: str | None) -> None:
    """Same rule for writing and reading: samples carry field coordinates.

    Requires SENSOR_INGEST_TOKEN in production (Vercel / SYLVA_REQUIRE_AUTH); locally
    without a configured token, access stays open for bench testing.
    """
    expected = os.getenv("SENSOR_INGEST_TOKEN", "").strip()
    require = (
        bool(expected)
        or os.getenv("VERCEL")
        or os.getenv("SYLVA_REQUIRE_AUTH", "").strip().lower() in {"1", "true", "yes", "on"}
    )
    if require:
        if not expected:
            raise HTTPException(
                status_code=503,
                detail="SENSOR_INGEST_TOKEN is not configured on the server",
            )
        if provided != expected:
            raise HTTPException(status_code=401, detail="invalid device token")


class SensorSample(BaseModel):
    ts_unix: int
    moisture_raw: Optional[int] = None
    moisture_pct: Optional[float] = None
    temp_c: Optional[float] = None
    ec_us_cm: Optional[float] = None
    battery_v: Optional[float] = None
    rssi: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class SensorBatch(BaseModel):
    device_id: str = Field(..., min_length=3, max_length=64)
    firmware: str = "0.1.0"
    samples: list[SensorSample] = Field(..., min_length=1, max_length=200)


@router.post("/sensors/ingest")
async def ingest_batch(
    batch: SensorBatch,
    x_device_token: str | None = Header(default=None),
    x_ingest_token: str | None = Header(default=None),
):
    """
    Accept a batch of soil-node readings.
    Requires SENSOR_INGEST_TOKEN in production (Vercel / SYLVA_REQUIRE_AUTH).
    """
    _check_token(x_device_token or x_ingest_token)

    # Re-insert so dict order tracks recency, then drop the stalest devices past the cap.
    bucket = _STORE.pop(batch.device_id, [])
    bucket.extend(s.model_dump() for s in batch.samples)
    _STORE[batch.device_id] = bucket[-MAX_SAMPLES_PER_DEVICE:]
    while len(_STORE) > MAX_DEVICES:
        dropped = next(iter(_STORE))
        del _STORE[dropped]
        LOG.warning("sensor store full: dropped least recently updated device %s", dropped)

    LOG.info("ingest %s: +%d samples (total %d)", batch.device_id, len(batch.samples), len(_STORE[batch.device_id]))
    return {
        "status": "ok",
        "device_id": batch.device_id,
        "accepted": len(batch.samples),
        "stored_total": len(_STORE[batch.device_id]),
    }


@router.get("/sensors/{device_id}/latest")
async def latest(
    device_id: str,
    n: int = Query(20, ge=1, le=MAX_LATEST),
    x_device_token: str | None = Header(default=None),
    x_ingest_token: str | None = Header(default=None),
):
    _check_token(x_device_token or x_ingest_token)
    rows = _STORE.get(device_id) or []
    return {"device_id": device_id, "samples": rows[-n:]}
