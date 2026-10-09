"""Operational endpoints: health, ready, version, metrics."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from src.api.config import get_settings
from src.api.deps import get_bundle_state, require_ready
from src.api.schemas import HealthResponse, ReadyResponse, VersionResponse

logger = logging.getLogger("api")
router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health():
    return {"status": "ok"}


@router.get("/ready", response_model=ReadyResponse, summary="Readiness probe")
async def ready():
    state = get_bundle_state()
    return {"ready": state["ready"]}


@router.get(
    "/version", response_model=VersionResponse, summary="Bundle and model metadata"
)
async def version(state: dict = Depends(require_ready)):
    settings = get_settings()
    bundle = state["bundle"]
    manifest = bundle.manifest
    metrics = manifest.get("model_metrics", {})
    return {
        "api_version": settings.API_VERSION,
        "git_commit": manifest.get("git_commit", settings.GIT_COMMIT),
        "bundle_as_of_week": manifest.get("as_of_week", 104),
        "model_map12_fold103": metrics.get("map12_fold103", 0.035832),
        "model_map12_holdout": metrics.get("map12_holdout", 0.036904),
        "n_features": len(bundle.feature_names),
    }


@router.get("/metrics", response_class=PlainTextResponse, summary="Prometheus metrics")
async def metrics():
    from prometheus_client import generate_latest

    from src.api.services.recommendation import cache_stats

    stats = cache_stats()
    output = generate_latest()
    # Append custom cache counters in Prometheus text format
    extra = (
        f"# HELP api_cache_hits_total LRU cache hits\n"
        f"# TYPE api_cache_hits_total counter\n"
        f"api_cache_hits_total {stats['hits']}\n"
        f"# HELP api_cache_misses_total LRU cache misses\n"
        f"# TYPE api_cache_misses_total counter\n"
        f"api_cache_misses_total {stats['misses']}\n"
        f"# HELP api_cache_size_current Current LRU cache size\n"
        f"# TYPE api_cache_size_current gauge\n"
        f"api_cache_size_current {stats['size']}\n"
    )
    return output.decode() + extra
