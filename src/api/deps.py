"""Bundle and service lifecycle dependencies."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Annotated

from fastapi import Depends, HTTPException

from src.api.config import Settings

logger = logging.getLogger("api")

_executor = ThreadPoolExecutor(max_workers=4)
_bundle_lock = Lock()
_bundle_state: dict = {"bundle": None, "recommender": None, "ready": False}


def get_bundle_state() -> dict:
    return _bundle_state


def load_bundle_at_startup(settings: Settings) -> None:
    """Called from app lifespan. Loads bundle + builds recommender; sets ready flag."""
    from src.api.services.experiment import set_ab_arrays
    from src.api.services.insights import set_bundle_reports
    from src.serving.bundle import load_bundle
    from src.serving.recommender import BundleRecommender

    with _bundle_lock:
        logger.info(f"Loading bundle from {settings.BUNDLE_PATH} ...")
        bundle = load_bundle(settings.BUNDLE_PATH)
        recommender = BundleRecommender(bundle)
        _bundle_state["bundle"] = bundle
        _bundle_state["recommender"] = recommender
        _bundle_state["ready"] = True
        set_bundle_reports(bundle.reports)
        set_ab_arrays(bundle.ab_arrays)
    logger.info("Bundle ready.")


def require_ready() -> dict:
    state = get_bundle_state()
    if not state["ready"]:
        raise HTTPException(
            status_code=503, detail="Service not ready — bundle loading"
        )
    return state


def get_recommender(state: Annotated[dict, Depends(require_ready)]):
    return state["recommender"]


def get_bundle(state: Annotated[dict, Depends(require_ready)]):
    return state["bundle"]


def get_executor() -> ThreadPoolExecutor:
    return _executor
