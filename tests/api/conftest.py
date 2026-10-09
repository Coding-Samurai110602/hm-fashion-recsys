"""Pytest fixtures for API tests: synthetic bundle TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

SYNTHETIC_BUNDLE = "artifacts/bundle_synthetic"
REAL_BUNDLE = "artifacts/bundle_week104"


@pytest.fixture(scope="module")
def synth_client():
    """TestClient backed by the synthetic bundle (no Kaggle data required)."""
    import os

    os.environ["BUNDLE_PATH"] = SYNTHETIC_BUNDLE

    # Reload settings and deps with new env
    from src.api.config import get_settings

    get_settings.cache_clear()

    # Reload dep state
    from src.api import deps

    deps._bundle_state["bundle"] = None
    deps._bundle_state["recommender"] = None
    deps._bundle_state["ready"] = False

    from src.api.main import create_app

    app = create_app()

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest.fixture(scope="module")
def real_client():
    """TestClient backed by the real bundle (skipped if bundle absent)."""
    import os
    from pathlib import Path

    if not Path(REAL_BUNDLE).exists():
        pytest.skip("Real bundle not present")

    os.environ["BUNDLE_PATH"] = REAL_BUNDLE
    from src.api.config import get_settings

    get_settings.cache_clear()

    from src.api import deps

    deps._bundle_state["bundle"] = None
    deps._bundle_state["recommender"] = None
    deps._bundle_state["ready"] = False

    from src.api.main import create_app

    app = create_app()

    with TestClient(app) as client:
        yield client
