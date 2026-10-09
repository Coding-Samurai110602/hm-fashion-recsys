"""Unit/API tests for operational endpoints: health, ready, version, metrics."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


class TestHealth:
    def test_health_200(self, synth_client):
        r = synth_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_no_auth_required(self, synth_client):
        """Health endpoint must respond without any authentication."""
        r = synth_client.get("/health")
        assert r.status_code == 200


class TestReady:
    def test_ready_true_after_startup(self, synth_client):
        r = synth_client.get("/ready")
        assert r.status_code == 200
        assert r.json()["ready"] is True


class TestVersion:
    def test_version_200(self, synth_client):
        r = synth_client.get("/version")
        assert r.status_code == 200
        data = r.json()
        assert "api_version" in data
        assert "bundle_as_of_week" in data
        assert "n_features" in data
        assert data["n_features"] == 68

    def test_version_bundle_as_of_week(self, synth_client):
        r = synth_client.get("/version")
        assert r.json()["bundle_as_of_week"] == 104


class TestMetrics:
    def test_metrics_200(self, synth_client):
        r = synth_client.get("/metrics")
        assert r.status_code == 200
        body = r.text
        assert "api_cache_hits_total" in body
        assert "api_cache_misses_total" in body


class TestNotReady:
    """503 before bundle is loaded (simulated by resetting state)."""

    def test_503_when_not_ready(self):
        import os

        os.environ["BUNDLE_PATH"] = "artifacts/bundle_synthetic"
        from src.api.config import get_settings

        get_settings.cache_clear()

        from src.api import deps

        old_state = dict(deps._bundle_state)
        deps._bundle_state.update({"bundle": None, "recommender": None, "ready": False})

        from src.api.main import create_app

        app = create_app()
        from fastapi.testclient import TestClient

        # Don't call lifespan — just create client without startup
        client = TestClient(app, raise_server_exceptions=False)
        # /health should still pass (no require_ready)
        r = client.get("/health")
        assert r.status_code == 200
        # /version requires ready → 503
        r = client.get("/version")
        assert r.status_code == 503

        # Restore
        deps._bundle_state.update(old_state)
