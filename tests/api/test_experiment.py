"""API tests for experiment power and simulation endpoints."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.api, pytest.mark.filterwarnings("ignore")]


class TestPower:
    def test_power_happy_path(self, synth_client):
        body = {
            "baseline_rate": 0.085,
            "relative_lift": 0.10,
            "alpha": 0.05,
            "power": 0.80,
        }
        r = synth_client.post("/api/v1/experiment/power", json=body)
        assert r.status_code == 200
        data = r.json()
        assert "n_per_arm" in data
        assert data["n_per_arm"] > 0
        assert "cohen_h" in data

    def test_power_with_weekly_traffic(self, synth_client):
        body = {
            "baseline_rate": 0.085,
            "relative_lift": 0.10,
            "weekly_traffic": 100_000,
        }
        r = synth_client.post("/api/v1/experiment/power", json=body)
        assert r.status_code == 200
        data = r.json()
        assert data.get("weeks_needed") is not None

    def test_power_with_cuped(self, synth_client):
        body = {
            "baseline_rate": 0.085,
            "relative_lift": 0.10,
            "cuped_variance_reduction": 0.026,
        }
        r = synth_client.post("/api/v1/experiment/power", json=body)
        assert r.status_code == 200
        data = r.json()
        assert data.get("n_per_arm_cuped") is not None

    def test_power_invalid_baseline(self, synth_client):
        body = {"baseline_rate": 1.5, "relative_lift": 0.10}
        r = synth_client.post("/api/v1/experiment/power", json=body)
        assert r.status_code == 422

    def test_power_invalid_lift(self, synth_client):
        body = {"baseline_rate": 0.085, "relative_lift": -0.1}
        r = synth_client.post("/api/v1/experiment/power", json=body)
        assert r.status_code == 422

    def test_power_matches_src_formula(self, synth_client):
        """Power endpoint result must match src.evaluation.power.required_n_per_arm.

        relative_lift is a fraction; the service derives p_treat = baseline * (1 + lift).
        """
        from src.evaluation.power import required_n_per_arm

        baseline_rate = 0.085
        relative_lift = 0.10
        p_treat = baseline_rate * (1.0 + relative_lift)
        expected_n = required_n_per_arm(baseline_rate, p_treat)

        body = {"baseline_rate": baseline_rate, "relative_lift": relative_lift}
        r = synth_client.post("/api/v1/experiment/power", json=body)
        assert r.status_code == 200
        actual_n = r.json()["n_per_arm"]
        assert actual_n == expected_n, f"Expected n={expected_n}, got {actual_n}"

    def test_power_lift_110pct_fraction_returns_200_not_500(self, synth_client):
        """Exact failing request body: relative_lift=1.1 (fraction = +110%).

        p_treat = 0.0857 * 2.1 = 0.17997 < 1 → must return 200, never 500.
        """
        body = {
            "baseline_rate": 0.0857,
            "relative_lift": 1.1,
            "alpha": 0.05,
            "power": 0.8,
            "weekly_traffic": 10000,
        }
        r = synth_client.post("/api/v1/experiment/power", json=body)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert data["n_per_arm"] > 0

    def test_power_p_treat_ge1_returns_422(self, synth_client):
        """baseline_rate=0.6, relative_lift=1.0 → p_treat=1.2 ≥ 1 → must be 422."""
        body = {"baseline_rate": 0.6, "relative_lift": 1.0}
        r = synth_client.post("/api/v1/experiment/power", json=body)
        assert r.status_code == 422
        data = r.json()
        assert "error" in data, f"Expected error envelope, got: {data}"

    def test_power_lift_above_schema_max_returns_422(self, synth_client):
        """relative_lift > 2 is rejected by Pydantic (le=2 constraint)."""
        body = {"baseline_rate": 0.085, "relative_lift": 2.5}
        r = synth_client.post("/api/v1/experiment/power", json=body)
        assert r.status_code == 422

    def test_power_result_contains_n_total(self, synth_client):
        """PowerResult must include n_total = n_per_arm * 2."""
        body = {"baseline_rate": 0.085, "relative_lift": 0.10}
        r = synth_client.post("/api/v1/experiment/power", json=body)
        assert r.status_code == 200
        data = r.json()
        assert "n_total" in data, "n_total missing from PowerResult"
        assert data["n_total"] == data["n_per_arm"] * 2

    def test_error_envelope_format_for_422(self, synth_client):
        """All non-200 responses must use {error: {code, message, request_id}} envelope."""
        body = {"baseline_rate": 0.6, "relative_lift": 1.0}
        r = synth_client.post("/api/v1/experiment/power", json=body)
        assert r.status_code == 422
        data = r.json()
        assert "error" in data
        err = data["error"]
        assert "code" in err
        assert "message" in err
        assert "request_id" in err

    def test_never_raw_500_from_domain_error(self):
        """Unhandled exceptions must return structured 500, not plain-text body.

        Uses raise_server_exceptions=False so the exception handlers can respond
        rather than having TestClient re-raise the exception in the test thread.
        """
        import os
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        os.environ["BUNDLE_PATH"] = "artifacts/bundle_synthetic"
        from src.api.config import get_settings

        get_settings.cache_clear()
        from src.api import deps

        deps._bundle_state["bundle"] = None
        deps._bundle_state["recommender"] = None
        deps._bundle_state["ready"] = False
        from src.api.main import create_app

        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as client, patch(
            "src.api.services.experiment.compute_power",
            side_effect=RuntimeError("synthetic crash"),
        ):
            r = client.post(
                "/api/v1/experiment/power",
                json={"baseline_rate": 0.1, "relative_lift": 0.1},
            )
        assert r.status_code == 500, f"Expected 500, got {r.status_code}: {r.text[:200]}"
        data = r.json()
        assert "error" in data, f"Raw 500 body, no envelope: {r.text[:200]}"
        assert data["error"]["code"] == "INTERNAL_ERROR"


class TestSimulate:
    def test_simulate_ab_happy(self, synth_client):
        body = {
            "test_type": "ab",
            "n_per_arm": 500,
            "n_sims": 10,
            "seed": 42,
        }
        r = synth_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 200
        data = r.json()
        assert data["test_type"] == "ab"
        assert data["n_per_arm"] == 500

    def test_simulate_aa_happy(self, synth_client):
        body = {
            "test_type": "aa",
            "n_per_arm": 200,
            "n_sims": 10,
            "seed": 99,
        }
        r = synth_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 200

    def test_simulate_n_sims_cap(self, synth_client):
        body = {"test_type": "ab", "n_per_arm": 100, "n_sims": 2001}
        r = synth_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 422

    def test_simulate_invalid_type(self, synth_client):
        body = {"test_type": "xyz", "n_per_arm": 100}
        r = synth_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 422


class TestSimulateConsistency:
    """Verify that simulations are deterministic given the same seed."""

    def test_aa_same_seed_same_result(self, synth_client):
        body = {"test_type": "aa", "n_per_arm": 100, "n_sims": 20, "seed": 42}
        r1 = synth_client.post("/api/v1/experiment/simulate", json=body)
        r2 = synth_client.post("/api/v1/experiment/simulate", json=body)
        assert r1.status_code == 200
        assert r2.status_code == 200
        d1, d2 = r1.json(), r2.json()
        assert d1["false_positive_rate"] == d2["false_positive_rate"], (
            f"A/A FPR not reproducible: {d1['false_positive_rate']} vs {d2['false_positive_rate']}"
        )

    def test_ab_same_seed_same_result(self, synth_client):
        body = {"test_type": "ab", "n_per_arm": 100, "n_sims": 20, "seed": 99}
        r1 = synth_client.post("/api/v1/experiment/simulate", json=body)
        r2 = synth_client.post("/api/v1/experiment/simulate", json=body)
        assert r1.status_code == 200
        assert r2.status_code == 200
        d1, d2 = r1.json(), r2.json()
        assert d1["empirical_power"] == d2["empirical_power"], (
            f"A/B power not reproducible: {d1['empirical_power']} vs {d2['empirical_power']}"
        )

    def test_aa_fpr_field_present(self, synth_client):
        body = {"test_type": "aa", "n_per_arm": 100, "n_sims": 10, "seed": 1}
        r = synth_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 200
        d = r.json()
        assert "false_positive_rate" in d
        assert d["false_positive_rate"] is not None or d["data_source"] == "unavailable"

    def test_ab_power_field_present(self, synth_client):
        body = {"test_type": "ab", "n_per_arm": 100, "n_sims": 10, "seed": 1}
        r = synth_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 200
        d = r.json()
        assert "empirical_power" in d

    def test_peeking_power_labeled_correctly(self, synth_client):
        """A/B with peeking=True → peeking_power is set; peeking_fpr is None."""
        body = {
            "test_type": "ab",
            "n_per_arm": 50,
            "n_sims": 5,
            "seed": 42,
            "peeking": True,
        }
        r = synth_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 200
        d = r.json()
        # A/B peeking → peeking_power populated; peeking_fpr stays None
        assert d.get("peeking_power") is not None
        assert d.get("peeking_fpr") is None

    def test_peeking_fpr_labeled_correctly_aa(self, synth_client):
        """A/A with peeking=True → peeking_fpr is set (FPR under null); peeking_power is None."""
        body = {
            "test_type": "aa",
            "n_per_arm": 50,
            "n_sims": 5,
            "seed": 42,
            "peeking": True,
        }
        r = synth_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 200
        d = r.json()
        # A/A peeking → peeking_fpr populated; peeking_power stays None
        assert d.get("peeking_fpr") is not None
        assert d.get("peeking_power") is None
