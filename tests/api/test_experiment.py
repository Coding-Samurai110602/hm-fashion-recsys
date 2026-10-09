"""API tests for experiment power and simulation endpoints."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


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
        """Power endpoint result must match src.evaluation.power.required_n_per_arm."""
        from src.evaluation.power import required_n_per_arm

        baseline_rate = 0.085
        relative_lift = 0.10
        expected_n = required_n_per_arm(baseline_rate, relative_lift)

        body = {"baseline_rate": baseline_rate, "relative_lift": relative_lift}
        r = synth_client.post("/api/v1/experiment/power", json=body)
        assert r.status_code == 200
        actual_n = r.json()["n_per_arm"]
        assert actual_n == expected_n, f"Expected n={expected_n}, got {actual_n}"


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
