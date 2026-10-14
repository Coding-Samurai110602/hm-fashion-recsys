"""API tests for insight endpoints backed by report JSON files."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.api, pytest.mark.filterwarnings("ignore")]

# Required fields for InsightsSummary (all must be non-null on any bundle).
_SUMMARY_REQUIRED = {
    "holdout_map12_ranker",
    "holdout_map12_heuristic",
    "holdout_lift_pct",
    "fold103_ranker_map12",
    "fold103_heuristic_map12",
    "fold103_ci_lo",
    "fold103_ci_hi",
}


class TestInsightsSummary:
    def test_summary_200(self, synth_client):
        r = synth_client.get("/api/v1/insights/summary")
        assert r.status_code == 200

    def test_summary_has_holdout_week(self, synth_client):
        r = synth_client.get("/api/v1/insights/summary")
        data = r.json()
        assert "holdout_week" in data

    def test_summary_no_null_metric_fields_synthetic(self, synth_client):
        """All required metric fields must be present and non-null on the synthetic bundle."""
        r = synth_client.get("/api/v1/insights/summary")
        data = r.json()
        for field in _SUMMARY_REQUIRED:
            assert field in data, f"Missing field: {field}"
            assert data[field] is not None, f"Null field: {field}"

    def test_summary_field_names_match_frontend_types(self, synth_client):
        """Old field names must no longer appear (they caused the blank-page crash)."""
        r = synth_client.get("/api/v1/insights/summary")
        data = r.json()
        assert "primary_map12" not in data
        assert "ranker_map12_fold103" not in data
        assert "heuristic_map12_fold103" not in data
        assert "bootstrap_ci_vs_heuristic" not in data


class TestInsightsWeeklyLift:
    def test_weekly_lift_200(self, synth_client):
        r = synth_client.get("/api/v1/insights/weekly-lift")
        assert r.status_code == 200

    def test_weekly_lift_returns_list(self, synth_client):
        r = synth_client.get("/api/v1/insights/weekly-lift")
        assert isinstance(r.json(), list)


class TestInsightsSegments:
    def test_segments_200(self, synth_client):
        r = synth_client.get("/api/v1/insights/segments")
        assert r.status_code == 200

    def test_segments_returns_list(self, synth_client):
        r = synth_client.get("/api/v1/insights/segments")
        assert isinstance(r.json(), list)


class TestInsightsAblations:
    def test_ablations_200(self, synth_client):
        r = synth_client.get("/api/v1/insights/ablations")
        assert r.status_code == 200

    def test_ablations_returns_list(self, synth_client):
        r = synth_client.get("/api/v1/insights/ablations")
        assert isinstance(r.json(), list)


class TestInsightsImportance:
    def test_importance_200(self, synth_client):
        r = synth_client.get("/api/v1/insights/importance")
        assert r.status_code == 200

    def test_importance_returns_list(self, synth_client):
        """Importance endpoint must return ImportanceEntry[] (array), not a dict."""
        r = synth_client.get("/api/v1/insights/importance")
        data = r.json()
        assert isinstance(data, list), f"Expected list, got {type(data).__name__}"

    def test_importance_entry_shape(self, synth_client):
        """When entries exist they have the required ImportanceEntry fields."""
        r = synth_client.get("/api/v1/insights/importance")
        data = r.json()
        for entry in data:
            assert "rank_shap" in entry
            assert "feature" in entry
            assert "mean_abs_shap" in entry
            assert "rank_gain" in entry


@pytest.mark.integration_real
class TestInsightsSummaryReal:
    """Integration tests against the real bundle — require real holdout_results.json."""

    def test_summary_real_holdout_map12_ranker(self, real_client):
        r = real_client.get("/api/v1/insights/summary")
        data = r.json()
        assert abs(data["holdout_map12_ranker"] - 0.036904) < 1e-4

    def test_summary_real_holdout_map12_heuristic(self, real_client):
        r = real_client.get("/api/v1/insights/summary")
        data = r.json()
        assert abs(data["holdout_map12_heuristic"] - 0.024999) < 1e-4

    def test_summary_real_ci_lo_hi(self, real_client):
        r = real_client.get("/api/v1/insights/summary")
        data = r.json()
        assert abs(data["fold103_ci_lo"] - 0.010563) < 1e-5
        assert abs(data["fold103_ci_hi"] - 0.011758) < 1e-5
