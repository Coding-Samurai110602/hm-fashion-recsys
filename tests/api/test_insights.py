"""API tests for insight endpoints backed by report JSON files."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


class TestInsightsSummary:
    def test_summary_200(self, synth_client):
        r = synth_client.get("/api/v1/insights/summary")
        assert r.status_code == 200

    def test_summary_has_holdout_week(self, synth_client):
        r = synth_client.get("/api/v1/insights/summary")
        data = r.json()
        assert "holdout_week" in data


class TestInsightsWeeklyLift:
    def test_weekly_lift_200(self, synth_client):
        r = synth_client.get("/api/v1/insights/weekly-lift")
        assert r.status_code == 200


class TestInsightsSegments:
    def test_segments_200(self, synth_client):
        r = synth_client.get("/api/v1/insights/segments")
        assert r.status_code == 200


class TestInsightsAblations:
    def test_ablations_200(self, synth_client):
        r = synth_client.get("/api/v1/insights/ablations")
        assert r.status_code == 200


class TestInsightsImportance:
    def test_importance_200(self, synth_client):
        r = synth_client.get("/api/v1/insights/importance")
        assert r.status_code == 200
        data = r.json()
        assert "top10_gain_importance" in data
        assert "top10_shap_all_candidates" in data
