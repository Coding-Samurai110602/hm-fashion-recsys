"""Unit tests for API services, schemas, and error handling.

Markers:
  unit - pure unit tests, no server startup required
"""

from __future__ import annotations

from datetime import date

import pytest

pytestmark = [pytest.mark.filterwarnings("ignore"), pytest.mark.unit]


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaValidation:
    def test_custom_history_future_date_rejected(self):
        from pydantic import ValidationError

        from src.api.schemas import CustomHistoryItem

        with pytest.raises(ValidationError):
            CustomHistoryItem(article_id="0108775015", t_dat=date(2025, 1, 1))

    def test_custom_history_valid_date_accepted(self):
        from src.api.schemas import CustomHistoryItem

        item = CustomHistoryItem(article_id="0108775015", t_dat=date(2020, 8, 1))
        assert item.t_dat == date(2020, 8, 1)

    def test_power_request_baseline_gt_0(self):
        from pydantic import ValidationError

        from src.api.schemas import PowerRequest

        with pytest.raises(ValidationError):
            PowerRequest(baseline_rate=0.0, relative_lift=0.1)

    def test_power_request_baseline_lt_1(self):
        from pydantic import ValidationError

        from src.api.schemas import PowerRequest

        with pytest.raises(ValidationError):
            PowerRequest(baseline_rate=1.0, relative_lift=0.1)

    def test_simulate_request_n_sims_cap(self):
        from pydantic import ValidationError

        from src.api.schemas import SimulateRequest

        with pytest.raises(ValidationError):
            SimulateRequest(test_type="ab", n_per_arm=100, n_sims=2001)

    def test_simulate_request_type_pattern(self):
        from pydantic import ValidationError

        from src.api.schemas import SimulateRequest

        with pytest.raises(ValidationError):
            SimulateRequest(test_type="invalid", n_per_arm=100)


# ─────────────────────────────────────────────────────────────────────────────
# Error envelope
# ─────────────────────────────────────────────────────────────────────────────


class TestErrorEnvelope:
    def test_make_error_shape(self):
        from src.api.errors import make_error

        env = make_error("NOT_FOUND", "Customer not found", "req-123")
        assert "error" in env
        assert env["error"]["code"] == "NOT_FOUND"
        assert env["error"]["message"] == "Customer not found"
        assert env["error"]["request_id"] == "req-123"

    def test_make_error_empty_request_id(self):
        from src.api.errors import make_error

        env = make_error("VALIDATION_ERROR", "bad input")
        assert env["error"]["request_id"] == ""


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


class TestConfig:
    def test_settings_defaults(self):
        import os

        os.environ["BUNDLE_PATH"] = "artifacts/bundle_synthetic"
        from src.api.config import Settings

        s = Settings()
        assert s.CACHE_SIZE == 512
        assert s.MAX_CUSTOM_HISTORY == 200
        assert s.API_VERSION == "v1"

    def test_cors_origins_parse_csv(self):
        from src.api.config import Settings

        s = Settings(CORS_ORIGINS="http://localhost:3000,http://localhost:5173")
        assert len(s.CORS_ORIGINS) == 2
        assert "http://localhost:3000" in s.CORS_ORIGINS


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation cache
# ─────────────────────────────────────────────────────────────────────────────


class TestRecommendationCache:
    def test_cache_stats_initial(self):
        from src.api.services.recommendation import cache_stats

        stats = cache_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats

    def test_cache_key_differs_by_include_reasons(self):
        from src.api.services.recommendation import _cache_key

        k1 = _cache_key(42, 12, True)
        k2 = _cache_key(42, 12, False)
        assert k1 != k2


# ─────────────────────────────────────────────────────────────────────────────
# Power analysis — matches src formula
# ─────────────────────────────────────────────────────────────────────────────


class TestPowerService:
    def test_compute_power_matches_src(self):
        from src.api.services.experiment import compute_power
        from src.evaluation.power import required_n_per_arm

        result = compute_power(
            baseline_rate=0.085,
            relative_lift=0.10,
            alpha=0.05,
            power=0.80,
            weekly_traffic=None,
            cuped_variance_reduction=0.0,
        )
        expected_n = required_n_per_arm(0.085, 0.10)
        assert result["n_per_arm"] == expected_n

    def test_compute_power_cuped_reduces_n(self):
        from src.api.services.experiment import compute_power

        result_no_cuped = compute_power(0.085, 0.10, 0.05, 0.80, None, 0.0)
        result_cuped = compute_power(0.085, 0.10, 0.05, 0.80, None, 0.05)
        assert result_cuped["n_per_arm_cuped"] is not None
        assert result_cuped["n_per_arm_cuped"] < result_no_cuped["n_per_arm"]

    def test_cohen_h_positive(self):
        from src.api.services.experiment import compute_power

        result = compute_power(0.085, 0.10, 0.05, 0.80, None, 0.0)
        assert result["cohen_h"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Insights service
# ─────────────────────────────────────────────────────────────────────────────


class TestInsightsService:
    def test_get_summary_returns_dict(self):
        from src.api.services.insights import _load_json_disk, get_summary

        # Clear lru_cache to get fresh data
        _load_json_disk.cache_clear()
        result = get_summary()
        assert isinstance(result, dict)
        # holdout_week is always set (defaults to 104 if file missing)
        assert "holdout_week" in result

    def test_get_importance_has_all_keys(self):
        from src.api.services.insights import _load_json_disk, get_importance

        _load_json_disk.cache_clear()
        result = get_importance()
        assert "top10_gain_importance" in result
        assert "top10_shap_all_candidates" in result
