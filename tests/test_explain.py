"""Tests for SHAP explainability and reason code modules.

Coverage:
  - Additivity on synthetic data
  - Additivity on real fold-103 sample (if model file present)
  - Reason template rendering on hand-built rows
  - Positive-only reason rule
  - Contrast computation on hand-built 3-item example
  - Faithfulness test function on synthetic data
"""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Shared constants
MODEL_PATH = Path("models/lgbm_ranker.txt")
FOLD103_PATH = Path("data/processed/features/fold_103.parquet")

pytestmark = pytest.mark.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_booster(n_features: int = 5, expected_val: float = 0.5):
    """Create a mock LightGBM booster that returns pred_contrib = ones * 0.1 + bias."""
    booster = MagicMock()

    def _predict(X, pred_contrib=False):
        n = len(X)
        if pred_contrib:
            contrib = np.full((n, n_features + 1), 0.1, dtype=np.float64)
            contrib[:, -1] = expected_val  # bias
            return contrib
        return np.full(n, 0.1 * n_features + expected_val, dtype=np.float64)

    booster.predict = _predict
    return booster


# ─────────────────────────────────────────────────────────────────────────────
# A. Additivity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAdditivity:
    def test_synthetic_additivity(self):
        """sum(SHAP) + expected == score for synthetic booster."""
        from src.explain.shap_values import additivity_check

        booster = _synthetic_booster(n_features=5, expected_val=0.5)
        X = np.random.default_rng(0).random((20, 5)).astype(np.float32)
        result = additivity_check(booster, X, tol=1e-5)
        assert result["passed"], f"max_error={result['max_error']}"
        assert result["n_rows"] == 20

    def test_compute_shap_shape(self):
        """compute_shap returns correct shapes."""
        from src.explain.shap_values import compute_shap

        booster = _synthetic_booster(n_features=4)
        X = np.ones((10, 4), dtype=np.float32)
        shap_vals, expected_val, max_err = compute_shap(booster, X)
        assert shap_vals.shape == (10, 4)
        assert expected_val.shape == (10,)
        assert max_err < 1e-5

    def test_additivity_real_sample(self):
        """Additivity on 50 rows from fold_103 with actual model."""
        if not MODEL_PATH.exists() or not FOLD103_PATH.exists():
            pytest.skip("model/fold_103 not available")

        import lightgbm as lgb
        import polars as pl

        from src.explain.shap_values import additivity_check
        from src.model.data import FEATURE_NAMES

        booster = lgb.Booster(model_file=str(MODEL_PATH))
        df = pl.read_parquet(FOLD103_PATH, n_rows=5000)
        X = (
            df.select(FEATURE_NAMES)
            .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
            .to_numpy(allow_copy=True)
        ).astype(np.float32)[:50]

        result = additivity_check(booster, X, tol=1e-5)
        assert result["passed"], f"max_error={result['max_error']:.2e}"


# ─────────────────────────────────────────────────────────────────────────────
# B. Contrast computation
# ─────────────────────────────────────────────────────────────────────────────

class TestContrast:
    def test_contrast_hand_built(self):
        """3-item example: contrast = SHAP - mean(SHAP) per customer."""
        from src.explain.shap_values import compute_within_list_contrast

        # 3 items, 2 features; 1 customer
        shap = np.array([[1.0, 2.0],
                         [3.0, 0.0],
                         [2.0, 4.0]], dtype=np.float32)
        custs = np.array([0, 0, 0], dtype=np.int32)
        result = compute_within_list_contrast(shap, custs)
        mean_shap = shap.mean(axis=0)   # [2.0, 2.0]
        expected = shap - mean_shap
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_contrast_two_customers(self):
        """Each customer's contrast is independent."""
        from src.explain.shap_values import compute_within_list_contrast

        shap = np.array([[1.0, 0.0],
                         [3.0, 0.0],
                         [10.0, 0.0],
                         [10.0, 0.0]], dtype=np.float32)
        custs = np.array([0, 0, 1, 1], dtype=np.int32)
        result = compute_within_list_contrast(shap, custs)
        # customer 0: mean=[2,0]; contrast row0=[-1,0], row1=[+1,0]
        np.testing.assert_allclose(result[0], [-1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(result[1], [+1.0, 0.0], atol=1e-5)
        # customer 1: both same → contrast=0
        np.testing.assert_allclose(result[2], [0.0, 0.0], atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# C. Reason template tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReasonTemplates:
    def _make_feature_vec(self, **overrides):
        """Build a feature vector dict with all NaN defaults."""
        from src.model.data import FEATURE_NAMES
        vals = {n: float("nan") for n in FEATURE_NAMES}
        vals.update(overrides)
        return np.array([vals[n] for n in FEATURE_NAMES], dtype=np.float32)

    def _all_positive_contrast(self):
        """Contrast vector with all +1.0."""
        from src.model.data import FEATURE_NAMES
        return np.ones(len(FEATURE_NAMES), dtype=np.float32)

    def test_days_since_article_template(self):
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = self._make_feature_vec(i_days_since_bought_article=5.0)
        contrast = self._all_positive_contrast()
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES)
        assert any("5 days ago" in r and "exact item" in r for r in reasons)

    def test_trend_ratio_template(self):
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = self._make_feature_vec(a_trend_ratio=2.0)  # 100% more than avg
        contrast = self._all_positive_contrast()
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES)
        assert any("Trending" in r for r in reasons)

    def test_product_code_template(self):
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = self._make_feature_vec(i_days_since_bought_product_code=30.0)
        contrast = self._all_positive_contrast()
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES)
        assert any("colour or size" in r.lower() for r in reasons)

    def test_share_template(self):
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = self._make_feature_vec(i_customer_share_product_group=0.35)
        contrast = self._all_positive_contrast()
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES)
        assert any("35%" in r for r in reasons)

    def test_age_gap_template(self):
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = self._make_feature_vec(i_age_gap=3.0)  # small gap → show reason
        contrast = self._all_positive_contrast()
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES)
        assert any("age" in r.lower() for r in reasons)

    def test_max_reasons_is_3(self):
        """generate_reasons returns at most 3 reasons."""
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = self._make_feature_vec(
            i_days_since_bought_article=5.0,
            i_days_since_bought_product_code=10.0,
            a_trend_ratio=2.5,
            i_customer_share_product_group=0.4,
            i_age_gap=2.0,
        )
        contrast = self._all_positive_contrast()
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES, top_k=3)
        assert len(reasons) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# G. New Part-1 rules: conflict suppression, thresholds, one-per-theme
# ─────────────────────────────────────────────────────────────────────────────

class TestReasonRules:
    """Tests for the three additional rules added in Part 1."""

    def _make_feature_vec(self, **overrides):
        from src.model.data import FEATURE_NAMES
        vals = {n: float("nan") for n in FEATURE_NAMES}
        vals.update(overrides)
        return np.array([vals[n] for n in FEATURE_NAMES], dtype=np.float32)

    def _all_positive_contrast(self):
        from src.model.data import FEATURE_NAMES
        return np.ones(len(FEATURE_NAMES), dtype=np.float32)

    def test_exact_article_suppresses_product_code(self):
        """When exact-item is eligible, product-code reason must not appear."""
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = self._make_feature_vec(
            i_days_since_bought_article=9.0,
            i_days_since_bought_product_code=30.0,
        )
        contrast = self._all_positive_contrast()
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES, top_k=3)

        has_exact = any("exact item" in r for r in reasons)
        has_pc = any("colour or size" in r for r in reasons)
        assert has_exact, f"Expected exact-item reason, got: {reasons}"
        assert not has_pc, f"Product-code reason must be suppressed when exact-item present, got: {reasons}"

    def test_product_code_shown_when_no_exact_article(self):
        """Product-code reason appears when exact-article is not eligible."""
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = self._make_feature_vec(
            i_days_since_bought_product_code=30.0,
            # i_days_since_bought_article left as NaN
        )
        contrast = self._all_positive_contrast()
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES, top_k=3)
        assert any("colour or size" in r for r in reasons), (
            f"Expected product-code reason, got: {reasons}"
        )

    def test_share_above_25_pct_says_you_buy_often(self):
        """share >= 25% yields 'you buy often'."""
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = self._make_feature_vec(i_customer_share_product_group=0.30)
        contrast = self._all_positive_contrast()
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES)
        assert any("you buy often" in r for r in reasons), (
            f"Expected 'you buy often' for 30% share, got: {reasons}"
        )

    def test_share_10_to_25_pct_says_bought_from_before(self):
        """10% <= share < 25% yields 'you've bought from before'."""
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = self._make_feature_vec(i_customer_share_product_group=0.15)
        contrast = self._all_positive_contrast()
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES)
        assert any("bought from before" in r for r in reasons), (
            f"Expected 'bought from before' for 15% share, got: {reasons}"
        )

    def test_share_below_10_pct_yields_no_reason(self):
        """share < 10% yields no category reason."""
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = self._make_feature_vec(i_customer_share_product_group=0.05)
        contrast = self._all_positive_contrast()
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES)
        # No category reason should appear for a <10% share
        assert not any("product category" in r for r in reasons), (
            f"Expected no category reason for 5% share, got: {reasons}"
        )

    def test_one_per_theme_category(self):
        """At most one category reason when both product_group and garment_group are eligible."""
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = self._make_feature_vec(
            i_customer_share_product_group=0.40,
            i_customer_share_garment_group=0.35,
        )
        contrast = self._all_positive_contrast()
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES, top_k=3)
        # Count how many reasons mention a category
        cat_count = sum(1 for r in reasons if "category" in r)
        assert cat_count <= 1, (
            f"Expected at most 1 category reason (one-per-theme), got {cat_count}: {reasons}"
        )

    def test_one_per_theme_repurchase(self):
        """At most one repurchase reason even when both exact-article and product-code eligible,
        AND exact-article suppresses product-code (so the only repurchase reason is exact-item)."""
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = self._make_feature_vec(
            i_days_since_bought_article=5.0,
            i_days_since_bought_product_code=10.0,
        )
        contrast = self._all_positive_contrast()
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES, top_k=3)
        # Should have exactly 1 repurchase reason (exact-item only)
        repurchase_count = sum(
            1 for r in reasons if "exact item" in r or "colour or size" in r
        )
        assert repurchase_count == 1, (
            f"Expected exactly 1 repurchase reason, got {repurchase_count}: {reasons}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# D. Positive-only rule
# ─────────────────────────────────────────────────────────────────────────────

class TestPositiveOnlyRule:
    def test_negative_contrast_excluded(self):
        """Features with negative contrast never appear as reasons."""
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = np.full(len(FEATURE_NAMES), 5.0, dtype=np.float32)
        # all contrast values are -1 → nothing should be a reason
        contrast = np.full(len(FEATURE_NAMES), -1.0, dtype=np.float32)
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES)
        assert reasons == [], f"expected empty, got {reasons}"

    def test_zero_contrast_excluded(self):
        """Features with zero contrast are excluded."""
        from src.model.data import FEATURE_NAMES
        from src.explain.reasons import generate_reasons

        fv = np.full(len(FEATURE_NAMES), 5.0, dtype=np.float32)
        contrast = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        reasons = generate_reasons(contrast, fv, FEATURE_NAMES)
        assert reasons == []


# ─────────────────────────────────────────────────────────────────────────────
# E. Faithfulness test on synthetic data
# ─────────────────────────────────────────────────────────────────────────────

class TestFaithfulnessSynthetic:
    def test_faithfulness_important_feature_drops_more(self):
        """On synthetic data where feature 0 has a clear linear effect,
        ablating it drops the score more than ablating a constant feature."""
        from src.explain.reasons import faithfulness_check, REASON_FEATURES
        from src.model.data import FEATURE_NAMES

        # We'll build a mock booster where score = 2.0 * X[:, 0] + X[:, 1]
        n_feats = len(FEATURE_NAMES)
        feat_0 = FEATURE_NAMES.index(REASON_FEATURES[0]) if REASON_FEATURES[0] in FEATURE_NAMES else 0

        class _MockBooster:
            def predict(self, X, pred_contrib=False):
                if pred_contrib:
                    n, f = X.shape
                    contrib = np.zeros((n, f + 1))
                    contrib[:, feat_0] = 2.0 * np.where(np.isnan(X[:, feat_0]), 0, X[:, feat_0])
                    contrib[:, 1] = np.where(np.isnan(X[:, 1]), 0, X[:, 1])
                    return contrib
                scores = np.zeros(len(X))
                scores += 2.0 * np.where(np.isnan(X[:, feat_0]), 0, X[:, feat_0])
                scores += np.where(np.isnan(X[:, 1]), 0, X[:, 1])
                return scores

        booster = _MockBooster()
        rng = np.random.default_rng(0)
        n = 50
        X = rng.random((n, n_feats)).astype(np.float32)
        X[:, 0] = rng.uniform(5, 20, n).astype(np.float32)  # large positive values → large score
        X[:, 1] = rng.uniform(0, 0.1, n).astype(np.float32)  # near-zero → small effect

        # Build contrast where feat_0 has large positive contrast
        contrast = np.zeros((n, n_feats), dtype=np.float32)
        contrast[:, feat_0] = 2.0

        result = faithfulness_check(booster, X, contrast, FEATURE_NAMES, seed=42)
        assert result["passed"], (
            f"top_mean_drop={result['top_mean_drop']:.4f}, "
            f"random_mean_drop={result['random_mean_drop']:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# F. numpy.int64 indexing fix (Polars 2.0 regression)
# ─────────────────────────────────────────────────────────────────────────────

class TestNumpyInt64IndexingFix:
    """Polars 2.0 rejects numpy.int64 as a Series index key.

    The fix: use Series.to_list() to get a plain Python list, then index with
    int(numpy_idx). This class documents and verifies the pattern used in the
    worked-examples loop of run_explain.py.
    """

    def test_polars_series_rejects_numpy_int64(self):
        """Polars Series[numpy.int64] raises TypeError (documents the regression)."""
        import polars as pl

        s = pl.Series("article_idx", [10, 20, 30], dtype=pl.Int32)
        numpy_idx = np.argsort(np.array([3.0, 1.0, 2.0]))[0]  # numpy.int64
        assert isinstance(numpy_idx, np.integer), "Expected numpy integer type"
        with pytest.raises(TypeError, match="cannot select elements using key of type"):
            _ = s[numpy_idx]

    def test_to_list_with_int_cast_works(self):
        """Series.to_list() + int() indexing is the correct fix."""
        import polars as pl

        article_ids = [101, 202, 303]
        s = pl.Series("article_idx", article_ids, dtype=pl.Int32)
        scores = np.array([0.5, 0.9, 0.3], dtype=np.float32)
        top_local = np.argsort(-scores)  # [1, 0, 2] as numpy.int64

        article_list = s.to_list()  # Python list of Python ints
        recs = []
        for local_i in top_local:
            local_i = int(local_i)   # numpy.int64 → Python int
            recs.append(article_list[local_i])

        # Highest score is index 1 → article_id 202
        assert recs == [202, 101, 303], f"Expected [202, 101, 303], got {recs}"
