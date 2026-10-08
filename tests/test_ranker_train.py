"""Tests for LightGBM ranker training (Part G)."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.model.data import FEATURE_NAMES, prepare_dataset
from src.model.train import DEFAULT_PARAMS, SEED, _build_lgb_dataset, train_inner


def _make_synthetic_fold(
    n_customers: int = 50,
    n_cands_per_customer: int = 20,
    pos_per_customer: int = 2,
    seed: int = 42,
) -> pl.DataFrame:
    """Build a minimal synthetic fold DataFrame for testing."""
    rng = np.random.default_rng(seed)

    rows = []
    for cust in range(n_customers):
        articles = rng.choice(10000, size=n_cands_per_customer, replace=False)
        pos_set = set(articles[:pos_per_customer])
        for rank, art in enumerate(articles, 1):
            label = 1 if art in pos_set else 0
            feat_vals = rng.random(len(FEATURE_NAMES)).astype(np.float32)
            row = {
                "customer_idx": cust,
                "article_idx": int(art),
                "label": label,
            }
            for name, val in zip(FEATURE_NAMES, feat_vals):
                row[name] = float(val)
            rows.append(row)

    schema = {
        "customer_idx": pl.Int32,
        "article_idx": pl.Int32,
        "label": pl.Int8,
        **{n: pl.Float32 for n in FEATURE_NAMES},
    }
    return pl.DataFrame(rows, schema=schema)


class TestRankerTrain:
    def test_end_to_end_synthetic(self):
        """Train end-to-end on synthetic data without errors."""
        import lightgbm as lgb

        df = _make_synthetic_fold(n_customers=50, n_cands_per_customer=20, pos_per_customer=2)
        X, y, groups = prepare_dataset(df, drop_zero_positive_groups=True)
        assert X.shape[1] == len(FEATURE_NAMES)
        assert len(y) == X.shape[0]
        assert groups.sum() == len(y)

        params = DEFAULT_PARAMS.copy()
        params["num_leaves"] = 7
        ds = _build_lgb_dataset(X, y, groups)
        booster = lgb.train(params, ds, num_boost_round=10,
                            callbacks=[lgb.log_evaluation(period=-1)])
        assert booster.num_trees() > 0

    def test_same_seed_same_predictions(self):
        """Same seed and data must produce bit-identical predictions."""
        import lightgbm as lgb

        df = _make_synthetic_fold(n_customers=40, seed=7)
        X, y, groups = prepare_dataset(df, drop_zero_positive_groups=True)

        params = DEFAULT_PARAMS.copy()
        params["num_leaves"] = 7

        def _train_once():
            ds = lgb.Dataset(X.copy(), label=y.copy(), group=groups.copy(),
                             free_raw_data=False)
            return lgb.train(params, ds, num_boost_round=10,
                             callbacks=[lgb.log_evaluation(period=-1)])

        b1 = _train_once()
        b2 = _train_once()
        p1 = b1.predict(X)
        p2 = b2.predict(X)
        np.testing.assert_array_equal(p1, p2, err_msg="Predictions differ across runs with same seed")

    def test_rejects_week_104_rows(self):
        """Training pipeline must reject any row from week 104 (leakage guard)."""
        # Week 104 rows would appear if the history filter was wrong.
        # The feature build pipeline (build_fold_features) already enforces week_idx < fold_week.
        # Here we verify the assertion in compute_article_features fires if violated.
        from src.features.article import compute_article_features
        from src.data_io import load_articles, load_customers
        from src.config import ANCHOR_EPOCH_DAYS

        # Build a history that includes week 104 rows
        history_with_leak = pl.DataFrame({
            "customer_idx": [0, 1],
            "article_idx": [10, 20],
            "t_dat": pl.Series([17793 + 104 * 7, 17793 + 100 * 7], dtype=pl.Date),
            "price": [0.05, 0.10],
            "sales_channel_id": pl.Series([1, 2], dtype=pl.Int8),
            "week_idx": pl.Series([104, 100], dtype=pl.Int16),
        }).lazy()

        articles_df = load_articles()
        customers_df = load_customers()

        # fold_week=103 means history must have week_idx < 103; week 104 should trigger assertion
        with pytest.raises(AssertionError):
            compute_article_features(
                history=history_with_leak,
                cutoff_week=103,
                articles_df=articles_df,
                customers_df=customers_df,
                anchor_epoch_days=ANCHOR_EPOCH_DAYS,
            )

    def test_drop_zero_positive_groups(self):
        """Groups with no positives are removed when requested."""
        # 10 customers, only 5 have positives
        rows = []
        for cust in range(10):
            for art in range(20):
                label = 1 if (cust < 5 and art == 0) else 0
                row = {"customer_idx": cust, "article_idx": art, "label": label}
                for n in FEATURE_NAMES:
                    row[n] = 0.5
                rows.append(row)
        schema = {"customer_idx": pl.Int32, "article_idx": pl.Int32, "label": pl.Int8,
                  **{n: pl.Float32 for n in FEATURE_NAMES}}
        df = pl.DataFrame(rows, schema=schema)

        X_full, _, g_full = prepare_dataset(df, drop_zero_positive_groups=False)
        X_drop, _, g_drop = prepare_dataset(df, drop_zero_positive_groups=True)

        assert len(g_full) == 10
        assert len(g_drop) == 5
        assert g_drop.sum() < g_full.sum()
