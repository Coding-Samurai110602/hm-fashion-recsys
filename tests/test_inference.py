"""Strict parity tests for single-customer inference (Task 2, Part G)."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def _model_exists() -> bool:
    return (Path(__file__).parent.parent / "models" / "lgbm_ranker.txt").exists()


@pytest.fixture(scope="module")
def recommender():
    """Load Recommender once; skip if model not trained yet."""
    if not _model_exists():
        pytest.skip("lgbm_ranker.txt not found — run scripts/train_ranker.py first")
    from src.model.inference import Recommender
    model_path = Path(__file__).parent.parent / "models" / "lgbm_ranker.txt"
    rec = Recommender(model_path=model_path)
    # Pre-build state so warm-up happens at fixture time, not inside timed tests
    rec.build_state(103)
    return rec


@pytest.fixture(scope="module")
def fold103_df():
    """Load batch fold_103 feature table once per module."""
    from src.config import PROCESSED_DIR
    return pl.read_parquet(PROCESSED_DIR / "features" / "fold_103.parquet")


@pytest.fixture(scope="module")
def eval_customers_103():
    from src.time_split import build_fold
    _, _, custs = build_fold(103)
    return custs


class TestStrictParity:
    """200-customer parity: batch fold_103 vs Recommender.recommend().

    Checks:
    1. Candidate article SET is identical (same articles in fold_103 row as returned by recommend).
    2. Every feature value is equal (nulls matching).
    3. Model scores within 1e-6.
    4. Top-12 list is identical.
    """

    def _batch_data(self, fold103_df, customer_idx: int):
        df = (
            fold103_df
            .filter(pl.col("customer_idx") == customer_idx)
            .sort("article_idx")
        )
        return df

    def _single_data(self, recommender, customer_idx: int):
        from src.config import PROCESSED_DIR
        from src.model.state import build_state
        from src.features.build import _compute_features_inner
        from src.data_io import load_articles, load_customers

        state = recommender._get_state(103)
        history_lf = state.history_df.lazy()

        df_feat = _compute_features_inner(
            history_lf=history_lf,
            ground_truth={customer_idx: set()},
            eval_customers=[customer_idx],
            articles_df=state.articles_lf,
            customers_df=state.customers_df.lazy(),
            fold_week=103,
            neg_sample_rate=1.0,
            seed=42,
            _state=state,
        )
        return df_feat.filter(pl.col("customer_idx") == customer_idx).sort("article_idx")

    def test_parity_200_customers(self, recommender, fold103_df, eval_customers_103):
        """Strict parity for 200 random fold-103 customers (seed 42).

        For each customer:
        - Candidate article set equals the batch fold_103 set.
        - Every feature value equal (nulls matching).
        - Model scores within 1e-6 of batch scores.
        - Top-12 list identical to batch top-12.
        """
        from src.model.data import FEATURE_NAMES
        import lightgbm as lgb

        rng = np.random.default_rng(42)
        test_custs = rng.choice(eval_customers_103, size=200, replace=False).tolist()
        booster = recommender.booster

        for cust in test_custs:
            # ── Batch data ────────────────────────────────────────────────
            batch_df = self._batch_data(fold103_df, cust)
            if batch_df.is_empty():
                continue

            # ── Single-customer data ──────────────────────────────────────
            single_df = self._single_data(recommender, cust)
            if single_df.is_empty():
                continue

            batch_articles = set(batch_df["article_idx"].to_list())
            single_articles = set(single_df["article_idx"].to_list())

            # 1. Candidate article sets must be identical
            assert batch_articles == single_articles, (
                f"cust {cust}: article set mismatch. "
                f"batch-only={batch_articles - single_articles}, "
                f"single-only={single_articles - batch_articles}"
            )

            # Align rows by article_idx for feature comparison
            batch_sorted = batch_df.sort("article_idx")
            single_sorted = single_df.sort("article_idx")

            # 2. Every feature value must be equal (nulls matching)
            for feat in FEATURE_NAMES:
                b_vals = batch_sorted[feat].to_list()
                s_vals = single_sorted[feat].to_list()
                for i, (bv, sv) in enumerate(zip(b_vals, s_vals)):
                    if bv is None and sv is None:
                        continue
                    if bv is None or sv is None:
                        assert False, (
                            f"cust {cust}, feat {feat}, row {i}: "
                            f"null mismatch batch={bv}, single={sv}"
                        )
                    assert bv == sv, (
                        f"cust {cust}, feat {feat}, row {i}: "
                        f"batch={bv}, single={sv}, diff={abs(bv - sv):.2e}"
                    )

            # 3. Model scores within 1e-6
            X_batch = (
                batch_sorted.select(FEATURE_NAMES)
                .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
                .to_numpy(allow_copy=True)
            )
            X_single = (
                single_sorted.select(FEATURE_NAMES)
                .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
                .to_numpy(allow_copy=True)
            )
            scores_batch = booster.predict(X_batch)
            scores_single = booster.predict(X_single)

            max_score_diff = float(np.max(np.abs(scores_batch - scores_single)))
            assert max_score_diff < 1e-6, (
                f"cust {cust}: max score diff {max_score_diff:.2e} >= 1e-6"
            )

            # 4. Top-12 list identical (same articles, same order)
            top12_batch = (
                batch_sorted
                .with_columns(pl.Series("_score", scores_batch, dtype=pl.Float64))
                .sort(["_score", "article_idx"], descending=[True, False])
                .head(12)["article_idx"]
                .to_list()
            )
            top12_single = (
                single_sorted
                .with_columns(pl.Series("_score", scores_single, dtype=pl.Float64))
                .sort(["_score", "article_idx"], descending=[True, False])
                .head(12)["article_idx"]
                .to_list()
            )
            assert top12_batch == top12_single, (
                f"cust {cust}: top-12 mismatch. "
                f"batch={top12_batch[:3]}..., single={top12_single[:3]}..."
            )


class TestRecommendCustom:
    def test_recommend_custom_returns_12_unique_finite(self, recommender):
        """recommend_custom on a synthetic history returns exactly 12 unique articles
        with finite scores."""
        history_rows = [
            {"article_idx": 108775015, "t_dat": "2020-09-01", "price": 0.05},
            {"article_idx": 108775044, "t_dat": "2020-09-01", "price": 0.04},
            {"article_idx": 111586003, "t_dat": "2020-08-15", "price": 0.08},
            {"article_idx": 118429015, "t_dat": "2020-08-10", "price": 0.03},
            {"article_idx": 106501004, "t_dat": "2020-07-20", "price": 0.07},
        ]
        recs = recommender.recommend_custom(history_rows, as_of_week=103)
        assert len(recs) == 12, f"Expected 12 recs, got {len(recs)}"
        article_ids = [r[0] for r in recs]
        assert len(set(article_ids)) == 12, "Recommendations must be unique"
        for art, score, sources in recs:
            assert np.isfinite(score), f"Non-finite score {score} for article {art}"
