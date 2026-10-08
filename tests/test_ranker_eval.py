"""Tests for ranker evaluation (Part G)."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.metrics import map_at_k, average_precision_at_k, per_customer_ap
from src.model.data import FEATURE_NAMES, prepare_dataset
from src.model.evaluate import compute_full_metrics


def _make_predictions_and_gt():
    """Hand-built example with known MAP@12."""
    # 3 customers, 12 predictions each
    # Customer 0: predictions [1,2,3,...], GT={1,5} → hits at pos 0,4 → AP=(1+2/5)/2=0.7
    # Customer 1: predictions [10,11,...], GT={10} → hit at pos 0 → AP=1.0
    # Customer 2: predictions [20,...], GT={99} → no hit → AP=0.0
    # MAP = (0.7 + 1.0 + 0.0) / 3 = 0.5667
    predictions = {
        0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        1: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        2: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    }
    ground_truth = {
        0: {1, 5},
        1: {10},
        2: {99},
    }
    expected_ap_0 = average_precision_at_k([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], {1, 5}, 12)
    expected_ap_1 = average_precision_at_k([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], {10}, 12)
    expected_ap_2 = 0.0
    expected_map = (expected_ap_0 + expected_ap_1 + expected_ap_2) / 3
    return predictions, ground_truth, expected_map


class TestEvalMetrics:
    def test_map_equals_src_metrics(self):
        """Ranker evaluation MAP@12 equals src.metrics.map_at_k on a hand-built example."""
        predictions, ground_truth, expected_map = _make_predictions_and_gt()
        result = compute_full_metrics(predictions, ground_truth, k=12)
        assert abs(result["map@12"] - expected_map) < 1e-10

    def test_per_customer_ap_matches_map(self):
        """per_customer_ap mean equals map_at_k."""
        predictions, ground_truth, _ = _make_predictions_and_gt()
        ap_dict = per_customer_ap(predictions, ground_truth, k=12)
        mean_ap = sum(ap_dict.values()) / len(ap_dict)
        map_val = map_at_k(predictions, ground_truth, k=12)
        assert abs(mean_ap - map_val) < 1e-10

    def test_heuristic_fold103_reproduces_0024675(self):
        """Heuristic ordering from fold_103.parquet gives MAP@12 = 0.024675 (regression)."""
        from src.config import PROCESSED_DIR
        from src.time_split import build_fold

        df = pl.read_parquet(PROCESSED_DIR / "features" / "fold_103.parquet")
        _, ground_truth_103, _ = build_fold(103)

        top12 = (
            df.filter(pl.col("final_rank") <= 12)
            .sort(["customer_idx", "final_rank"])
            .group_by("customer_idx", maintain_order=True)
            .agg(pl.col("article_idx").alias("articles"))
        )
        predictions = {
            row["customer_idx"]: list(row["articles"])
            for row in top12.iter_rows(named=True)
        }
        result_map = map_at_k(predictions, ground_truth_103, k=12)
        assert abs(result_map - 0.024675) < 1e-4, (
            f"Heuristic MAP@12 = {result_map:.6f}, expected 0.024675"
        )


class TestScoreFoldShuffle:
    """score_fold must give identical top-k per customer regardless of input row order."""

    def _make_fold(self, n_customers: int = 20, n_cands: int = 15, pos: int = 2) -> pl.DataFrame:
        import numpy as np
        rng = np.random.default_rng(77)
        rows = []
        for cust in range(n_customers):
            articles = rng.choice(5000, size=n_cands, replace=False).tolist()
            pos_set = set(articles[:pos])
            for art in articles:
                row = {
                    "customer_idx": cust,
                    "article_idx": int(art),
                    "label": 1 if art in pos_set else 0,
                }
                for name in FEATURE_NAMES:
                    row[name] = float(rng.random())
                rows.append(row)
        schema = {
            "customer_idx": pl.Int32,
            "article_idx": pl.Int32,
            "label": pl.Int8,
            **{n: pl.Float32 for n in FEATURE_NAMES},
        }
        return pl.DataFrame(rows, schema=schema).sort(["customer_idx", "article_idx"])

    def test_score_fold_shuffle_invariant(self):
        import lightgbm as lgb
        from src.model.train import DEFAULT_PARAMS, SEED, _build_lgb_dataset
        from src.model.evaluate import score_fold

        df = self._make_fold()
        X, y, groups = prepare_dataset(df, drop_zero_positive_groups=True)
        params = DEFAULT_PARAMS.copy()
        params["num_leaves"] = 7
        ds = _build_lgb_dataset(X, y, groups, free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=5,
                            callbacks=[lgb.log_evaluation(period=-1)])

        k = 5
        preds_sorted, _ = score_fold(booster, df, k=k, sanity_check=False)
        # Shuffle the input
        df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=99)
        preds_shuffled, _ = score_fold(booster, df_shuffled, k=k, sanity_check=False)

        assert set(preds_sorted.keys()) == set(preds_shuffled.keys()), \
            "Customer sets differ after shuffle"
        for cust in preds_sorted:
            assert set(preds_sorted[cust]) == set(preds_shuffled[cust]), (
                f"Customer {cust}: sorted top-{k}={preds_sorted[cust]}, "
                f"shuffled top-{k}={preds_shuffled[cust]}"
            )


class TestBootstrap:
    def test_ci_contains_zero_for_identical(self):
        """CI for identical AP dicts (zero diff) should contain zero."""
        from src.model.evaluate import bootstrap_paired_diff

        ap_a = {c: 0.1 for c in range(100)}
        result = bootstrap_paired_diff(ap_a, ap_a, n=100, seed=42)
        assert result["mean_diff"] == pytest.approx(0.0, abs=1e-10)
        assert result["ci_lo"] <= 0.0 <= result["ci_hi"]

    def test_ci_excludes_zero_for_clear_winner(self):
        """When A always outperforms B, 95% CI should exclude zero (both bounds positive)."""
        from src.model.evaluate import bootstrap_paired_diff

        rng = np.random.default_rng(0)
        n = 1000
        ap_a = {c: min(1.0, float(rng.random()) * 0.3 + 0.1) for c in range(n)}
        ap_b = {c: float(rng.random()) * 0.05 for c in range(n)}  # clearly lower
        result = bootstrap_paired_diff(ap_a, ap_b, n=500, seed=42)
        assert result["mean_diff"] > 0
        assert result["ci_lo"] > 0
        assert result["ci_excludes_zero"]

    def test_ci_excludes_zero_for_clear_loser(self):
        """When A always underperforms B (ablation case), CI should exclude zero (both bounds negative)."""
        from src.model.evaluate import bootstrap_paired_diff

        rng = np.random.default_rng(0)
        n = 1000
        # A (ablated) is clearly worse than B (full model)
        ap_a = {c: float(rng.random()) * 0.05 for c in range(n)}  # clearly lower
        ap_b = {c: min(1.0, float(rng.random()) * 0.3 + 0.1) for c in range(n)}
        result = bootstrap_paired_diff(ap_a, ap_b, n=500, seed=42)
        assert result["mean_diff"] < 0
        assert result["ci_hi"] < 0, "Both CI bounds must be negative when effect is clearly negative"
        assert result["ci_excludes_zero"], "ci_excludes_zero must be True when CI is entirely negative"
