"""Evaluation of the LightGBM ranker on a fold."""
from __future__ import annotations

import numpy as np
import polars as pl
import lightgbm as lgb

from src.config import HOLDOUT_WEEK, SEGMENT_LABELS
from src.metrics import (
    average_precision_at_k,
    map_at_k,
    mean_ndcg_at_k,
    mean_precision_at_k,
    mean_recall_user_at_k,
    per_customer_ap,
)
from src.model.data import FEATURE_NAMES, prepare_dataset
from src.time_split import build_fold


def score_fold(
    booster: lgb.Booster,
    df: pl.DataFrame,
    k: int = 12,
    sanity_check: bool = True,
) -> tuple[dict[int, list[int]], np.ndarray]:
    """Score all candidates in df and return top-k predictions per customer.

    Parameters
    ----------
    booster: Trained LightGBM booster.
    df: Feature DataFrame (all candidates, not downsampled).
    k: Number of items to recommend per customer.
    sanity_check: If True, assert every customer gets exactly k predictions.

    Returns
    -------
    predictions: dict[customer_idx, list[article_idx]] (top-k ordered by score)
    raw_scores: np.ndarray of raw scores (same row order as df, sorted by customer)
    """
    df_sorted = df.sort("customer_idx")

    X = (
        df_sorted.select(FEATURE_NAMES)
        .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
        .to_numpy(allow_copy=True)
    )
    raw_scores = booster.predict(X)
    assert np.all(np.isfinite(raw_scores)), "Ranker produced non-finite scores"

    df_scored = df_sorted.with_columns(
        pl.Series("_score", raw_scores, dtype=pl.Float64)
    )

    # Top-k per customer: sort desc by score, article_idx ASC for stable tiebreaking
    top_k_df = (
        df_scored
        .sort(["customer_idx", "_score", "article_idx"], descending=[False, True, False])
        .with_columns(
            (pl.int_range(pl.len(), dtype=pl.Int32).over("customer_idx")).alias("_row_num")
        )
        .filter(pl.col("_row_num") < k)
        .sort(["customer_idx", "_row_num"])
    )

    # Build predictions dict
    pred_rows = (
        top_k_df
        .group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_idx").alias("articles"))
    )
    predictions: dict[int, list[int]] = {
        row["customer_idx"]: list(row["articles"])
        for row in pred_rows.iter_rows(named=True)
    }

    if sanity_check:
        for cust, arts in predictions.items():
            assert len(arts) == k, (
                f"Customer {cust} has {len(arts)} predictions, expected {k}"
            )
        # Verify predictions are subsets of candidates
        cand_pairs = set(
            zip(df["customer_idx"].to_list(), df["article_idx"].to_list())
        )
        for cust, arts in predictions.items():
            for art in arts:
                assert (cust, art) in cand_pairs, (
                    f"Ranker predicted article {art} for customer {cust} "
                    "that was not in the candidate list"
                )

    return predictions, raw_scores


def compute_full_metrics(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 12,
) -> dict:
    """Compute MAP@k, NDCG@k, recall@k, precision@k over all ground_truth customers."""
    return {
        "map@12": map_at_k(predictions, ground_truth, k),
        "ndcg@12": mean_ndcg_at_k(predictions, ground_truth, k),
        "recall@12": mean_recall_user_at_k(predictions, ground_truth, k),
        "precision@12": mean_precision_at_k(predictions, ground_truth, k),
        "n_customers": len(ground_truth),
    }


def bootstrap_paired_diff(
    ap_a: dict[int, float],
    ap_b: dict[int, float],
    n: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap CI for MAP@12 difference (a - b) over shared customers.

    Resamples customers with replacement n times. Returns mean diff, 95% CI,
    and relative lift vs b.
    """
    shared = sorted(set(ap_a) & set(ap_b))
    arr_a = np.array([ap_a[c] for c in shared])
    arr_b = np.array([ap_b[c] for c in shared])
    diff = arr_a - arr_b

    rng = np.random.default_rng(seed)
    n_customers = len(shared)
    boot_means = np.zeros(n)
    for i in range(n):
        idx = rng.integers(0, n_customers, size=n_customers)
        boot_means[i] = diff[idx].mean()

    mean_diff = diff.mean()
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    map_b = arr_b.mean()
    relative_lift = mean_diff / map_b if map_b > 0 else float("nan")

    return {
        "mean_diff": float(mean_diff),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "ci_excludes_zero": ci_lo > 0 or ci_hi < 0,
        "relative_lift": float(relative_lift),
        "n_customers": n_customers,
        "n_resamples": n,
    }


def segment_map(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    segments_df: pl.DataFrame,
    k: int = 12,
) -> list[dict]:
    """MAP@k per customer segment.

    segments_df must have columns [customer_idx (Int32), segment (Utf8)].
    """
    results = []
    for seg in ["0", "1-4", "5-19", "20+"]:
        seg_custs = set(
            segments_df.filter(pl.col("segment") == seg)["customer_idx"].to_list()
        )
        seg_gt = {c: gt for c, gt in ground_truth.items() if c in seg_custs}
        if not seg_gt:
            results.append({"segment": seg, "n_customers": 0, "map@12": float("nan")})
            continue
        seg_pred = {c: predictions.get(c, []) for c in seg_gt}
        results.append({
            "segment": seg,
            "n_customers": len(seg_gt),
            "map@12": map_at_k(seg_pred, seg_gt, k),
        })
    return results


def segment_map_with_bootstrap(
    predictions_ranker: dict[int, list[int]],
    predictions_heuristic: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    segments_df: pl.DataFrame,
    k: int = 12,
    n_boot: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """MAP@k per segment with paired bootstrap CI (ranker vs heuristic).

    segments_df must have columns [customer_idx (Int32), segment (Utf8)].
    """
    ap_ranker = per_customer_ap(predictions_ranker, ground_truth, k)
    ap_heuristic = per_customer_ap(predictions_heuristic, ground_truth, k)

    results = []
    for seg in ["0", "1-4", "5-19", "20+"]:
        seg_custs = set(
            segments_df.filter(pl.col("segment") == seg)["customer_idx"].to_list()
        )
        seg_gt = {c: ground_truth[c] for c in ground_truth if c in seg_custs}
        if not seg_gt:
            results.append({"segment": seg, "n_customers": 0})
            continue

        seg_pred_r = {c: predictions_ranker.get(c, []) for c in seg_gt}
        seg_pred_h = {c: predictions_heuristic.get(c, []) for c in seg_gt}
        seg_ap_r = {c: ap_ranker.get(c, 0.0) for c in seg_gt}
        seg_ap_h = {c: ap_heuristic.get(c, 0.0) for c in seg_gt}

        boot = bootstrap_paired_diff(seg_ap_r, seg_ap_h, n=n_boot, seed=seed)
        results.append({
            "segment": seg,
            "n_customers": len(seg_gt),
            "map@12_ranker": map_at_k(seg_pred_r, seg_gt, k),
            "map@12_heuristic": map_at_k(seg_pred_h, seg_gt, k),
            "bootstrap_ranker_vs_heuristic": boot,
        })
    return results


def eval_heuristic_fold103() -> dict[int, list[int]]:
    """Return heuristic top-12 predictions for fold 103 by reading fold_103 parquet.

    final_rank ≤ 12 gives the same ordering as the heuristic. The Session 3
    univariate regression confirms this reproduces MAP@12 = 0.024675 exactly.
    """
    from src.config import PROCESSED_DIR
    df = pl.read_parquet(PROCESSED_DIR / "features" / "fold_103.parquet")
    top12 = (
        df.filter(pl.col("final_rank") <= 12)
        .sort(["customer_idx", "final_rank"])
        .group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_idx").alias("articles"))
    )
    return {
        row["customer_idx"]: list(row["articles"])
        for row in top12.iter_rows(named=True)
    }


def eval_baseline_b_fold103() -> dict[int, list[int]]:
    """Return Baseline B top-12 predictions for fold 103 from fold_103.parquet.

    Baseline B = recency-ordered repurchase (most recent article first), then
    global last-week popularity fill.  Derived from candidate feature columns:
      - repurchase items sorted by repurchase_score DESC (last purchase date), article_idx ASC
      - non-repurchase items sorted by popularity_last_week_rank ASC, article_idx ASC
    This matches the recency_only ordering used in the regression check.
    """
    from src.config import PROCESSED_DIR
    df = pl.read_parquet(PROCESSED_DIR / "features" / "fold_103.parquet")

    # Tag each row with its Baseline B sort group and key.
    # Feature columns use the naming convention: in_<source>, <source>_rank, <source>_score.
    # Group 0 (repurchase): sort by -repurchase_score ASC, article_idx ASC (recency_only)
    # Group 1 (popularity): sort by popularity_last_week_rank ASC (nulls last), article_idx ASC
    pop_rank_fill = (
        df["popularity_last_week_rank"]
        .fill_null(999_999)
        .cast(pl.Int32)
    )
    df_b = df.with_columns([
        pl.when(pl.col("in_repurchase") == 1)
        .then(pl.lit(0, dtype=pl.Int8))
        .otherwise(pl.lit(1, dtype=pl.Int8))
        .alias("_grp"),
        pl.when(pl.col("in_repurchase") == 1)
        .then(-pl.col("repurchase_score").cast(pl.Float64))
        .otherwise(pop_rank_fill.cast(pl.Float64))
        .alias("_sort_key"),
    ]).sort(
        ["customer_idx", "_grp", "_sort_key", "article_idx"],
        descending=[False, False, False, False],
    ).with_columns(
        pl.int_range(pl.len(), dtype=pl.Int32).over("customer_idx").alias("_rn")
    ).filter(pl.col("_rn") < 12)

    preds = {
        row["customer_idx"]: list(row["articles"])
        for row in df_b
        .group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_idx").alias("articles"))
        .iter_rows(named=True)
    }
    return preds
