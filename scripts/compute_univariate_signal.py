"""Compute univariate feature signal on fold 103 (validation, no downsampling).

For each feature in FEATURE_LIST:
  - Direction: chosen to maximize AUC computed on the concatenated training folds
               (folds 100, 101, 102) — NOT on fold 103 itself, to avoid direction
               being chosen with knowledge of the validation set.
  - AUC: global ROC AUC with the fixed direction, computed on fold 103 (non-null rows only).
  - MAP@12: rank each customer's 200 candidates by the feature, take top 12,
            compute MAP@12 averaged over ALL eval customers (including those with 0 hits).
            Denominator = n_eval_customers from build_fold(103), NOT n_customers_with_hits.

Saves: reports/features/univariate_signal.json
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from src.config import PROCESSED_DIR, REPORTS_DIR
from src.features.registry import FEATURE_LIST
from src.metrics import map_at_k
from src.time_split import build_fold

FOLD_103_PATH = PROCESSED_DIR / "features" / "fold_103.parquet"
TRAIN_FOLD_PATHS = [
    PROCESSED_DIR / "features" / "fold_100.parquet",
    PROCESSED_DIR / "features" / "fold_101.parquet",
    PROCESSED_DIR / "features" / "fold_102.parquet",
]
OUT_PATH = REPORTS_DIR / "features" / "univariate_signal.json"


def _sorted_top12_by_feature(
    df: pl.DataFrame,
    feat: str,
    ascending: bool,
) -> pl.DataFrame:
    """Return (customer_idx, article_idx) for the top-12 candidates per customer.

    Nulls are always placed last (least informative candidates fill the tail).
    For ascending: nulls get +inf sentinel; for descending: nulls get -inf sentinel.
    """
    FLOAT_MAX = float("inf")
    FLOAT_MIN = float("-inf")

    fill_val = FLOAT_MAX if ascending else FLOAT_MIN

    sort_col = "_sv"
    df_with_sort = df.with_columns(
        pl.col(feat).cast(pl.Float64).fill_null(fill_val).alias(sort_col)
    )

    if ascending:
        df_sorted = df_with_sort.sort(["customer_idx", sort_col, "article_idx"])
    else:
        df_sorted = df_with_sort.sort(
            ["customer_idx", sort_col, "article_idx"],
            descending=[False, True, False],
        )

    # Row number within each customer group (1-indexed, preserves sort order).
    # cum_count() on a non-null column gives sequential 1,2,3,... per group.
    df_ranked = df_sorted.with_columns(
        pl.col("article_idx").cum_count().over("customer_idx").alias("_rnum")
    )

    return df_ranked.filter(pl.col("_rnum") <= 12).select(["customer_idx", "article_idx"])


def _compute_auc(
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, bool]:
    """Return (auc, ascending) choosing the direction that gives AUC >= 0.5.

    Non-null entries only. Returns (0.5, True) if insufficient data.
    """
    non_null = ~np.isnan(values)
    if non_null.sum() < 10:
        return 0.5, True

    v = values[non_null]
    y = labels[non_null]

    if y.sum() == 0 or y.sum() == len(y):
        return 0.5, True

    try:
        auc_desc = float(roc_auc_score(y, v))
    except Exception:
        return 0.5, True

    auc_asc = 1.0 - auc_desc

    if auc_desc >= auc_asc:
        return auc_desc, False  # descending (higher = more positive)
    else:
        return auc_asc, True    # ascending (lower = more positive)


def main() -> None:
    if not FOLD_103_PATH.exists():
        print(f"ERROR: {FOLD_103_PATH} not found. Run scripts/build_features.py first.")
        sys.exit(1)

    missing_train = [p for p in TRAIN_FOLD_PATHS if not p.exists()]
    if missing_train:
        print(f"ERROR: Training fold parquets not found: {missing_train}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load training folds (100, 101, 102) for direction determination
    # ------------------------------------------------------------------
    print("Loading training folds (100, 101, 102) to determine feature directions ...")
    t0 = time.time()
    train_dfs = [pl.read_parquet(p) for p in TRAIN_FOLD_PATHS]
    train_df = pl.concat(train_dfs, how="vertical")
    print(f"  Training concat: {len(train_df):,} rows, {train_df['customer_idx'].n_unique():,} customers "
          f"({time.time()-t0:.1f}s)")

    train_labels_np = train_df["label"].to_numpy()

    print("Loading fold_103.parquet ...")
    t0_val = time.time()
    df = pl.read_parquet(FOLD_103_PATH)
    print(f"  {len(df):,} rows, {df['customer_idx'].n_unique():,} customers "
          f"({time.time()-t0_val:.1f}s)")

    print("Loading ground truth from build_fold(103) ...")
    _, gt_dict, eval_customers = build_fold(103)
    n_eval = len(eval_customers)
    print(f"  {n_eval:,} eval customers")

    # Materialise label column as numpy (fold 103, for AUC reporting)
    labels_np = df["label"].to_numpy()

    results = []
    for i, spec in enumerate(FEATURE_LIST):
        fname = spec.name
        t_feat = time.time()

        # ------------------------------------------------------------------
        # 1. Determine direction using AUC on TRAINING folds (100-102)
        #    This prevents the direction choice from seeing fold 103 labels.
        # ------------------------------------------------------------------
        train_col = train_df[fname]

        try:
            train_values_np = train_col.cast(pl.Float64).to_numpy(allow_copy=True).astype(float)
            train_values_np = np.where(train_col.is_null().to_numpy(), np.nan, train_values_np)
        except Exception:
            train_values_np = np.full(len(train_df), np.nan)

        # Direction fixed from training folds
        _, ascending = _compute_auc(train_values_np, train_labels_np)

        # ------------------------------------------------------------------
        # AUC on fold 103 with the fixed direction (for reporting only)
        # ------------------------------------------------------------------
        col_series = df[fname]
        null_rate = float(col_series.is_null().mean())

        try:
            values_np = col_series.cast(pl.Float64).to_numpy(allow_copy=True).astype(float)
            values_np = np.where(col_series.is_null().to_numpy(), np.nan, values_np)
        except Exception:
            values_np = np.full(len(df), np.nan)

        # Compute AUC on fold 103 with the fixed direction
        non_null = ~np.isnan(values_np)
        if non_null.sum() >= 10:
            v = values_np[non_null]
            y = labels_np[non_null]
            if y.sum() > 0 and y.sum() < len(y):
                try:
                    from sklearn.metrics import roc_auc_score as _roc
                    auc_desc = float(_roc(y, v))
                    auc = auc_desc if not ascending else (1.0 - auc_desc)
                except Exception:
                    auc = 0.5
            else:
                auc = 0.5
        else:
            auc = 0.5

        # ------------------------------------------------------------------
        # 2. Compute MAP@12 over ALL eval customers
        # ------------------------------------------------------------------
        top12_df = _sorted_top12_by_feature(df, fname, ascending)

        preds_raw = (
            top12_df
            .group_by("customer_idx")
            .agg(pl.col("article_idx").alias("articles"))
        )

        predictions: dict[int, list[int]] = {
            row["customer_idx"]: row["articles"]
            for row in preds_raw.iter_rows(named=True)
        }

        # Customers with no candidates get empty prediction → AP = 0
        for c in eval_customers:
            if c not in predictions:
                predictions[c] = []

        # Use src.metrics.map_at_k (denominator = len(gt_dict) = n_eval customers)
        map12 = map_at_k(predictions, gt_dict, k=12)

        results.append({
            "feature": fname,
            "group": spec.group,
            "auc": round(auc, 4),
            "map12": round(map12, 6),
            "null_rate": round(null_rate, 4),
            "direction": "ascending" if ascending else "descending",
        })

        direction_str = "asc" if ascending else "desc"
        print(f"  [{i+1:2d}/{len(FEATURE_LIST)}] {fname:<42s} "
              f"AUC={auc:.4f} ({direction_str})  MAP@12={map12:.6f}  "
              f"null={null_rate:.3f}  ({time.time()-t_feat:.1f}s)")

    # Sort by MAP@12 descending for readability
    results.sort(key=lambda x: x["map12"], reverse=True)

    # ------------------------------------------------------------------
    # Sanity check: final_rank MAP@12 must match the heuristic (0.024675)
    # ------------------------------------------------------------------
    final_rank_row = next((r for r in results if r["feature"] == "final_rank"), None)
    if final_rank_row:
        heuristic = 0.024675292792042714  # from fold_103_results.json
        diff = abs(final_rank_row["map12"] - heuristic)
        status = "PASS" if diff < 1e-4 else f"FAIL (diff={diff:.6f})"
        print(f"\nSanity check — final_rank MAP@12 = heuristic 0.024675: [{status}]")
        if diff >= 1e-4:
            print("  WARNING: final_rank MAP@12 does not match heuristic. "
                  "Check map_at_k denominator and candidate ordering.")

    out = {"feature_signal": results}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\nSaved {len(results)} features -> {OUT_PATH}")
    print(f"Total runtime: {time.time()-t0:.1f}s")

    # Print top-10 for review
    print("\nTop 10 features by MAP@12 (corrected denominator = all eval customers):")
    for r in results[:10]:
        print(f"  {r['feature']:<42s} MAP@12={r['map12']:.6f}  AUC={r['auc']:.4f}")


if __name__ == "__main__":
    main()
