"""SHAP value computation via LightGBM's exact TreeSHAP (pred_contrib=True).

LightGBM's pred_contrib returns shape (n_rows, n_features + 1):
  - columns 0..n_features-1  : per-feature SHAP values
  - column n_features          : expected value (bias)
Additivity: sum(shap[:, :-1], axis=1) + shap[:, -1]  ==  model_score  (within 1e-5)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from src.model.data import FEATURE_NAMES


def compute_shap(
    booster,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute exact TreeSHAP for a 2-D float32 feature matrix.

    Parameters
    ----------
    booster : lgb.Booster
        Trained LightGBM model.
    X : np.ndarray
        Shape (n_rows, n_features); float32, NaN for missing.

    Returns
    -------
    shap_vals : np.ndarray
        Shape (n_rows, n_features); SHAP value per feature per row.
    expected_val : np.ndarray
        Shape (n_rows,); expected value (bias) per row — same for all rows,
        but returned as an array for easy broadcasting.
    max_additivity_error : float
        Maximum absolute error across all rows: |sum(shap) + expected - score|.
    """
    contrib = booster.predict(X, pred_contrib=True)  # (n_rows, n_features+1)
    shap_vals = contrib[:, :-1].astype(np.float64)
    expected_val = contrib[:, -1].astype(np.float64)  # same scalar repeated per row
    scores = booster.predict(X).astype(np.float64)
    reconstructed = shap_vals.sum(axis=1) + expected_val
    max_err = float(np.max(np.abs(reconstructed - scores)))
    return shap_vals.astype(np.float32), expected_val.astype(np.float32), max_err


def load_fold103_sample(
    fold_path: str,
    booster,
    n_customers: int = 5000,
    seed: int = 42,
    batch_size: int = 50_000,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load fold_103 parquet, sample n_customers, compute SHAP for all candidates.

    TreeSHAP is expensive (~1.44s per 1k rows with 421 trees). For 5000 customers
    × 200 candidates = 1M rows, computation takes ~24 minutes. Progress is printed
    every batch.

    Parameters
    ----------
    fold_path : str
        Path to fold_103.parquet.
    booster : lgb.Booster
        Trained model.
    n_customers : int
        Number of eval customers to sample.
    seed : int
        RNG seed.
    batch_size : int
        Number of rows per pred_contrib batch (controls progress granularity).

    Returns
    -------
    df_sample : pl.DataFrame
        Sorted by [customer_idx, article_idx].
    shap_vals : np.ndarray
        Shape (n_rows, n_features); SHAP per feature.
    expected_val : np.ndarray
        Shape (n_rows,); expected value.
    scores : np.ndarray
        Shape (n_rows,); raw model scores.
    """
    import math
    import time

    df = pl.read_parquet(fold_path)
    rng = np.random.default_rng(seed)
    all_customers = df["customer_idx"].unique().to_list()
    chosen = rng.choice(all_customers, size=min(n_customers, len(all_customers)), replace=False)
    chosen_set = set(int(c) for c in chosen)

    df_sample = df.filter(pl.col("customer_idx").is_in(list(chosen_set))).sort(
        ["customer_idx", "article_idx"]
    )

    X = (
        df_sample.select(FEATURE_NAMES)
        .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
        .to_numpy(allow_copy=True)
    ).astype(np.float32)
    print(f"  [SHAP] X shape: {X.shape}", flush=True)

    # Regular scores (fast: ~5s for 1M rows)
    scores = booster.predict(X).astype(np.float32)
    print(f"  [SHAP] Regular predict done.", flush=True)

    # TreeSHAP in batches (slow: ~1.4s per 1k rows)
    n_rows, n_feats = X.shape
    n_batches = math.ceil(n_rows / batch_size)
    shap_vals = np.empty((n_rows, n_feats), dtype=np.float32)
    expected_val = np.empty(n_rows, dtype=np.float32)
    t_start = time.time()

    print(f"  [SHAP] TreeSHAP: {n_rows:,} rows in {n_batches} batches of {batch_size:,}...",
          flush=True)
    for bi in range(n_batches):
        s = bi * batch_size
        e = min(s + batch_size, n_rows)
        contrib = booster.predict(X[s:e], pred_contrib=True)
        shap_vals[s:e] = contrib[:, :-1].astype(np.float32)
        expected_val[s:e] = contrib[:, -1].astype(np.float32)
        elapsed = time.time() - t_start
        rows_done = e
        rate = rows_done / elapsed if elapsed > 0 else 1
        eta = (n_rows - rows_done) / rate if rate > 0 else 0
        print(f"  [SHAP] Batch {bi+1}/{n_batches}: {e:,}/{n_rows:,} rows done  "
              f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    total_t = time.time() - t_start
    print(f"  [SHAP] TreeSHAP complete in {total_t:.0f}s.", flush=True)

    return df_sample, shap_vals, expected_val, scores


def additivity_check(
    booster,
    X: np.ndarray,
    tol: float = 1e-5,
) -> dict:
    """Verify additivity: sum(SHAP) + expected == score for every row.

    Returns dict with max_error, mean_error, n_rows, passed (bool).
    """
    shap_vals, expected_val, max_err = compute_shap(booster, X)
    scores = booster.predict(X).astype(np.float64)
    sv64 = shap_vals.astype(np.float64)
    ev64 = expected_val.astype(np.float64)
    errors = np.abs(sv64.sum(axis=1) + ev64 - scores)
    return {
        "max_error": float(errors.max()),
        "mean_error": float(errors.mean()),
        "n_rows": int(len(X)),
        "tol": tol,
        "passed": bool(errors.max() <= tol),
    }


def get_top12_mask(
    df_sample: pl.DataFrame,
    scores: np.ndarray,
) -> np.ndarray:
    """Return boolean mask for the top-12 candidates per customer.

    Parameters
    ----------
    df_sample : pl.DataFrame
        Must contain 'customer_idx' sorted by [customer_idx, article_idx].
    scores : np.ndarray
        Model scores for each row (same order as df_sample).

    Returns
    -------
    mask : np.ndarray of bool, shape (n_rows,)
    """
    mask = np.zeros(len(df_sample), dtype=bool)
    customers = df_sample["customer_idx"].to_numpy()
    unique_custs, starts = np.unique(customers, return_index=True)
    ends = np.append(starts[1:], len(customers))

    for start, end in zip(starts, ends):
        seg_scores = scores[start:end]
        # top-12 indices within this customer's block
        k = min(12, end - start)
        top_k_local = np.argpartition(-seg_scores, k - 1)[:k]
        top_k_global = start + top_k_local
        mask[top_k_global] = True

    return mask


def mean_abs_shap_by_feature(
    shap_vals: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Mean |SHAP| per feature; returns {feature_name: mean_abs_shap} sorted descending."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    result = {name: float(v) for name, v in zip(feature_names, mean_abs)}
    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))


def compute_within_list_contrast(
    shap_vals: np.ndarray,
    customer_indices: np.ndarray,
) -> np.ndarray:
    """Compute per-row SHAP contrast relative to within-list mean.

    For each customer, contrast_i = SHAP_i - mean(SHAP for all of customer's rows).
    Positive contrast → this item is above average for that feature (potential reason).

    LambdaRank scores are only meaningful relative to the other items in a customer's
    list. The contrast tells you why this item ranks above the customer's alternatives,
    not why it scores above the global average.

    Parameters
    ----------
    shap_vals : np.ndarray, shape (n_rows, n_features)
    customer_indices : np.ndarray, shape (n_rows,); int customer IDs in same row order

    Returns
    -------
    contrast : np.ndarray, shape (n_rows, n_features)
    """
    contrast = np.empty_like(shap_vals, dtype=np.float32)
    unique_custs, starts = np.unique(customer_indices, return_index=True)
    ends = np.append(starts[1:], len(customer_indices))
    for start, end in zip(starts, ends):
        seg = shap_vals[start:end]
        mean_shap = seg.mean(axis=0, keepdims=True)
        contrast[start:end] = seg - mean_shap
    return contrast


def mean_abs_contrast_shap_by_feature(
    contrast: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Mean |within-list contrast SHAP| per feature; returns {feature_name: value} sorted descending.

    Unlike raw |SHAP|, contrast SHAP subtracts the customer's within-list mean before
    taking the absolute value. This shows what actually drives *ranking* between candidates
    for the same customer, rather than what shifts all of a customer's scores together.

    Customer-level features (e.g. c_age, c_n_purchases) tend to have high raw |SHAP| but
    low contrast |SHAP| because they shift every candidate's score by the same amount and
    therefore do not change the ranking within a customer's list.
    """
    mean_abs = np.abs(contrast).mean(axis=0)
    result = {name: float(v) for name, v in zip(feature_names, mean_abs)}
    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))


def group_shap(
    shap_vals: np.ndarray,
    feature_names: list[str],
    feature_groups: dict[str, str],
) -> dict[str, float]:
    """Sum of mean |SHAP| per feature group.

    Parameters
    ----------
    feature_groups : dict mapping feature_name → group_name
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    group_sums: dict[str, float] = {}
    for fname, mv in zip(feature_names, mean_abs):
        grp = feature_groups.get(fname, "other")
        group_sums[grp] = group_sums.get(grp, 0.0) + float(mv)
    return dict(sorted(group_sums.items(), key=lambda kv: kv[1], reverse=True))
