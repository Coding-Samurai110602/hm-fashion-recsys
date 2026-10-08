"""Extended segment analysis with Holm correction for Part 1C."""
from __future__ import annotations

import datetime

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from src.config import ANCHOR_DATE
from src.evaluation.suite import bootstrap_ci, per_customer_hit
from src.metrics import map_at_k, per_customer_ap
from src.model.evaluate import bootstrap_paired_diff


# ─────────────────────────────────────────────────────────────────────────────
# Segment definitions
# ─────────────────────────────────────────────────────────────────────────────

def _age_bucket(age: float | None) -> str:
    if age is None or np.isnan(float(age)):
        return "missing"
    age = float(age)
    if age < 25:
        return "<25"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55+"


def _recency_bucket(days: int | None) -> str:
    """Map days since last purchase to recency bucket."""
    if days is None:
        return "never"
    if days <= 7:
        return "<=7"
    if days <= 30:
        return "8-30"
    if days <= 90:
        return "31-90"
    return "91+"


def _online_tertile_thresholds(online_shares: list[float]) -> tuple[float, float]:
    """Return (p33, p67) of the distribution of online shares."""
    arr = np.array(online_shares)
    return float(np.percentile(arr, 33.33)), float(np.percentile(arr, 66.67))


def compute_extended_segments(
    tx_lf: pl.LazyFrame,
    customers_lf: pl.LazyFrame,
    cutoff_week: int,
    eval_customers: list[int],
    ground_truth: dict[int, set[int]],
) -> dict[str, dict[str, list[int]]]:
    """Compute all segment definitions for fold analysis.

    Returns dict: segment_family -> {segment_label -> [customer_idx]}.
    Families: prior_purchases, age_bucket, channel_tertile, recency, new_article_buyer.
    """
    eval_set = set(eval_customers)
    eval_series = pl.Series("customer_idx", eval_customers, dtype=pl.Int32)

    # ── History ──────────────────────────────────────────────────────────────
    history_df = (
        tx_lf
        .filter(pl.col("week_idx") < cutoff_week)
        .filter(pl.col("customer_idx").is_in(eval_customers))
        .collect()
    )

    # ── 1. Prior purchase count ───────────────────────────────────────────────
    prior_counts = (
        history_df
        .group_by("customer_idx")
        .agg(pl.len().alias("prior_count"))
    )
    all_df = pl.DataFrame({"customer_idx": eval_series})
    prior_df = all_df.join(prior_counts, on="customer_idx", how="left").with_columns(
        pl.col("prior_count").fill_null(0)
    )
    prior_segments: dict[str, list[int]] = {"0": [], "1-4": [], "5-19": [], "20+": []}
    for row in prior_df.iter_rows(named=True):
        c, cnt = row["customer_idx"], row["prior_count"]
        if cnt == 0:
            prior_segments["0"].append(c)
        elif cnt <= 4:
            prior_segments["1-4"].append(c)
        elif cnt <= 19:
            prior_segments["5-19"].append(c)
        else:
            prior_segments["20+"].append(c)

    # ── 2. Age bucket ─────────────────────────────────────────────────────────
    age_df = (
        customers_lf
        .select(["customer_idx", "age"])
        .filter(pl.col("customer_idx").is_in(eval_customers))
        .collect()
    )
    age_segments: dict[str, list[int]] = {
        "<25": [], "25-34": [], "35-44": [], "45-54": [], "55+": [], "missing": []
    }
    age_map = {row["customer_idx"]: row["age"] for row in age_df.iter_rows(named=True)}
    for c in eval_customers:
        age = age_map.get(c)
        bucket = _age_bucket(age)
        age_segments[bucket].append(c)

    # ── 3. Channel preference (online share tertiles) ─────────────────────────
    channel_df = (
        history_df
        .group_by("customer_idx")
        .agg([
            (pl.col("sales_channel_id") == 2).sum().alias("online_count"),
            pl.len().alias("total_count"),
        ])
    )
    chan_with_all = all_df.join(channel_df, on="customer_idx", how="left")
    # Customers with no history: no purchases before cutoff → share = 0
    chan_with_all = chan_with_all.with_columns([
        pl.col("online_count").fill_null(0),
        pl.col("total_count").fill_null(0),
    ]).with_columns(
        pl.when(pl.col("total_count") > 0)
        .then(pl.col("online_count").cast(pl.Float64) / pl.col("total_count").cast(pl.Float64))
        .otherwise(pl.lit(None, dtype=pl.Float64))
        .alias("online_share")
    )
    # Compute tertile thresholds from customers with history
    shares_with_hist = chan_with_all.filter(pl.col("online_share").is_not_null())["online_share"].to_list()
    if len(shares_with_hist) >= 3:
        p33, p67 = _online_tertile_thresholds(shares_with_hist)
    else:
        p33, p67 = 0.33, 0.67

    channel_segments: dict[str, list[int]] = {"low_online": [], "mid_online": [], "high_online": [], "no_history": []}
    for row in chan_with_all.iter_rows(named=True):
        c, share = row["customer_idx"], row["online_share"]
        if share is None:
            channel_segments["no_history"].append(c)
        elif share <= p33:
            channel_segments["low_online"].append(c)
        elif share <= p67:
            channel_segments["mid_online"].append(c)
        else:
            channel_segments["high_online"].append(c)

    # ── 4. Recency since last purchase ────────────────────────────────────────
    cutoff_start = ANCHOR_DATE + datetime.timedelta(weeks=cutoff_week)
    if history_df.is_empty():
        recency_df = pl.DataFrame({"customer_idx": pl.Series(dtype=pl.Int32), "max_t_dat": pl.Series(dtype=pl.Date)})
    else:
        recency_df = (
            history_df
            .group_by("customer_idx")
            .agg(pl.col("t_dat").max().alias("max_t_dat"))
        )
    rec_with_all = all_df.join(recency_df, on="customer_idx", how="left")
    recency_segments: dict[str, list[int]] = {"<=7": [], "8-30": [], "31-90": [], "91+": [], "never": []}
    for row in rec_with_all.iter_rows(named=True):
        c, last_t = row["customer_idx"], row["max_t_dat"]
        if last_t is None:
            days = None
        else:
            days = (cutoff_start - last_t).days
        bucket = _recency_bucket(days)
        recency_segments[bucket].append(c)

    # ── 5. New-article buyers ─────────────────────────────────────────────────
    # Articles first sold in the target week (no prior transaction)
    target_articles = set()
    for arts in ground_truth.values():
        target_articles.update(arts)
    if history_df.is_empty():
        articles_in_history = set()
    else:
        articles_in_history = set(history_df["article_idx"].unique().to_list())
    new_articles = target_articles - articles_in_history

    buyer_new: list[int] = []
    buyer_not_new: list[int] = []
    for c in eval_customers:
        bought = ground_truth.get(c, set())
        if bought & new_articles:
            buyer_new.append(c)
        else:
            buyer_not_new.append(c)
    new_article_segments = {"buys_new_article": buyer_new, "no_new_article": buyer_not_new}

    return {
        "prior_purchases": prior_segments,
        "age_bucket": age_segments,
        "channel_tertile": channel_segments,
        "recency": recency_segments,
        "new_article_buyer": new_article_segments,
    }


# ─────────────────────────────────────────────────────────────────────────────
# P-value from bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def p_value_from_paired_ttest(arr_r: np.ndarray, arr_h: np.ndarray) -> float:
    """Two-sided paired t-test p-value on per-customer AP differences.

    H0: mean(AP_ranker - AP_heuristic) = 0.
    More powerful than bootstrap at B=1000 (resolution-limited to 2/(B+1)≈0.002).
    """
    diff = arr_r - arr_h
    if len(diff) < 2:
        return 1.0
    _, p = scipy_stats.ttest_1samp(diff, 0.0)
    return float(p)


def p_value_from_bootstrap(boot_means: np.ndarray, observed_mean: float) -> float:
    """Two-sided bootstrap p-value using the smooth +1/(B+1) estimator.

    Under H0 the mean diff = 0. The bootstrap distribution is centered at
    observed_mean. The one-sided fraction uses (count + 1) / (B + 1) so that
    the p-value is never exactly 0; the minimum possible value is 2 / (B + 1).
    """
    n = len(boot_means)
    if n == 0:
        return 1.0
    if observed_mean > 0:
        count = int(np.sum(boot_means <= 0))
    else:
        count = int(np.sum(boot_means >= 0))
    frac = (count + 1) / (n + 1)
    return float(min(2.0 * frac, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Holm-Bonferroni correction
# ─────────────────────────────────────────────────────────────────────────────

def holm_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction; return adjusted p-values in the same order.

    Steps (0-indexed ranks):
      1. Sort p-values ascending.
      2. Adjusted[i] = min(1, max(adjusted[i-1], (m - i) × p[i]))   for rank i.
    """
    n = len(p_values)
    if n == 0:
        return []
    idx_sorted = np.argsort(p_values)
    adjusted = np.zeros(n)
    max_so_far = 0.0
    for i, orig_idx in enumerate(idx_sorted):
        corrected = min(1.0, (n - i) * p_values[orig_idx])
        corrected = max(corrected, max_so_far)
        adjusted[orig_idx] = corrected
        max_so_far = corrected
    return adjusted.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Full segment analysis run
# ─────────────────────────────────────────────────────────────────────────────

def _run_one_segment(
    seg_label: str,
    seg_custs: list[int],
    predictions_ranker: dict[int, list[int]],
    predictions_heuristic: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 12,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict | None:
    """Compute MAP@12 + bootstrap CI + p-value for a single segment."""
    seg_gt = {c: ground_truth[c] for c in seg_custs if c in ground_truth}
    n = len(seg_gt)
    if n == 0:
        return None

    seg_preds_r = {c: predictions_ranker.get(c, []) for c in seg_gt}
    seg_preds_h = {c: predictions_heuristic.get(c, []) for c in seg_gt}

    map_r = map_at_k(seg_preds_r, seg_gt, k)
    map_h = map_at_k(seg_preds_h, seg_gt, k)

    ap_r = per_customer_ap(seg_preds_r, seg_gt, k)
    ap_h = per_customer_ap(seg_preds_h, seg_gt, k)
    boot = bootstrap_paired_diff(ap_r, ap_h, n=n_boot, seed=seed)

    arr_r = np.array([ap_r[c] for c in seg_gt])
    arr_h = np.array([ap_h[c] for c in seg_gt])
    diff = arr_r - arr_h
    n_custs = len(diff)

    # Bootstrap p-value (kept for backward compat; resolution-limited at B=1000)
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        diff[rng.integers(0, n_custs, size=n_custs)].mean()
        for _ in range(n_boot)
    ])
    p_bootstrap = p_value_from_bootstrap(boot_means, float(diff.mean()))

    # Paired t-test p-value (continuous; preferred for Holm correction)
    p_ttest = p_value_from_paired_ttest(arr_r, arr_h)

    return {
        "n": n,
        "map@12_ranker": map_r,
        "map@12_heuristic": map_h,
        "ci_lo": boot["ci_lo"],
        "ci_hi": boot["ci_hi"],
        "ci_excludes_zero": boot["ci_excludes_zero"],
        "p_raw": p_bootstrap,
        "p_ttest": p_ttest,
    }


def run_segment_analysis(
    predictions_ranker: dict[int, list[int]],
    predictions_heuristic: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    segments: dict[str, dict[str, list[int]]],
    k: int = 12,
    n_boot: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """Run segment analysis with Holm correction across all segment tests.

    Parameters
    ----------
    segments: family -> {label -> [customer_idx]} from compute_extended_segments()
    """
    rows: list[dict] = []
    p_raws: list[float] = []
    row_indices: list[int] = []

    # Collect all segment results
    for family, seg_dict in segments.items():
        for label, custs in seg_dict.items():
            res = _run_one_segment(
                label, custs, predictions_ranker, predictions_heuristic,
                ground_truth, k, n_boot, seed,
            )
            if res is None:
                continue
            row = {"family": family, "segment": label, **res}
            rows.append(row)
            p_raws.append(res["p_raw"])
            row_indices.append(len(rows) - 1)

    # Holm correction on bootstrap p-values (kept for backward compat)
    if p_raws:
        p_adj_boot = holm_correction(p_raws)
        for i, row_idx in enumerate(row_indices):
            rows[row_idx]["p_adj"] = p_adj_boot[i]
            rows[row_idx]["significant_adj"] = p_adj_boot[i] < 0.05
            rows[row_idx]["flagged"] = not rows[row_idx]["ci_excludes_zero"]

    # Holm correction on paired t-test p-values (preferred; continuous values)
    p_ttests = [rows[row_idx]["p_ttest"] for row_idx in row_indices]
    if p_ttests:
        p_adj_ttest = holm_correction(p_ttests)
        for i, row_idx in enumerate(row_indices):
            rows[row_idx]["p_ttest_adj"] = p_adj_ttest[i]
            rows[row_idx]["significant_ttest_adj"] = p_adj_ttest[i] < 0.05

    return rows
