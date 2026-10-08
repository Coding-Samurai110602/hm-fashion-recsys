"""Part D: Simulated A/B test (offline replay on fold 103) — vectorized.

All per-customer hit@12 and AP@12 arrays are computed once; every simulation
(A/B split, empirical power, A/A, peeking, CUPED) runs as numpy index/
permutation operations on those arrays.

Usage:
    caffeinate -i python -u scripts/run_experiment.py --n-sims 1000 \\
        2>&1 | tee reports/evaluation/ab_run.log
    python scripts/run_experiment.py --n-sims 200   # quick test

Writes: reports/evaluation/ab_test.json
        reports/evaluation/segment_analysis.json  (updated with t-test p-values)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# A/A test always uses 10,000 splits for a tight binomial CI (±0.43pp at p=0.05).
N_AA_SIMS = 10_000

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy import stats as scipy_stats
from statsmodels.stats.proportion import proportion_confint

from src.config import PROCESSED_DIR, REPORTS_DIR
from src.data_io import load_customers, load_transactions
from src.evaluation.ab_sim import (
    OFFLINE_REPLAY_ASSUMPTION,
    cuped_adjustment,
    two_proportion_z_test,
    welch_t_test,
)
from src.evaluation.power import (
    analytical_power,
    mde_curve,
    required_n_per_arm,
    statsmodels_n_per_arm,
    weeks_needed,
)
from src.evaluation.segments import (
    compute_extended_segments,
    holm_correction,
    p_value_from_paired_ttest,
    run_segment_analysis,
)
from src.metrics import map_at_k, per_customer_ap
from src.model.evaluate import bootstrap_paired_diff, score_fold
from src.time_split import build_fold

EVAL_DIR = REPORTS_DIR / "evaluation"
FEATURES_DIR = PROCESSED_DIR / "features"
MODELS_DIR = Path(__file__).parent.parent / "models"

norm_cdf = scipy_stats.norm.cdf


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized primitives (all operate on pre-built numpy arrays)
# ─────────────────────────────────────────────────────────────────────────────

def _vz_reject(sum_a: np.ndarray, n_a: int,
               sum_b: np.ndarray, n_b: int, alpha: float) -> np.ndarray:
    """Vectorized two-proportion z-test rejection. Returns bool array."""
    p_a = sum_a / n_a
    p_b = sum_b / n_b
    p_pool = (sum_a + sum_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    z = np.where(se > 0.0, (p_b - p_a) / se, 0.0)
    return 2.0 * (1.0 - norm_cdf(np.abs(z))) < alpha


def _vz_stats(sum_a: float, n_a: int, sum_b: float, n_b: int) -> dict:
    """Single-scalar z-test (for reporting the one main A/B split)."""
    return two_proportion_z_test(int(sum_a), n_a, int(sum_b), n_b)


def _aa_test_vec(hits: np.ndarray, n_sims: int, seed: int, alpha: float) -> dict:
    """Vectorized A/A test: n_sims random 50/50 splits, both arms heuristic."""
    rng = np.random.default_rng(seed)
    n = len(hits)
    mid = n // 2
    n_b = n - mid
    rejections = 0
    BATCH = 50
    for i in range(0, n_sims, BATCH):
        bs = min(BATCH, n_sims - i)
        # (bs, n) array of random floats; argsort gives bs independent permutations
        perm = rng.random((bs, n)).argsort(axis=1)
        sa = hits[perm[:, :mid]].sum(axis=1)
        sb = hits[perm[:, mid:]].sum(axis=1)
        rejections += int(_vz_reject(sa, mid, sb, n_b, alpha).sum())
    fpr = rejections / n_sims
    return {
        "n_sims": n_sims,
        "alpha": alpha,
        "false_positive_rate": float(fpr),
        "rejections": rejections,
        "expected_fpr": alpha,
        "within_expected": abs(fpr - alpha) < 0.02,
    }


def _peeking_vec(hits: np.ndarray, n_days: int,
                 n_sims: int, seed: int, alpha: float) -> dict:
    """Vectorized peeking simulation under H0 (both arms = heuristic hits)."""
    rng = np.random.default_rng(seed)
    n = len(hits)
    bucket_size = n // n_days
    BATCH = 50
    peek_rej = 0
    fixed_rej = 0

    for i in range(0, n_sims, BATCH):
        bs = min(BATCH, n_sims - i)
        perm = rng.random((bs, n)).argsort(axis=1)   # (bs, n)
        cum_a = np.zeros(bs)
        cum_b = np.zeros(bs)
        cum_na = 0
        cum_nb = 0
        peeked = np.zeros(bs, dtype=bool)

        for day in range(n_days):
            start = day * bucket_size
            end = start + bucket_size if day < n_days - 1 else n
            day_n = end - start
            day_mid = day_n // 2
            day_custs = perm[:, start:end]           # (bs, day_n)
            cum_a += hits[day_custs[:, :day_mid]].sum(axis=1)
            cum_b += hits[day_custs[:, day_mid:]].sum(axis=1)
            cum_na += day_mid
            cum_nb += day_n - day_mid

            if cum_na > 0 and cum_nb > 0:
                reject = _vz_reject(cum_a, cum_na, cum_b, cum_nb, alpha)
                peeked |= reject
                if day == n_days - 1:
                    fixed_rej += int(reject.sum())

        peek_rej += int(peeked.sum())

    peek_fpr = peek_rej / n_sims
    fixed_fpr = fixed_rej / n_sims
    return {
        "n_sims": n_sims,
        "n_days": n_days,
        "alpha": alpha,
        "peeking_false_positive_rate": float(peek_fpr),
        "fixed_horizon_false_positive_rate": float(fixed_fpr),
        "inflation_factor": float(peek_fpr / fixed_fpr) if fixed_fpr > 0 else float("nan"),
    }


def _emp_power_vec(hit_h: np.ndarray, hit_r: np.ndarray,
                   n_values: list[int], n_sims: int, seed: int, alpha: float) -> list[dict]:
    """Vectorized empirical power curve."""
    rng = np.random.default_rng(seed)
    n = len(hit_h)
    results = []
    for n_per_arm in n_values:
        if 2 * n_per_arm > n:
            results.append({"n_per_arm": n_per_arm,
                             "empirical_power": float("nan"),
                             "note": "exceeds pool"})
            continue
        rejections = 0
        BATCH = 50
        for i in range(0, n_sims, BATCH):
            bs = min(BATCH, n_sims - i)
            # Each row: 2*n_per_arm unique indices from n
            perm = rng.random((bs, n)).argsort(axis=1)[:, :2 * n_per_arm]
            sa = hit_h[perm[:, :n_per_arm]].sum(axis=1)
            sb = hit_r[perm[:, n_per_arm:]].sum(axis=1)
            rejections += int(_vz_reject(sa, n_per_arm, sb, n_per_arm, alpha).sum())
        results.append({"n_per_arm": n_per_arm,
                         "empirical_power": float(rejections / n_sims)})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_from_parquet(path: Path) -> dict[int, list[int]]:
    df = pl.read_parquet(path)
    top12 = (
        df.filter(pl.col("final_rank") <= 12)
        .sort(["customer_idx", "final_rank"])
        .group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_idx").alias("articles"))
    )
    return {r["customer_idx"]: list(r["articles"]) for r in top12.iter_rows(named=True)}


def _build_article_pop_ranks(tx_lf: pl.LazyFrame, cutoff_week: int) -> dict[int, int]:
    pop_df = (
        tx_lf
        .filter(pl.col("week_idx") < cutoff_week)
        .group_by("article_idx")
        .agg(pl.len().alias("sales_count"))
        .sort("sales_count", descending=True)
        .with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("pop_rank"))
        .collect()
    )
    return {r["article_idx"]: r["pop_rank"] for r in pop_df.iter_rows(named=True)}


def _build_sold_articles(tx_lf: pl.LazyFrame, cutoff_week: int) -> set[int]:
    return set(
        tx_lf.filter(pl.col("week_idx") < cutoff_week)
        .select("article_idx").unique().collect()["article_idx"].to_list()
    )


def _guardrail_stats(preds: dict[int, list[int]],
                     customers: list[int],
                     sold_articles: set[int],
                     article_pop_ranks: dict[int, int]) -> dict:
    arm_preds = {c: preds.get(c, []) for c in customers if c in preds}
    rec_set: set[int] = set()
    for arts in arm_preds.values():
        rec_set.update(arts)
    coverage = len(rec_set) / len(sold_articles) if sold_articles else 0.0
    pop_vals = []
    for arts in arm_preds.values():
        ranks = [article_pop_ranks[a] for a in arts if a in article_pop_ranks]
        if ranks:
            pop_vals.append(float(np.mean(ranks)))
    pop_bias = float(np.mean(pop_vals)) if pop_vals else float("nan")
    return {"catalog_coverage": coverage, "popularity_bias_mean_rank": pop_bias}


def _realistic_power_table(p_ctrl: float,
                            weekly_buyers: int,
                            var_reduction_pct: float) -> list[dict]:
    """Required customers per arm and weeks for relative lifts; with/without CUPED."""
    adj_factor = 1.0 - (var_reduction_pct / 100.0)
    weekly_per_arm = weekly_buyers / 2.0
    rows = []
    lifts = [0.01, 0.02, 0.05, 0.10, None]  # None = observed lift
    for lift in lifts:
        if lift is None:
            label = "observed"
            p_treat = None  # filled after we know the observed hit rate
        else:
            label = f"{int(lift*100)}%"
            p_treat = p_ctrl * (1.0 + lift)
        rows.append({"lift_label": label, "p_control": p_ctrl,
                     "p_treat": p_treat, "adj_factor": adj_factor,
                     "weekly_per_arm": weekly_per_arm})
    return rows


def _fill_power_table(rows: list[dict], p_ranker: float, weekly_per_arm: float) -> list[dict]:
    """Fill in required_n and weeks_needed for each row; uses p_ranker for observed lift."""
    out = []
    for r in rows:
        p_ctrl = r["p_control"]
        p_treat = r["p_treat"] if r["p_treat"] is not None else p_ranker
        lift_label = r["lift_label"]
        adj = r["adj_factor"]
        try:
            n_base = required_n_per_arm(p_ctrl, p_treat)
            n_cuped = max(1, round(n_base * adj))
            wks_base = weeks_needed(n_base, weekly_per_arm)
            wks_cuped = weeks_needed(n_cuped, weekly_per_arm)
        except (ValueError, ZeroDivisionError):
            n_base = n_cuped = wks_base = wks_cuped = float("nan")
        out.append({
            "relative_lift": lift_label,
            "p_control": p_ctrl,
            "p_treat": p_treat,
            "n_per_arm_no_cuped": n_base if not math.isnan(float(n_base)) else None,
            "n_per_arm_cuped": n_cuped if not math.isnan(float(n_cuped)) else None,
            "weeks_no_cuped": round(wks_base, 2) if not math.isnan(float(wks_base)) else None,
            "weeks_cuped": round(wks_cuped, 2) if not math.isnan(float(wks_cuped)) else None,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(n_sims: int = 1000) -> None:
    print("=" * 60, flush=True)
    print("PART D — Simulated A/B Test (fold 103 offline replay)", flush=True)
    print("=" * 60, flush=True)
    print(f"  Assumption: {OFFLINE_REPLAY_ASSUMPTION}", flush=True)
    print(f"  n_sims={n_sims}", flush=True)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── [1/9] Load model + score fold 103 ────────────────────────────────────
    print("\n  [1/9] Loading model and scoring fold_103.parquet ...", flush=True)
    t1 = time.time()
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_ranker.txt"))
    df_103 = pl.read_parquet(FEATURES_DIR / "fold_103.parquet")
    print(f"    fold_103 loaded: {len(df_103):,} rows in {time.time()-t1:.1f}s", flush=True)
    _, gt_103, eval_custs = build_fold(103)
    print(f"    Ground truth: {len(gt_103):,} buyers", flush=True)
    preds_ranker, _ = score_fold(booster, df_103, k=12, sanity_check=False)
    preds_heuristic = _heuristic_from_parquet(FEATURES_DIR / "fold_103.parquet")
    print(f"    Predictions ready in {time.time()-t1:.1f}s total", flush=True)

    # ── [2/9] Build all per-customer arrays once ──────────────────────────────
    print("\n  [2/9] Building per-customer arrays ...", flush=True)
    customers = sorted(gt_103.keys())
    n_custs = len(customers)

    ap_h_map = per_customer_ap(preds_heuristic, gt_103, 12)
    ap_r_map = per_customer_ap(preds_ranker, gt_103, 12)

    # Per-customer hit indicators (1 if ≥1 hit in top-12)
    hit_h = np.array([
        float(any(a in gt_103.get(c, set()) for a in preds_heuristic.get(c, [])[:12]))
        for c in customers
    ])
    hit_r = np.array([
        float(any(a in gt_103.get(c, set()) for a in preds_ranker.get(c, [])[:12]))
        for c in customers
    ])
    ap_h = np.array([ap_h_map.get(c, 0.0) for c in customers])
    ap_r = np.array([ap_r_map.get(c, 0.0) for c in customers])

    p_heuristic = float(hit_h.mean())
    p_ranker    = float(hit_r.mean())
    print(f"    Heuristic hit rate@12: {p_heuristic:.4f}", flush=True)
    print(f"    Ranker hit rate@12:    {p_ranker:.4f}", flush=True)
    print(f"    n_eval_customers: {n_custs:,}", flush=True)

    # ── [3/9] Load covariates + transaction data ──────────────────────────────
    print("\n  [3/9] Loading covariates + transaction metadata ...", flush=True)
    df_feats = df_103  # already loaded

    # Covariate (a): 4-week purchase count before week-103 cutoff
    covar_df = (
        df_feats
        .group_by("customer_idx")
        .agg(pl.col("c_n_purchases_4w").first())
        .select(["customer_idx", "c_n_purchases_4w"])
    )
    covar_map = {
        r["customer_idx"]: float(r["c_n_purchases_4w"]) if r["c_n_purchases_4w"] is not None else 0.0
        for r in covar_df.iter_rows(named=True)
    }
    covar = np.array([covar_map.get(c, 0.0) for c in customers])
    print(f"    Covariate (a) c_n_purchases_4w: loaded for {len(covar_map):,} customers", flush=True)

    # Covariate (b): heuristic hit@12 in week 102 (pre-period same metric)
    #   Handling: customers in week-103 eval set who did NOT buy in week 102
    #   have no week-102 predictions; their covariate is set to 0.0 and
    #   tracked via n_no_pre_period_data.
    _, gt_102, _ = build_fold(102)
    preds_h102 = _heuristic_from_parquet(FEATURES_DIR / "fold_102_full.parquet")
    covar_pre_map: dict[int, float] = {}
    n_no_pre_period = 0
    for c in customers:
        if c in gt_102 and c in preds_h102:
            pred102 = preds_h102[c][:12]
            covar_pre_map[c] = float(any(a in gt_102[c] for a in pred102))
        else:
            covar_pre_map[c] = 0.0
            n_no_pre_period += 1
    covar_pre = np.array([covar_pre_map[c] for c in customers])
    n_with_pre = n_custs - n_no_pre_period
    print(f"    Covariate (b) heuristic_hit12_week102: {n_with_pre:,} with week-102 data, "
          f"{n_no_pre_period:,} set to 0 (no pre-period purchase)", flush=True)

    tx_lf = load_transactions()
    article_pop_ranks = _build_article_pop_ranks(tx_lf, 103)
    sold_articles = _build_sold_articles(tx_lf, 103)
    print(f"    Covariates + tx metadata ready in {time.time()-t0:.1f}s total", flush=True)

    # ── [4/9] Power analysis ──────────────────────────────────────────────────
    print("\n  [4/9] Power analysis ...", flush=True)
    n_req = required_n_per_arm(p_heuristic, p_ranker, alpha=0.05, power=0.8)
    sm = statsmodels_n_per_arm(p_heuristic, p_ranker, alpha=0.05, power=0.8)
    weekly_per_arm = n_custs / 2.0
    weeks_base = weeks_needed(n_req, weekly_per_arm)

    n_curve_vals = [500, 1000, 2000, 5000, 10000, 20000, 36000, 72000]
    mde_pts = mde_curve(p_heuristic, n_curve_vals)

    print(f"    Required n/arm (formula):     {n_req:,}", flush=True)
    print(f"    Required n/arm (statsmodels): {sm['statsmodels_n_per_arm']:,}", flush=True)
    print(f"    Cohen h:                      {sm['effect_size_h']:.4f}", flush=True)
    print(f"    Weekly customers per arm:     {weekly_per_arm:,.0f}", flush=True)
    print(f"    Weeks needed (50/50 split):   {weeks_base:.2f}", flush=True)

    power_analysis = {
        "p_control": p_heuristic,
        "p_treat": p_ranker,
        "alpha": 0.05,
        "power": 0.8,
        "n_per_arm_formula": n_req,
        "n_per_arm_statsmodels": sm["statsmodels_n_per_arm"],
        "effect_size_h": sm["effect_size_h"],
        "weekly_buying_customers_fold103": n_custs,
        "weekly_customers_per_arm": weekly_per_arm,
        "weeks_needed": weeks_base,
        "mde_curve": mde_pts,
        "population_caveat": (
            "The offline metric is computed over customers who purchased in the target week. "
            "An online test would randomize all active visitors, lowering the baseline hit rate "
            "and increasing the required sample size."
        ),
    }

    # ── [5/9] Main A/B simulation (one seed=42 split) ─────────────────────────
    print("\n  [5/9] Main A/B simulation (seed=42, 50/50 split) ...", flush=True)
    rng42 = np.random.default_rng(42)
    perm42 = rng42.permutation(n_custs)
    mid = n_custs // 2
    idx_a = perm42[:mid]
    idx_b = perm42[mid:]

    hits_arm_a = hit_h[idx_a].sum()
    hits_arm_b = hit_r[idx_b].sum()
    n_a, n_b = len(idx_a), len(idx_b)

    z_res = _vz_stats(hits_arm_a, n_a, hits_arm_b, n_b)
    arr_a = ap_h[idx_a]
    arr_b = ap_r[idx_b]
    t_res = welch_t_test(arr_a, arr_b)

    arm_a_custs = [customers[i] for i in idx_a]
    arm_b_custs = [customers[i] for i in idx_b]
    guardrail_a = _guardrail_stats(preds_heuristic, arm_a_custs, sold_articles, article_pop_ranks)
    guardrail_b = _guardrail_stats(preds_ranker, arm_b_custs, sold_articles, article_pop_ranks)

    print(f"    Hit rate z-test: z={z_res['z']:.3f}, p={z_res['p_value']:.4f}, "
          f"effect={z_res['effect']:+.4f}", flush=True)
    print(f"    AP Welch t-test: t={t_res['t']:.3f}, p={t_res['p_value']:.4f}", flush=True)
    print(f"    Guardrail arm_a coverage={guardrail_a['catalog_coverage']:.4f} "
          f"pop_rank={guardrail_a['popularity_bias_mean_rank']:.0f}", flush=True)
    print(f"    Guardrail arm_b coverage={guardrail_b['catalog_coverage']:.4f} "
          f"pop_rank={guardrail_b['popularity_bias_mean_rank']:.0f}", flush=True)

    simulation = {
        "assumption": OFFLINE_REPLAY_ASSUMPTION,
        "seed": 42,
        "n_a": n_a,
        "n_b": n_b,
        "hit_rate_z_test": z_res,
        "ap_welch_t_test": t_res,
        "guardrail_arm_a": guardrail_a,
        "guardrail_arm_b": guardrail_b,
    }

    # ── [6/9] A/A test (10,000 splits) ────────────────────────────────────────
    print(f"\n  [6/9] A/A test ({N_AA_SIMS:,} splits, both arms heuristic) ...", flush=True)
    t_aa = time.time()
    aa_result = _aa_test_vec(hit_h, n_sims=N_AA_SIMS, seed=42, alpha=0.05)
    k_rej = aa_result["rejections"]
    # 95% Clopper-Pearson binomial CI on the observed FPR
    ci_lo_aa, ci_hi_aa = proportion_confint(k_rej, N_AA_SIMS, alpha=0.05, method="binom_test")
    alpha_in_ci = bool(ci_lo_aa <= 0.05 <= ci_hi_aa)
    aa_result["fpr_ci_lo"] = float(ci_lo_aa)
    aa_result["fpr_ci_hi"] = float(ci_hi_aa)
    aa_result["alpha_in_ci"] = alpha_in_ci
    aa_result["within_expected"] = alpha_in_ci  # now CI-based, not ±2pp heuristic
    fpr = aa_result["false_positive_rate"]
    print(f"    False-positive rate: {fpr:.4f} ({k_rej}/{N_AA_SIMS}) "
          f"in {time.time()-t_aa:.1f}s", flush=True)
    print(f"    95% binomial CI: [{ci_lo_aa:.4f}, {ci_hi_aa:.4f}]  "
          f"alpha=5% within CI: {alpha_in_ci}", flush=True)
    if not alpha_in_ci:
        print("    WARNING: nominal alpha (5%) is OUTSIDE the 95% CI — investigating below.",
              flush=True)
        # Check 1: Is the z-test approximation well-behaved?
        #   n*p = N_AA_SIMS * 0.05 = 500 >> 5 → normal approximation valid.
        # Check 2: split logic — each split is an independent rng.random() argsort.
        #   With large n (72k) and balanced split (36k each), the z-test SE is
        #   sqrt(0.0857*0.9143/36009 + 0.0857*0.9143/36010) ≈ 0.00147; any
        #   observed p̂ imbalance across arms exceeds the SE by chance at rate 5%.
        # Conclusion: if CI excludes 5%, report that FPR ≈ {fpr:.1%} and note
        #   variance from MC sampling; with 10k sims the MC error is ≈ 0.43pp.
        print(f"    Note: z-test approximation is valid (n*alpha=500 >> 5). "
              f"FPR {fpr:.1%} is {abs(fpr-0.05)*100:.2f}pp from nominal.",
              flush=True)

    # ── [7/9] Peeking simulation ──────────────────────────────────────────────
    print(f"\n  [7/9] Peeking simulation (14-day, {n_sims} sims) ...", flush=True)
    t_pk = time.time()
    peek_result = _peeking_vec(hit_h, n_days=14, n_sims=n_sims, seed=42, alpha=0.05)
    print(f"    Fixed-horizon FPR: {peek_result['fixed_horizon_false_positive_rate']:.3f}", flush=True)
    print(f"    Peeking FPR:       {peek_result['peeking_false_positive_rate']:.3f}", flush=True)
    print(f"    Inflation factor:  {peek_result['inflation_factor']:.2f}x "
          f"in {time.time()-t_pk:.1f}s", flush=True)

    # ── [8/9] Empirical power + analytical power ──────────────────────────────
    print(f"\n  [8/9] Empirical power curve ({n_sims} sims × 7 n_values) ...", flush=True)
    t_ep = time.time()
    emp_n_values = [500, 1000, 2000, 5000, 10000, 20000, 36000]
    emp_power = _emp_power_vec(
        hit_h, hit_r, n_values=emp_n_values, n_sims=n_sims, seed=42, alpha=0.05
    )
    # Annotate each point with analytical power and |empirical - analytical|
    for pt in emp_power:
        n_val = pt["n_per_arm"]
        anl = analytical_power(p_heuristic, p_ranker, n_val, alpha=0.05)
        pt["analytical_power"] = float(anl)
        ep = pt.get("empirical_power", float("nan"))
        pt["abs_diff"] = float(abs(ep - anl)) if not math.isnan(ep) else float("nan")
    print(f"    {'n/arm':>8s}  {'empirical':>10s}  {'analytical':>11s}  {'|diff|':>8s}",
          flush=True)
    for pt in emp_power:
        ep = pt.get("empirical_power", float("nan"))
        anl = pt["analytical_power"]
        diff = pt["abs_diff"]
        note = pt.get("note", "")
        ep_s = f"{ep:.3f}" if not math.isnan(ep) else "n/a"
        print(f"    {pt['n_per_arm']:>8,}  {ep_s:>10s}  {anl:>11.3f}  {diff:>8.3f}  {note}",
              flush=True)
    print(f"    Empirical power elapsed: {time.time()-t_ep:.1f}s", flush=True)

    # ── CUPED (primary metric: hit rate@12) ────────────────────────────────────
    print("\n  [8b/9] CUPED on hit rate@12 — comparing two covariates ...", flush=True)

    # Covariate (a): 4-week purchase count before week-103 cutoff
    cov_a_ctrl = covar[idx_a]
    cov_a_trt  = covar[idx_b]
    cuped_a = cuped_adjustment(hit_r[idx_b], hit_h[idx_a], cov_a_trt, cov_a_ctrl)
    cuped_a["n_required"] = max(1, round(n_req * cuped_a["adj_factor_for_n"]))

    # Covariate (b): heuristic hit@12 in week 102
    #   Customers not in week-102 GT: covariate set to 0 (no_pre_period_data).
    cov_b_ctrl = covar_pre[idx_a]
    cov_b_trt  = covar_pre[idx_b]
    cuped_b = cuped_adjustment(hit_r[idx_b], hit_h[idx_a], cov_b_trt, cov_b_ctrl)
    cuped_b["n_required"] = max(1, round(n_req * cuped_b["adj_factor_for_n"]))

    print(f"    Covariate (a) c_n_purchases_4w:", flush=True)
    print(f"      corr(X, hit_rate@12)={cuped_a['covariate_correlation_with_metric']:.4f}  "
          f"var_red={cuped_a['variance_reduction_pct']:.2f}%  "
          f"n_required={cuped_a['n_required']:,}", flush=True)
    print(f"    Covariate (b) heuristic_hit12_week102 "
          f"({n_no_pre_period:,}/{n_custs:,} set to 0):", flush=True)
    print(f"      corr(X, hit_rate@12)={cuped_b['covariate_correlation_with_metric']:.4f}  "
          f"var_red={cuped_b['variance_reduction_pct']:.2f}%  "
          f"n_required={cuped_b['n_required']:,}", flush=True)

    # Best covariate = lower adj_factor (more variance reduction on hit rate)
    if cuped_a["adj_factor_for_n"] <= cuped_b["adj_factor_for_n"]:
        best_covar_id = "a"
        best_cuped_hit = cuped_a
    else:
        best_covar_id = "b"
        best_cuped_hit = cuped_b
    n_cuped = best_cuped_hit["n_required"]
    print(f"    Better covariate: ({best_covar_id})  n_required={n_cuped:,} vs {n_req:,} without CUPED",
          flush=True)

    # Secondary: AP@12 CUPED with covariate (a) for reference
    cuped_ap12 = cuped_adjustment(arr_b, arr_a, cov_a_trt, cov_a_ctrl)
    cuped_ap12["n_required_cuped"] = max(1, round(n_req * cuped_ap12["adj_factor_for_n"]))
    cuped_ap12["n_required_unadjusted"] = n_req

    cuped_out = {
        "primary_metric": "hit_rate@12",
        "covariate_a": {
            "name": "c_n_purchases_4w",
            "description": "4-week purchase count before week-103 cutoff",
            "theta": cuped_a["theta"],
            "x_mean": cuped_a["x_mean"],
            "corr_with_hit_rate": cuped_a["covariate_correlation_with_metric"],
            "r_squared": cuped_a["r_squared"],
            "variance_reduction_pct": cuped_a["variance_reduction_pct"],
            "adj_factor_for_n": cuped_a["adj_factor_for_n"],
            "n_required": cuped_a["n_required"],
        },
        "covariate_b": {
            "name": "heuristic_hit12_week102",
            "description": "Heuristic hit@12 in week 102 (pre-period, same metric)",
            "no_pre_period_handling": (
                "Customers not in week-102 GT have no pre-period prediction; "
                "covariate set to 0.0. See n_no_pre_period_data."
            ),
            "n_no_pre_period_data": n_no_pre_period,
            "n_with_pre_period_data": n_custs - n_no_pre_period,
            "theta": cuped_b["theta"],
            "x_mean": cuped_b["x_mean"],
            "corr_with_hit_rate": cuped_b["covariate_correlation_with_metric"],
            "r_squared": cuped_b["r_squared"],
            "variance_reduction_pct": cuped_b["variance_reduction_pct"],
            "adj_factor_for_n": cuped_b["adj_factor_for_n"],
            "n_required": cuped_b["n_required"],
        },
        "best_covariate": best_covar_id,
        "best_covariate_name": (
            "c_n_purchases_4w" if best_covar_id == "a" else "heuristic_hit12_week102"
        ),
        "n_required_unadjusted": n_req,
        "n_required_cuped": n_cuped,
        "secondary_ap12_covariate_a": cuped_ap12,
    }

    # ── Realistic power table (uses best covariate on hit rate) ──────────────
    var_red_pct = best_cuped_hit["variance_reduction_pct"]
    power_table_rows = _realistic_power_table(p_heuristic, n_custs, var_red_pct)
    power_table = _fill_power_table(power_table_rows, p_ranker, weekly_per_arm)
    for row in power_table:
        row["cuped_covariate"] = best_covar_id
    best_name = cuped_out["best_covariate_name"]
    print(f"\n  Realistic power table (CUPED covariate: ({best_covar_id}) {best_name}, "
          f"weekly_buyers={n_custs:,}, α=0.05, power=0.8):", flush=True)
    print(f"    {'Lift':>10s}  {'n/arm (no CUPED)':>18s}  {'weeks':>8s}  "
          f"{'n/arm (CUPED)':>15s}  {'weeks':>8s}", flush=True)
    for row in power_table:
        n_b = row["n_per_arm_no_cuped"] or "—"
        n_c = row["n_per_arm_cuped"] or "—"
        w_b = f"{row['weeks_no_cuped']:.2f}" if row["weeks_no_cuped"] is not None else "—"
        w_c = f"{row['weeks_cuped']:.2f}" if row["weeks_cuped"] is not None else "—"
        print(f"    {row['relative_lift']:>10s}  {str(n_b):>18s}  {w_b:>8s}  "
              f"{str(n_c):>15s}  {w_c:>8s}", flush=True)

    # ── Hand-example z-test verification ─────────────────────────────────────
    hand = two_proportion_z_test(hits_a=100, n_a=1000, hits_b=150, n_b=1000)
    assert abs(hand["z"] - 3.381) < 0.01, f"Hand z mismatch: {hand['z']:.4f}"
    print("\n  Hand-example z-test: PASS", flush=True)

    # ── [9/9] Segment p-values (paired t-test) ────────────────────────────────
    print("\n  [9/9] Recomputing segment p-values (paired t-test) ...", flush=True)
    t_seg = time.time()
    seg_path = EVAL_DIR / "segment_analysis.json"
    if seg_path.exists():
        with open(seg_path) as f:
            old_segs = json.load(f)
        updated_segs = _recompute_segment_ttests(
            old_segs, preds_ranker, preds_heuristic, gt_103
        )
        with open(seg_path, "w") as f:
            json.dump(updated_segs, f, indent=2)
        print(f"    segment_analysis.json updated in {time.time()-t_seg:.1f}s", flush=True)
    else:
        print("    segment_analysis.json not found; skipping segment update.", flush=True)

    # ── Assemble and write ab_test.json ───────────────────────────────────────
    runtime_s = round(time.time() - t0, 1)
    output = {
        "assumption": OFFLINE_REPLAY_ASSUMPTION,
        "fold": 103,
        "runtime_seconds": runtime_s,
        "power_analysis": power_analysis,
        "simulation": simulation,
        "aa_test": aa_result,
        "peeking": peek_result,
        "empirical_power_curve": emp_power,
        "cuped": cuped_out,
        "realistic_power_table": power_table,
        "hand_example_z_test": hand,
    }

    with open(EVAL_DIR / "ab_test.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {EVAL_DIR / 'ab_test.json'}", flush=True)
    print(f"  Total runtime: {runtime_s:.1f}s ({runtime_s/60:.1f}min)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Segment t-test recomputation (in-memory, no re-scoring)
# ─────────────────────────────────────────────────────────────────────────────

def _recompute_segment_ttests(
    old_segs: dict,
    preds_ranker: dict[int, list[int]],
    preds_heuristic: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 12,
) -> dict:
    """Recompute paired t-test p-values and Holm adjustment for each segment.

    Uses per-customer AP arrays computed from existing predictions; does not
    re-run the full segment pipeline.
    """
    from src.metrics import per_customer_ap
    ap_r_all = per_customer_ap(preds_ranker, ground_truth, k)
    ap_h_all = per_customer_ap(preds_heuristic, ground_truth, k)

    results = old_segs.get("results", [])
    # We need the customer lists per segment to compute per-customer AP.
    # Since segment_analysis.json stores only aggregated MAP@12 (not customer lists),
    # we approximate using the full population AP for t-test on the n customers
    # who belong to each segment — but we can't recover the exact segment members
    # from the JSON. Instead, recompute from ground_truth customer IDs.
    # The segment family/label mapping can be inferred from the JSON metadata.
    # For a statistically valid t-test we need per-segment customer AP diffs.
    # Fallback: keep bootstrap p_raw, add note.
    #
    # To get per-segment customers, we need to recompute segments — which requires
    # transaction data + customers parquet.  Do that here.
    tx_lf = load_transactions()
    customers_lf = load_customers()
    eval_customers = sorted(ground_truth.keys())

    from src.evaluation.segments import compute_extended_segments
    try:
        segments = compute_extended_segments(
            tx_lf, customers_lf, cutoff_week=103,
            eval_customers=eval_customers, ground_truth=ground_truth,
        )
    except Exception as e:
        print(f"    WARNING: Could not recompute segments ({e}); using fallback.", flush=True)
        return old_segs

    # Build family→label→customer_idx_list lookup
    seg_lookup: dict[str, dict[str, list[int]]] = segments

    new_results = []
    p_ttests = []
    row_indices = []

    for row in results:
        fam = row.get("family", "")
        lab = row.get("segment", "")
        custs = seg_lookup.get(fam, {}).get(lab, [])
        seg_gt = {c: ground_truth[c] for c in custs if c in ground_truth}
        if not seg_gt:
            new_results.append(row)
            continue
        arr_r = np.array([ap_r_all.get(c, 0.0) for c in seg_gt])
        arr_h = np.array([ap_h_all.get(c, 0.0) for c in seg_gt])
        p_tt = p_value_from_paired_ttest(arr_r, arr_h)
        row = dict(row)
        row["p_ttest"] = p_tt
        new_results.append(row)
        p_ttests.append(p_tt)
        row_indices.append(len(new_results) - 1)

    # Holm on t-test p-values
    if p_ttests:
        adj = holm_correction(p_ttests)
        for i, ri in enumerate(row_indices):
            new_results[ri]["p_ttest_adj"] = adj[i]
            new_results[ri]["significant_ttest_adj"] = adj[i] < 0.05

    print(f"    Segments with t-test: {len(p_ttests)}", flush=True)
    if p_ttests:
        print(f"    p_ttest range: [{min(p_ttests):.2e}, {max(p_ttests):.2e}]", flush=True)
        print(f"    p_ttest_adj range: [{min(adj):.2e}, {max(adj):.2e}]", flush=True)

    return {**old_segs, "results": new_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=1000)
    args = parser.parse_args()
    main(n_sims=args.n_sims)
