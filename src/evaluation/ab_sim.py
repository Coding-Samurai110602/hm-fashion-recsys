"""Offline A/B test simulation on fold-103 replay.

Assumption (stated explicitly in all outputs): offline replay assumes customers
purchase the same items regardless of which list they were shown (no feedback
effect). Online lift may differ.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from src.evaluation.suite import per_customer_hit
from src.metrics import per_customer_ap

OFFLINE_REPLAY_ASSUMPTION = (
    "Offline replay assumes customers purchase the same items regardless of "
    "which list was shown (no feedback effect); online lift may differ."
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_split(
    customers: list[int], seed: int
) -> tuple[list[int], list[int]]:
    """Random 50/50 split into arm A and arm B."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(customers))
    mid = len(customers) // 2
    idx_a = order[:mid]
    idx_b = order[mid:]
    return [customers[i] for i in idx_a], [customers[i] for i in idx_b]


def two_proportion_z_test(
    hits_a: int, n_a: int, hits_b: int, n_b: int
) -> dict:
    """Two-sided two-proportion z-test (pooled).

    Returns z, p_value, effect (p_B - p_A), 95% CI on the difference.
    """
    p_a = hits_a / n_a
    p_b = hits_b / n_b
    p_pool = (hits_a + hits_b) / (n_a + n_b)
    se_h0 = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    z = (p_b - p_a) / se_h0 if se_h0 > 0 else 0.0
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    # 95% CI on p_B - p_A (using unpooled SE for CI)
    se_ci = math.sqrt(p_a * (1.0 - p_a) / n_a + p_b * (1.0 - p_b) / n_b)
    z95 = stats.norm.ppf(0.975)
    effect = p_b - p_a
    return {
        "z": float(z),
        "p_value": float(p_val),
        "effect": float(effect),
        "ci_lo": float(effect - z95 * se_ci),
        "ci_hi": float(effect + z95 * se_ci),
        "p_a": float(p_a),
        "p_b": float(p_b),
        "n_a": n_a,
        "n_b": n_b,
    }


def welch_t_test(values_a: np.ndarray, values_b: np.ndarray) -> dict:
    """Welch's t-test (unequal variances, two-sided) on AP@12."""
    t, p = stats.ttest_ind(values_b, values_a, equal_var=False)
    effect = float(values_b.mean() - values_a.mean())
    se = math.sqrt(values_a.var(ddof=1) / len(values_a) + values_b.var(ddof=1) / len(values_b))
    z95 = stats.norm.ppf(0.975)
    return {
        "t": float(t),
        "p_value": float(p),
        "effect": effect,
        "ci_lo": float(effect - z95 * se),
        "ci_hi": float(effect + z95 * se),
        "mean_a": float(values_a.mean()),
        "mean_b": float(values_b.mean()),
        "n_a": len(values_a),
        "n_b": len(values_b),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Guardrails
# ─────────────────────────────────────────────────────────────────────────────

def _guardrail_stats(
    preds: dict[int, list[int]],
    customers: list[int],
    sold_articles: set[int],
    article_pop_ranks: dict[int, int],
) -> dict:
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


# ─────────────────────────────────────────────────────────────────────────────
# Main A/B simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_ab_simulation(
    preds_a: dict[int, list[int]],
    preds_b: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    sold_articles: set[int],
    article_pop_ranks: dict[int, int],
    seed: int = 42,
    k: int = 12,
) -> dict:
    """One A/B simulation: random 50/50 split, arm A=heuristic, arm B=ranker.

    Returns z-test on hit rate@k, Welch t-test on AP@k, guardrail values.
    """
    customers = sorted(ground_truth.keys())
    arm_a, arm_b = _random_split(customers, seed=seed)

    # Hit rate per arm (binary)
    hits_a = sum(per_customer_hit(preds_a, ground_truth, k).get(c, 0.0) for c in arm_a)
    hits_b = sum(per_customer_hit(preds_b, ground_truth, k).get(c, 0.0) for c in arm_b)
    n_a, n_b = len(arm_a), len(arm_b)

    z_res = two_proportion_z_test(int(hits_a), n_a, int(hits_b), n_b)

    # AP@12 per arm
    ap_a_all = per_customer_ap(preds_a, ground_truth, k)
    ap_b_all = per_customer_ap(preds_b, ground_truth, k)
    arr_a = np.array([ap_a_all.get(c, 0.0) for c in arm_a])
    arr_b = np.array([ap_b_all.get(c, 0.0) for c in arm_b])
    t_res = welch_t_test(arr_a, arr_b)

    guardrail_a = _guardrail_stats(preds_a, arm_a, sold_articles, article_pop_ranks)
    guardrail_b = _guardrail_stats(preds_b, arm_b, sold_articles, article_pop_ranks)

    return {
        "assumption": OFFLINE_REPLAY_ASSUMPTION,
        "seed": seed,
        "n_a": n_a,
        "n_b": n_b,
        "hit_rate_z_test": z_res,
        "ap_welch_t_test": t_res,
        "guardrail_arm_a": guardrail_a,
        "guardrail_arm_b": guardrail_b,
    }


# ─────────────────────────────────────────────────────────────────────────────
# A/A test: false positive rate
# ─────────────────────────────────────────────────────────────────────────────

def run_aa_test(
    preds_heuristic: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    n_sims: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    k: int = 12,
) -> dict:
    """A/A test: both arms heuristic, 1000 random splits.

    Returns the false-positive rate (should be ≈ α = 5%).
    """
    rng = np.random.default_rng(seed)
    customers = sorted(ground_truth.keys())
    n = len(customers)
    hit_vals = per_customer_hit(preds_heuristic, ground_truth, k)
    hits_arr = np.array([hit_vals.get(c, 0.0) for c in customers])

    rejections = 0
    for _ in range(n_sims):
        perm = rng.permutation(n)
        mid = n // 2
        idx_a = perm[:mid]
        idx_b = perm[mid:]
        ha = float(hits_arr[idx_a].sum())
        hb = float(hits_arr[idx_b].sum())
        res = two_proportion_z_test(int(ha), len(idx_a), int(hb), len(idx_b))
        if res["p_value"] < alpha:
            rejections += 1

    fpr = rejections / n_sims
    return {
        "n_sims": n_sims,
        "alpha": alpha,
        "false_positive_rate": float(fpr),
        "rejections": rejections,
        "expected_fpr": alpha,
        "within_expected": abs(fpr - alpha) < 0.02,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Peeking: daily looks with stop-at-first-significance
# ─────────────────────────────────────────────────────────────────────────────

def run_peeking_simulation(
    preds_a: dict[int, list[int]],
    preds_b: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    n_days: int = 14,
    n_sims: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    k: int = 12,
) -> dict:
    """Peeking simulation: stop as soon as p < α on any daily look.

    Under H0 (A=B=heuristic), measure how peeking inflates the false-positive
    rate relative to a fixed-horizon test at day 14.
    Fixed-horizon false-positive rate is computed from the same sims but only
    reading the final-day result.
    """
    rng = np.random.default_rng(seed)
    customers = sorted(ground_truth.keys())
    n = len(customers)

    # Precompute hit indicators for arm A (heuristic = preds_a)
    hit_vals_a = per_customer_hit(preds_a, ground_truth, k)
    hit_vals_b = per_customer_hit(preds_b, ground_truth, k)
    hits_a_arr = np.array([hit_vals_a.get(c, 0.0) for c in customers])
    hits_b_arr = np.array([hit_vals_b.get(c, 0.0) for c in customers])

    peeking_rejections = 0
    fixed_rejections = 0
    # Daily bucket sizes: divide n customers evenly across n_days
    bucket_size = n // n_days

    for _ in range(n_sims):
        perm = rng.permutation(n)
        # 50/50 assignment within each day's arrivals
        peeked_reject = False
        cum_arm_a_hits = 0.0
        cum_arm_b_hits = 0.0
        cum_na = 0
        cum_nb = 0

        for day in range(n_days):
            start = day * bucket_size
            end = start + bucket_size if day < n_days - 1 else n
            day_custs = perm[start:end]
            day_n = len(day_custs)
            day_mid = day_n // 2
            day_a = day_custs[:day_mid]
            day_b = day_custs[day_mid:]

            # Under H0: both arms use preds_a (heuristic)
            cum_arm_a_hits += hits_a_arr[day_a].sum()
            cum_arm_b_hits += hits_a_arr[day_b].sum()
            cum_na += len(day_a)
            cum_nb += len(day_b)

            if cum_na > 0 and cum_nb > 0:
                res = two_proportion_z_test(
                    int(cum_arm_a_hits), cum_na,
                    int(cum_arm_b_hits), cum_nb,
                )
                if not peeked_reject and res["p_value"] < alpha:
                    peeked_reject = True
                if day == n_days - 1:
                    if res["p_value"] < alpha:
                        fixed_rejections += 1

        if peeked_reject:
            peeking_rejections += 1

    peeking_fpr = peeking_rejections / n_sims
    fixed_fpr = fixed_rejections / n_sims
    return {
        "n_sims": n_sims,
        "n_days": n_days,
        "alpha": alpha,
        "peeking_false_positive_rate": float(peeking_fpr),
        "fixed_horizon_false_positive_rate": float(fixed_fpr),
        "inflation_factor": float(peeking_fpr / fixed_fpr) if fixed_fpr > 0 else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Empirical power at several sample sizes
# ─────────────────────────────────────────────────────────────────────────────

def empirical_power_curve(
    preds_a: dict[int, list[int]],
    preds_b: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    n_values: list[int],
    n_sims: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    k: int = 12,
) -> list[dict]:
    """Empirical power at each per-arm sample size via 1000 repeated splits.

    For each n, randomly subsample n customers per arm from the full pool,
    run the z-test n_sims times, count rejections = empirical power.
    """
    rng = np.random.default_rng(seed)
    customers = sorted(ground_truth.keys())
    n_total = len(customers)

    hit_vals_a = per_customer_hit(preds_a, ground_truth, k)
    hit_vals_b = per_customer_hit(preds_b, ground_truth, k)
    hits_a_arr = np.array([hit_vals_a.get(c, 0.0) for c in customers])
    hits_b_arr = np.array([hit_vals_b.get(c, 0.0) for c in customers])

    results = []
    for n_per_arm in n_values:
        if 2 * n_per_arm > n_total:
            results.append({"n_per_arm": n_per_arm, "empirical_power": float("nan"), "note": "exceeds pool"})
            continue

        rejections = 0
        for _ in range(n_sims):
            idx = rng.choice(n_total, size=2 * n_per_arm, replace=False)
            idx_a = idx[:n_per_arm]
            idx_b = idx[n_per_arm:]
            ha = int(hits_a_arr[idx_a].sum())
            hb = int(hits_b_arr[idx_b].sum())
            res = two_proportion_z_test(ha, n_per_arm, hb, n_per_arm)
            if res["p_value"] < alpha:
                rejections += 1
        results.append({
            "n_per_arm": n_per_arm,
            "empirical_power": float(rejections / n_sims),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CUPED: covariate-adjusted pre-experiment estimator
# ─────────────────────────────────────────────────────────────────────────────

def cuped_adjustment(
    metric_treat: np.ndarray,
    metric_control: np.ndarray,
    covariate_treat: np.ndarray,
    covariate_control: np.ndarray,
) -> dict:
    """CUPED (Controlled-experiment Using Pre-Experiment Data).

    Adjusts metric by subtracting the linear projection onto the covariate.
    θ is estimated from the control arm (or pooled); applied to both arms.

    Y_adj = Y − θ · (X − E[X])   where θ = Cov(X, Y) / Var(X).

    Returns variance reduction (%) and the adjusted Welch t-test result.
    """
    # Pool covariate and metric for θ estimation
    X_all = np.concatenate([covariate_treat, covariate_control])
    Y_all = np.concatenate([metric_treat, metric_control])
    var_x = float(np.var(X_all, ddof=1))
    cov_xy = float(np.cov(X_all, Y_all, ddof=1)[0, 1])
    theta = cov_xy / var_x if var_x > 0 else 0.0

    x_mean = float(X_all.mean())
    metric_treat_adj = metric_treat - theta * (covariate_treat - x_mean)
    metric_control_adj = metric_control - theta * (covariate_control - x_mean)

    # Variance reduction
    var_unadj = float(np.var(np.concatenate([metric_treat, metric_control]), ddof=1))
    var_adj = float(np.var(np.concatenate([metric_treat_adj, metric_control_adj]), ddof=1))
    var_reduction_pct = float(100.0 * (1.0 - var_adj / var_unadj)) if var_unadj > 0 else 0.0

    t_res_unadj = welch_t_test(metric_control, metric_treat)
    t_res_adj = welch_t_test(metric_control_adj, metric_treat_adj)

    from src.evaluation.power import required_n_per_arm
    # Adjusted required n: multiply unadjusted n by (1 - R²) = var_adj / var_unadj
    corr = float(np.corrcoef(X_all, Y_all)[0, 1])
    r_sq = corr**2
    n_unadj_approx = None  # caller must supply
    adj_factor = 1.0 - r_sq

    return {
        "theta": float(theta),
        "x_mean": float(x_mean),
        "covariate_correlation_with_metric": float(corr),
        "r_squared": float(r_sq),
        "variance_reduction_pct": float(var_reduction_pct),
        "var_unadjusted": float(var_unadj),
        "var_adjusted": float(var_adj),
        "adj_factor_for_n": float(adj_factor),
        "t_test_unadjusted": t_res_unadj,
        "t_test_adjusted": t_res_adj,
    }
