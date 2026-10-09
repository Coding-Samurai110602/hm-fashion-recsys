"""Power analysis and A/B simulation service."""

from __future__ import annotations

import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger("api")

# Bundle ab_arrays populated at startup.
_ab_arrays = None  # polars DataFrame or None


def set_ab_arrays(ab_arrays) -> None:
    """Called at startup from deps.load_bundle_at_startup."""
    global _ab_arrays
    _ab_arrays = ab_arrays


def compute_power(
    baseline_rate: float,
    relative_lift: float,
    alpha: float,
    power: float,
    weekly_traffic: int | None,
    cuped_variance_reduction: float,
) -> dict:
    from src.evaluation.power import required_n_per_arm, weeks_needed

    n = required_n_per_arm(baseline_rate, relative_lift, alpha=alpha, power=power)
    n_sm = n
    try:
        from src.evaluation.power import statsmodels_n_per_arm

        result_sm = statsmodels_n_per_arm(
            baseline_rate, relative_lift, alpha=alpha, power=power
        )
        if isinstance(result_sm, dict):
            n_sm = int(result_sm.get("statsmodels_n_per_arm", n))
        else:
            n_sm = int(result_sm)
    except Exception:
        pass

    n_cuped = None
    if cuped_variance_reduction > 0:
        effective_n = int(math.ceil(n * (1 - cuped_variance_reduction)))
        n_cuped = effective_n

    weeks = None
    if weekly_traffic and weekly_traffic > 0:
        weeks = weeks_needed(n, weekly_traffic)

    treatment_rate = baseline_rate * (1 + relative_lift)
    p1, p2 = baseline_rate, treatment_rate
    h = 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))

    return {
        "n_per_arm": n,
        "n_per_arm_statsmodels": n_sm,
        "n_per_arm_cuped": n_cuped,
        "weeks_needed": weeks,
        "cohen_h": abs(h),
    }


async def run_simulation(
    test_type: str,
    n_per_arm: int,
    n_sims: int,
    peeking: bool,
    n_days: int,
    seed: int,
    executor: ThreadPoolExecutor,
    bundle=None,
) -> dict:
    loop = asyncio.get_event_loop()

    def _simulate():
        result = {
            "test_type": test_type,
            "n_per_arm": n_per_arm,
            "n_sims": n_sims,
            "false_positive_rate": None,
            "empirical_power": None,
            "peeking_fpr": None,
            "data_source": "unavailable",
        }

        # Prefer bundle ab_arrays for proper permutation simulation
        ab_df = (
            _ab_arrays
            if _ab_arrays is not None
            else (bundle.ab_arrays if bundle is not None else None)
        )

        if ab_df is not None and len(ab_df) > 0:
            result["data_source"] = "bundle_ab_arrays"
            # AP@12 arrays available for paired t-test simulations (not used in current z-test)
            ranker_hit = ab_df["ranker_hit12"].to_numpy().astype(np.float64)
            heuristic_hit = ab_df["heuristic_hit12"].to_numpy().astype(np.float64)

            rng = np.random.default_rng(seed)

            if test_type == "aa":
                # A/A test: both arms from same heuristic distribution
                combined = heuristic_hit
                n_pop = len(combined)
                rejections = 0
                for _ in range(n_sims):
                    idx_a = rng.choice(n_pop, size=n_per_arm, replace=True)
                    idx_b = rng.choice(n_pop, size=n_per_arm, replace=True)
                    rate_a = combined[idx_a].mean()
                    rate_b = combined[idx_b].mean()
                    p_pool = (combined[idx_a].sum() + combined[idx_b].sum()) / (
                        2 * n_per_arm
                    )
                    se = (
                        math.sqrt(p_pool * (1 - p_pool) * (2 / n_per_arm))
                        if p_pool > 0
                        else 1e-9
                    )
                    z = abs(rate_b - rate_a) / se if se > 0 else 0.0
                    from scipy.stats import norm as _norm

                    p_val = 2 * (1 - _norm.cdf(abs(z)))
                    if p_val < 0.05:
                        rejections += 1
                result["false_positive_rate"] = rejections / n_sims
            else:
                # A/B test: arm A = heuristic, arm B = ranker
                n_pop = len(ranker_hit)
                rejections = 0
                for _ in range(n_sims):
                    idx_a = rng.choice(n_pop, size=n_per_arm, replace=True)
                    idx_b = rng.choice(n_pop, size=n_per_arm, replace=True)
                    rate_a = heuristic_hit[idx_a].mean()
                    rate_b = ranker_hit[idx_b].mean()
                    p_pool = (heuristic_hit[idx_a].sum() + ranker_hit[idx_b].sum()) / (
                        2 * n_per_arm
                    )
                    se = (
                        math.sqrt(p_pool * (1 - p_pool) * (2 / n_per_arm))
                        if p_pool > 0
                        else 1e-9
                    )
                    z = (rate_b - rate_a) / se if se > 0 else 0.0
                    from scipy.stats import norm as _norm

                    p_val = 1 - _norm.cdf(z)
                    if p_val < 0.05:
                        rejections += 1
                result["empirical_power"] = rejections / n_sims

                if peeking:
                    # Peeking simulation: 14-day sequential peeking
                    daily_n = max(1, n_per_arm // n_days)
                    peeking_rejections = 0
                    for _ in range(n_sims):
                        idx_a = rng.choice(n_pop, size=n_per_arm, replace=True)
                        idx_b = rng.choice(n_pop, size=n_per_arm, replace=True)
                        # Shuffle to simulate daily arrival
                        rng.shuffle(idx_a)
                        rng.shuffle(idx_b)
                        rejected = False
                        for d in range(1, n_days + 1):
                            n_d = min(d * daily_n, n_per_arm)
                            rate_a = heuristic_hit[idx_a[:n_d]].mean()
                            rate_b = ranker_hit[idx_b[:n_d]].mean()
                            p_pool = (
                                heuristic_hit[idx_a[:n_d]].sum()
                                + ranker_hit[idx_b[:n_d]].sum()
                            ) / (2 * n_d)
                            se = (
                                math.sqrt(p_pool * (1 - p_pool) * (2 / n_d))
                                if p_pool > 0
                                else 1e-9
                            )
                            z = abs(rate_b - rate_a) / se if se > 0 else 0.0
                            from scipy.stats import norm as _norm

                            if 2 * (1 - _norm.cdf(z)) < 0.05:
                                rejected = True
                                break
                        if rejected:
                            peeking_rejections += 1
                    result["peeking_fpr"] = peeking_rejections / n_sims

        return result

    return await loop.run_in_executor(executor, _simulate)
