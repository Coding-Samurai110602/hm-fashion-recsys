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
    except ImportError:
        pass  # statsmodels optional; n_sm falls back to formula value
    except (ValueError, TypeError) as e:
        logger.warning("statsmodels power computation failed: %s", e)

    n_cuped = None
    if cuped_variance_reduction > 0:
        effective_n = math.ceil(n * (1 - cuped_variance_reduction))
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
        from src.evaluation.ab_sim import (
            empirical_power_curve_vec,
            run_aa_test_vec,
            run_peeking_null_vec,
            run_peeking_power_vec,
        )

        result: dict = {
            "test_type": test_type,
            "n_per_arm": n_per_arm,
            "n_sims": n_sims,
            "false_positive_rate": None,
            "empirical_power": None,
            "peeking_fpr": None,
            "peeking_power": None,
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
            ranker_hit = ab_df["ranker_hit12"].to_numpy().astype(np.float64)
            heuristic_hit = ab_df["heuristic_hit12"].to_numpy().astype(np.float64)

            if test_type == "aa":
                # A/A: both arms heuristic, permutation splits (no replacement)
                aa_res = run_aa_test_vec(
                    heuristic_hit, n_sims=n_sims, seed=seed, alpha=0.05
                )
                result["false_positive_rate"] = aa_res["false_positive_rate"]
                if peeking:
                    peek_res = run_peeking_null_vec(
                        heuristic_hit,
                        n_days=n_days,
                        n_sims=n_sims,
                        seed=seed,
                        alpha=0.05,
                    )
                    # A/A peeking: false-positive rate under the null
                    result["peeking_fpr"] = peek_res["peeking_false_positive_rate"]
            else:
                # A/B: arm A = heuristic, arm B = ranker, permutation subsampling
                power_res = empirical_power_curve_vec(
                    heuristic_hit,
                    ranker_hit,
                    n_values=[n_per_arm],
                    n_sims=n_sims,
                    seed=seed,
                    alpha=0.05,
                )
                raw_power = power_res[0]["empirical_power"] if power_res else None
                import math as _math

                result["empirical_power"] = (
                    None
                    if raw_power is None
                    or (isinstance(raw_power, float) and _math.isnan(raw_power))
                    else raw_power
                )
                if peeking:
                    peek_res = run_peeking_power_vec(
                        heuristic_hit,
                        ranker_hit,
                        n_days=n_days,
                        n_sims=n_sims,
                        seed=seed,
                        alpha=0.05,
                    )
                    result["peeking_power"] = peek_res["peeking_power"]

        return result

    return await loop.run_in_executor(executor, _simulate)
