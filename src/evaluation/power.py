"""Power analysis for the offline A/B test: sample size, MDE curve, weeks needed."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────────
# Core formula: required n per arm (two-sided two-proportion z-test)
# ─────────────────────────────────────────────────────────────────────────────

def required_n_per_arm(
    p_control: float,
    p_treat: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Minimum customers per arm for a two-sided two-proportion z-test.

    Formula:
        n = ⌈ (z_{α/2}·√(2·p̄·q̄) + z_β·√(p_A·q_A + p_B·q_B))² / δ² ⌉
    where δ = |p_B − p_A| and p̄ = (p_A + p_B)/2.
    """
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_beta = norm.ppf(power)
    p_bar = (p_control + p_treat) / 2.0
    q_bar = 1.0 - p_bar
    delta = abs(p_treat - p_control)
    if delta == 0:
        raise ValueError("p_control and p_treat must differ")

    numerator = (
        z_alpha * math.sqrt(2.0 * p_bar * q_bar)
        + z_beta * math.sqrt(
            p_control * (1.0 - p_control) + p_treat * (1.0 - p_treat)
        )
    ) ** 2
    return math.ceil(numerator / delta**2)


def statsmodels_n_per_arm(
    p_control: float,
    p_treat: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict:
    """Cross-check via statsmodels.stats.proportion / NormalIndPower.

    Returns dict with statsmodels n and Cohen's h effect size.
    """
    from statsmodels.stats.proportion import proportion_effectsize
    from statsmodels.stats.power import NormalIndPower

    h = proportion_effectsize(p_treat, p_control)
    analysis = NormalIndPower()
    n_sm = analysis.solve_power(
        effect_size=abs(h), alpha=alpha, power=power, alternative="two-sided"
    )
    return {"statsmodels_n_per_arm": math.ceil(n_sm), "effect_size_h": float(h)}


# ─────────────────────────────────────────────────────────────────────────────
# MDE vs sample-size curve
# ─────────────────────────────────────────────────────────────────────────────

def mde_at_n(
    p_control: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Minimum detectable absolute effect at a given n per arm.

    Approximate via the closed-form: MDE ≈ (z_{α/2} + z_β) · √(2·p·q / n)
    where p = p_control (approximation valid when MDE is small relative to p).
    """
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_beta = norm.ppf(power)
    p, q = p_control, 1.0 - p_control
    return float((z_alpha + z_beta) * math.sqrt(2.0 * p * q / n_per_arm))


def mde_curve(
    p_control: float,
    n_values: list[int],
    alpha: float = 0.05,
    power: float = 0.8,
) -> list[dict]:
    """MDE (absolute lift) for each sample size per arm."""
    return [
        {"n_per_arm": n, "mde_absolute": mde_at_n(p_control, n, alpha, power)}
        for n in n_values
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Analytical power (inverse of required_n_per_arm)
# ─────────────────────────────────────────────────────────────────────────────

def analytical_power(
    p_control: float,
    p_treat: float,
    n_per_arm: int,
    alpha: float = 0.05,
) -> float:
    """Analytical power for the two-sided two-proportion z-test.

    Inverse of required_n_per_arm:
        z_β = (δ·√n − z_{α/2}·√(2·p̄·q̄)) / √(p_A·q_A + p_B·q_B)
        power = Φ(z_β)
    """
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    p_bar = (p_control + p_treat) / 2.0
    delta = abs(p_treat - p_control)
    if delta == 0 or n_per_arm <= 0:
        return float(alpha)
    se_h0_scaled = math.sqrt(2.0 * p_bar * (1.0 - p_bar))
    se_h1_scaled = math.sqrt(
        p_control * (1.0 - p_control) + p_treat * (1.0 - p_treat)
    )
    if se_h1_scaled == 0:
        return 1.0
    z_beta = (delta * math.sqrt(n_per_arm) - z_alpha * se_h0_scaled) / se_h1_scaled
    return float(norm.cdf(z_beta))


# ─────────────────────────────────────────────────────────────────────────────
# Weeks needed
# ─────────────────────────────────────────────────────────────────────────────

def weeks_needed(n_per_arm: int, weekly_customers_per_arm: float) -> float:
    """Weeks of traffic needed assuming a constant arrival rate."""
    return n_per_arm / weekly_customers_per_arm
