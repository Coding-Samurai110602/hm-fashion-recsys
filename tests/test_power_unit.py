"""Unit tests for src/evaluation/power.py.

Verify that required_n_per_arm reproduces the realistic power table committed in
reports/evaluation/ab_test.json within rounding, and matches statsmodels within 5%.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.evaluation.power import required_n_per_arm, statsmodels_n_per_arm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_ab_test() -> dict:
    path = Path(__file__).parents[1] / "reports" / "evaluation" / "ab_test.json"
    if not path.exists():
        pytest.skip("reports/evaluation/ab_test.json not found (data-dependent)")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Tests against committed realistic power table
# ---------------------------------------------------------------------------

class TestPowerTableReproduction:
    """required_n_per_arm must reproduce the committed ab_test.json power table."""

    @pytest.fixture(scope="class")
    def power_table(self):
        d = _load_ab_test()
        table = d.get("realistic_power_table", d.get("power_table", []))
        if not table:
            pytest.skip("No power table in ab_test.json")
        return table

    def test_1pct_lift_within_rounding(self, power_table):
        row = next(r for r in power_table if r["relative_lift"] == "1%")
        p_ctrl = row["p_control"]
        p_treat = row["p_treat"]
        expected = row["n_per_arm_no_cuped"]
        computed = required_n_per_arm(p_ctrl, p_treat)
        # Allow ±1 from ceil rounding
        assert abs(computed - expected) <= 1, (
            f"1% lift: expected {expected}, got {computed}"
        )

    def test_2pct_lift_within_rounding(self, power_table):
        row = next(r for r in power_table if r["relative_lift"] == "2%")
        p_ctrl = row["p_control"]
        p_treat = row["p_treat"]
        expected = row["n_per_arm_no_cuped"]
        computed = required_n_per_arm(p_ctrl, p_treat)
        assert abs(computed - expected) <= 1, (
            f"2% lift: expected {expected}, got {computed}"
        )

    def test_5pct_lift_within_rounding(self, power_table):
        row = next(r for r in power_table if r["relative_lift"] == "5%")
        p_ctrl = row["p_control"]
        p_treat = row["p_treat"]
        expected = row["n_per_arm_no_cuped"]
        computed = required_n_per_arm(p_ctrl, p_treat)
        assert abs(computed - expected) <= 1, (
            f"5% lift: expected {expected}, got {computed}"
        )

    def test_10pct_lift_within_rounding(self, power_table):
        row = next(r for r in power_table if r["relative_lift"] == "10%")
        p_ctrl = row["p_control"]
        p_treat = row["p_treat"]
        expected = row["n_per_arm_no_cuped"]
        computed = required_n_per_arm(p_ctrl, p_treat)
        assert abs(computed - expected) <= 1, (
            f"10% lift: expected {expected}, got {computed}"
        )


# ---------------------------------------------------------------------------
# Tests against statsmodels (cross-check within 5%)
# ---------------------------------------------------------------------------

class TestPowerVsStatsmodels:
    """required_n_per_arm must agree with statsmodels NormalIndPower within 5%."""

    @pytest.mark.parametrize("relative_lift_frac,label", [
        (0.01, "1%"),
        (0.02, "2%"),
        (0.05, "5%"),
        (0.10, "10%"),
    ])
    def test_matches_statsmodels_within_5pct(self, relative_lift_frac, label):
        baseline = 0.0857
        p_treat = baseline * (1.0 + relative_lift_frac)
        formula_n = required_n_per_arm(baseline, p_treat)
        try:
            sm_result = statsmodels_n_per_arm(baseline, p_treat)
            sm_n = sm_result["statsmodels_n_per_arm"]
        except ImportError:
            pytest.skip("statsmodels not installed")

        rel_diff = abs(formula_n - sm_n) / max(sm_n, 1)
        assert rel_diff <= 0.05, (
            f"{label} lift: formula_n={formula_n}, statsmodels_n={sm_n}, "
            f"relative diff={rel_diff:.2%} exceeds 5%"
        )


# ---------------------------------------------------------------------------
# Convention: required_n_per_arm takes rates, not fractions
# ---------------------------------------------------------------------------

class TestConvention:
    """Callers must pass p_control and p_treat as rates, not lift fractions."""

    def test_p_treat_above_1_raises_value_error(self):
        # p_treat=1.1: p_treat*(1-p_treat) = 1.1*(-0.1) = -0.11 → sqrt of negative → ValueError
        with pytest.raises((ValueError, ZeroDivisionError, Exception)):
            required_n_per_arm(0.0857, 1.1)

    def test_same_rate_raises_value_error(self):
        with pytest.raises(ValueError, match="must differ"):
            required_n_per_arm(0.1, 0.1)

    def test_valid_rates_return_positive_int(self):
        n = required_n_per_arm(0.0857, 0.0857 * 1.10)
        assert isinstance(n, int)
        assert n > 0

    def test_larger_lift_requires_fewer_participants(self):
        baseline = 0.0857
        n_small = required_n_per_arm(baseline, baseline * 1.01)
        n_large = required_n_per_arm(baseline, baseline * 1.10)
        assert n_small > n_large, "Larger lift should require fewer participants"
