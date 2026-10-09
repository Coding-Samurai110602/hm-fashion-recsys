"""D1 tests for Part 1 evaluation suite and segment analysis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.suite import (
    bootstrap_ci,
    compute_catalog_coverage,
    compute_hit_rate,
    compute_novelty_share,
    compute_popularity_bias,
    compute_repeat_share,
    per_customer_hit,
    per_customer_novelty_share,
    per_customer_pop_bias,
    per_customer_repeat_share,
)
from src.evaluation.segments import holm_correction, p_value_from_bootstrap
from src.model.evaluate import bootstrap_paired_diff

EVAL_DIR = Path(__file__).parent.parent / "reports" / "evaluation"


# ─────────────────────────────────────────────────────────────────────────────
# Hit rate
# ─────────────────────────────────────────────────────────────────────────────

class TestHitRate:
    def test_all_hit(self):
        preds = {1: [10, 11], 2: [20, 21]}
        gt = {1: {10}, 2: {20}}
        assert compute_hit_rate(preds, gt, k=2) == pytest.approx(1.0)

    def test_no_hit(self):
        preds = {1: [10, 11], 2: [20, 21]}
        gt = {1: {99}, 2: {88}}
        assert compute_hit_rate(preds, gt, k=2) == pytest.approx(0.0)

    def test_partial_hit(self):
        preds = {1: [10, 11], 2: [20, 21]}
        gt = {1: {10}, 2: {99}}
        # Customer 1: hit, Customer 2: miss → mean=0.5
        assert compute_hit_rate(preds, gt, k=2) == pytest.approx(0.5)

    def test_k_truncation(self):
        preds = {1: [10, 11, 12]}
        gt = {1: {12}}
        # With k=2, item 12 is not considered → no hit
        assert compute_hit_rate(preds, gt, k=2) == pytest.approx(0.0)
        # With k=3, item 12 is included → hit
        assert compute_hit_rate(preds, gt, k=3) == pytest.approx(1.0)

    def test_per_customer_values(self):
        preds = {1: [10, 11], 2: [20, 21]}
        gt = {1: {10}, 2: {99}}
        vals = per_customer_hit(preds, gt, k=2)
        assert vals[1] == pytest.approx(1.0)
        assert vals[2] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Catalog coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestCatalogCoverage:
    def test_full_coverage(self):
        preds = {1: [10, 11], 2: [12, 13]}
        sold = {10, 11, 12, 13}
        assert compute_catalog_coverage(preds, sold) == pytest.approx(1.0)

    def test_partial_coverage(self):
        preds = {1: [10, 11]}
        sold = {10, 11, 12, 13}
        # 2 of 4 articles recommended
        assert compute_catalog_coverage(preds, sold) == pytest.approx(0.5)

    def test_empty_sold(self):
        preds = {1: [10]}
        assert compute_catalog_coverage(preds, set()) == 0.0

    def test_overlap_dedup(self):
        # Both customers recommend same items
        preds = {1: [10, 11], 2: [10, 11]}
        sold = {10, 11, 12}
        assert compute_catalog_coverage(preds, sold) == pytest.approx(2 / 3)


# ─────────────────────────────────────────────────────────────────────────────
# Popularity bias
# ─────────────────────────────────────────────────────────────────────────────

class TestPopularityBias:
    def setup_method(self):
        # Rank 1 = most popular, rank 5 = least popular
        self.ranks = {10: 1, 11: 2, 12: 3, 13: 4, 14: 5}

    def test_all_popular(self):
        preds = {1: [10, 11], 2: [10, 11]}
        gt = {1: {99}, 2: {99}}
        # Per customer mean rank: (1+2)/2=1.5; overall mean = 1.5
        assert compute_popularity_bias(preds, self.ranks, gt) == pytest.approx(1.5)

    def test_mixed_popularity(self):
        preds = {1: [10], 2: [14]}
        gt = {1: {99}, 2: {99}}
        # Customer 1 mean=1, Customer 2 mean=5; overall=(1+5)/2=3.0
        assert compute_popularity_bias(preds, self.ranks, gt) == pytest.approx(3.0)

    def test_per_customer_values(self):
        preds = {1: [10, 12], 2: [13, 14]}
        gt = {1: {99}, 2: {99}}
        vals = per_customer_pop_bias(preds, self.ranks, gt)
        assert vals[1] == pytest.approx((1 + 3) / 2)
        assert vals[2] == pytest.approx((4 + 5) / 2)


# ─────────────────────────────────────────────────────────────────────────────
# Repeat share
# ─────────────────────────────────────────────────────────────────────────────

class TestRepeatShare:
    def test_all_repeat(self):
        preds = {1: [10, 11]}
        history = {1: {10, 11, 12}}
        gt = {1: {99}}
        assert compute_repeat_share(preds, history, gt) == pytest.approx(1.0)

    def test_no_repeat(self):
        preds = {1: [10, 11]}
        history = {1: {20, 21}}
        gt = {1: {99}}
        assert compute_repeat_share(preds, history, gt) == pytest.approx(0.0)

    def test_partial_repeat(self):
        preds = {1: [10, 20]}
        history = {1: {10}}
        gt = {1: {99}}
        # 1 of 2 items is repeat → 0.5
        assert compute_repeat_share(preds, history, gt) == pytest.approx(0.5)

    def test_no_history(self):
        preds = {1: [10, 20]}
        history = {}
        gt = {1: {99}}
        assert compute_repeat_share(preds, history, gt) == pytest.approx(0.0)

    def test_per_customer_values(self):
        preds = {1: [10, 11], 2: [20, 21]}
        history = {1: {10}, 2: {20, 21}}
        gt = {1: {99}, 2: {99}}
        vals = per_customer_repeat_share(preds, history, gt)
        assert vals[1] == pytest.approx(0.5)
        assert vals[2] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Novelty share
# ─────────────────────────────────────────────────────────────────────────────

class TestNoveltyShare:
    def test_all_novel(self):
        preds = {1: [10, 11]}
        before = {99, 100}  # articles before cutoff (not 10 or 11)
        gt = {1: {99}}
        assert compute_novelty_share(preds, before, gt) == pytest.approx(1.0)

    def test_no_novelty(self):
        preds = {1: [10, 11]}
        before = {10, 11, 12}
        gt = {1: {99}}
        assert compute_novelty_share(preds, before, gt) == pytest.approx(0.0)

    def test_partial_novelty(self):
        preds = {1: [10, 99]}
        before = {10}
        gt = {1: {99}}
        # Article 99 is novel (not in before) → 1/2=0.5
        assert compute_novelty_share(preds, before, gt) == pytest.approx(0.5)

    def test_per_customer_values(self):
        preds = {1: [10, 11], 2: [20, 21]}
        before = {10, 20}
        gt = {1: {99}, 2: {99}}
        vals = per_customer_novelty_share(preds, before, gt)
        # Customer 1: 11 novel → 0.5; Customer 2: 21 novel → 0.5
        assert vals[1] == pytest.approx(0.5)
        assert vals[2] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Holm correction
# ─────────────────────────────────────────────────────────────────────────────

class TestHolmCorrection:
    def test_hand_worked_example(self):
        # p = [0.01, 0.04, 0.03], n=3
        # Sorted: [0.01, 0.03, 0.04]
        # Adjusted (0-indexed ranks):
        #   rank 0: min(1, 3*0.01)=0.03
        #   rank 1: min(1, max(0.03, 2*0.03))=0.06
        #   rank 2: min(1, max(0.06, 1*0.04))=0.06
        p = [0.01, 0.04, 0.03]
        adj = holm_correction(p)
        assert adj[0] == pytest.approx(0.03)
        assert adj[1] == pytest.approx(0.06)
        assert adj[2] == pytest.approx(0.06)

    def test_single_pvalue(self):
        p = [0.02]
        adj = holm_correction(p)
        assert adj[0] == pytest.approx(0.02)

    def test_empty(self):
        assert holm_correction([]) == []

    def test_monotone_nondecreasing(self):
        p = [0.001, 0.01, 0.05, 0.1, 0.5]
        adj = holm_correction(p)
        for i in range(len(adj) - 1):
            assert adj[i] <= adj[i + 1] + 1e-12, f"Holm not non-decreasing at i={i}: {adj}"

    def test_capped_at_one(self):
        p = [0.4, 0.5, 0.6]
        adj = holm_correction(p)
        assert all(v <= 1.0 for v in adj)

    def test_matches_statsmodels(self):
        """Cross-check against statsmodels.stats.multitest.multipletests."""
        pytest.importorskip("statsmodels")
        from statsmodels.stats.multitest import multipletests
        import numpy as np
        rng = np.random.default_rng(99)
        p_vals = rng.uniform(0, 0.3, size=20).tolist()
        _, p_adj_sm, _, _ = multipletests(p_vals, method="holm")
        p_adj_ours = holm_correction(p_vals)
        for sm, ours in zip(p_adj_sm, p_adj_ours):
            assert abs(sm - ours) < 1e-10, f"Mismatch: statsmodels={sm:.8f} ours={ours:.8f}"


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap sanity
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapSanity:
    def test_identical_inputs_ci_contains_zero(self):
        """Identical AP arrays: bootstrap CI for mean diff should contain 0."""
        rng = np.random.default_rng(0)
        ap = {i: float(v) for i, v in enumerate(rng.uniform(0, 0.05, size=500))}
        boot = bootstrap_paired_diff(ap, ap, n=1000, seed=42)
        assert boot["ci_lo"] <= 0 <= boot["ci_hi"], (
            f"Identical inputs should give CI containing 0; got [{boot['ci_lo']:.6f}, {boot['ci_hi']:.6f}]"
        )

    def test_clear_winner_ci_excludes_zero(self):
        """Clear winner (a >> b): bootstrap CI should exclude 0."""
        rng = np.random.default_rng(1)
        ap_a = {i: 0.05 + float(v) for i, v in enumerate(rng.uniform(0, 0.01, size=500))}
        ap_b = {i: float(v) for i, v in enumerate(rng.uniform(0, 0.001, size=500))}
        boot = bootstrap_paired_diff(ap_a, ap_b, n=1000, seed=42)
        assert boot["ci_lo"] > 0, (
            f"Clear winner should give CI entirely above 0; got [{boot['ci_lo']:.6f}, {boot['ci_hi']:.6f}]"
        )
        assert boot["ci_excludes_zero"], "ci_excludes_zero should be True for clear winner"

    def test_bootstrap_ci_symmetry(self):
        """Negative effect (b >> a): CI should be entirely below 0."""
        rng = np.random.default_rng(2)
        ap_a = {i: float(v) for i, v in enumerate(rng.uniform(0, 0.001, size=500))}
        ap_b = {i: 0.05 + float(v) for i, v in enumerate(rng.uniform(0, 0.01, size=500))}
        boot = bootstrap_paired_diff(ap_a, ap_b, n=1000, seed=42)
        assert boot["ci_hi"] < 0, (
            f"Clear loser should give CI entirely below 0; got [{boot['ci_lo']:.6f}, {boot['ci_hi']:.6f}]"
        )
        assert boot["ci_excludes_zero"], "ci_excludes_zero should be True for clear loser too"


# ─────────────────────────────────────────────────────────────────────────────
# Fold 103 regression: MAP@12 must match eval_fold103.json to 6 decimals
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.requires_data
class TestFold103Regression:
    EXPECTED_MAP12 = 0.035832

    @pytest.fixture(scope="class")
    @staticmethod
    def eval_103():
        path = Path(__file__).parent.parent / "reports" / "ranker" / "eval_fold103.json"
        if not path.exists():
            pytest.skip("eval_fold103.json not found")
        with open(path) as f:
            return json.load(f)

    def test_ranker_map12_matches_json(self, eval_103):
        """Metric suite fold-103 ranker MAP@12 must equal eval_fold103.json to 6 decimals."""
        from src.time_split import build_fold
        from src.model.evaluate import score_fold
        from src.metrics import map_at_k
        import lightgbm as lgb
        import polars as pl

        model_path = Path(__file__).parent.parent / "models" / "lgbm_ranker.txt"
        if not model_path.exists():
            pytest.skip("lgbm_ranker.txt not found")

        feat_path = Path(__file__).parent.parent / "data" / "processed" / "features" / "fold_103.parquet"
        if not feat_path.exists():
            pytest.skip("fold_103.parquet not found")

        booster = lgb.Booster(model_file=str(model_path))
        df = pl.read_parquet(feat_path)
        _, gt, _ = build_fold(103)
        preds, _ = score_fold(booster, df, k=12, sanity_check=False)
        computed = map_at_k(preds, gt, 12)

        expected = eval_103["ranker"]["map@12"]
        assert abs(computed - expected) < 1e-6, (
            f"MAP@12 mismatch: computed={computed:.8f} expected={expected:.8f}"
        )
        # Also check against the hard-coded target
        assert abs(computed - self.EXPECTED_MAP12) < 1e-4, (
            f"MAP@12={computed:.6f} differs from target {self.EXPECTED_MAP12:.6f} by more than 1e-4"
        )


# ─────────────────────────────────────────────────────────────────────────────
# P-value from bootstrap
# ─────────────────────────────────────────────────────────────────────────────

class TestPValueFromBootstrap:
    def test_large_positive_effect_minimum_p(self):
        B = 1000
        boot_means = np.full(B, 0.05)  # all positive, count=0
        p = p_value_from_bootstrap(boot_means, observed_mean=0.05)
        # With (count+1)/(B+1): frac = 1/1001, two-sided p = 2/1001
        assert p == pytest.approx(2 / (B + 1)), (
            f"All-positive boot should give p=2/(B+1)={2/(B+1):.6f}; got {p}"
        )

    def test_minimum_p_value_is_one_over_b_plus_one(self):
        """Minimum possible p-value (one-sided fraction) is 1/(B+1), never 0."""
        B = 1000
        boot_means = np.full(B, 0.05)  # all on the same side as observed
        p = p_value_from_bootstrap(boot_means, observed_mean=0.05)
        # The one-sided fraction (count+1)/(B+1) >= 1/(B+1) always
        assert p >= 1 / (B + 1), f"p={p} is less than minimum 1/(B+1)={1/(B+1):.6f}"
        # Also verify it's not exactly 0
        assert p > 0.0, "p-value must never be exactly 0"

    def test_null_effect_pvalue_near_1(self):
        # Half above, half below 0
        boot_means = np.linspace(-0.05, 0.05, 1000)
        p = p_value_from_bootstrap(boot_means, observed_mean=0.001)
        # ~half above 0 → p ≈ 1.0
        assert p > 0.8, f"Symmetric boot around 0 should give p≈1; got {p}"

    def test_p_bounded(self):
        rng = np.random.default_rng(7)
        boot = rng.normal(0.01, 0.02, 1000)
        p = p_value_from_bootstrap(boot, observed_mean=float(boot.mean()))
        assert 0.0 <= p <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Power analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestPowerAnalysis:
    def test_required_n_vs_statsmodels(self):
        """required_n_per_arm must agree with statsmodels within 5%."""
        from src.evaluation.power import required_n_per_arm, statsmodels_n_per_arm
        p_ctrl, p_treat = 0.0857, 0.1471
        n_ours = required_n_per_arm(p_ctrl, p_treat, alpha=0.05, power=0.8)
        sm = statsmodels_n_per_arm(p_ctrl, p_treat, alpha=0.05, power=0.8)
        n_sm = sm["statsmodels_n_per_arm"]
        rel_diff = abs(n_ours - n_sm) / n_sm
        assert rel_diff < 0.05, (
            f"required_n_per_arm={n_ours} vs statsmodels={n_sm} "
            f"rel_diff={rel_diff:.3f} > 5%"
        )

    def test_mde_decreases_with_n(self):
        """MDE must decrease as sample size increases."""
        from src.evaluation.power import mde_curve
        ns = [1000, 5000, 10000, 50000]
        pts = mde_curve(0.0857, ns)
        mdes = [pt["mde_absolute"] for pt in pts]
        for i in range(len(mdes) - 1):
            assert mdes[i] > mdes[i + 1], f"MDE not decreasing at index {i}: {mdes}"

    def test_weeks_needed(self):
        """weeks_needed = n_per_arm / weekly_customers_per_arm."""
        from src.evaluation.power import weeks_needed
        assert weeks_needed(500, 250) == pytest.approx(2.0)
        assert weeks_needed(1000, 500) == pytest.approx(2.0)

    def test_analytical_power_at_required_n_is_approx_08(self):
        """At n = required_n_per_arm, analytical power must be ≥ 80%."""
        from src.evaluation.power import analytical_power, required_n_per_arm
        p_ctrl, p_treat = 0.0857, 0.1471
        n = required_n_per_arm(p_ctrl, p_treat, alpha=0.05, power=0.8)
        pw = analytical_power(p_ctrl, p_treat, n, alpha=0.05)
        assert pw >= 0.80, f"power={pw:.4f} < 0.80 at n={n}"

    def test_analytical_power_increases_with_n(self):
        """Analytical power must be monotonically increasing in n."""
        from src.evaluation.power import analytical_power
        p_ctrl, p_treat = 0.0857, 0.1471
        ns = [200, 427, 1000, 5000]
        powers = [analytical_power(p_ctrl, p_treat, n) for n in ns]
        for i in range(len(powers) - 1):
            assert powers[i] <= powers[i + 1], (
                f"power not increasing: powers[{i}]={powers[i]:.4f} > powers[{i+1}]={powers[i+1]:.4f}"
            )

    def test_analytical_power_consistent_with_empirical(self):
        """Analytical power at n=500 must be within 5pp of empirical 0.876."""
        from src.evaluation.power import analytical_power
        pw = analytical_power(0.0857, 0.1471, 500, alpha=0.05)
        empirical = 0.876
        assert abs(pw - empirical) < 0.05, (
            f"analytical_power={pw:.4f} deviates from empirical={empirical} by >{0.05}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Two-proportion z-test hand example
# ─────────────────────────────────────────────────────────────────────────────

class TestZTest:
    def test_hand_example(self):
        """Verify z-test on a hand-computed example: p_A=0.10, p_B=0.15, n=1000 each."""
        from src.evaluation.ab_sim import two_proportion_z_test
        res = two_proportion_z_test(hits_a=100, n_a=1000, hits_b=150, n_b=1000)
        # p_pool=0.125, se_h0=sqrt(0.125*0.875*2/1000)=sqrt(0.000219)≈0.01479
        # z = 0.05/0.01479 ≈ 3.381
        assert abs(res["z"] - 3.381) < 0.01, f"z={res['z']:.4f} expected ≈3.381"
        assert res["p_value"] < 0.001, f"p_value={res['p_value']} should be < 0.001"
        assert res["effect"] == pytest.approx(0.05)

    def test_no_effect_pvalue_large(self):
        from src.evaluation.ab_sim import two_proportion_z_test
        res = two_proportion_z_test(100, 1000, 100, 1000)
        assert res["p_value"] == pytest.approx(1.0), f"No-effect p={res['p_value']}"
        assert abs(res["effect"]) < 1e-10

    def test_ci_contains_effect(self):
        from src.evaluation.ab_sim import two_proportion_z_test
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = int(rng.integers(100, 2000))
            p_a = float(rng.uniform(0.05, 0.4))
            p_b = float(rng.uniform(0.05, 0.4))
            ha = int(rng.binomial(n, p_a))
            hb = int(rng.binomial(n, p_b))
            res = two_proportion_z_test(ha, n, hb, n)
            assert res["ci_lo"] <= res["effect"] <= res["ci_hi"]


# ─────────────────────────────────────────────────────────────────────────────
# A/A false-positive rate test (lightweight synthetic)
# ─────────────────────────────────────────────────────────────────────────────

class TestAATest:
    def test_aa_fpr_within_3_to_7_pct(self):
        """A/A false-positive rate must be within [3%, 7%] at 1000 sims."""
        from src.evaluation.ab_sim import run_aa_test
        # Synthetic: 5000 customers, hit indicator Bernoulli(0.1), seed=0 for stability
        rng = np.random.default_rng(0)
        n_custs = 5000
        hit_rate = 0.1
        hits = rng.binomial(1, hit_rate, size=n_custs)
        preds = {i: [0] for i in range(n_custs)}
        gt = {i: ({0} if h else {999}) for i, h in enumerate(hits)}
        aa = run_aa_test(preds, gt, n_sims=1000, seed=0, alpha=0.05, k=1)
        fpr = aa["false_positive_rate"]
        assert 0.03 <= fpr <= 0.07, (
            f"A/A false-positive rate={fpr:.3f} outside [3%, 7%]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CUPED: variance reduction on a synthetic correlated example
# ─────────────────────────────────────────────────────────────────────────────

class TestCUPED:
    def test_cuped_reduces_variance_on_correlated_covariate(self):
        """CUPED must reduce variance when covariate is correlated with metric."""
        from src.evaluation.ab_sim import cuped_adjustment
        rng = np.random.default_rng(99)
        n = 500
        n_half = n // 2
        X = rng.normal(10, 3, size=n)
        X_ctrl = X[:n_half]
        X_treat = X[n_half:]
        # Y correlated with X (rho ≈ 0.7) + treatment effect
        Y_ctrl = 0.5 * X_ctrl + rng.normal(0, 1, size=n_half)
        Y_treat = 0.5 * X_treat + 0.2 + rng.normal(0, 1, size=n - n_half)
        res = cuped_adjustment(Y_treat, Y_ctrl, X_treat, X_ctrl)
        # CUPED should reduce variance meaningfully when rho is large
        assert res["variance_reduction_pct"] > 0, (
            f"CUPED should reduce variance; got {res['variance_reduction_pct']:.1f}%"
        )
        assert res["r_squared"] > 0.1, "Expected |corr|>0.32 for this synthetic example"


# ─────────────────────────────────────────────────────────────────────────────
# Holdout protocol integrity: run_holdout.py must never write to protocol file
# ─────────────────────────────────────────────────────────────────────────────

class TestHoldoutProtocol:
    HOLDOUT_SCRIPT = Path(__file__).parent.parent / "scripts" / "run_holdout.py"
    EVAL_DIR = Path(__file__).parent.parent / "reports" / "evaluation"

    def test_run_holdout_source_checks_protocol_exists(self):
        """run_holdout.py source must contain a FileNotFoundError guard for the protocol file."""
        source = self.HOLDOUT_SCRIPT.read_text()
        assert "FileNotFoundError" in source, (
            "run_holdout.py must raise FileNotFoundError if holdout_protocol.json is missing"
        )
        assert "holdout_protocol.json" in source, (
            "run_holdout.py must reference 'holdout_protocol.json' in its existence check"
        )

    def test_run_holdout_never_opens_protocol_write_mode(self):
        """Source of run_holdout.py must not open holdout_protocol.json in write mode."""
        source = self.HOLDOUT_SCRIPT.read_text()
        import ast
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # Match: open(protocol_path, "w") or open(..., "w")
                if (isinstance(func, ast.Name) and func.id == "open"
                        or isinstance(func, ast.Attribute) and func.attr == "open"):
                    if len(node.args) >= 2:
                        mode_arg = node.args[1]
                        if isinstance(mode_arg, ast.Constant) and "w" in str(mode_arg.value):
                            # Check if any argument references the protocol file
                            call_src = ast.unparse(node)
                            assert "protocol" not in call_src.lower(), (
                                f"run_holdout.py opens a file in write mode at a path "
                                f"referencing 'protocol': {call_src}"
                            )

    def test_protocol_file_has_no_labels_read_field(self):
        """holdout_protocol.json must not contain labels_read_timestamp_utc (post-read field)."""
        path = self.EVAL_DIR / "holdout_protocol.json"
        if not path.exists():
            pytest.skip("holdout_protocol.json not found")
        with open(path) as f:
            p = json.load(f)
        assert "labels_read_timestamp_utc" not in p, (
            "holdout_protocol.json must not contain labels_read_timestamp_utc "
            "(this field belongs in holdout_results.json only)"
        )
        assert "labels_sha256" not in p, (
            "holdout_protocol.json must not contain labels_sha256"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized A/A matches old code within MC noise (small-scale)
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorizedAAMatchesOld:
    """Vectorized A/A FPR and empirical power must agree with original ab_sim
    functions on a small synthetic dataset within Monte Carlo noise."""

    @staticmethod
    def _make_synthetic(n: int = 1000, p_h: float = 0.09, p_r: float = 0.15,
                        seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        hit_h = rng.binomial(1, p_h, size=n).astype(float)
        hit_r = rng.binomial(1, p_r, size=n).astype(float)
        return hit_h, hit_r

    def test_aa_fpr_vec_within_noise_of_old(self):
        """Vectorized A/A FPR must be within ±3 sigma of old code FPR on the same seed."""
        from src.evaluation.ab_sim import run_aa_test
        hit_h, _ = self._make_synthetic(n=1000, seed=7)

        # Reconstruct dicts for old code
        n = len(hit_h)
        preds_h = {i: ([0] if hit_h[i] > 0.5 else [999]) for i in range(n)}
        gt = {i: {0} for i in range(n)}

        n_sims = 500
        old = run_aa_test(preds_h, gt, n_sims=n_sims, seed=42, alpha=0.05, k=1)
        old_fpr = old["false_positive_rate"]

        # Vectorized version
        from scripts.run_experiment import _aa_test_vec
        new = _aa_test_vec(hit_h, n_sims=n_sims, seed=42, alpha=0.05)
        new_fpr = new["false_positive_rate"]

        # Within ±3 sigma of a Binomial(n_sims, alpha)
        sigma = np.sqrt(n_sims * 0.05 * 0.95) / n_sims
        assert abs(new_fpr - old_fpr) <= 3 * sigma + 0.01, (
            f"Vectorized FPR={new_fpr:.3f} vs old FPR={old_fpr:.3f} "
            f"differ by more than 3-sigma ({3*sigma:.3f})"
        )

    def test_aa_fpr_vec_calibrated(self):
        """Vectorized A/A FPR must be within [3%, 7%] for Bernoulli(0.1) hits."""
        hit_h, _ = self._make_synthetic(n=2000, p_h=0.1, seed=3)
        from scripts.run_experiment import _aa_test_vec
        res = _aa_test_vec(hit_h, n_sims=1000, seed=42, alpha=0.05)
        fpr = res["false_positive_rate"]
        assert 0.03 <= fpr <= 0.07, f"Vectorized A/A FPR={fpr:.3f} outside [3%, 7%]"


# ─────────────────────────────────────────────────────────────────────────────
# Paired t-test p-value from segments
# ─────────────────────────────────────────────────────────────────────────────

class TestPairedTtest:
    def test_clear_effect_small_p(self):
        """Paired t-test gives small p when ranker clearly beats heuristic."""
        from src.evaluation.segments import p_value_from_paired_ttest
        rng = np.random.default_rng(5)
        n = 500
        arr_h = rng.uniform(0.0, 0.03, size=n)
        arr_r = arr_h + 0.02 + rng.normal(0, 0.005, size=n)
        p = p_value_from_paired_ttest(arr_r, arr_h)
        assert p < 0.001, f"Expected p<0.001 for clear effect; got {p:.6f}"

    def test_no_effect_large_p(self):
        """Paired t-test gives large p when there is no effect (same distribution)."""
        from src.evaluation.segments import p_value_from_paired_ttest
        rng = np.random.default_rng(6)
        n = 500
        # Same distribution, different realisations → no systematic effect → large p
        arr_h = rng.uniform(0.0, 0.05, size=n)
        arr_r = rng.uniform(0.0, 0.05, size=n)
        p = p_value_from_paired_ttest(arr_r, arr_h)
        # Not asserting p>0.05 (would fail ~5% of the time); just assert it's not tiny
        assert p > 0.01, f"No-effect t-test gave suspiciously small p={p:.6f}"

    def test_returns_float_in_unit_interval(self):
        from src.evaluation.segments import p_value_from_paired_ttest
        rng = np.random.default_rng(8)
        a = rng.uniform(0, 0.05, 200)
        b = rng.uniform(0, 0.05, 200)
        p = p_value_from_paired_ttest(a, b)
        assert 0.0 <= p <= 1.0
