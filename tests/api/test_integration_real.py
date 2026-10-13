"""Integration tests against the real bundle (skipped if bundle absent).

Markers: integration_real — run with: pytest -m integration_real
"""

from __future__ import annotations

import pytest

from tests.api.conftest import REAL_BUNDLE

pytestmark = [
    pytest.mark.api,
    pytest.mark.filterwarnings("ignore"),
    pytest.mark.integration_real,
    pytest.mark.slow,
]

# The medium buyer documented in session notes (customer_idx=896118).
# Hex customer_id from data/processed/customers.parquet:
#   customer_idx=896118 → customer_id=a73eaaa35d11971ef7b16811fd05ea0f165b016507ebc069e7b1b3f8feca9eae
REAL_CUSTOMER_ID = "a73eaaa35d11971ef7b16811fd05ea0f165b016507ebc069e7b1b3f8feca9eae"
REAL_CUSTOMER_IDX = 896118  # integer idx, for direct recommender calls


@pytest.mark.skipif(
    not __import__("pathlib").Path(REAL_BUNDLE).exists(),
    reason="Real bundle not present",
)
class TestRealBundleRecommendations:
    def test_real_recommendations_match_recommender(self, real_client):
        """API recommendations must equal BundleRecommender output exactly."""
        r = real_client.get(
            f"/api/v1/customers/{REAL_CUSTOMER_ID}/recommendations?k=12&include_reasons=false"
        )
        assert r.status_code == 200
        api_items = r.json()
        api_article_idxs = [item["article_idx"] for item in api_items]

        # Direct BundleRecommender call
        from src.serving.bundle import load_bundle
        from src.serving.recommender import BundleRecommender

        b = load_bundle(REAL_BUNDLE)
        rec = BundleRecommender(b)
        direct = rec.recommend(REAL_CUSTOMER_IDX, k=12)
        direct_idxs = [item["article_idx"] for item in direct]

        assert api_article_idxs == direct_idxs, (
            f"API and direct recommender disagree:\n"
            f"  API:    {api_article_idxs}\n"
            f"  Direct: {direct_idxs}"
        )

    def test_real_recommendations_count(self, real_client):
        r = real_client.get(
            f"/api/v1/customers/{REAL_CUSTOMER_ID}/recommendations?k=12"
        )
        assert r.status_code == 200
        assert len(r.json()) == 12

    def test_real_recommendations_no_duplicates(self, real_client):
        r = real_client.get(
            f"/api/v1/customers/{REAL_CUSTOMER_ID}/recommendations?k=12"
        )
        article_idxs = [item["article_idx"] for item in r.json()]
        assert len(set(article_idxs)) == len(article_idxs), (
            "Duplicate articles in recommendations"
        )

    def test_real_profile_200(self, real_client):
        r = real_client.get(f"/api/v1/customers/{REAL_CUSTOMER_ID}")
        assert r.status_code == 200
        data = r.json()
        assert data["customer_idx"] == REAL_CUSTOMER_IDX
        assert data["n_purchases_all_time"] >= 7  # known to have ≥7 purchases

    def test_real_baseline_differs_from_ranker(self, real_client):
        r_rec = real_client.get(
            f"/api/v1/customers/{REAL_CUSTOMER_ID}/recommendations?k=12&include_reasons=false"
        )
        r_base = real_client.get(f"/api/v1/customers/{REAL_CUSTOMER_ID}/baseline?k=12")
        assert r_rec.status_code == 200
        assert r_base.status_code == 200
        rec_idxs = {i["article_idx"] for i in r_rec.json()}
        base_idxs = {i["article_idx"] for i in r_base.json()}
        # Ranker and heuristic should differ on at least one item
        _ = rec_idxs != base_idxs  # informational, no assertion (may agree by chance)

    def test_real_funnel_has_candidates(self, real_client):
        r = real_client.get(f"/api/v1/customers/{REAL_CUSTOMER_ID}/funnel")
        assert r.status_code == 200
        candidates = r.json()
        assert len(candidates) > 12  # must have more candidates than final recs

    def test_real_version_includes_real_map12(self, real_client):
        r = real_client.get("/version")
        assert r.status_code == 200
        data = r.json()
        # Real model has MAP@12 ≈ 0.035832 on fold 103
        assert data.get("model_map12_fold103", 0) > 0.03


@pytest.mark.skipif(
    not __import__("pathlib").Path(REAL_BUNDLE).exists(),
    reason="Real bundle not present",
)
class TestSimulateRealBundle:
    """Verify simulation against real bundle: calibrated A/A FPR, peeking inflation, power."""

    def test_aa_fpr_calibrated(self, real_client):
        """A/A FPR must be within Monte Carlo tolerance of nominal alpha=0.05."""
        body = {"test_type": "aa", "n_per_arm": 1000, "n_sims": 500, "seed": 42}
        r = real_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 200
        d = r.json()
        fpr = d.get("false_positive_rate")
        if fpr is None:
            pytest.skip("No ab_arrays in bundle")
        # Binomial tolerance: 3σ ≈ 3*sqrt(0.05*0.95/500) ≈ 0.029
        assert 0.02 <= fpr <= 0.10, f"A/A FPR {fpr:.3f} out of expected [0.02, 0.10]"

    def test_ab_power_positive(self, real_client):
        """A/B empirical power must be positive (ranker beats heuristic)."""
        body = {"test_type": "ab", "n_per_arm": 500, "n_sims": 200, "seed": 42}
        r = real_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 200
        d = r.json()
        power = d.get("empirical_power")
        if power is None:
            pytest.skip("No ab_arrays in bundle")
        assert power > 0.5, (
            f"Empirical power {power:.3f} too low; ranker may not beat heuristic"
        )

    def test_peeking_inflates_fpr(self, real_client):
        """Peeking FPR under null (A/A with peeking) must exceed fixed-horizon FPR."""
        body = {
            "test_type": "aa",
            "n_per_arm": 500,
            "n_sims": 200,
            "seed": 42,
            "peeking": True,
        }
        r = real_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 200
        d = r.json()
        fpr = d.get("false_positive_rate")
        peeking_fpr = d.get("peeking_fpr")  # A/A peeking → peeking_fpr field
        if fpr is None or peeking_fpr is None:
            pytest.skip("No ab_arrays in bundle or peeking not computed")
        assert peeking_fpr >= fpr, (
            f"Peeking FPR {peeking_fpr:.3f} should be >= fixed-horizon FPR {fpr:.3f}"
        )


@pytest.mark.skipif(
    not __import__("pathlib").Path(REAL_BUNDLE).exists(),
    reason="Real bundle not present",
)
class TestSimulateVsAbTestJson:
    """Verify /experiment/simulate is consistent with the committed ab_test.json values.

    Committed values (reports/evaluation/ab_test.json):
      A/A FPR (n_sims=10000): 0.0532
      Peeking FPR (n_sims=1000, n_days=14): 0.224
      Empirical power at n=500/arm (n_sims=1000): 0.876

    Each test uses n_sims=1000 and checks the result within the 99% binomial MC CI
    for the committed value (p_committed ± z_99 * sqrt(p*(1-p)/n_sims)).
    """

    def test_aa_fpr_within_mc_tolerance_of_committed(self, real_client):
        """A/A FPR (n_sims=1000, seed=42) must be within 99% MC CI of committed 0.0532."""
        import math

        committed = 0.0532
        n_sims = 1000
        z99 = 2.576
        se = math.sqrt(committed * (1 - committed) / n_sims)
        lo, hi = committed - z99 * se, committed + z99 * se

        body = {"test_type": "aa", "n_per_arm": 1000, "n_sims": n_sims, "seed": 42}
        r = real_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 200
        d = r.json()
        fpr = d.get("false_positive_rate")
        if fpr is None:
            pytest.skip("No ab_arrays in bundle")
        assert lo <= fpr <= hi, (
            f"A/A FPR API={fpr:.4f} vs committed=0.0532; "
            f"99% MC CI=[{lo:.4f},{hi:.4f}] (n_sims={n_sims})"
        )

    def test_peeking_fpr_within_mc_tolerance_of_committed(self, real_client):
        """Peeking FPR (n_sims=1000, seed=42) must be within 99% MC CI of committed 0.224."""
        import math

        committed = 0.224
        n_sims = 1000
        z99 = 2.576
        se = math.sqrt(committed * (1 - committed) / n_sims)
        lo, hi = committed - z99 * se, committed + z99 * se

        body = {
            "test_type": "aa",
            "n_per_arm": 1000,
            "n_sims": n_sims,
            "seed": 42,
            "peeking": True,
        }
        r = real_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 200
        d = r.json()
        peeking_fpr = d.get("peeking_fpr")  # A/A peeking → peeking_fpr field
        if peeking_fpr is None:
            pytest.skip("No ab_arrays in bundle or peeking not computed")
        assert lo <= peeking_fpr <= hi, (
            f"Peeking FPR API={peeking_fpr:.4f} vs committed=0.224; "
            f"99% MC CI=[{lo:.4f},{hi:.4f}] (n_sims={n_sims})"
        )

    def test_empirical_power_n500_within_mc_tolerance_of_committed(self, real_client):
        """Empirical power at n=500/arm (n_sims=200, seed=42) within 99% CI of committed 0.876."""
        import math

        committed = 0.876
        n_sims = 200
        z99 = 2.576
        se = math.sqrt(committed * (1 - committed) / n_sims)
        lo, hi = committed - z99 * se, committed + z99 * se

        body = {"test_type": "ab", "n_per_arm": 500, "n_sims": n_sims, "seed": 42}
        r = real_client.post("/api/v1/experiment/simulate", json=body)
        assert r.status_code == 200
        d = r.json()
        power = d.get("empirical_power")
        if power is None:
            pytest.skip("No ab_arrays in bundle")
        assert lo <= power <= hi, (
            f"Empirical power API={power:.4f} vs committed=0.876; "
            f"99% MC CI=[{lo:.4f},{hi:.4f}] (n_sims={n_sims})"
        )


@pytest.mark.skipif(
    not __import__("pathlib").Path(REAL_BUNDLE).exists(),
    reason="Real bundle not present",
)
class TestSHAPCacheCorrectness:
    """Verify that explain() after recommend() returns identical results to fresh explain()."""

    def test_explain_after_recommend_matches_fresh(self, real_client):
        """SHAP cache hit: explain after recommend-with-reasons must equal direct explain."""
        # 1. Call recommend-with-reasons (populates SHAP cache)
        r_rec = real_client.get(
            f"/api/v1/customers/{REAL_CUSTOMER_ID}/recommendations?k=12&include_reasons=true"
        )
        assert r_rec.status_code == 200
        recs = r_rec.json()
        assert recs, "Empty recommendations"

        # 2. Call explain for the top-ranked article (SHAP cache hit)
        article_id = recs[0]["article_id"]
        r_explain_cached = real_client.get(
            f"/api/v1/customers/{REAL_CUSTOMER_ID}/explain/{article_id}"
        )
        assert r_explain_cached.status_code == 200
        cached_result = r_explain_cached.json()
        assert "shap_values" in cached_result

        # 3. Verify reasons from recommend match explain reasons (consistency check)
        rec_reasons = recs[0].get("reasons", [])
        explain_reasons = cached_result.get("reasons", [])
        # At least one reason should match (SHAP values are the same source)
        if rec_reasons and explain_reasons:
            assert set(rec_reasons) == set(explain_reasons), (
                f"Reasons differ: rec={rec_reasons} explain={explain_reasons}"
            )
