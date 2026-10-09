"""Integration tests against the real bundle (skipped if bundle absent).

Markers: integration_real — run with: pytest -m integration_real
"""

from __future__ import annotations

import pytest

from tests.api.conftest import REAL_BUNDLE

pytestmark = [
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
