"""API tests for customer and recommendation endpoints."""

from __future__ import annotations

import pytest

from tests.api.conftest import SYNTHETIC_BUNDLE

pytestmark = pytest.mark.filterwarnings("ignore")


def _first_customer(synth_client) -> tuple[str, int]:
    """Return (customer_id hex str, customer_idx int) for the first synthetic bundle customer."""
    from src.serving.bundle import load_bundle

    b = load_bundle(SYNTHETIC_BUNDLE)
    row = b.customers.head(1).to_dicts()[0]
    return str(row["customer_id"]), int(row["customer_idx"])


class TestSampleCustomers:
    def test_sample_returns_list(self, synth_client):
        r = synth_client.get("/api/v1/customers/sample")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)

    def test_sample_n_param(self, synth_client):
        r = synth_client.get("/api/v1/customers/sample?n=3")
        assert r.status_code == 200
        data = r.json()
        assert len(data) <= 3

    def test_sample_segment_medium(self, synth_client):
        r = synth_client.get("/api/v1/customers/sample?segment=medium&n=5")
        assert r.status_code == 200

    def test_sample_invalid_n(self, synth_client):
        r = synth_client.get("/api/v1/customers/sample?n=100")
        assert r.status_code == 422

    def test_sample_schema(self, synth_client):
        r = synth_client.get("/api/v1/customers/sample?n=2")
        assert r.status_code == 200
        for item in r.json():
            assert "customer_idx" in item
            assert "customer_id" in item
            assert "age_bucket" in item

    def test_sample_customer_id_is_hex_string(self, synth_client):
        """customer_id returned by /sample must be the Kaggle hex string, not an integer."""
        r = synth_client.get("/api/v1/customers/sample?n=1")
        assert r.status_code == 200
        data = r.json()
        if data:
            cid = data[0]["customer_id"]
            # Must not be a plain integer — must be a hex string
            assert not cid.isdigit(), f"customer_id looks like integer idx: {cid!r}"


class TestCustomerProfile:
    def test_profile_hex_customer_id(self, synth_client):
        """All requests must use Kaggle hex customer_id; integer idx is rejected."""
        cid, cidx = _first_customer(synth_client)
        r = synth_client.get(f"/api/v1/customers/{cid}")
        assert r.status_code == 200
        data = r.json()
        assert data["customer_idx"] == cidx
        assert "age_bucket" in data
        assert "n_purchases_all_time" in data

    def test_profile_integer_idx_rejected(self, synth_client):
        """Integer-string customer_id is no longer accepted (Mode 1 removed)."""
        _, cidx = _first_customer(synth_client)
        r = synth_client.get(f"/api/v1/customers/{cidx}")
        # Must be 404 (not found in customer_id column) or 503 (no mapping available)
        assert r.status_code in (404, 503)

    def test_profile_unknown_customer_returns_404(self, synth_client):
        r = synth_client.get(
            "/api/v1/customers/deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        )
        assert r.status_code == 404
        assert "error" in r.json()

    def test_profile_error_envelope_format(self, synth_client):
        r = synth_client.get(
            "/api/v1/customers/nonexistent_hex_id_that_is_long_enough_abcdef"
        )
        assert r.status_code in (404, 422, 503)
        if r.status_code != 200:
            body = r.json()
            assert "error" in body


class TestRecommendations:
    def test_recommendations_returns_list(self, synth_client):
        cid, _ = _first_customer(synth_client)
        r = synth_client.get(f"/api/v1/customers/{cid}/recommendations")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)

    def test_recommendations_k12(self, synth_client):
        cid, _ = _first_customer(synth_client)
        r = synth_client.get(f"/api/v1/customers/{cid}/recommendations?k=12")
        data = r.json()
        assert len(data) <= 12

    def test_recommendations_include_reasons_true(self, synth_client):
        cid, _ = _first_customer(synth_client)
        r = synth_client.get(
            f"/api/v1/customers/{cid}/recommendations?include_reasons=true"
        )
        assert r.status_code == 200
        data = r.json()
        for item in data:
            assert "reasons" in item

    def test_recommendations_include_reasons_false(self, synth_client):
        cid, _ = _first_customer(synth_client)
        r = synth_client.get(
            f"/api/v1/customers/{cid}/recommendations?include_reasons=false"
        )
        assert r.status_code == 200
        data = r.json()
        for item in data:
            assert "reasons" not in item

    def test_recommendations_k_max_50(self, synth_client):
        cid, _ = _first_customer(synth_client)
        r = synth_client.get(f"/api/v1/customers/{cid}/recommendations?k=51")
        assert r.status_code == 422

    def test_recommendations_item_schema(self, synth_client):
        cid, _ = _first_customer(synth_client)
        r = synth_client.get(f"/api/v1/customers/{cid}/recommendations?k=5")
        data = r.json()
        for item in data:
            assert "rank" in item
            assert "article_idx" in item
            assert "sources" in item

    def test_recommendations_ranks_sequential(self, synth_client):
        cid, _ = _first_customer(synth_client)
        r = synth_client.get(f"/api/v1/customers/{cid}/recommendations?k=5")
        data = r.json()
        if data:
            assert data[0]["rank"] == 1


class TestBaseline:
    def test_baseline_returns_list(self, synth_client):
        cid, _ = _first_customer(synth_client)
        r = synth_client.get(f"/api/v1/customers/{cid}/baseline")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_baseline_k_12(self, synth_client):
        cid, _ = _first_customer(synth_client)
        r = synth_client.get(f"/api/v1/customers/{cid}/baseline?k=12")
        data = r.json()
        assert len(data) <= 12


class TestActualPurchases:
    def test_actual_purchases_200(self, synth_client):
        cid, _ = _first_customer(synth_client)
        r = synth_client.get(f"/api/v1/customers/{cid}/actual-purchases")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


class TestFunnel:
    def test_funnel_returns_list(self, synth_client):
        cid, _ = _first_customer(synth_client)
        r = synth_client.get(f"/api/v1/customers/{cid}/funnel")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)

    def test_funnel_item_schema(self, synth_client):
        cid, _ = _first_customer(synth_client)
        r = synth_client.get(f"/api/v1/customers/{cid}/funnel")
        for item in r.json():
            assert "article_idx" in item
            assert "model_score" in item
            assert "final_rank" in item


class TestCustomRecommendations:
    def _get_valid_article_id(self) -> str:
        from src.serving.bundle import load_bundle

        b = load_bundle(SYNTHETIC_BUNDLE)
        return str(b.article_meta["article_id"][0])

    def test_custom_recommendations_happy_path(self, synth_client):
        article_id = self._get_valid_article_id()
        body = {
            "history": [
                {"article_id": article_id, "t_dat": "2020-08-01"},
            ]
        }
        r = synth_client.post("/api/v1/recommendations/custom", json=body)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_custom_future_date_rejected(self, synth_client):
        article_id = self._get_valid_article_id()
        body = {
            "history": [
                {"article_id": article_id, "t_dat": "2025-01-01"},
            ]
        }
        r = synth_client.post("/api/v1/recommendations/custom", json=body)
        assert r.status_code == 422

    def test_custom_empty_history_rejected(self, synth_client):
        r = synth_client.post("/api/v1/recommendations/custom", json={"history": []})
        assert r.status_code == 422
