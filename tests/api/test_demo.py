"""API tests for the demo shoppers endpoint."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.api, pytest.mark.filterwarnings("ignore")]

_VALID_AGE_RANGES = {"<25", "25-34", "35-44", "45-54", "55+"}
_VALID_CHANNELS = {"online", "store"}

_REQUIRED_FIELDS = {
    "customer_id",
    "customer_idx",
    "age_range",
    "channel",
    "purchase_count",
    "top_product_groups",
    "summary",
}


class TestDemoShoppers:
    def test_demo_shoppers_returns_list(self, synth_client):
        r = synth_client.get("/demo/shoppers")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)

    def test_demo_shoppers_has_items(self, synth_client):
        r = synth_client.get("/demo/shoppers")
        assert r.status_code == 200
        data = r.json()
        # Synthetic bundle has 200 customers with 5–150 purchases; expect at least 1
        assert len(data) >= 1

    def test_demo_shoppers_schema(self, synth_client):
        """Each item has required fields."""
        r = synth_client.get("/demo/shoppers")
        assert r.status_code == 200
        data = r.json()
        for item in data:
            for field in _REQUIRED_FIELDS:
                assert field in item, f"Missing field {field!r} in {item}"

    def test_demo_shoppers_age_range_values(self, synth_client):
        """age_range must be a valid bucket."""
        r = synth_client.get("/demo/shoppers")
        assert r.status_code == 200
        for item in r.json():
            assert item["age_range"] in _VALID_AGE_RANGES, (
                f"Invalid age_range {item['age_range']!r}"
            )

    def test_demo_shoppers_channel_values(self, synth_client):
        """channel must be 'online' or 'store'."""
        r = synth_client.get("/demo/shoppers")
        assert r.status_code == 200
        for item in r.json():
            assert item["channel"] in _VALID_CHANNELS, (
                f"Invalid channel {item['channel']!r}"
            )

    def test_demo_shoppers_purchase_count_positive(self, synth_client):
        r = synth_client.get("/demo/shoppers")
        assert r.status_code == 200
        for item in r.json():
            assert item["purchase_count"] >= 0

    def test_demo_shoppers_top_product_groups_is_list(self, synth_client):
        r = synth_client.get("/demo/shoppers")
        assert r.status_code == 200
        for item in r.json():
            assert isinstance(item["top_product_groups"], list)

    def test_demo_shoppers_summary_is_string(self, synth_client):
        r = synth_client.get("/demo/shoppers")
        assert r.status_code == 200
        for item in r.json():
            assert isinstance(item["summary"], str)
            assert len(item["summary"]) > 0

    def test_demo_shoppers_customer_id_nonempty(self, synth_client):
        r = synth_client.get("/demo/shoppers")
        assert r.status_code == 200
        for item in r.json():
            assert isinstance(item["customer_id"], str)
            assert len(item["customer_id"]) > 0

    def test_demo_shoppers_cached(self, synth_client):
        """Two calls should return identical results (cached)."""
        r1 = synth_client.get("/demo/shoppers")
        r2 = synth_client.get("/demo/shoppers")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json() == r2.json()
