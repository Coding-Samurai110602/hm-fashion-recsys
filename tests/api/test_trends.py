"""API tests for the /insights/trends endpoint."""

from __future__ import annotations

import re

import pytest

pytestmark = [pytest.mark.api, pytest.mark.filterwarnings("ignore")]

_EXPECTED_AGE_BUCKETS = {"<25", "25-34", "35-44", "45-54", "55+"}
_DECAY_LABELS = {"Just launched", "Recently launched", "Established", "Mature"}

# Regex for detecting calendar dates or year numbers in response values
_DATE_PATTERN = re.compile(r"\b(19|20)\d{2}\b|\b\d{4}-\d{2}-\d{2}\b|\bweek\s+\d+\b", re.IGNORECASE)


class TestTrendsStructure:
    def test_trends_200(self, synth_client):
        r = synth_client.get("/api/v1/insights/trends")
        assert r.status_code == 200

    def test_trends_structure(self, synth_client):
        r = synth_client.get("/api/v1/insights/trends")
        assert r.status_code == 200
        data = r.json()
        assert "trending_now" in data
        assert "trending_by_age" in data
        assert "top_product_groups" in data
        assert "popularity_decay" in data

    def test_trends_trending_now_is_list(self, synth_client):
        r = synth_client.get("/api/v1/insights/trends")
        data = r.json()
        assert isinstance(data["trending_now"], list)

    def test_trends_trending_now_has_items(self, synth_client):
        r = synth_client.get("/api/v1/insights/trends")
        data = r.json()
        assert len(data["trending_now"]) > 0

    def test_trends_trending_now_item_schema(self, synth_client):
        r = synth_client.get("/api/v1/insights/trends")
        data = r.json()
        required = {"article_idx", "article_id", "prod_name", "product_group", "colour", "image_url"}
        for item in data["trending_now"]:
            for field in required:
                assert field in item, f"Missing {field!r} in trending_now item"

    def test_trends_by_age_has_expected_buckets(self, synth_client):
        r = synth_client.get("/api/v1/insights/trends")
        data = r.json()
        by_age = data["trending_by_age"]
        assert isinstance(by_age, dict)
        # Must have at least 2 age buckets (synthetic bundle has all 5)
        present_buckets = set(by_age.keys())
        assert len(present_buckets & _EXPECTED_AGE_BUCKETS) >= 2, (
            f"Expected at least 2 age buckets, got: {present_buckets}"
        )

    def test_trends_by_age_bucket_items(self, synth_client):
        r = synth_client.get("/api/v1/insights/trends")
        data = r.json()
        by_age = data["trending_by_age"]
        for bucket, items in by_age.items():
            assert isinstance(items, list), f"Bucket {bucket!r} should be a list"
            for item in items:
                assert "article_idx" in item
                assert "article_id" in item

    def test_trends_top_product_groups_is_list(self, synth_client):
        r = synth_client.get("/api/v1/insights/trends")
        data = r.json()
        assert isinstance(data["top_product_groups"], list)

    def test_trends_top_product_groups_schema(self, synth_client):
        r = synth_client.get("/api/v1/insights/trends")
        data = r.json()
        for item in data["top_product_groups"]:
            assert "product_group" in item
            assert "share" in item
            assert "count" in item

    def test_trends_decay_is_list(self, synth_client):
        r = synth_client.get("/api/v1/insights/trends")
        data = r.json()
        assert isinstance(data["popularity_decay"], list)

    def test_trends_decay_shares_sum_to_one(self, synth_client):
        r = synth_client.get("/api/v1/insights/trends")
        data = r.json()
        decay = data["popularity_decay"]
        assert len(decay) == 4, f"Expected 4 decay buckets, got {len(decay)}"
        total_share = sum(item["share"] for item in decay)
        assert abs(total_share - 1.0) < 0.01, f"Shares sum to {total_share}, expected ~1.0"

    def test_trends_decay_labels(self, synth_client):
        r = synth_client.get("/api/v1/insights/trends")
        data = r.json()
        labels = {item["age_label"] for item in data["popularity_decay"]}
        assert labels == _DECAY_LABELS, f"Unexpected labels: {labels}"

    def test_trends_decay_last_bucket_has_null_age_days_max(self, synth_client):
        r = synth_client.get("/api/v1/insights/trends")
        data = r.json()
        decay = data["popularity_decay"]
        mature = next((d for d in decay if d["age_label"] == "Mature"), None)
        assert mature is not None
        assert mature["age_days_max"] is None


class TestTrendsNoDates:
    def _extract_string_values(self, obj) -> list[str]:
        """Recursively collect all string values from a JSON-decoded object."""
        results = []
        if isinstance(obj, str):
            results.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                results.extend(self._extract_string_values(v))
        elif isinstance(obj, list):
            for item in obj:
                results.extend(self._extract_string_values(item))
        return results

    def test_trends_no_dates_in_response(self, synth_client):
        """Ensure no 4-digit years or date strings appear in string values."""
        r = synth_client.get("/api/v1/insights/trends")
        assert r.status_code == 200
        data = r.json()
        string_values = self._extract_string_values(data)
        for val in string_values:
            # Allow image URLs (they won't have dates) and product names
            # but flag any year-like patterns in non-URL strings
            if "image_url" in str(val) or val.startswith("/api/"):
                continue
            match = _DATE_PATTERN.search(val)
            assert match is None, (
                f"Found date-like pattern {match.group()!r} in response value {val!r}"
            )
