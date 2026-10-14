"""API tests for catalog endpoints: trending, browse, filters."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.api, pytest.mark.filterwarnings("ignore")]


class TestTrending:
    def test_trending_returns_items(self, synth_client):
        r = synth_client.get("/api/v1/catalog/trending")
        assert r.status_code == 200
        items = r.json()
        assert isinstance(items, list)
        assert len(items) > 0

    def test_trending_k_parameter(self, synth_client):
        r = synth_client.get("/api/v1/catalog/trending?k=5")
        assert r.status_code == 200
        items = r.json()
        assert len(items) <= 5

    def test_trending_default_k_is_20(self, synth_client):
        r = synth_client.get("/api/v1/catalog/trending")
        assert r.status_code == 200
        items = r.json()
        assert len(items) <= 20

    def test_trending_item_schema(self, synth_client):
        r = synth_client.get("/api/v1/catalog/trending?k=3")
        assert r.status_code == 200
        for item in r.json():
            assert "article_idx" in item
            assert "article_id" in item
            assert "prod_name" in item
            assert "product_group" in item
            assert "colour" in item
            assert "image_url" in item
            assert "popularity_rank" in item
            assert item["image_url"].startswith("/api/v1/articles/")

    def test_trending_k_max(self, synth_client):
        r = synth_client.get("/api/v1/catalog/trending?k=200")
        assert r.status_code == 200

    def test_trending_k_above_max_rejected(self, synth_client):
        r = synth_client.get("/api/v1/catalog/trending?k=201")
        assert r.status_code == 422


class TestBrowse:
    def test_browse_no_filter(self, synth_client):
        r = synth_client.get("/api/v1/catalog/browse")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert data["page"] == 1
        assert data["page_size"] == 24

    def test_browse_by_product_group(self, synth_client):
        # Get a product group from filters first
        r_filters = synth_client.get("/api/v1/catalog/filters")
        filters = r_filters.json()
        groups = filters["product_groups"]
        if not groups:
            pytest.skip("No product groups in catalog")
        group = groups[0]
        r = synth_client.get(f"/api/v1/catalog/browse?product_group={group}")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["items"], list)
        # All returned items should match the filter
        for item in data["items"]:
            assert item["product_group"].lower() == group.lower()

    def test_browse_by_colour(self, synth_client):
        r_filters = synth_client.get("/api/v1/catalog/filters")
        filters = r_filters.json()
        colours = filters["colours"]
        if not colours:
            pytest.skip("No colours in catalog")
        colour = colours[0]
        r = synth_client.get(f"/api/v1/catalog/browse?colour={colour}")
        assert r.status_code == 200
        data = r.json()
        for item in data["items"]:
            assert item["colour"].lower() == colour.lower()

    def test_browse_search_q(self, synth_client):
        r = synth_client.get("/api/v1/catalog/browse?q=Article")
        assert r.status_code == 200
        data = r.json()
        # Synthetic articles are named "Article N"
        assert len(data["items"]) > 0
        for item in data["items"]:
            assert "article" in item["prod_name"].lower()

    def test_browse_search_q_no_results(self, synth_client):
        r = synth_client.get("/api/v1/catalog/browse?q=XYZNONEXISTENT99999")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_browse_pagination(self, synth_client):
        r1 = synth_client.get("/api/v1/catalog/browse?page=1&page_size=5")
        r2 = synth_client.get("/api/v1/catalog/browse?page=2&page_size=5")
        assert r1.status_code == 200
        assert r2.status_code == 200
        d1 = r1.json()
        d2 = r2.json()
        assert d1["page"] == 1
        assert d2["page"] == 2
        assert d1["total"] == d2["total"]
        # Different items on different pages
        ids1 = {item["article_id"] for item in d1["items"]}
        ids2 = {item["article_id"] for item in d2["items"]}
        assert ids1.isdisjoint(ids2) or len(ids1) == 0

    def test_browse_page_size_max(self, synth_client):
        r = synth_client.get("/api/v1/catalog/browse?page_size=100")
        assert r.status_code == 200

    def test_browse_page_size_above_max_rejected(self, synth_client):
        r = synth_client.get("/api/v1/catalog/browse?page_size=101")
        assert r.status_code == 422

    def test_browse_item_has_image_url(self, synth_client):
        r = synth_client.get("/api/v1/catalog/browse?page_size=3")
        assert r.status_code == 200
        for item in r.json()["items"]:
            assert "image_url" in item
            assert item["image_url"].startswith("/api/v1/articles/")


class TestFilters:
    def test_filters_structure(self, synth_client):
        r = synth_client.get("/api/v1/catalog/filters")
        assert r.status_code == 200
        data = r.json()
        assert "product_groups" in data
        assert "colours" in data
        assert isinstance(data["product_groups"], list)
        assert isinstance(data["colours"], list)

    def test_filters_has_data(self, synth_client):
        r = synth_client.get("/api/v1/catalog/filters")
        assert r.status_code == 200
        data = r.json()
        assert len(data["product_groups"]) > 0
        assert len(data["colours"]) > 0

    def test_filters_sorted(self, synth_client):
        r = synth_client.get("/api/v1/catalog/filters")
        data = r.json()
        groups = data["product_groups"]
        colours = data["colours"]
        assert groups == sorted(groups)
        assert colours == sorted(colours)
