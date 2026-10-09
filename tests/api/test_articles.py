"""API tests for article catalog endpoints."""

from __future__ import annotations

import pytest

from tests.api.conftest import SYNTHETIC_BUNDLE

pytestmark = pytest.mark.filterwarnings("ignore")


def _get_article_id() -> str:
    from src.serving.bundle import load_bundle

    b = load_bundle(SYNTHETIC_BUNDLE)
    return str(b.article_meta["article_id"][0])


class TestArticleSearch:
    def test_search_no_params(self, synth_client):
        r = synth_client.get("/api/v1/articles/search")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_search_limit(self, synth_client):
        r = synth_client.get("/api/v1/articles/search?limit=5")
        assert r.status_code == 200
        assert len(r.json()) <= 5

    def test_search_by_q(self, synth_client):
        r = synth_client.get("/api/v1/articles/search?q=Article")
        assert r.status_code == 200

    def test_search_limit_too_large(self, synth_client):
        r = synth_client.get("/api/v1/articles/search?limit=101")
        assert r.status_code == 422

    def test_search_schema(self, synth_client):
        r = synth_client.get("/api/v1/articles/search?limit=3")
        for item in r.json():
            assert "article_idx" in item
            assert "article_id" in item
            assert "prod_name" in item


class TestArticleGet:
    def test_get_existing_article(self, synth_client):
        aid = _get_article_id()
        r = synth_client.get(f"/api/v1/articles/{aid}")
        assert r.status_code == 200
        data = r.json()
        assert data["article_id"] == aid

    def test_get_unknown_article_404(self, synth_client):
        r = synth_client.get("/api/v1/articles/0000000000")
        assert r.status_code == 404
        assert "error" in r.json()

    def test_get_article_schema(self, synth_client):
        aid = _get_article_id()
        r = synth_client.get(f"/api/v1/articles/{aid}")
        data = r.json()
        assert "article_idx" in data
        assert "colour" in data
        assert "department" in data


class TestArticleImage:
    def test_image_valid_id_returns_image(self, synth_client):
        aid = _get_article_id()
        r = synth_client.get(f"/api/v1/articles/{aid}/image")
        # Returns placeholder PNG or real JPEG; either is fine
        assert r.status_code == 200
        assert r.headers["content-type"] in ("image/jpeg", "image/png")

    def test_image_path_traversal_rejected(self, synth_client):
        r = synth_client.get("/api/v1/articles/../../../etc/passwd/image")
        assert r.status_code in (404, 422)

    def test_image_wrong_format_rejected(self, synth_client):
        r = synth_client.get("/api/v1/articles/abc/image")
        assert r.status_code == 422
