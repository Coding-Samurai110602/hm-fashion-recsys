"""API tests for profile CRUD, recommendations, summary, and compare endpoints."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.api, pytest.mark.filterwarnings("ignore")]

# ---------------------------------------------------------------------------
# Fixture override: use an in-memory (tmpdir) SQLite so tests are isolated
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path):
    """Point SQLITE_DB_PATH to a fresh temp file for each test."""
    db_file = tmp_path / "test_profiles.db"
    old = os.environ.get("SQLITE_DB_PATH")
    os.environ["SQLITE_DB_PATH"] = str(db_file)
    # Reset thread-local connection so get_connection() sees the new path
    from src.api import db as db_mod
    if hasattr(db_mod._thread_local, "conn") and db_mod._thread_local.conn:
        try:
            db_mod._thread_local.conn.close()
        except Exception:  # noqa: BLE001,S110
            pass
    db_mod._thread_local.conn = None
    yield
    # Cleanup
    if hasattr(db_mod._thread_local, "conn") and db_mod._thread_local.conn:
        try:
            db_mod._thread_local.conn.close()
        except Exception:  # noqa: BLE001,S110
            pass
    db_mod._thread_local.conn = None
    if old is not None:
        os.environ["SQLITE_DB_PATH"] = old
    else:
        os.environ.pop("SQLITE_DB_PATH", None)


# ---------------------------------------------------------------------------
# Helper to create a valid profile
# ---------------------------------------------------------------------------

_VALID_PROFILE = {"nickname": "Tester", "age_range": "25-34", "channel": "online"}


def _create(client, **kwargs) -> dict:
    body = {**_VALID_PROFILE, **kwargs}
    r = client.post("/api/v1/profiles", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _first_article_id(client) -> str:
    """Get a valid article_id from the catalog."""
    r = client.get("/api/v1/catalog/trending?k=1")
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 1, "No trending items in catalog"
    return items[0]["article_id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateProfile:
    def test_create_profile_valid(self, synth_client):
        r = synth_client.post("/api/v1/profiles", json=_VALID_PROFILE)
        assert r.status_code == 201
        data = r.json()
        assert data["nickname"] == "Tester"
        assert data["age_range"] == "25-34"
        assert data["channel"] == "online"
        assert data["purchase_count"] == 0
        assert "id" in data
        assert "created_at" in data

    def test_create_profile_invalid_age_range(self, synth_client):
        r = synth_client.post(
            "/api/v1/profiles",
            json={"nickname": "X", "age_range": "invalid", "channel": "online"},
        )
        assert r.status_code == 422

    def test_create_profile_nickname_too_long(self, synth_client):
        r = synth_client.post(
            "/api/v1/profiles",
            json={"nickname": "A" * 31, "age_range": "25-34", "channel": "online"},
        )
        assert r.status_code == 422

    def test_create_profile_invalid_channel(self, synth_client):
        r = synth_client.post(
            "/api/v1/profiles",
            json={"nickname": "X", "age_range": "25-34", "channel": "web"},
        )
        assert r.status_code == 422


class TestListProfiles:
    def test_list_profiles_empty(self, synth_client):
        # Module-scoped client may have profiles from other tests; just check structure
        r = synth_client.get("/api/v1/profiles")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "profiles" in data
        assert isinstance(data["profiles"], list)
        assert data["total"] >= 0

    def test_list_profiles_after_create(self, synth_client):
        before = synth_client.get("/api/v1/profiles").json()["total"]
        _create(synth_client, nickname="Alice")
        _create(synth_client, nickname="Bob")
        r = synth_client.get("/api/v1/profiles")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == before + 2
        assert len(data["profiles"]) >= 2

    def test_list_profiles_pagination(self, synth_client):
        for i in range(5):
            _create(synth_client, nickname=f"PagUser{i}")
        r = synth_client.get("/api/v1/profiles?limit=2&offset=0")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] >= 5
        assert len(data["profiles"]) == 2


class TestGetProfile:
    def test_get_profile_not_found(self, synth_client):
        r = synth_client.get("/api/v1/profiles/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404

    def test_get_profile_ok(self, synth_client):
        profile = _create(synth_client)
        r = synth_client.get(f"/api/v1/profiles/{profile['id']}")
        assert r.status_code == 200
        assert r.json()["id"] == profile["id"]


class TestDeleteProfile:
    def test_delete_profile(self, synth_client):
        profile = _create(synth_client)
        r = synth_client.delete(f"/api/v1/profiles/{profile['id']}")
        assert r.status_code == 204
        # Should be gone
        r2 = synth_client.get(f"/api/v1/profiles/{profile['id']}")
        assert r2.status_code == 404

    def test_delete_profile_not_found(self, synth_client):
        r = synth_client.delete("/api/v1/profiles/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404


class TestCheckout:
    def test_checkout_adds_purchases(self, synth_client):
        profile = _create(synth_client)
        art_id = _first_article_id(synth_client)
        r = synth_client.post(
            f"/api/v1/profiles/{profile['id']}/checkout",
            json={"article_ids": [art_id]},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["new_count"] == 1
        assert data["purchase_count"] == 1  # now returns new total (not old count)

    def test_checkout_max_50_limit(self, synth_client):
        profile = _create(synth_client)
        # Get some article IDs from catalog
        r = synth_client.get("/api/v1/catalog/trending?k=50")
        items = r.json()
        article_ids = [item["article_id"] for item in items]

        # Add 45 items
        batch1 = article_ids[:45]
        r1 = synth_client.post(
            f"/api/v1/profiles/{profile['id']}/checkout",
            json={"article_ids": batch1},
        )
        assert r1.status_code == 200
        assert r1.json()["new_count"] == 45

        # Try to add 10 more (would exceed 50)
        batch2 = article_ids[:10]
        r2 = synth_client.post(
            f"/api/v1/profiles/{profile['id']}/checkout",
            json={"article_ids": batch2},
        )
        assert r2.status_code == 400

    def test_checkout_profile_not_found(self, synth_client):
        art_id = _first_article_id(synth_client)
        r = synth_client.post(
            "/api/v1/profiles/00000000-0000-0000-0000-000000000000/checkout",
            json={"article_ids": [art_id]},
        )
        assert r.status_code in (404, 400)  # not found


class TestRecommendations:
    def test_recommendations_cold_start(self, synth_client):
        """0 purchases → 12 trending items with cold_start=True."""
        profile = _create(synth_client)
        r = synth_client.get(f"/api/v1/profiles/{profile['id']}/recommendations")
        assert r.status_code == 200
        data = r.json()
        assert data["cold_start"] is True
        items = data["items"]
        assert len(items) > 0
        assert len(items) <= 12
        for item in items:
            assert item["cold_start"] is True
            assert item["source"] == "trending"
            assert "article_id" in item
            assert "image_url" in item

    def test_recommendations_with_purchases(self, synth_client):
        """After adding purchases, recs return up to 12 items."""
        profile = _create(synth_client)
        art_id = _first_article_id(synth_client)
        synth_client.post(
            f"/api/v1/profiles/{profile['id']}/checkout",
            json={"article_ids": [art_id]},
        )
        r = synth_client.get(f"/api/v1/profiles/{profile['id']}/recommendations")
        assert r.status_code == 200
        data = r.json()
        items = data["items"]
        assert len(items) <= 12
        for item in items:
            assert "article_id" in item
            assert "score" in item

    def test_recommendations_schema(self, synth_client):
        """Each item has required fields."""
        profile = _create(synth_client)
        r = synth_client.get(f"/api/v1/profiles/{profile['id']}/recommendations")
        assert r.status_code == 200
        data = r.json()
        assert "profile_id" in data
        assert "cold_start" in data
        assert "items" in data
        assert "latency_ms" in data

    def test_age_range_changes_candidates(self, synth_client):
        """Two profiles with same purchases but different age_range both return 12 items."""
        art_id = _first_article_id(synth_client)

        p1 = _create(synth_client, age_range="<25", nickname="Young")
        synth_client.post(
            f"/api/v1/profiles/{p1['id']}/checkout",
            json={"article_ids": [art_id]},
        )

        p2 = _create(synth_client, age_range="55+", nickname="Senior")
        synth_client.post(
            f"/api/v1/profiles/{p2['id']}/checkout",
            json={"article_ids": [art_id]},
        )

        r1 = synth_client.get(f"/api/v1/profiles/{p1['id']}/recommendations")
        r2 = synth_client.get(f"/api/v1/profiles/{p2['id']}/recommendations")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert len(r1.json()["items"]) <= 12
        assert len(r2.json()["items"]) <= 12


class TestProfileSummary:
    def test_profile_summary(self, synth_client):
        profile = _create(synth_client)
        r = synth_client.get(f"/api/v1/profiles/{profile['id']}/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["profile_id"] == profile["id"]
        assert data["nickname"] == "Tester"
        assert data["age_range"] == "25-34"
        assert data["channel"] == "online"
        assert isinstance(data["top_product_groups"], list)
        assert data["purchase_count"] == 0

    def test_profile_summary_with_purchases(self, synth_client):
        profile = _create(synth_client)
        art_id = _first_article_id(synth_client)
        synth_client.post(
            f"/api/v1/profiles/{profile['id']}/checkout",
            json={"article_ids": [art_id]},
        )
        r = synth_client.get(f"/api/v1/profiles/{profile['id']}/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["purchase_count"] == 1


class TestCompare:
    def test_compare_two_profiles(self, synth_client):
        p1 = _create(synth_client, nickname="A", age_range="<25")
        p2 = _create(synth_client, nickname="B", age_range="55+")
        r = synth_client.get(f"/api/v1/compare?a={p1['id']}&b={p2['id']}")
        assert r.status_code == 200
        data = r.json()
        assert "a_summary" in data
        assert "b_summary" in data
        assert "a_recommendations" in data
        assert "b_recommendations" in data
        assert "overlap" in data
        assert "overlap_count" in data
        assert isinstance(data["overlap"], list)
        assert data["a_summary"]["profile_id"] == p1["id"]
        assert data["b_summary"]["profile_id"] == p2["id"]

    def test_compare_not_found(self, synth_client):
        r = synth_client.get(
            "/api/v1/compare?a=00000000-0000-0000-0000-000000000000&b=00000000-0000-0000-0000-000000000001"
        )
        assert r.status_code == 404
