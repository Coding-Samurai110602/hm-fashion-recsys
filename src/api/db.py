"""SQLite profile store for user-created shopping profiles.

DB path is read from env var SQLITE_DB_PATH; defaults to data/profiles.db
relative to the project root (parent of src/).
Thread-safe via threading.local() per-thread connections.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_VALID_AGE_RANGES = {"<25", "25-34", "35-44", "45-54", "55+"}
_VALID_CHANNELS = {"online", "store"}
_MAX_PROFILES = 500
_MAX_PURCHASES_PER_PROFILE = 50

_PROJECT_ROOT = Path(__file__).parent.parent.parent  # hm-fashion-recsys/
_thread_local = threading.local()


def get_db_path() -> Path:
    """Return the absolute profiles DB path.

    Priority: SQLITE_DB_PATH env var (resolved to absolute) → project_root/data/profiles.db.
    Using .resolve() ensures relative paths in the env var are interpreted from cwd,
    not from this file's location, matching user expectations.
    """
    env_val = os.environ.get("SQLITE_DB_PATH", "")
    if env_val:
        return Path(env_val).resolve()
    return (_PROJECT_ROOT / "data" / "profiles.db").resolve()


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_PROFILES = """
CREATE TABLE IF NOT EXISTS profiles (
    id          TEXT PRIMARY KEY,
    nickname    TEXT NOT NULL,
    age_range   TEXT NOT NULL,
    channel     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""

_CREATE_PURCHASES = """
CREATE TABLE IF NOT EXISTS profile_purchases (
    id          TEXT PRIMARY KEY,
    profile_id  TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    article_idx INTEGER NOT NULL,
    article_id  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""

_CREATE_IDX_PURCHASES_PROFILE = """
CREATE INDEX IF NOT EXISTS idx_purchases_profile_id ON profile_purchases(profile_id);
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


def get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection; create tables on first use."""
    if not hasattr(_thread_local, "conn") or _thread_local.conn is None:
        db_path = get_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(_CREATE_PROFILES)
        conn.execute(_CREATE_PURCHASES)
        conn.execute(_CREATE_IDX_PURCHASES_PROFILE)
        conn.commit()
        _thread_local.conn = conn
    return _thread_local.conn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _validate_profile_input(nickname: str, age_range: str, channel: str) -> str:
    """Validate and return cleaned nickname; raises ValueError on bad input."""
    nickname = nickname.strip()
    if not nickname or len(nickname) > 30:
        raise ValueError("nickname must be 1–30 characters (stripped)")
    if age_range not in _VALID_AGE_RANGES:
        raise ValueError(f"age_range must be one of {sorted(_VALID_AGE_RANGES)}")
    if channel not in _VALID_CHANNELS:
        raise ValueError(f"channel must be one of {sorted(_VALID_CHANNELS)}")
    return nickname


def _row_to_profile(row: sqlite3.Row, purchase_count: int = 0) -> dict:
    return {
        "id": row["id"],
        "nickname": row["nickname"],
        "age_range": row["age_range"],
        "channel": row["channel"],
        "created_at": row["created_at"],
        "purchase_count": purchase_count,
    }


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def create_profile(nickname: str, age_range: str, channel: str) -> dict:
    """Validate inputs, insert a new profile, and return the full profile dict."""
    nickname = _validate_profile_input(nickname, age_range, channel)

    conn = get_connection()
    # Cap at _MAX_PROFILES
    total = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    if total >= _MAX_PROFILES:
        raise OverflowError(f"Profile store is full ({_MAX_PROFILES} max)")

    profile_id = str(uuid.uuid4())
    created_at = _now_iso()
    conn.execute(
        "INSERT INTO profiles (id, nickname, age_range, channel, created_at) VALUES (?,?,?,?,?)",
        (profile_id, nickname, age_range, channel, created_at),
    )
    conn.commit()
    return {
        "id": profile_id,
        "nickname": nickname,
        "age_range": age_range,
        "channel": channel,
        "created_at": created_at,
        "purchase_count": 0,
    }


def get_profile(profile_id: str) -> dict | None:
    """Return profile dict with purchase_count, or None if not found."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    if row is None:
        return None
    count = conn.execute(
        "SELECT COUNT(*) FROM profile_purchases WHERE profile_id = ?", (profile_id,)
    ).fetchone()[0]
    return _row_to_profile(row, count)


def list_profiles(limit: int = 50, offset: int = 0) -> tuple[list[dict], int]:
    """Return (profiles_list, total_count). Each profile includes purchase_count."""
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    rows = conn.execute(
        "SELECT * FROM profiles ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    result = []
    for row in rows:
        count = conn.execute(
            "SELECT COUNT(*) FROM profile_purchases WHERE profile_id = ?", (row["id"],)
        ).fetchone()[0]
        result.append(_row_to_profile(row, count))
    return result, total


def delete_profile(profile_id: str) -> bool:
    """Delete profile and its purchases. Returns True if deleted, False if not found."""
    conn = get_connection()
    cur = conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
    conn.commit()
    return cur.rowcount > 0


def add_purchases(
    profile_id: str,
    article_idx_list: list[int],
    article_id_list: list[str],
) -> int:
    """Append purchases to a profile (max 50 total). Returns new purchase_count.

    Raises ValueError if adding these items would exceed the per-profile cap.
    Raises LookupError if profile not found.
    """
    if len(article_idx_list) != len(article_id_list):
        raise ValueError(
            "article_idx_list and article_id_list must have the same length"
        )

    conn = get_connection()
    row = conn.execute("SELECT id FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    if row is None:
        raise LookupError(f"Profile {profile_id!r} not found")

    current_count = conn.execute(
        "SELECT COUNT(*) FROM profile_purchases WHERE profile_id = ?", (profile_id,)
    ).fetchone()[0]
    new_total = current_count + len(article_idx_list)
    if new_total > _MAX_PURCHASES_PER_PROFILE:
        raise ValueError(
            f"Adding {len(article_idx_list)} items would exceed the "
            f"{_MAX_PURCHASES_PER_PROFILE}-purchase limit "
            f"(currently {current_count})"
        )

    now = _now_iso()
    conn.executemany(
        "INSERT INTO profile_purchases (id, profile_id, article_idx, article_id, created_at) "
        "VALUES (?,?,?,?,?)",
        [
            (str(uuid.uuid4()), profile_id, int(aidx), str(aid), now)
            for aidx, aid in zip(article_idx_list, article_id_list)
        ],
    )
    conn.commit()
    return new_total


def get_purchases(profile_id: str) -> list[dict]:
    """Return list of {article_idx, article_id, created_at} for a profile."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT article_idx, article_id, created_at FROM profile_purchases "
        "WHERE profile_id = ? ORDER BY created_at ASC",
        (profile_id,),
    ).fetchall()
    return [dict(r) for r in rows]
