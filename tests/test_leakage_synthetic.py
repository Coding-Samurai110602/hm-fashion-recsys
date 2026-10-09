"""Candidate source leakage tests using purely synthetic data — no Kaggle data required.

Verifies that each candidate source correctly filters week_idx < cutoff_week
and never surfaces the sentinel article that lives in week == cutoff_week.

These tests run in CI (no requires_data marker).
The full real-data leakage tests are in test_leakage.py (requires_data).
"""
import polars as pl
import pytest

from src.config import ANCHOR_EPOCH_DAYS

_SENTINEL_ARTICLE = 777777
_SENTINEL_CUSTOMER = 666666
_CUTOFF = 5  # small week index for fast arithmetic


def _make_history() -> pl.LazyFrame:
    """Purely synthetic transactions: history in weeks 2-4, sentinel in week 5.

    Customers 1 and 2 bought articles 100/101 in weeks 2-4.
    The sentinel article (777777) is purchased in week 5 (== cutoff) — every
    source must exclude it because it filters week_idx < cutoff_week.
    """
    anchor = ANCHOR_EPOCH_DAYS
    rows = [
        # Customer 1 history (weeks < cutoff)
        (1, 100, anchor + 2 * 7, 0.05, 1, 2),
        (1, 101, anchor + 3 * 7, 0.04, 1, 3),
        (1, 100, anchor + 4 * 7, 0.05, 2, 4),
        # Customer 2 history (co-buyer of 100 — copurchase signal)
        (2, 100, anchor + 3 * 7, 0.05, 1, 3),
        (2, 101, anchor + 4 * 7, 0.04, 1, 4),
        # Sentinel: in target week (week == cutoff) — must never appear in candidates
        (_SENTINEL_CUSTOMER, _SENTINEL_ARTICLE, anchor + 5 * 7, 0.99, 1, 5),
    ]
    df = pl.DataFrame(
        {
            "customer_idx": pl.Series([r[0] for r in rows], dtype=pl.Int32),
            "article_idx": pl.Series([r[1] for r in rows], dtype=pl.Int32),
            "t_dat": pl.Series([r[2] for r in rows], dtype=pl.Int32).cast(pl.Date),
            "price": pl.Series([r[3] for r in rows], dtype=pl.Float32),
            "sales_channel_id": pl.Series([r[4] for r in rows], dtype=pl.Int8),
            "week_idx": pl.Series([r[5] for r in rows], dtype=pl.Int16),
        }
    )
    return df.lazy()


def _assert_clean(df: pl.DataFrame, source: str) -> None:
    found = (df["article_idx"] == _SENTINEL_ARTICLE).any()
    assert not found, (
        f"[{source}] Sentinel article {_SENTINEL_ARTICLE} appeared in candidates "
        "— week_idx >= cutoff_week filter is missing!"
    )


_CUSTOMERS = [1, 2, _SENTINEL_CUSTOMER]


def test_sentinel_week_present_in_synthetic_history():
    """Confirm the sentinel IS in the history at week == cutoff so the test is meaningful."""
    hist = _make_history().filter(pl.col("week_idx") == _CUTOFF).collect()
    assert len(hist) >= 1, "Sentinel row missing from synthetic history"
    assert _SENTINEL_ARTICLE in hist["article_idx"].to_list()


def test_repurchase_no_leakage_synthetic():
    from src.candidates.repurchase import generate

    df = generate(_make_history(), _CUTOFF, _CUSTOMERS, k=20)
    _assert_clean(df, "repurchase")


def test_popularity_last_week_no_leakage_synthetic():
    from src.candidates.popularity import generate_global_last_week

    df = generate_global_last_week(_make_history(), _CUTOFF, _CUSTOMERS, k=20)
    _assert_clean(df, "popularity_last_week")


def test_popularity_decayed_no_leakage_synthetic():
    from src.candidates.popularity import generate_global_decayed

    df = generate_global_decayed(_make_history(), _CUTOFF, _CUSTOMERS, k=20)
    _assert_clean(df, "popularity_decayed")


def test_copurchase_no_leakage_synthetic():
    from src.candidates.copurchase import generate

    df = generate(_make_history(), _CUTOFF, _CUSTOMERS, k=20)
    _assert_clean(df, "copurchase")
