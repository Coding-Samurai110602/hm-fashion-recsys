"""Leakage tests: candidate sources must not use data from week >= cutoff_week.

Strategy: build a synthetic history with a known sentinel transaction at week==cutoff_week
(i.e., inside the target period). Each source filters week_idx < cutoff_week; the sentinel
must never appear in candidate output or scores.
"""
from datetime import date, timedelta

import polars as pl
import pytest

from src.config import ANCHOR_DATE, ANCHOR_EPOCH_DAYS
from src.data_io import load_transactions

# Sentinel: a fake article_idx unlikely to appear in real data
SENTINEL_ARTICLE = 999999
SENTINEL_CUSTOMER = 888888
CUTOFF_WEEK = 102

# The sentinel transaction is dated in week CUTOFF_WEEK (should be excluded)
# week start = ANCHOR_DATE + timedelta(weeks=CUTOFF_WEEK)
_sentinel_date = ANCHOR_DATE + timedelta(weeks=CUTOFF_WEEK)
_sentinel_date_int = (date(*_sentinel_date.timetuple()[:3]) - date(1970, 1, 1)).days


@pytest.fixture(scope="module")
def synthetic_history():
    """A minimal transactions LazyFrame with a sentinel row in the target week."""
    # Load a tiny slice of real history (to give sources something to work with)
    real_slice = (
        load_transactions()
        .filter(pl.col("week_idx") < CUTOFF_WEEK)
        .head(50000)
        .collect()
    )

    # Compute sentinel week_idx to verify it equals CUTOFF_WEEK
    sentinel_week = (_sentinel_date_int - ANCHOR_EPOCH_DAYS) // 7
    assert sentinel_week == CUTOFF_WEEK, (
        f"Sentinel at week {sentinel_week}, expected {CUTOFF_WEEK}"
    )

    # Append the sentinel row (at week == CUTOFF_WEEK — should be excluded by all sources)
    sentinel_row = pl.DataFrame({
        "t_dat": [date(*_sentinel_date.timetuple()[:3])],
        "customer_idx": pl.Series([SENTINEL_CUSTOMER], dtype=pl.Int32),
        "article_idx": pl.Series([SENTINEL_ARTICLE], dtype=pl.Int32),
        "price": pl.Series([0.1], dtype=pl.Float32),
        "sales_channel_id": pl.Series([1], dtype=pl.Int8),
        "week_idx": pl.Series([sentinel_week], dtype=pl.Int16),
    })

    combined = pl.concat([real_slice, sentinel_row], how="diagonal").lazy()
    return combined


def _assert_no_sentinel(df: pl.DataFrame, source_name: str) -> None:
    has_sentinel_article = (df["article_idx"] == SENTINEL_ARTICLE).any()
    assert not has_sentinel_article, (
        f"[{source_name}] Sentinel article {SENTINEL_ARTICLE} found in candidates — leakage!"
    )


def test_repurchase_no_leakage(synthetic_history):
    from src.candidates.repurchase import generate
    # Use the sentinel customer itself to ensure it's included in evaluation
    customers = [SENTINEL_CUSTOMER] + list(range(0, 100))
    df = generate(synthetic_history, CUTOFF_WEEK, customers, k=20)
    _assert_no_sentinel(df, "repurchase")


def test_product_code_no_leakage(synthetic_history):
    from src.candidates.product_code import generate
    customers = [SENTINEL_CUSTOMER] + list(range(0, 100))
    df = generate(synthetic_history, CUTOFF_WEEK, customers, k=20)
    _assert_no_sentinel(df, "product_code")


def test_popularity_last_week_no_leakage(synthetic_history):
    from src.candidates.popularity import generate_global_last_week
    customers = [SENTINEL_CUSTOMER] + list(range(0, 100))
    df = generate_global_last_week(synthetic_history, CUTOFF_WEEK, customers, k=20)
    _assert_no_sentinel(df, "popularity_last_week")


def test_popularity_decayed_no_leakage(synthetic_history):
    from src.candidates.popularity import generate_global_decayed
    customers = [SENTINEL_CUSTOMER] + list(range(0, 100))
    df = generate_global_decayed(synthetic_history, CUTOFF_WEEK, customers, k=20)
    _assert_no_sentinel(df, "popularity_decayed")


def test_segment_popular_no_leakage(synthetic_history):
    from src.candidates.popularity import generate_segment_popular
    customers = [SENTINEL_CUSTOMER] + list(range(0, 100))
    df = generate_segment_popular(synthetic_history, CUTOFF_WEEK, customers, k=20)
    _assert_no_sentinel(df, "segment_popular")


def test_copurchase_no_leakage(synthetic_history):
    from src.candidates.copurchase import generate
    customers = [SENTINEL_CUSTOMER] + list(range(0, 100))
    df = generate(synthetic_history, CUTOFF_WEEK, customers, k=20)
    _assert_no_sentinel(df, "copurchase")


def test_no_source_reads_cutoff_week(synthetic_history):
    """Each source must filter week_idx < cutoff_week; verify the sentinel is excluded."""
    from src.candidates.repurchase import generate as rep_gen
    from src.candidates.popularity import generate_global_last_week as pop_gen

    # Check that both sources produce no sentinel article from week == cutoff_week
    customers = list(range(0, 200))
    df_rep = rep_gen(synthetic_history, CUTOFF_WEEK, customers, k=50)
    df_pop = pop_gen(synthetic_history, CUTOFF_WEEK, customers, k=50)

    _assert_no_sentinel(df_rep, "repurchase")
    _assert_no_sentinel(df_pop, "popularity_last_week")


def test_history_filter_strict(synthetic_history):
    """Verify that our synthetic history contains a row with week_idx == CUTOFF_WEEK."""
    sentinel_rows = (
        synthetic_history
        .filter(pl.col("week_idx") == CUTOFF_WEEK)
        .collect()
    )
    assert len(sentinel_rows) >= 1, "Synthetic history should contain a sentinel row at cutoff week"
