"""Tests for candidate sources: schema, dtypes, no dups, determinism.

Uses a small sample of ~500 eval_customers from fold 103 to keep runtime under 90s.
"""
import random

import polars as pl
import pytest

from src.candidates.base import OUTPUT_COLUMNS, OUTPUT_DTYPES
from src.time_split import build_fold

SAMPLE_SIZE = 500
SEED = 42
K = 50
FOLD_WEEK = 103


@pytest.fixture(scope="module")
def fold_data():
    history, ground_truth, eval_customers = build_fold(FOLD_WEEK)
    rng = random.Random(SEED)
    sampled = sorted(rng.sample(eval_customers, min(SAMPLE_SIZE, len(eval_customers))))
    return history, ground_truth, sampled


def _assert_schema(df: pl.DataFrame, source_name: str) -> None:
    for col in OUTPUT_COLUMNS:
        assert col in df.columns, f"[{source_name}] Missing column: {col}"
    for col, expected in OUTPUT_DTYPES.items():
        assert df.schema[col] == expected, (
            f"[{source_name}] Column '{col}': expected {expected}, got {df.schema[col]}"
        )


def _assert_no_dups(df: pl.DataFrame, source_name: str) -> None:
    n = len(df)
    n_unique = df.select(["customer_idx", "article_idx"]).n_unique()
    assert n == n_unique, (
        f"[{source_name}] {n - n_unique} duplicate (customer_idx, article_idx) pairs"
    )


def _assert_max_k(df: pl.DataFrame, k: int, source_name: str) -> None:
    if len(df) == 0:
        return
    max_rank = (
        df.group_by("customer_idx")
        .agg(pl.col("source_rank").max())
        ["source_rank"]
        .max()
    )
    assert max_rank <= k, f"[{source_name}] max source_rank {max_rank} > k={k}"


def _assert_deterministic(df1: pl.DataFrame, df2: pl.DataFrame, source_name: str) -> None:
    key = ["customer_idx", "source_rank"]
    s1 = df1.sort(key)
    s2 = df2.sort(key)
    assert s1.equals(s2), f"[{source_name}] Two runs produced different results"


# ---------------------------------------------------------------------------
# Repurchase
# ---------------------------------------------------------------------------

def test_repurchase_schema(fold_data):
    from src.candidates.repurchase import generate
    history, _, customers = fold_data
    df = generate(history, FOLD_WEEK, customers, K)
    _assert_schema(df, "repurchase")


def test_repurchase_no_dups(fold_data):
    from src.candidates.repurchase import generate
    history, _, customers = fold_data
    df = generate(history, FOLD_WEEK, customers, K)
    _assert_no_dups(df, "repurchase")


def test_repurchase_max_k(fold_data):
    from src.candidates.repurchase import generate
    history, _, customers = fold_data
    df = generate(history, FOLD_WEEK, customers, K)
    _assert_max_k(df, K, "repurchase")


def test_repurchase_deterministic(fold_data):
    from src.candidates.repurchase import generate
    history, _, customers = fold_data
    df1 = generate(history, FOLD_WEEK, customers, K)
    df2 = generate(history, FOLD_WEEK, customers, K)
    _assert_deterministic(df1, df2, "repurchase")


# ---------------------------------------------------------------------------
# Product code
# ---------------------------------------------------------------------------

def test_product_code_schema(fold_data):
    from src.candidates.product_code import generate
    history, _, customers = fold_data
    df = generate(history, FOLD_WEEK, customers, K)
    _assert_schema(df, "product_code")


def test_product_code_no_dups(fold_data):
    from src.candidates.product_code import generate
    history, _, customers = fold_data
    df = generate(history, FOLD_WEEK, customers, K)
    _assert_no_dups(df, "product_code")


def test_product_code_max_k(fold_data):
    from src.candidates.product_code import generate
    history, _, customers = fold_data
    df = generate(history, FOLD_WEEK, customers, K)
    _assert_max_k(df, K, "product_code")


def test_product_code_deterministic(fold_data):
    from src.candidates.product_code import generate
    history, _, customers = fold_data
    df1 = generate(history, FOLD_WEEK, customers, K)
    df2 = generate(history, FOLD_WEEK, customers, K)
    _assert_deterministic(df1, df2, "product_code")


# ---------------------------------------------------------------------------
# Popularity: global_last_week
# ---------------------------------------------------------------------------

def test_popularity_last_week_schema(fold_data):
    from src.candidates.popularity import generate_global_last_week
    history, _, customers = fold_data
    df = generate_global_last_week(history, FOLD_WEEK, customers, K)
    _assert_schema(df, "popularity_last_week")


def test_popularity_last_week_no_dups(fold_data):
    from src.candidates.popularity import generate_global_last_week
    history, _, customers = fold_data
    df = generate_global_last_week(history, FOLD_WEEK, customers, K)
    _assert_no_dups(df, "popularity_last_week")


def test_popularity_last_week_max_k(fold_data):
    from src.candidates.popularity import generate_global_last_week
    history, _, customers = fold_data
    df = generate_global_last_week(history, FOLD_WEEK, customers, K)
    _assert_max_k(df, K, "popularity_last_week")


def test_popularity_last_week_deterministic(fold_data):
    from src.candidates.popularity import generate_global_last_week
    history, _, customers = fold_data
    df1 = generate_global_last_week(history, FOLD_WEEK, customers, K)
    df2 = generate_global_last_week(history, FOLD_WEEK, customers, K)
    _assert_deterministic(df1, df2, "popularity_last_week")


# ---------------------------------------------------------------------------
# Popularity: global_decayed
# ---------------------------------------------------------------------------

def test_popularity_decayed_schema(fold_data):
    from src.candidates.popularity import generate_global_decayed
    history, _, customers = fold_data
    df = generate_global_decayed(history, FOLD_WEEK, customers, K)
    _assert_schema(df, "popularity_decayed")


def test_popularity_decayed_no_dups(fold_data):
    from src.candidates.popularity import generate_global_decayed
    history, _, customers = fold_data
    df = generate_global_decayed(history, FOLD_WEEK, customers, K)
    _assert_no_dups(df, "popularity_decayed")


def test_popularity_decayed_max_k(fold_data):
    from src.candidates.popularity import generate_global_decayed
    history, _, customers = fold_data
    df = generate_global_decayed(history, FOLD_WEEK, customers, K)
    _assert_max_k(df, K, "popularity_decayed")


def test_popularity_decayed_deterministic(fold_data):
    from src.candidates.popularity import generate_global_decayed
    history, _, customers = fold_data
    df1 = generate_global_decayed(history, FOLD_WEEK, customers, K)
    df2 = generate_global_decayed(history, FOLD_WEEK, customers, K)
    _assert_deterministic(df1, df2, "popularity_decayed")


# ---------------------------------------------------------------------------
# Popularity: segment_popular
# ---------------------------------------------------------------------------

def test_segment_popular_schema(fold_data):
    from src.candidates.popularity import generate_segment_popular
    history, _, customers = fold_data
    df = generate_segment_popular(history, FOLD_WEEK, customers, K)
    _assert_schema(df, "segment_popular")


def test_segment_popular_no_dups(fold_data):
    from src.candidates.popularity import generate_segment_popular
    history, _, customers = fold_data
    df = generate_segment_popular(history, FOLD_WEEK, customers, K)
    _assert_no_dups(df, "segment_popular")


def test_segment_popular_max_k(fold_data):
    from src.candidates.popularity import generate_segment_popular
    history, _, customers = fold_data
    df = generate_segment_popular(history, FOLD_WEEK, customers, K)
    _assert_max_k(df, K, "segment_popular")


# ---------------------------------------------------------------------------
# Repurchase: recency_only ordering
# ---------------------------------------------------------------------------

def test_repurchase_recency_only_ordering():
    """recency_only mode: ordering is last_date DESC, article_idx ASC (no purchase_count key).

    Synthetic example: two customers each with two articles.
    Customer 10: article 100 bought on day 100, article 200 bought on day 50.
      → recency_only order: 100, 200
    Customer 20: article 300 bought on day 80 (once), article 400 bought on day 80 (3 times).
      → tie on last_date=80; recency_only tie-break: article_idx ASC → 300, 400
      → production tie-break: purchase_count DESC (400 first), then article_idx ASC → 400, 300
    """
    import datetime
    from src.candidates.repurchase import generate

    anchor = datetime.date(2018, 9, 19)
    def to_epoch(d: datetime.date) -> int:
        return (d - datetime.date(1970, 1, 1)).days

    # Build a minimal transactions LazyFrame
    # week_idx doesn't matter as long as it's < cutoff_week=10
    rows = [
        # customer 10
        (10, 100, to_epoch(anchor + datetime.timedelta(days=100)), 0),
        (10, 200, to_epoch(anchor + datetime.timedelta(days=50)),  0),
        # customer 20: article 400 bought 3 times on day 80 (same date → purchase_count=3 after agg)
        (20, 300, to_epoch(anchor + datetime.timedelta(days=80)),  0),
        (20, 400, to_epoch(anchor + datetime.timedelta(days=80)),  0),
        (20, 400, to_epoch(anchor + datetime.timedelta(days=80)),  0),
        (20, 400, to_epoch(anchor + datetime.timedelta(days=80)),  0),
    ]
    history = pl.DataFrame({
        "customer_idx": pl.Series([r[0] for r in rows], dtype=pl.Int32),
        "article_idx":  pl.Series([r[1] for r in rows], dtype=pl.Int32),
        "t_dat":        pl.Series([r[2] for r in rows], dtype=pl.Int32).cast(pl.Date),
        "week_idx":     pl.Series([r[3] for r in rows], dtype=pl.Int32),
    }).lazy()

    cutoff_week = 10
    customers = [10, 20]

    df_recency = generate(history, cutoff_week, customers, k=2, ordering="recency_only")
    df_prod    = generate(history, cutoff_week, customers, k=2, ordering="production")

    def top_articles(df: pl.DataFrame, cust: int) -> list[int]:
        return (
            df.filter(pl.col("customer_idx") == cust)
            .sort("source_rank")
            ["article_idx"]
            .to_list()
        )

    # recency_only: customer 10 → [100, 200] (day 100 > day 50)
    assert top_articles(df_recency, 10) == [100, 200], (
        f"recency_only cust 10: expected [100,200], got {top_articles(df_recency, 10)}"
    )
    # recency_only: customer 20 → [300, 400] (same date, article_idx ASC)
    assert top_articles(df_recency, 20) == [300, 400], (
        f"recency_only cust 20: expected [300,400], got {top_articles(df_recency, 20)}"
    )
    # production: customer 20 → [400, 300] (same date, purchase_count DESC → 400 first)
    assert top_articles(df_prod, 20) == [400, 300], (
        f"production cust 20: expected [400,300], got {top_articles(df_prod, 20)}"
    )


# ---------------------------------------------------------------------------
# Copurchase
# ---------------------------------------------------------------------------

def test_copurchase_schema(fold_data):
    from src.candidates.copurchase import generate
    history, _, customers = fold_data
    df = generate(history, FOLD_WEEK, customers, K)
    _assert_schema(df, "copurchase")


def test_copurchase_no_dups(fold_data):
    from src.candidates.copurchase import generate
    history, _, customers = fold_data
    df = generate(history, FOLD_WEEK, customers, K)
    _assert_no_dups(df, "copurchase")


def test_copurchase_max_k(fold_data):
    from src.candidates.copurchase import generate
    history, _, customers = fold_data
    df = generate(history, FOLD_WEEK, customers, K)
    _assert_max_k(df, K, "copurchase")


def test_copurchase_deterministic(fold_data):
    from src.candidates.copurchase import generate
    history, _, customers = fold_data
    df1 = generate(history, FOLD_WEEK, customers, K)
    df2 = generate(history, FOLD_WEEK, customers, K)
    _assert_deterministic(df1, df2, "copurchase")
