"""Repurchase recency-only ordering: synthetic test, always runs in CI."""

import datetime

import polars as pl
import pytest

from src.candidates.repurchase import generate


def test_repurchase_recency_only_ordering():
    """recency_only mode: ordering is last_date DESC, article_idx ASC (no purchase_count key).

    Synthetic example: two customers each with two articles.
    Customer 10: article 100 bought on day 100, article 200 bought on day 50.
      → recency_only order: 100, 200
    Customer 20: article 300 bought on day 80 (once), article 400 bought on day 80 (3 times).
      → tie on last_date=80; recency_only tie-break: article_idx ASC → 300, 400
      → production tie-break: purchase_count DESC (400 first), then article_idx ASC → 400, 300
    """
    anchor = datetime.date(2018, 9, 19)

    def to_epoch(d: datetime.date) -> int:
        return (d - datetime.date(1970, 1, 1)).days

    rows = [
        (10, 100, to_epoch(anchor + datetime.timedelta(days=100)), 0),
        (10, 200, to_epoch(anchor + datetime.timedelta(days=50)), 0),
        (20, 300, to_epoch(anchor + datetime.timedelta(days=80)), 0),
        (20, 400, to_epoch(anchor + datetime.timedelta(days=80)), 0),
        (20, 400, to_epoch(anchor + datetime.timedelta(days=80)), 0),
        (20, 400, to_epoch(anchor + datetime.timedelta(days=80)), 0),
    ]
    history = pl.DataFrame(
        {
            "customer_idx": pl.Series([r[0] for r in rows], dtype=pl.Int32),
            "article_idx": pl.Series([r[1] for r in rows], dtype=pl.Int32),
            "t_dat": pl.Series([r[2] for r in rows], dtype=pl.Int32).cast(pl.Date),
            "week_idx": pl.Series([r[3] for r in rows], dtype=pl.Int32),
        }
    ).lazy()

    cutoff_week = 10
    customers = [10, 20]

    df_recency = generate(history, cutoff_week, customers, k=2, ordering="recency_only")
    df_prod = generate(history, cutoff_week, customers, k=2, ordering="production")

    def top_articles(df: pl.DataFrame, cust: int) -> list[int]:
        return (
            df.filter(pl.col("customer_idx") == cust)
            .sort("source_rank")
            .get_column("article_idx")
            .to_list()
        )

    # Customer 10: article 100 first (more recent)
    assert top_articles(df_recency, 10) == [100, 200], "recency_only: most recent first"
    assert top_articles(df_prod, 10) == [100, 200], "production: same when unambiguous"

    # Customer 20: tied last_date; recency_only tie-break = article_idx ASC
    assert top_articles(df_recency, 20) == [300, 400], "recency_only: article_idx ASC on tie"
    # Production: purchase_count DESC first (400 has 3 vs 300 has 1)
    assert top_articles(df_prod, 20) == [400, 300], "production: purchase_count DESC on tie"
