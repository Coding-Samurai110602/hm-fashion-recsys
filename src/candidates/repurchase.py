"""Repurchase candidate source: articles a customer has bought before the cutoff."""
import polars as pl

from src.candidates.base import OUTPUT_DTYPES, validate_candidates


def generate(
    history: pl.LazyFrame,
    cutoff_week: int,
    customers: list[int],
    k: int,
    lookback_weeks: int | None = None,
    ordering: str = "production",
) -> pl.DataFrame:
    """Return up to k previously purchased articles per customer.

    Parameters
    ----------
    ordering : str
        'production' (default): sort by last_date DESC, purchase_count DESC,
            article_idx ASC. Captures both recency and frequency.
        'recency_only': sort by last_date DESC, article_idx ASC.
            Matches the EDA Baseline B definition exactly (no purchase_count key),
            used for regression testing.
    """
    lf = history.filter(pl.col("week_idx") < cutoff_week)

    # Filter to the requested customer set
    if len(customers) > 50000:
        customer_series = pl.Series("customer_idx", customers, dtype=pl.Int32)
        lf = lf.filter(pl.col("customer_idx").is_in(customer_series.to_list()))
    else:
        lf = lf.filter(pl.col("customer_idx").is_in(customers))

    # Optional lookback window
    if lookback_weeks is not None:
        lf = lf.filter(pl.col("week_idx") >= cutoff_week - lookback_weeks)

    # Aggregate per (customer_idx, article_idx)
    lf = (
        lf.group_by(["customer_idx", "article_idx"])
        .agg(
            pl.col("t_dat").cast(pl.Int32).max().alias("last_date"),
            pl.len().cast(pl.Int32).alias("purchase_count"),
        )
    )

    if ordering == "recency_only":
        # Sort: last_date DESC, article_idx ASC — matches EDA Baseline B exactly
        lf = lf.sort(
            ["customer_idx", "last_date", "article_idx"],
            descending=[False, True, False],
        )
    else:
        # Sort: recency DESC, purchase_count DESC, article_idx ASC (tie-breaker)
        lf = lf.sort(
            ["customer_idx", "last_date", "purchase_count", "article_idx"],
            descending=[False, True, True, False],
        )

    df = lf.collect()

    # Compute source_rank = cumulative count within customer (1-based)
    df = df.with_columns(
        pl.cum_count("article_idx")
        .over("customer_idx")
        .cast(pl.Int16)
        .alias("source_rank")
    )

    # Keep only top-k per customer
    df = df.filter(pl.col("source_rank") <= k)

    # Build final output
    df = (
        df.with_columns(
            pl.col("last_date").cast(pl.Float32).alias("score"),
            pl.lit("repurchase").cast(pl.Utf8).alias("source"),
        )
        .select(
            [
                pl.col("customer_idx").cast(OUTPUT_DTYPES["customer_idx"]),
                pl.col("article_idx").cast(OUTPUT_DTYPES["article_idx"]),
                pl.col("source").cast(OUTPUT_DTYPES["source"]),
                pl.col("score").cast(OUTPUT_DTYPES["score"]),
                pl.col("source_rank").cast(OUTPUT_DTYPES["source_rank"]),
            ]
        )
    )

    validate_candidates(df, k, "repurchase")
    return df
