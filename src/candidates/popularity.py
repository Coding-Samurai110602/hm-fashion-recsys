"""Popularity candidate sources: global last-week, time-decayed, and age-segment."""
import polars as pl

from src.candidates.base import OUTPUT_DTYPES, validate_candidates
from src.config import ANCHOR_EPOCH_DAYS
from src.data_io import get_customer_age_buckets


def generate_global_last_week(
    history: pl.LazyFrame,
    cutoff_week: int,
    customers: list[int],
    k: int,
) -> pl.DataFrame:
    """Return top-k articles by sales count in the week before cutoff, for all customers."""
    # Compute top-k article list
    top_k = (
        history
        .filter(pl.col("week_idx") == cutoff_week - 1)
        .group_by("article_idx")
        .agg(pl.len().cast(pl.Float32).alias("score"))
        .sort(["score", "article_idx"], descending=[True, False])
        .head(k)
        .collect()
    )

    # Add sequential rank (already sorted)
    top_k = top_k.with_columns(
        (pl.int_range(pl.len(), dtype=pl.Int16) + pl.lit(1, dtype=pl.Int16))
        .alias("source_rank")
    )

    # Cross with all customers
    customers_df = pl.DataFrame(
        {"customer_idx": pl.Series(customers, dtype=pl.Int32)}
    )
    result = customers_df.join(top_k, how="cross")

    result = (
        result
        .with_columns(pl.lit("popularity_last_week").cast(pl.Utf8).alias("source"))
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

    validate_candidates(result, k, "popularity_last_week")
    return result


def generate_global_decayed(
    history: pl.LazyFrame,
    cutoff_week: int,
    customers: list[int],
    k: int,
    halflife_days: float = 7.0,
) -> pl.DataFrame:
    """Return top-k articles by inverse-time-decayed sales over last 4 weeks.

    Decay weight = 1 / (1 + days_before_cutoff). Inverse-time decay outperformed
    exponential decay (7-day half-life) by ~4.6% in recall@100 across folds
    100-103 (inverse-time: 0.0952 vs exp: 0.0910). Both were evaluated; inverse-time retained.
    """
    # cutoff_date as epoch days
    cutoff_epoch: int = ANCHOR_EPOCH_DAYS + cutoff_week * 7

    top_k = (
        history
        .filter(
            (pl.col("week_idx") >= cutoff_week - 4)
            & (pl.col("week_idx") < cutoff_week)
        )
        .with_columns(
            (
                1.0 / (
                    1.0 + (pl.lit(cutoff_epoch) - pl.col("t_dat").cast(pl.Int32))
                    .cast(pl.Float64)
                )
            ).alias("decay_weight")
        )
        .group_by("article_idx")
        .agg(pl.col("decay_weight").sum().alias("decayed_score"))
        .sort(["decayed_score", "article_idx"], descending=[True, False])
        .head(k)
        .collect()
    )

    # Add sequential rank
    top_k = top_k.with_columns(
        (pl.int_range(pl.len(), dtype=pl.Int16) + pl.lit(1, dtype=pl.Int16))
        .alias("source_rank")
    ).rename({"decayed_score": "score"})

    # Cross with all customers
    customers_df = pl.DataFrame(
        {"customer_idx": pl.Series(customers, dtype=pl.Int32)}
    )
    result = customers_df.join(top_k, how="cross")

    result = (
        result
        .with_columns(pl.lit("popularity_decayed").cast(pl.Utf8).alias("source"))
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

    validate_candidates(result, k, "popularity_decayed")
    return result


def generate_segment_popular(
    history: pl.LazyFrame,
    cutoff_week: int,
    customers: list[int],
    k: int,
    customers_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return top-k articles per age-bucket, assigned to each customer by their bucket."""
    if customers_df is None:
        customers_df = get_customer_age_buckets()

    # Age bucket for ASSIGNMENT: which eval_customers to give recommendations to
    cust_buckets = customers_df.filter(
        pl.col("customer_idx").is_in(customers)
    )

    # Build age-bucket -> top-k articles from transactions in last 2 weeks.
    # Popularity is computed using ALL customers (not filtered to eval_customers)
    # so that scores are globally consistent across batch and single-customer calls.
    bucket_top_k = (
        history
        .filter(
            (pl.col("week_idx") >= cutoff_week - 2)
            & (pl.col("week_idx") < cutoff_week)
        )
        .join(
            customers_df.lazy().select(["customer_idx", "age_bucket"]),
            on="customer_idx",
            how="inner",
        )
        .group_by(["age_bucket", "article_idx"])
        .agg(pl.len().cast(pl.Float32).alias("score"))
        .sort(["age_bucket", "score", "article_idx"], descending=[False, True, False])
        .collect()
    )

    # Keep top-k per age_bucket
    bucket_top_k = (
        bucket_top_k
        .with_columns(
            pl.cum_count("article_idx")
            .over("age_bucket")
            .cast(pl.Int16)
            .alias("source_rank")
        )
        .filter(pl.col("source_rank") <= k)
    )

    # Join customers with their bucket's top-k articles
    result = (
        cust_buckets
        .join(bucket_top_k, on="age_bucket", how="inner")
        .select(["customer_idx", "article_idx", "score", "source_rank"])
    )

    # A customer may appear in multiple buckets if customers_df has duplicates —
    # deduplicate keeping highest score to be safe
    result = (
        result
        .sort(["customer_idx", "score", "article_idx"], descending=[False, True, False])
        .unique(subset=["customer_idx", "article_idx"], keep="first", maintain_order=True)
    )

    # Re-rank after dedup (rank may shift if article appeared multiple times)
    result = result.sort(
        ["customer_idx", "score", "article_idx"], descending=[False, True, False]
    ).with_columns(
        pl.cum_count("article_idx")
        .over("customer_idx")
        .cast(pl.Int16)
        .alias("source_rank")
    ).filter(pl.col("source_rank") <= k)

    result = (
        result
        .with_columns(pl.lit("segment_popular").cast(pl.Utf8).alias("source"))
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

    validate_candidates(result, k, "segment_popular")
    return result
