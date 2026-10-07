"""Centralized data loading with week_idx computed once."""
import polars as pl
from src.config import PROCESSED_DIR, ANCHOR_EPOCH_DAYS


def load_transactions() -> pl.LazyFrame:
    """Scan transactions parquet; add week_idx = (t_dat - anchor).days // 7."""
    return (
        pl.scan_parquet(PROCESSED_DIR / "transactions_train.parquet")
        .with_columns(
            ((pl.col("t_dat").cast(pl.Int32) - ANCHOR_EPOCH_DAYS) // 7)
            .cast(pl.Int16)
            .alias("week_idx")
        )
    )


def load_articles() -> pl.LazyFrame:
    """Scan articles parquet (includes article_idx and product_code string)."""
    return pl.scan_parquet(PROCESSED_DIR / "articles.parquet")


def load_customers() -> pl.LazyFrame:
    """Scan customers parquet (includes customer_idx and age float)."""
    return pl.scan_parquet(PROCESSED_DIR / "customers.parquet")


def get_customer_age_buckets() -> pl.DataFrame:
    """Return customer_idx -> age_bucket mapping for segment_popular source."""
    return (
        load_customers()
        .select(["customer_idx", "age"])
        .with_columns(
            pl.when(pl.col("age").is_null())
            .then(pl.lit("missing"))
            .when(pl.col("age") < 25)
            .then(pl.lit("<25"))
            .when(pl.col("age") < 35)
            .then(pl.lit("25-34"))
            .when(pl.col("age") < 45)
            .then(pl.lit("35-44"))
            .when(pl.col("age") < 55)
            .then(pl.lit("45-54"))
            .otherwise(pl.lit("55+"))
            .alias("age_bucket")
        )
        .drop("age")
        .collect()
    )


def get_article_product_codes() -> pl.DataFrame:
    """Return article_idx -> product_code mapping."""
    return (
        load_articles()
        .select(["article_idx", "product_code"])
        .collect()
    )
