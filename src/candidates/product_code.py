"""Product-code candidate source: articles sharing a product_code with recent purchases."""
import polars as pl

from src.candidates.base import OUTPUT_DTYPES, validate_candidates
from src.data_io import load_articles


def generate(
    history: pl.LazyFrame,
    cutoff_week: int,
    customers: list[int],
    k: int,
    lookback_weeks: int = 12,
) -> pl.DataFrame:
    """Return up to k same-product-code articles not yet bought, scored by last-week sales."""
    # Step 1: customer's recent purchased (customer_idx, article_idx) pairs
    base_lf = history.filter(pl.col("week_idx") < cutoff_week)
    if len(customers) > 50000:
        customer_series = pl.Series("customer_idx", customers, dtype=pl.Int32)
        base_lf = base_lf.filter(
            pl.col("customer_idx").is_in(customer_series.to_list())
        )
    else:
        base_lf = base_lf.filter(pl.col("customer_idx").is_in(customers))

    # All-history purchases for exclusion (step 4)
    all_history_pairs = (
        base_lf
        .select(["customer_idx", "article_idx"])
        .unique(subset=["customer_idx", "article_idx"])
    )

    # Recent purchases for product-code lookup
    recent_lf = base_lf.filter(
        pl.col("week_idx") >= cutoff_week - lookback_weeks
    )
    recent_pairs = (
        recent_lf
        .select(["customer_idx", "article_idx"])
        .unique(subset=["customer_idx", "article_idx"])
    )

    # Step 2: join with articles to get product_code for each purchased article
    articles_lf = load_articles().select(["article_idx", "product_code"])
    recent_with_code = recent_pairs.join(
        articles_lf, on="article_idx", how="left"
    )

    # Step 3: all articles sharing same product_code
    # Join product_codes back to articles to expand to all sibling articles
    customer_product_codes = (
        recent_with_code
        .select(["customer_idx", "product_code"])
        .unique(subset=["customer_idx", "product_code"])
    )
    candidate_articles = customer_product_codes.join(
        articles_lf, on="product_code", how="left"
    ).select(["customer_idx", "article_idx"])

    # Step 4: exclude articles the customer bought in ALL of history (anti-join)
    all_history_collected = all_history_pairs.collect()
    candidate_collected = candidate_articles.collect()

    candidates = candidate_collected.join(
        all_history_collected,
        on=["customer_idx", "article_idx"],
        how="anti",
    )

    # Step 5: score = article sales count in week (cutoff_week - 1)
    prev_week_sales = (
        history
        .filter(pl.col("week_idx") == cutoff_week - 1)
        .group_by("article_idx")
        .agg(pl.len().cast(pl.Int32).alias("sales_count"))
        .collect()
    )

    # Left join to get score; fill missing with 0
    candidates = candidates.join(
        prev_week_sales, on="article_idx", how="left"
    ).with_columns(
        pl.col("sales_count").fill_null(0).cast(pl.Float32).alias("score")
    )

    # Drop duplicate (customer_idx, article_idx) rows (a customer may share multiple
    # product_codes with the same sibling article — keep row with highest score)
    candidates = (
        candidates
        .sort(["customer_idx", "score", "article_idx"], descending=[False, True, False])
        .unique(subset=["customer_idx", "article_idx"], keep="first", maintain_order=True)
    )

    # Step 6: sort, rank, filter
    candidates = candidates.sort(
        ["customer_idx", "score", "article_idx"],
        descending=[False, True, False],
    )
    candidates = candidates.with_columns(
        pl.cum_count("article_idx")
        .over("customer_idx")
        .cast(pl.Int16)
        .alias("source_rank")
    )
    candidates = candidates.filter(pl.col("source_rank") <= k)

    # Step 7: add source and cast to output dtypes
    result = (
        candidates
        .with_columns(pl.lit("product_code").cast(pl.Utf8).alias("source"))
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

    validate_candidates(result, k, "product_code")
    return result
