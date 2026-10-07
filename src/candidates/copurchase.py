"""Copurchase candidate source: articles frequently bought together with a customer's history."""
import duckdb
import polars as pl

from src.candidates.base import OUTPUT_DTYPES, validate_candidates


def generate(
    history: pl.LazyFrame,
    cutoff_week: int,
    customers: list[int],
    k: int,
    lookback_weeks: int = 8,
    customer_lookback_weeks: int = 4,
    min_count: int = 5,
) -> pl.DataFrame:
    """Return up to k co-purchased article candidates per customer using basket cosine similarity."""
    # --- Build basket data for the co-purchase matrix ---
    # Baskets: transactions in [cutoff - lookback_weeks, cutoff) from all customers
    basket_lf = (
        history
        .filter(
            (pl.col("week_idx") >= cutoff_week - lookback_weeks)
            & (pl.col("week_idx") < cutoff_week)
        )
        .select(["customer_idx", "t_dat", "article_idx"])
    )

    # Filter to requested customers for the customer-side lookup
    if len(customers) > 50000:
        customer_series = pl.Series("customer_idx", customers, dtype=pl.Int32)
        customer_filter_list = customer_series.to_list()
    else:
        customer_filter_list = customers

    # Customer recent purchases (last customer_lookback_weeks before cutoff)
    customer_recent_lf = (
        history
        .filter(
            (pl.col("week_idx") >= cutoff_week - customer_lookback_weeks)
            & (pl.col("week_idx") < cutoff_week)
        )
        .filter(pl.col("customer_idx").is_in(customer_filter_list))
        .select(["customer_idx", "article_idx"])
        .unique(subset=["customer_idx", "article_idx"])
    )

    # All-history purchases for exclusion (week < cutoff, filtered customers)
    all_history_lf = (
        history
        .filter(pl.col("week_idx") < cutoff_week)
        .filter(pl.col("customer_idx").is_in(customer_filter_list))
        .select(["customer_idx", "article_idx"])
        .unique(subset=["customer_idx", "article_idx"])
    )

    # Collect to Arrow for DuckDB registration
    # Pre-convert t_dat to Int32 epoch days — DuckDB cannot cast DATE -> BIGINT
    baskets_df = (
        basket_lf
        .with_columns(pl.col("t_dat").cast(pl.Int32).alias("t_dat_int"))
        .drop("t_dat")
        .collect()
    )
    customer_recent_df = customer_recent_lf.collect()
    all_history_df = all_history_lf.collect()

    if baskets_df.is_empty() or customer_recent_df.is_empty():
        return pl.DataFrame(schema={
            "customer_idx": pl.Int32,
            "article_idx": pl.Int32,
            "source": pl.Utf8,
            "score": pl.Float32,
            "source_rank": pl.Int16,
        })

    conn = duckdb.connect()
    conn.register("baskets", baskets_df.to_arrow())
    conn.register("customer_recent", customer_recent_df.to_arrow())
    conn.register("all_history", all_history_df.to_arrow())

    # Build co-purchase matrix via basket self-join
    # pair_count = number of distinct (customer_idx, t_dat) baskets containing both a and b
    # item_count_a / item_count_b = distinct baskets containing just a or b
    copurchase_sql = """
    WITH basket_pairs AS (
        SELECT
            b1.article_idx AS article_a,
            b2.article_idx AS article_b,
            COUNT(DISTINCT (CAST(b1.customer_idx AS BIGINT) * 1000000 + CAST(b1.t_dat_int AS BIGINT))) AS pair_count
        FROM baskets b1
        JOIN baskets b2
            ON b1.customer_idx = b2.customer_idx
            AND b1.t_dat_int = b2.t_dat_int
            AND b1.article_idx < b2.article_idx
        GROUP BY b1.article_idx, b2.article_idx
        HAVING COUNT(DISTINCT (CAST(b1.customer_idx AS BIGINT) * 1000000 + CAST(b1.t_dat_int AS BIGINT))) >= {min_count}
    ),
    item_counts AS (
        SELECT
            article_idx,
            COUNT(DISTINCT (CAST(customer_idx AS BIGINT) * 1000000 + CAST(t_dat_int AS BIGINT))) AS basket_count
        FROM baskets
        GROUP BY article_idx
    ),
    cosine_matrix AS (
        SELECT
            bp.article_a,
            bp.article_b,
            bp.pair_count,
            CAST(bp.pair_count AS DOUBLE) / SQRT(CAST(ia.basket_count AS DOUBLE) * CAST(ib.basket_count AS DOUBLE)) AS cosine_score
        FROM basket_pairs bp
        JOIN item_counts ia ON bp.article_a = ia.article_idx
        JOIN item_counts ib ON bp.article_b = ib.article_idx
    ),
    -- Symmetrize: for each pair (a, b), also create (b, a)
    cosine_symmetric AS (
        SELECT article_a AS seed_article, article_b AS candidate_article, cosine_score FROM cosine_matrix
        UNION ALL
        SELECT article_b AS seed_article, article_a AS candidate_article, cosine_score FROM cosine_matrix
    ),
    -- For each customer, sum cosine scores over their recently purchased seed articles
    customer_scores AS (
        SELECT
            cr.customer_idx,
            cs.candidate_article AS article_idx,
            SUM(cs.cosine_score) AS score
        FROM customer_recent cr
        JOIN cosine_symmetric cs ON cr.article_idx = cs.seed_article
        GROUP BY cr.customer_idx, cs.candidate_article
    ),
    -- Exclude articles the customer has bought in all history
    filtered AS (
        SELECT
            csc.customer_idx,
            csc.article_idx,
            csc.score
        FROM customer_scores csc
        LEFT JOIN all_history ah
            ON csc.customer_idx = ah.customer_idx
            AND csc.article_idx = ah.article_idx
        WHERE ah.article_idx IS NULL
    )
    SELECT
        CAST(customer_idx AS INTEGER) AS customer_idx,
        CAST(article_idx AS INTEGER) AS article_idx,
        CAST(score AS FLOAT) AS score
    FROM filtered
    ORDER BY customer_idx ASC, score DESC, article_idx ASC
    """.format(min_count=min_count)

    copurchase_result = conn.execute(copurchase_sql).pl()
    conn.close()

    if copurchase_result.is_empty():
        return pl.DataFrame(schema={
            "customer_idx": pl.Int32,
            "article_idx": pl.Int32,
            "source": pl.Utf8,
            "score": pl.Float32,
            "source_rank": pl.Int16,
        })

    # Assign source_rank within each customer
    result = copurchase_result.with_columns(
        pl.cum_count("article_idx")
        .over("customer_idx")
        .cast(pl.Int16)
        .alias("source_rank")
    ).filter(pl.col("source_rank") <= k)

    result = (
        result
        .with_columns(pl.lit("copurchase").cast(pl.Utf8).alias("source"))
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

    validate_candidates(result, k, "copurchase")
    return result
