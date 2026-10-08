"""Precomputed customer-independent state for fast single-customer inference."""
from __future__ import annotations

import resource
import time
from dataclasses import dataclass, field

import duckdb
import polars as pl

from src.config import (
    ANCHOR_EPOCH_DAYS,
    COPURCHASE_LOOKBACK,
    MIN_COPURCHASE_COUNT,
    SOURCE_K,
)


@dataclass
class RecommenderState:
    """All customer-independent precomputed data for a given cutoff week.

    Built once via build_state() and reused across all single-customer calls.
    The batch feature build path uses the same state so batch and single-customer
    results are guaranteed identical.
    """

    as_of_week: int
    history_df: pl.DataFrame          # tx filtered to week_idx < as_of_week
    article_feats: pl.DataFrame        # per-article features (all articles in history)
    customers_df: pl.DataFrame         # ALL customers table (collected)
    articles_lf: pl.LazyFrame          # articles LazyFrame (for interaction features)
    all_cust_buckets: pl.DataFrame    # (customer_idx, age_bucket) for ALL 1.37M customers
    bucket_totals: pl.DataFrame        # (age_bucket, _bucket_total) for last 2 weeks
    pop_last_week_top: pl.DataFrame   # (article_idx, score, source_rank) top-k articles
    pop_decayed_top: pl.DataFrame     # (article_idx, score, source_rank) top-k articles
    seg_pop_by_bucket: dict           # str → DataFrame(article_idx, score, source_rank)
    copurchase_symmetric: pl.DataFrame  # (seed_article, candidate_article, cosine_score)
    prev_week_sales: pl.DataFrame      # (article_idx, sales_count)
    article_product_codes: pl.DataFrame  # (article_idx, product_code)
    total_tx_last_week: float          # denominator for popularity_last_week_share
    total_decayed_tx: float            # denominator for popularity_decayed_share
    build_time_s: float = 0.0
    rss_mb: float = 0.0


def _age_bucket_expr() -> pl.Expr:
    """Age bucket expression matching get_customer_age_buckets() in data_io.py."""
    return (
        pl.when(pl.col("age").is_null()).then(pl.lit("missing"))
        .when(pl.col("age") < 25).then(pl.lit("<25"))
        .when(pl.col("age") < 35).then(pl.lit("25-34"))
        .when(pl.col("age") < 45).then(pl.lit("35-44"))
        .when(pl.col("age") < 55).then(pl.lit("45-54"))
        .otherwise(pl.lit("55+"))
        .alias("age_bucket")
    )


def build_state(
    as_of_week: int,
    history_df: pl.DataFrame,
    articles_lf: pl.LazyFrame,
    customers_lf: pl.LazyFrame,
) -> RecommenderState:
    """Build all customer-independent precomputed state for the given cutoff week.

    Parameters
    ----------
    as_of_week:
        Cutoff week index (exclusive). All precomputed data uses week_idx < as_of_week.
    history_df:
        Collected transaction DataFrame with week_idx column, pre-filtered to
        week_idx < as_of_week. Must NOT contain future rows.
    articles_lf:
        LazyFrame from load_articles().
    customers_lf:
        LazyFrame from load_customers().
    """
    t0 = time.perf_counter()
    history_lf = history_df.lazy()
    cutoff_epoch: int = ANCHOR_EPOCH_DAYS + as_of_week * 7

    # ── Article features (expensive full-history scan — precomputed once) ──────
    print("  [state] article features ...", flush=True)
    from src.features.article import compute_article_features
    article_feats = compute_article_features(
        history=history_lf,
        cutoff_week=as_of_week,
        articles_df=articles_lf,
        customers_df=customers_lf,
        anchor_epoch_days=ANCHOR_EPOCH_DAYS,
    )

    # ── Customer age buckets for all 1.37M customers ──────────────────────────
    print("  [state] customer buckets ...", flush=True)
    customers_df = customers_lf.collect()
    all_cust_buckets = (
        customers_df
        .select(["customer_idx", "age"])
        .with_columns(_age_bucket_expr())
        .drop("age")
    )

    # ── Bucket totals for segment_popular_share (last 2 weeks, all customers) ─
    bucket_totals = (
        history_lf
        .filter((pl.col("week_idx") >= as_of_week - 2) & (pl.col("week_idx") < as_of_week))
        .join(all_cust_buckets.lazy().select(["customer_idx", "age_bucket"]),
              on="customer_idx", how="inner")
        .group_by("age_bucket")
        .agg(pl.len().cast(pl.Float32).alias("_bucket_total"))
        .collect()
    )

    # ── Popularity last-week top-k (no customer cross-join yet) ───────────────
    print("  [state] popularity last-week ...", flush=True)
    pop_last_week_top = (
        history_lf
        .filter(pl.col("week_idx") == as_of_week - 1)
        .group_by("article_idx")
        .agg(pl.len().cast(pl.Float32).alias("score"))
        .sort(["score", "article_idx"], descending=[True, False])
        .head(SOURCE_K["popularity_last_week"])
        .collect()
        .with_columns(
            (pl.int_range(pl.len(), dtype=pl.Int16) + pl.lit(1, dtype=pl.Int16))
            .alias("source_rank")
        )
    )

    # ── Popularity decayed top-k ───────────────────────────────────────────────
    print("  [state] popularity decayed ...", flush=True)
    pop_decayed_top = (
        history_lf
        .filter((pl.col("week_idx") >= as_of_week - 4) & (pl.col("week_idx") < as_of_week))
        .with_columns(
            (
                1.0 / (1.0 + (pl.lit(cutoff_epoch) - pl.col("t_dat").cast(pl.Int32)).cast(pl.Float64))
            ).alias("decay_weight")
        )
        .group_by("article_idx")
        .agg(pl.col("decay_weight").sum().alias("score"))
        .sort(["score", "article_idx"], descending=[True, False])
        .head(SOURCE_K["popularity_decayed"])
        .collect()
        .with_columns(
            (pl.int_range(pl.len(), dtype=pl.Int16) + pl.lit(1, dtype=pl.Int16))
            .alias("source_rank")
        )
    )

    # ── Segment popular top-k per age bucket ──────────────────────────────────
    print("  [state] segment popular ...", flush=True)
    k_seg = SOURCE_K["segment_popular"]
    seg_pop_raw = (
        history_lf
        .filter((pl.col("week_idx") >= as_of_week - 2) & (pl.col("week_idx") < as_of_week))
        .join(all_cust_buckets.lazy().select(["customer_idx", "age_bucket"]),
              on="customer_idx", how="inner")
        .group_by(["age_bucket", "article_idx"])
        .agg(pl.len().cast(pl.Float32).alias("score"))
        .sort(["age_bucket", "score", "article_idx"], descending=[False, True, False])
        .collect()
    )
    seg_pop_raw = (
        seg_pop_raw
        .with_columns(
            pl.cum_count("article_idx").over("age_bucket").cast(pl.Int16).alias("source_rank")
        )
        .filter(pl.col("source_rank") <= k_seg)
    )
    seg_pop_by_bucket: dict[str, pl.DataFrame] = {}
    for bucket in seg_pop_raw["age_bucket"].unique().to_list():
        seg_pop_by_bucket[bucket] = (
            seg_pop_raw
            .filter(pl.col("age_bucket") == bucket)
            .select(["article_idx", "score", "source_rank"])
        )

    # ── Copurchase symmetric matrix (DuckDB pair-matrix, precomputed once) ────
    print("  [state] copurchase symmetric matrix ...", flush=True)
    copurchase_symmetric = _build_copurchase_symmetric(history_df, as_of_week)
    print(f"  [state] copurchase: {len(copurchase_symmetric):,} pairs", flush=True)

    # ── Previous-week sales (for product_code scoring) ────────────────────────
    prev_week_sales = (
        history_lf
        .filter(pl.col("week_idx") == as_of_week - 1)
        .group_by("article_idx")
        .agg(pl.len().cast(pl.Int32).alias("sales_count"))
        .collect()
    )

    # ── Article product codes (static, from articles table) ───────────────────
    article_product_codes = articles_lf.select(["article_idx", "product_code"]).collect()

    # ── Scalar denominators for share features ────────────────────────────────
    total_tx_last_week = float(max(
        history_df.filter(pl.col("week_idx") == as_of_week - 1).height, 1
    ))
    total_decayed_tx = float(max(
        history_lf
        .filter((pl.col("week_idx") >= as_of_week - 4) & (pl.col("week_idx") < as_of_week))
        .with_columns(
            (1.0 / (1.0 + (pl.lit(cutoff_epoch) - pl.col("t_dat").cast(pl.Int32)).cast(pl.Float64)))
            .alias("_dw")
        )
        .select(pl.col("_dw").sum())
        .collect()
        .item(),
        1e-6,
    ))

    build_time = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"  [state] build complete in {build_time:.1f}s, RSS={rss:.0f}MB", flush=True)

    return RecommenderState(
        as_of_week=as_of_week,
        history_df=history_df,
        article_feats=article_feats,
        customers_df=customers_df,
        articles_lf=articles_lf,
        all_cust_buckets=all_cust_buckets,
        bucket_totals=bucket_totals,
        pop_last_week_top=pop_last_week_top,
        pop_decayed_top=pop_decayed_top,
        seg_pop_by_bucket=seg_pop_by_bucket,
        copurchase_symmetric=copurchase_symmetric,
        prev_week_sales=prev_week_sales,
        article_product_codes=article_product_codes,
        total_tx_last_week=total_tx_last_week,
        total_decayed_tx=total_decayed_tx,
        build_time_s=build_time,
        rss_mb=rss,
    )


def _build_copurchase_symmetric(
    history_df: pl.DataFrame,
    as_of_week: int,
    lookback_weeks: int = COPURCHASE_LOOKBACK,
    min_count: int = MIN_COPURCHASE_COUNT,
) -> pl.DataFrame:
    """Build symmetric article co-purchase cosine matrix using DuckDB.

    Equivalent to the first four CTEs of copurchase.generate() (up to cosine_symmetric).
    Customer-specific scoring is done per-call in _copurchase_from_state().
    The result is sorted by (seed_article, candidate_article) for deterministic order.
    """
    basket_df = (
        history_df
        .filter(
            (pl.col("week_idx") >= as_of_week - lookback_weeks)
            & (pl.col("week_idx") < as_of_week)
        )
        .with_columns(pl.col("t_dat").cast(pl.Int32).alias("t_dat_int"))
        .select(["customer_idx", "article_idx", "t_dat_int"])
    )

    if basket_df.is_empty():
        return pl.DataFrame(schema={
            "seed_article": pl.Int32,
            "candidate_article": pl.Int32,
            "cosine_score": pl.Float64,
        })

    conn = duckdb.connect()
    conn.register("baskets", basket_df.to_arrow())

    sql = f"""
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
            CAST(bp.pair_count AS DOUBLE) / SQRT(CAST(ia.basket_count AS DOUBLE) * CAST(ib.basket_count AS DOUBLE)) AS cosine_score
        FROM basket_pairs bp
        JOIN item_counts ia ON bp.article_a = ia.article_idx
        JOIN item_counts ib ON bp.article_b = ib.article_idx
    )
    SELECT
        CAST(article_a AS INTEGER) AS seed_article,
        CAST(article_b AS INTEGER) AS candidate_article,
        cosine_score
    FROM cosine_matrix
    UNION ALL
    SELECT
        CAST(article_b AS INTEGER) AS seed_article,
        CAST(article_a AS INTEGER) AS candidate_article,
        cosine_score
    FROM cosine_matrix
    ORDER BY seed_article, candidate_article
    """

    result_df = conn.execute(sql).pl()
    conn.close()
    if result_df.is_empty():
        return pl.DataFrame(schema={
            "seed_article": pl.Int32,
            "candidate_article": pl.Int32,
            "cosine_score": pl.Float32,
        })
    return result_df
