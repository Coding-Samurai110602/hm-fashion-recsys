"""Interaction feature computation for a single fold."""
import polars as pl


def compute_interaction_features(
    history: pl.LazyFrame,
    candidates: pl.DataFrame,
    cutoff_week: int,
    article_features: pl.DataFrame,
    customer_features: pl.DataFrame,
    articles_df: pl.LazyFrame,
    anchor_epoch_days: int,
) -> pl.DataFrame:
    """Compute per-(customer_idx, article_idx) interaction features.

    Parameters
    ----------
    history:
        LazyFrame of transactions already filtered to week_idx < cutoff_week.
    candidates:
        DataFrame with columns customer_idx (Int32) and article_idx (Int32).
    cutoff_week:
        The fold's cutoff week index.
    article_features:
        Output of compute_article_features; must contain article_idx,
        a_mean_price_4w, a_mean_buyer_age_12w, _article_online_share_4w.
    customer_features:
        Output of compute_customer_features; must contain customer_idx,
        c_mean_price, c_online_share, c_age.
    articles_df:
        LazyFrame from load_articles() for product_code and group lookups.
    anchor_epoch_days:
        ANCHOR_EPOCH_DAYS constant (17793).

    Returns
    -------
    pl.DataFrame
        Keyed by (customer_idx, article_idx) with all i_ columns.
    """
    assert history.filter(pl.col("week_idx") >= cutoff_week).collect().height == 0, \
        f"history contains rows with week_idx >= {cutoff_week}"

    cutoff_epoch: int = anchor_epoch_days + cutoff_week * 7

    hist = history.collect()

    # Article lookup: article_idx -> product_code, product_group_name, garment_group_name
    art_lookup = (
        articles_df
        .select(["article_idx", "product_code", "product_group_name", "garment_group_name"])
        .collect()
    )

    # ------------------------------------------------------------------
    # i_times_bought and i_days_since_bought_article
    # ------------------------------------------------------------------
    pair_agg = (
        hist
        .join(
            candidates.select(["customer_idx", "article_idx"]),
            on=["customer_idx", "article_idx"],
            how="inner",
        )
        .group_by(["customer_idx", "article_idx"])
        .agg(
            pl.len().cast(pl.Int32).alias("i_times_bought"),
            pl.col("t_dat").cast(pl.Int32).max().alias("_last_bought_epoch"),
        )
    )

    pair_agg = pair_agg.with_columns(
        (pl.lit(cutoff_epoch, dtype=pl.Int32) - pl.col("_last_bought_epoch"))
        .cast(pl.Float32)
        .alias("i_days_since_bought_article")
    ).drop("_last_bought_epoch")

    result = candidates.join(pair_agg, on=["customer_idx", "article_idx"], how="left")

    result = result.with_columns(
        pl.col("i_times_bought").fill_null(0).cast(pl.Int32),
    )

    # ------------------------------------------------------------------
    # i_bought_same_product_code and i_days_since_bought_product_code
    # ------------------------------------------------------------------
    # Map each candidate article to its product_code
    cand_with_code = candidates.join(
        art_lookup.select(["article_idx", "product_code"]),
        on="article_idx",
        how="left",
    )

    # For each (customer_idx, product_code) pair in candidates, aggregate history
    hist_with_code = hist.join(
        art_lookup.select(["article_idx", "product_code"]),
        on="article_idx",
        how="left",
    )

    cand_codes = (
        cand_with_code
        .select(["customer_idx", "product_code"])
        .unique(subset=["customer_idx", "product_code"])
    )

    code_agg = (
        hist_with_code
        .join(cand_codes, on=["customer_idx", "product_code"], how="inner")
        .group_by(["customer_idx", "product_code"])
        .agg(
            pl.len().cast(pl.Int32).alias("i_bought_same_product_code"),
            pl.col("t_dat").cast(pl.Int32).max().alias("_last_code_epoch"),
        )
    )

    code_agg = code_agg.with_columns(
        (pl.lit(cutoff_epoch, dtype=pl.Int32) - pl.col("_last_code_epoch"))
        .cast(pl.Float32)
        .alias("i_days_since_bought_product_code")
    ).drop("_last_code_epoch")

    result = result.join(
        cand_with_code.select(["customer_idx", "article_idx", "product_code"]),
        on=["customer_idx", "article_idx"],
        how="left",
    )
    result = result.join(code_agg, on=["customer_idx", "product_code"], how="left")
    result = result.drop("product_code")

    result = result.with_columns(
        pl.col("i_bought_same_product_code").fill_null(0).cast(pl.Int32),
    )

    # ------------------------------------------------------------------
    # i_customer_share_product_group and i_customer_share_garment_group
    # ------------------------------------------------------------------
    # Join each transaction to its article's group columns
    hist_groups = hist.join(
        art_lookup.select(["article_idx", "product_group_name", "garment_group_name"]),
        on="article_idx",
        how="left",
    )

    # Total purchases per customer (all history)
    # Use Float64 accumulation: Polars' scalar Float32 path can give 1-ULP different
    # results from the vectorised SIMD path for small datasets (≤ a few hundred rows).
    # Computing in Float64 then casting to Float32 gives a correctly-rounded result
    # independent of dataset size, ensuring batch == single-customer parity.
    cust_total = (
        hist_groups
        .group_by("customer_idx")
        .agg(pl.len().cast(pl.Float64).alias("_total_purchases"))
    )

    # Purchases per (customer_idx, product_group_name)
    pg_counts = (
        hist_groups
        .group_by(["customer_idx", "product_group_name"])
        .agg(pl.len().cast(pl.Float64).alias("_pg_count"))
        .join(cust_total, on="customer_idx", how="left")
        .with_columns(
            (pl.col("_pg_count") / pl.col("_total_purchases"))
            .cast(pl.Float32)
            .alias("i_customer_share_product_group")
        )
        .select(["customer_idx", "product_group_name", "i_customer_share_product_group"])
    )

    # Purchases per (customer_idx, garment_group_name)
    gg_counts = (
        hist_groups
        .group_by(["customer_idx", "garment_group_name"])
        .agg(pl.len().cast(pl.Float64).alias("_gg_count"))
        .join(cust_total, on="customer_idx", how="left")
        .with_columns(
            (pl.col("_gg_count") / pl.col("_total_purchases"))
            .cast(pl.Float32)
            .alias("i_customer_share_garment_group")
        )
        .select(["customer_idx", "garment_group_name", "i_customer_share_garment_group"])
    )

    # Map candidate articles to their groups
    result = result.join(
        art_lookup.select(["article_idx", "product_group_name", "garment_group_name"]),
        on="article_idx",
        how="left",
    )

    result = result.join(pg_counts, on=["customer_idx", "product_group_name"], how="left")
    result = result.join(gg_counts, on=["customer_idx", "garment_group_name"], how="left")
    result = result.drop(["product_group_name", "garment_group_name"])

    # ------------------------------------------------------------------
    # i_age_gap, i_price_ratio, i_channel_gap (from precomputed features)
    # ------------------------------------------------------------------
    cust_subset = customer_features.select(["customer_idx", "c_age", "c_mean_price", "c_online_share"])
    art_subset = article_features.select([
        "article_idx",
        "a_mean_buyer_age_12w",
        "a_mean_price_4w",
        "_article_online_share_4w",
    ])

    result = result.join(cust_subset, on="customer_idx", how="left")
    result = result.join(art_subset, on="article_idx", how="left")

    # i_age_gap = |customer_age - a_mean_buyer_age_12w| (null if either null)
    result = result.with_columns(
        pl.when(pl.col("c_age").is_null() | pl.col("a_mean_buyer_age_12w").is_null())
        .then(pl.lit(None, dtype=pl.Float32))
        .otherwise(
            (pl.col("c_age") - pl.col("a_mean_buyer_age_12w")).abs().cast(pl.Float32)
        )
        .alias("i_age_gap")
    )

    # i_price_ratio = a_mean_price_4w / c_mean_price (null if denominator 0 or null)
    # Use Float64 intermediate: Polars scalar vs SIMD Float32 paths can differ by 1 ULP.
    result = result.with_columns(
        pl.when(
            pl.col("c_mean_price").is_null()
            | (pl.col("c_mean_price") == 0.0)
            | pl.col("a_mean_price_4w").is_null()
        )
        .then(pl.lit(None, dtype=pl.Float32))
        .otherwise(
            (pl.col("a_mean_price_4w").cast(pl.Float64) / pl.col("c_mean_price").cast(pl.Float64))
            .cast(pl.Float32)
        )
        .alias("i_price_ratio")
    )

    # i_channel_gap = |customer_online_share - article_online_share_4w| (null if either null)
    result = result.with_columns(
        pl.when(
            pl.col("c_online_share").is_null() | pl.col("_article_online_share_4w").is_null()
        )
        .then(pl.lit(None, dtype=pl.Float32))
        .otherwise(
            (pl.col("c_online_share") - pl.col("_article_online_share_4w")).abs().cast(pl.Float32)
        )
        .alias("i_channel_gap")
    )

    # ------------------------------------------------------------------
    # Final column selection and dtype enforcement
    # ------------------------------------------------------------------
    result = result.select([
        pl.col("customer_idx").cast(pl.Int32),
        pl.col("article_idx").cast(pl.Int32),
        pl.col("i_times_bought").cast(pl.Int32),
        pl.col("i_days_since_bought_article").cast(pl.Float32),
        pl.col("i_bought_same_product_code").cast(pl.Int32),
        pl.col("i_days_since_bought_product_code").cast(pl.Float32),
        pl.col("i_customer_share_product_group").cast(pl.Float32),
        pl.col("i_customer_share_garment_group").cast(pl.Float32),
        pl.col("i_age_gap").cast(pl.Float32),
        pl.col("i_price_ratio").cast(pl.Float32),
        pl.col("i_channel_gap").cast(pl.Float32),
    ])

    return result
