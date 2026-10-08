"""Article feature computation for a single fold."""
import polars as pl


def compute_article_features(
    history: pl.LazyFrame,
    cutoff_week: int,
    articles_df: pl.LazyFrame,
    customers_df: pl.LazyFrame,
    anchor_epoch_days: int,
    censored_cutoff_epoch: int = 17821,
) -> pl.DataFrame:
    """Compute per-article features using only transactions with week_idx < cutoff_week.

    Parameters
    ----------
    history:
        LazyFrame of transactions already filtered to week_idx < cutoff_week.
    cutoff_week:
        The fold's cutoff week index.
    articles_df:
        LazyFrame from load_articles().
    customers_df:
        LazyFrame from load_customers() for buyer age join.
    anchor_epoch_days:
        ANCHOR_EPOCH_DAYS constant (17793).
    censored_cutoff_epoch:
        Epoch day threshold for censored flag. Default = 17793 + 28 = 17821 (2018-10-17).

    Returns
    -------
    pl.DataFrame
        One row per article_idx. Columns: article_idx and all a_ features.
    """
    assert history.filter(pl.col("week_idx") >= cutoff_week).collect().height == 0, \
        f"history contains rows with week_idx >= {cutoff_week}"

    cutoff_epoch: int = anchor_epoch_days + cutoff_week * 7

    hist = history.collect()

    # ------------------------------------------------------------------
    # Sales window aggregations
    # ------------------------------------------------------------------
    def _sales_window(df: pl.DataFrame, weeks_back: int) -> pl.DataFrame:
        return (
            df.filter(pl.col("week_idx") >= cutoff_week - weeks_back)
            .group_by("article_idx")
            .agg(pl.len().cast(pl.Int32).alias(f"a_sales_{weeks_back}w"))
        )

    sales_1w = _sales_window(hist, 1)
    sales_2w = _sales_window(hist, 2)
    sales_4w = _sales_window(hist, 4)

    unique_buyers_1w = (
        hist.filter(pl.col("week_idx") >= cutoff_week - 1)
        .group_by("article_idx")
        .agg(pl.col("customer_idx").n_unique().cast(pl.Int32).alias("a_unique_buyers_1w"))
    )

    # ------------------------------------------------------------------
    # First and last sale dates (epoch days)
    # ------------------------------------------------------------------
    sale_dates = (
        hist
        .group_by("article_idx")
        .agg(
            pl.col("t_dat").cast(pl.Int32).min().alias("_first_sale_epoch"),
            pl.col("t_dat").cast(pl.Int32).max().alias("_last_sale_epoch"),
        )
    )

    # ------------------------------------------------------------------
    # Price features
    # ------------------------------------------------------------------
    price_4w = (
        hist.filter(pl.col("week_idx") >= cutoff_week - 4)
        .group_by("article_idx")
        .agg(pl.col("price").cast(pl.Float32).mean().alias("a_mean_price_4w"))
    )

    price_1w = (
        hist.filter(pl.col("week_idx") >= cutoff_week - 1)
        .group_by("article_idx")
        .agg(pl.col("price").cast(pl.Float32).mean().alias("_mean_price_1w"))
    )

    price_all = (
        hist
        .group_by("article_idx")
        .agg(pl.col("price").cast(pl.Float32).mean().alias("_mean_price_all"))
    )

    # ------------------------------------------------------------------
    # Repurchase rate
    # ------------------------------------------------------------------
    # For each (customer_idx, article_idx), find if there are >= 2 distinct purchase dates
    # Rate = distinct buyers with repurchase / distinct buyers total
    buyer_dates = (
        hist
        .group_by(["article_idx", "customer_idx"])
        .agg(pl.col("t_dat").cast(pl.Int32).n_unique().alias("_n_dates"))
    )

    total_buyers = (
        buyer_dates
        .group_by("article_idx")
        .agg(pl.len().cast(pl.Float32).alias("_total_buyers"))
    )

    repurchase_buyers = (
        buyer_dates
        .filter(pl.col("_n_dates") >= 2)
        .group_by("article_idx")
        .agg(pl.len().cast(pl.Float32).alias("_repurchase_buyers"))
    )

    repurchase_rate = (
        total_buyers
        .join(repurchase_buyers, on="article_idx", how="left")
        .with_columns(
            pl.when(pl.col("_total_buyers") > 0)
            .then(
                (pl.col("_repurchase_buyers").fill_null(0.0) / pl.col("_total_buyers"))
                .cast(pl.Float32)
            )
            .otherwise(pl.lit(None, dtype=pl.Float32))
            .alias("a_repurchase_rate")
        )
        .select(["article_idx", "a_repurchase_rate"])
    )

    # ------------------------------------------------------------------
    # Buyer age in last 12 weeks
    # ------------------------------------------------------------------
    cust_ages = (
        customers_df
        .select(["customer_idx", "age"])
        .collect()
    )

    buyers_12w = (
        hist.filter(pl.col("week_idx") >= cutoff_week - 12)
        .select(["article_idx", "customer_idx"])
        .join(cust_ages, on="customer_idx", how="left")
        .group_by("article_idx")
        .agg(
            pl.col("age").cast(pl.Float32).mean().alias("a_mean_buyer_age_12w"),
            pl.col("age").cast(pl.Float32).std().alias("a_std_buyer_age_12w"),
        )
    )

    # ------------------------------------------------------------------
    # Article channel online share in last 4 weeks (stored for interaction features)
    # ------------------------------------------------------------------
    channel_4w = (
        hist.filter(pl.col("week_idx") >= cutoff_week - 4)
        .group_by("article_idx")
        .agg(
            (
                (pl.col("sales_channel_id").cast(pl.Int32) == 2).cast(pl.Float32).mean()
            ).alias("_article_online_share_4w")
        )
    )

    # ------------------------------------------------------------------
    # Article metadata from articles table
    # ------------------------------------------------------------------
    art_meta = (
        articles_df
        .select([
            "article_idx",
            "product_type_no",
            "product_group_name",
            "index_group_name",
            "garment_group_name",
            "colour_group_name",
            "section_name",
            "department_no",
        ])
        .collect()
    )

    # Build deterministic ordinal encodings from all unique values in articles table
    def _ordinal_encode(df: pl.DataFrame, src_col: str, dst_col: str) -> pl.DataFrame:
        unique_vals = sorted(
            [v for v in art_meta[src_col].unique().to_list() if v is not None]
        )
        val_to_code = {v: i for i, v in enumerate(unique_vals)}
        mapping = pl.DataFrame({
            src_col: pl.Series(unique_vals, dtype=pl.Utf8),
            dst_col: pl.Series(list(range(len(unique_vals))), dtype=pl.Int8),
        })
        return df.join(mapping, on=src_col, how="left").drop(src_col)

    art_meta = _ordinal_encode(art_meta, "product_group_name", "a_product_group_name")
    art_meta = _ordinal_encode(art_meta, "index_group_name", "a_index_group_name")
    art_meta = _ordinal_encode(art_meta, "garment_group_name", "a_garment_group_name")
    art_meta = _ordinal_encode(art_meta, "colour_group_name", "a_colour_group_name")
    art_meta = _ordinal_encode(art_meta, "section_name", "a_section_name")

    art_meta = art_meta.rename({
        "product_type_no": "a_product_type_no",
        "department_no": "a_department_no",
    })

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    result = (
        art_meta
        .join(sales_1w, on="article_idx", how="left")
        .join(sales_2w, on="article_idx", how="left")
        .join(sales_4w, on="article_idx", how="left")
        .join(unique_buyers_1w, on="article_idx", how="left")
        .join(sale_dates, on="article_idx", how="left")
        .join(price_4w, on="article_idx", how="left")
        .join(price_1w, on="article_idx", how="left")
        .join(price_all, on="article_idx", how="left")
        .join(repurchase_rate, on="article_idx", how="left")
        .join(buyers_12w, on="article_idx", how="left")
        .join(channel_4w, on="article_idx", how="left")
    )

    # Fill window counts with 0 for articles with no activity in that window
    result = result.with_columns(
        pl.col("a_sales_1w").fill_null(0).cast(pl.Int32),
        pl.col("a_sales_2w").fill_null(0).cast(pl.Int32),
        pl.col("a_sales_4w").fill_null(0).cast(pl.Int32),
        pl.col("a_unique_buyers_1w").fill_null(0).cast(pl.Int32),
    )

    # Trend ratio: a_sales_1w / (a_sales_4w / 4.0); null if a_sales_4w == 0
    result = result.with_columns(
        pl.when(pl.col("a_sales_4w") == 0)
        .then(pl.lit(None, dtype=pl.Float32))
        .otherwise(
            (pl.col("a_sales_1w").cast(pl.Float32) / (pl.col("a_sales_4w").cast(pl.Float32) / 4.0))
            .cast(pl.Float32)
        )
        .alias("a_trend_ratio")
    )

    # Days since first sale (null if never sold)
    result = result.with_columns(
        pl.when(pl.col("_first_sale_epoch").is_null())
        .then(pl.lit(None, dtype=pl.Float32))
        .otherwise(
            (pl.lit(cutoff_epoch, dtype=pl.Int32) - pl.col("_first_sale_epoch"))
            .cast(pl.Float32)
        )
        .alias("a_days_since_first_sale")
    )

    # Censored flag: 1 if first_sale_epoch < censored_cutoff_epoch, else 0, null if never sold
    result = result.with_columns(
        pl.when(pl.col("_first_sale_epoch").is_null())
        .then(pl.lit(None, dtype=pl.Int8))
        .when(pl.col("_first_sale_epoch") < censored_cutoff_epoch)
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("a_censored_flag")
    )

    # Days since last sale (null if never sold)
    result = result.with_columns(
        pl.when(pl.col("_last_sale_epoch").is_null())
        .then(pl.lit(None, dtype=pl.Float32))
        .otherwise(
            (pl.lit(cutoff_epoch, dtype=pl.Int32) - pl.col("_last_sale_epoch"))
            .cast(pl.Float32)
        )
        .alias("a_days_since_last_sale")
    )

    # Price vs own history: last-week mean / all-history mean (null if missing)
    result = result.with_columns(
        pl.when(
            pl.col("_mean_price_1w").is_null() | pl.col("_mean_price_all").is_null()
        )
        .then(pl.lit(None, dtype=pl.Float32))
        .otherwise(
            (pl.col("_mean_price_1w") / pl.col("_mean_price_all")).cast(pl.Float32)
        )
        .alias("a_price_vs_own_history")
    )

    # ------------------------------------------------------------------
    # Final column selection and dtype enforcement
    # ------------------------------------------------------------------
    result = result.select([
        pl.col("article_idx").cast(pl.Int32),
        pl.col("a_sales_1w").cast(pl.Int32),
        pl.col("a_sales_2w").cast(pl.Int32),
        pl.col("a_sales_4w").cast(pl.Int32),
        pl.col("a_unique_buyers_1w").cast(pl.Int32),
        pl.col("a_trend_ratio").cast(pl.Float32),
        pl.col("a_days_since_first_sale").cast(pl.Float32),
        pl.col("a_censored_flag").cast(pl.Int8),
        pl.col("a_days_since_last_sale").cast(pl.Float32),
        pl.col("a_mean_price_4w").cast(pl.Float32),
        pl.col("a_price_vs_own_history").cast(pl.Float32),
        pl.col("a_repurchase_rate").cast(pl.Float32),
        pl.col("a_mean_buyer_age_12w").cast(pl.Float32),
        pl.col("a_std_buyer_age_12w").cast(pl.Float32),
        pl.col("a_product_type_no").cast(pl.Int16),
        pl.col("a_product_group_name").cast(pl.Int8),
        pl.col("a_index_group_name").cast(pl.Int8),
        pl.col("a_garment_group_name").cast(pl.Int8),
        pl.col("a_colour_group_name").cast(pl.Int8),
        pl.col("a_section_name").cast(pl.Int8),
        pl.col("a_department_no").cast(pl.Int16),
        pl.col("_article_online_share_4w").cast(pl.Float32),
    ])

    return result
