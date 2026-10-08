"""Customer feature computation for a single fold."""
import polars as pl


def compute_customer_features(
    history: pl.LazyFrame,
    cutoff_week: int,
    customers_df: pl.LazyFrame,
    anchor_epoch_days: int,
) -> pl.DataFrame:
    """Compute per-customer features using only transactions with week_idx < cutoff_week.

    Parameters
    ----------
    history:
        LazyFrame of transactions already filtered to week_idx < cutoff_week.
    cutoff_week:
        The fold's cutoff week index. Used only to compute day offsets.
    customers_df:
        LazyFrame from load_customers() with columns:
        customer_idx (Int32), customer_id (str), FN (float), Active (float),
        club_member_status (str), fashion_news_frequency (str), age (float).
    anchor_epoch_days:
        ANCHOR_EPOCH_DAYS constant (days since 1970-01-01 to 2018-09-19 = 17793).

    Returns
    -------
    pl.DataFrame
        One row per customer_idx present in history. Columns:
        customer_idx, c_n_purchases, c_n_purchases_1w, c_n_purchases_4w,
        c_n_purchases_12w, c_n_unique_articles, c_days_since_last_purchase,
        c_n_active_weeks, c_avg_basket_size, c_mean_price, c_std_price,
        c_online_share, c_age, c_club_member_status, c_fashion_news_frequency,
        c_FN, c_Active.
    """
    assert history.filter(pl.col("week_idx") >= cutoff_week).collect().height == 0, \
        f"history contains rows with week_idx >= {cutoff_week}"

    cutoff_epoch: int = anchor_epoch_days + cutoff_week * 7

    hist = history.collect()

    # ------------------------------------------------------------------
    # Counts and basic aggregations (all history)
    # ------------------------------------------------------------------
    agg_all = (
        hist
        .group_by("customer_idx")
        .agg(
            pl.len().cast(pl.Int32).alias("c_n_purchases"),
            pl.col("article_idx").n_unique().cast(pl.Int32).alias("c_n_unique_articles"),
            pl.col("t_dat").cast(pl.Int32).max().alias("_last_date_epoch"),
            pl.col("week_idx").cast(pl.Int32).n_unique().cast(pl.Int32).alias("c_n_active_weeks"),
            pl.col("price").cast(pl.Float32).mean().alias("c_mean_price"),
            pl.col("price").cast(pl.Float32).std().alias("c_std_price"),
            (
                (pl.col("sales_channel_id").cast(pl.Int32) == 2).cast(pl.Float32).mean()
            ).alias("c_online_share"),
        )
    )

    agg_all = agg_all.with_columns(
        (pl.lit(cutoff_epoch, dtype=pl.Int32) - pl.col("_last_date_epoch"))
        .cast(pl.Float32)
        .alias("c_days_since_last_purchase")
    ).drop("_last_date_epoch")

    # ------------------------------------------------------------------
    # Window purchase counts
    # ------------------------------------------------------------------
    def _window_count(df: pl.DataFrame, weeks_back: int, col_name: str) -> pl.DataFrame:
        return (
            df.filter(pl.col("week_idx") >= cutoff_week - weeks_back)
            .group_by("customer_idx")
            .agg(pl.len().cast(pl.Int32).alias(col_name))
        )

    cnt_1w = _window_count(hist, 1, "c_n_purchases_1w")
    cnt_4w = _window_count(hist, 4, "c_n_purchases_4w")
    cnt_12w = _window_count(hist, 12, "c_n_purchases_12w")

    # ------------------------------------------------------------------
    # Average basket size: mean articles per (customer_idx, t_dat) day
    # ------------------------------------------------------------------
    basket_sizes = (
        hist
        .group_by(["customer_idx", "t_dat"])
        .agg(pl.len().cast(pl.Float32).alias("day_count"))
        .group_by("customer_idx")
        .agg(pl.col("day_count").mean().cast(pl.Float32).alias("c_avg_basket_size"))
    )

    # ------------------------------------------------------------------
    # Join all transaction-derived features
    # ------------------------------------------------------------------
    result = (
        agg_all
        .join(cnt_1w, on="customer_idx", how="left")
        .join(cnt_4w, on="customer_idx", how="left")
        .join(cnt_12w, on="customer_idx", how="left")
        .join(basket_sizes, on="customer_idx", how="left")
    )

    # Fill window counts with 0 for customers who had no activity in that window
    result = result.with_columns(
        pl.col("c_n_purchases_1w").fill_null(0).cast(pl.Int32),
        pl.col("c_n_purchases_4w").fill_null(0).cast(pl.Int32),
        pl.col("c_n_purchases_12w").fill_null(0).cast(pl.Int32),
    )

    # ------------------------------------------------------------------
    # Join customer metadata
    # ------------------------------------------------------------------
    cust_meta = (
        customers_df
        .select(["customer_idx", "age", "club_member_status", "fashion_news_frequency", "FN", "Active"])
        .collect()
    )

    result = result.join(cust_meta, on="customer_idx", how="left")

    # ------------------------------------------------------------------
    # Encode categoricals
    # ------------------------------------------------------------------
    # club_member_status: "ACTIVE"->1, "PRE-CREATE"->0, else 0; null->-1
    result = result.with_columns(
        pl.when(pl.col("club_member_status").is_null())
        .then(pl.lit(-1, dtype=pl.Int8))
        .when(pl.col("club_member_status") == "ACTIVE")
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("c_club_member_status")
    )

    # fashion_news_frequency: "Regularly"->2, "Monthly"->1, "NONE"->0, else 0; null->-1
    result = result.with_columns(
        pl.when(pl.col("fashion_news_frequency").is_null())
        .then(pl.lit(-1, dtype=pl.Int8))
        .when(pl.col("fashion_news_frequency") == "Regularly")
        .then(pl.lit(2, dtype=pl.Int8))
        .when(pl.col("fashion_news_frequency") == "Monthly")
        .then(pl.lit(1, dtype=pl.Int8))
        .otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("c_fashion_news_frequency")
    )

    # FN: 1 or 0 or null->-1
    result = result.with_columns(
        pl.when(pl.col("FN").is_null())
        .then(pl.lit(-1, dtype=pl.Int8))
        .otherwise(pl.col("FN").cast(pl.Int8))
        .alias("c_FN")
    )

    # Active: 1 or 0 or null->-1
    result = result.with_columns(
        pl.when(pl.col("Active").is_null())
        .then(pl.lit(-1, dtype=pl.Int8))
        .otherwise(pl.col("Active").cast(pl.Int8))
        .alias("c_Active")
    )

    # ------------------------------------------------------------------
    # Final column selection and dtype enforcement
    # ------------------------------------------------------------------
    result = result.select([
        pl.col("customer_idx").cast(pl.Int32),
        pl.col("c_n_purchases").cast(pl.Int32),
        pl.col("c_n_purchases_1w").cast(pl.Int32),
        pl.col("c_n_purchases_4w").cast(pl.Int32),
        pl.col("c_n_purchases_12w").cast(pl.Int32),
        pl.col("c_n_unique_articles").cast(pl.Int32),
        pl.col("c_days_since_last_purchase").cast(pl.Float32),
        pl.col("c_n_active_weeks").cast(pl.Int32),
        pl.col("c_avg_basket_size").cast(pl.Float32),
        pl.col("c_mean_price").cast(pl.Float32),
        pl.col("c_std_price").cast(pl.Float32),
        pl.col("c_online_share").cast(pl.Float32),
        pl.col("age").cast(pl.Float32).alias("c_age"),
        pl.col("c_club_member_status").cast(pl.Int8),
        pl.col("c_fashion_news_frequency").cast(pl.Int8),
        pl.col("c_FN").cast(pl.Int8),
        pl.col("c_Active").cast(pl.Int8),
    ])

    return result
