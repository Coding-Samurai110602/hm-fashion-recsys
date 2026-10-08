"""Feature registry: defines the canonical ordered list of features for the LightGBM ranker."""
from typing import NamedTuple

import polars as pl


class FeatureSpec(NamedTuple):
    name: str
    dtype: type
    group: str
    description: str


FEATURE_LIST: list[FeatureSpec] = [
    # ------------------------------------------------------------------
    # Candidate group
    # ------------------------------------------------------------------
    FeatureSpec("final_rank", pl.Int32, "candidate", "Rank after merge (1=highest priority)"),
    FeatureSpec("n_sources", pl.Int8, "candidate", "Number of sources producing this pair"),
    FeatureSpec("in_repurchase", pl.Int8, "candidate", "1 if produced by repurchase source, else 0"),
    FeatureSpec("repurchase_rank", pl.Int16, "candidate", "Rank within repurchase (null if not in source)"),
    FeatureSpec("repurchase_score", pl.Float32, "candidate", "Score from repurchase source (null if not in source)"),
    FeatureSpec("in_product_code", pl.Int8, "candidate", "1 if produced by product_code source, else 0"),
    FeatureSpec("product_code_rank", pl.Int16, "candidate", "Rank within product_code (null if not in source)"),
    FeatureSpec("product_code_score", pl.Float32, "candidate", "Score from product_code source (null if not in source)"),
    FeatureSpec("in_copurchase", pl.Int8, "candidate", "1 if produced by copurchase source, else 0"),
    FeatureSpec("copurchase_rank", pl.Int16, "candidate", "Rank within copurchase (null if not in source)"),
    FeatureSpec("copurchase_score", pl.Float32, "candidate", "Score from copurchase source (null if not in source)"),
    FeatureSpec("in_popularity_last_week", pl.Int8, "candidate", "1 if produced by popularity_last_week source, else 0"),
    FeatureSpec("popularity_last_week_rank", pl.Int16, "candidate", "Rank within popularity_last_week (null if not in source)"),
    FeatureSpec("popularity_last_week_score", pl.Float32, "candidate", "Score from popularity_last_week source (null if not in source)"),
    FeatureSpec("in_popularity_decayed", pl.Int8, "candidate", "1 if produced by popularity_decayed source, else 0"),
    FeatureSpec("popularity_decayed_rank", pl.Int16, "candidate", "Rank within popularity_decayed (null if not in source)"),
    FeatureSpec("popularity_decayed_score", pl.Float32, "candidate", "Score from popularity_decayed source (null if not in source)"),
    FeatureSpec("in_segment_popular", pl.Int8, "candidate", "1 if produced by segment_popular source, else 0"),
    FeatureSpec("segment_popular_rank", pl.Int16, "candidate", "Rank within segment_popular (null if not in source)"),
    FeatureSpec("segment_popular_score", pl.Float32, "candidate", "Score from segment_popular source (null if not in source)"),
    # ------------------------------------------------------------------
    # Customer group
    # ------------------------------------------------------------------
    FeatureSpec("c_n_purchases", pl.Int32, "customer", "Total purchase count (all history)"),
    FeatureSpec("c_n_purchases_1w", pl.Int32, "customer", "Purchases in last 1 week before cutoff"),
    FeatureSpec("c_n_purchases_4w", pl.Int32, "customer", "Purchases in last 4 weeks before cutoff"),
    FeatureSpec("c_n_purchases_12w", pl.Int32, "customer", "Purchases in last 12 weeks before cutoff"),
    FeatureSpec("c_n_unique_articles", pl.Int32, "customer", "Unique article count (all history)"),
    FeatureSpec("c_days_since_last_purchase", pl.Float32, "customer", "Days from last purchase to cutoff epoch"),
    FeatureSpec("c_n_active_weeks", pl.Int32, "customer", "Distinct week_idx count in history"),
    FeatureSpec("c_avg_basket_size", pl.Float32, "customer", "Mean articles per shopping day"),
    FeatureSpec("c_mean_price", pl.Float32, "customer", "Mean price all history"),
    FeatureSpec("c_std_price", pl.Float32, "customer", "Std dev price all history"),
    FeatureSpec("c_online_share", pl.Float32, "customer", "Share of channel-2 (online) purchases"),
    FeatureSpec("c_age", pl.Float32, "customer", "Customer age from customers table"),
    FeatureSpec("c_club_member_status", pl.Int8, "customer", "Encoded club member status (null→-1, else ordinal code)"),
    FeatureSpec("c_fashion_news_frequency", pl.Int8, "customer", "Encoded fashion news frequency (null→-1, else ordinal code)"),
    FeatureSpec("c_FN", pl.Int8, "customer", "FN flag (null→-1)"),
    FeatureSpec("c_Active", pl.Int8, "customer", "Active flag (null→-1)"),
    # ------------------------------------------------------------------
    # Article group
    # ------------------------------------------------------------------
    FeatureSpec("a_sales_1w", pl.Int32, "article", "Sales count in last 1 week before cutoff"),
    FeatureSpec("a_sales_2w", pl.Int32, "article", "Sales count in last 2 weeks before cutoff"),
    FeatureSpec("a_sales_4w", pl.Int32, "article", "Sales count in last 4 weeks before cutoff"),
    FeatureSpec("a_unique_buyers_1w", pl.Int32, "article", "Unique buyer count last 1 week"),
    FeatureSpec("a_trend_ratio", pl.Float32, "article", "a_sales_1w / (a_sales_4w / 4.0), null if a_sales_4w==0"),
    FeatureSpec("a_days_since_first_sale", pl.Float32, "article", "Days from first sale to cutoff (null if never sold before cutoff)"),
    FeatureSpec("a_censored_flag", pl.Int8, "article", "1 if first sale is before censored window, 0 otherwise, null if never sold"),
    FeatureSpec("a_days_since_last_sale", pl.Float32, "article", "Days from last sale to cutoff (null if never sold)"),
    FeatureSpec("a_mean_price_4w", pl.Float32, "article", "Mean price in last 4 weeks (null if no sales in window)"),
    FeatureSpec("a_price_vs_own_history", pl.Float32, "article", "last-week mean price / all-history mean price (null if missing)"),
    FeatureSpec("a_repurchase_rate", pl.Float32, "article", "Share of buyers who bought again on a strictly later date (null if no buyers)"),
    FeatureSpec("a_mean_buyer_age_12w", pl.Float32, "article", "Mean buyer age in last 12 weeks"),
    FeatureSpec("a_std_buyer_age_12w", pl.Float32, "article", "Std dev buyer age in last 12 weeks"),
    FeatureSpec("a_product_type_no", pl.Int16, "article", "Product type number (from articles table, as-is)"),
    FeatureSpec("a_product_group_name", pl.Int8, "article", "Ordinal code for product_group_name"),
    FeatureSpec("a_index_group_name", pl.Int8, "article", "Ordinal code for index_group_name"),
    FeatureSpec("a_garment_group_name", pl.Int8, "article", "Ordinal code for garment_group_name"),
    FeatureSpec("a_colour_group_name", pl.Int8, "article", "Ordinal code for colour_group_name"),
    FeatureSpec("a_section_name", pl.Int8, "article", "Ordinal code for section_name"),
    FeatureSpec("a_department_no", pl.Int16, "article", "Department number (from articles table, as-is)"),
    # ------------------------------------------------------------------
    # Interaction group
    # ------------------------------------------------------------------
    FeatureSpec("i_times_bought", pl.Int32, "interaction", "Count of times customer bought this exact article (all history)"),
    FeatureSpec("i_days_since_bought_article", pl.Float32, "interaction", "Days since customer last bought this article (null if never)"),
    FeatureSpec("i_bought_same_product_code", pl.Int32, "interaction", "Count of customer purchases of same product_code (all history)"),
    FeatureSpec("i_days_since_bought_product_code", pl.Float32, "interaction", "Days since last purchase of same product_code (null if never)"),
    FeatureSpec("i_customer_share_product_group", pl.Float32, "interaction", "Customer's fraction of ALL purchases in article's product_group_name"),
    FeatureSpec("i_customer_share_garment_group", pl.Float32, "interaction", "Customer's fraction of ALL purchases in article's garment_group_name"),
    FeatureSpec("i_age_gap", pl.Float32, "interaction", "|customer_age - a_mean_buyer_age_12w| (null if either is null)"),
    FeatureSpec("i_price_ratio", pl.Float32, "interaction", "a_mean_price_4w / c_mean_price (null if denominator is 0 or null)"),
    FeatureSpec("i_channel_gap", pl.Float32, "interaction", "|customer_online_share - article_online_share_4w| (null if either is null)"),
]
