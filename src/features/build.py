"""Main feature build pipeline for a single fold or holdout week."""
import polars as pl

from src.config import (
    ANCHOR_EPOCH_DAYS,
    COPURCHASE_CUSTOMER_LOOKBACK,
    COPURCHASE_LOOKBACK,
    HOLDOUT_WEEK,
    MERGE_N,
    MERGE_PRIORITY,
    MIN_COPURCHASE_COUNT,
    PROCESSED_DIR,
    PRODUCT_CODE_LOOKBACK,
    REPURCHASE_WINDOWS,
    SOURCE_K,
    VALIDATION_WEEKS,
)
from src.data_io import load_articles, load_customers, load_transactions
from src.time_split import build_fold, build_holdout_fold

from src.candidates import repurchase as repurchase_mod
from src.candidates import product_code as product_code_mod
from src.candidates import copurchase as copurchase_mod
from src.candidates import popularity as popularity_mod
from src.candidates.merge import merge_candidates

from src.features.candidate import compute_candidate_features
from src.features.customer import compute_customer_features
from src.features.article import compute_article_features
from src.features.interaction import compute_interaction_features
from src.features.registry import FEATURE_LIST


def _generate_candidates(
    history: pl.LazyFrame,
    cutoff_week: int,
    eval_customers: list[int],
) -> pl.DataFrame:
    """Run all candidate sources and merge into a single candidate set."""
    source_dfs: dict[str, pl.DataFrame] = {}

    # Repurchase (uses all windows; take the one that produces the most candidates
    # by collecting each window and picking the union — here we use the production
    # default with no lookback, which subsumes shorter windows by design)
    repurchase_df = repurchase_mod.generate(
        history=history,
        cutoff_week=cutoff_week,
        customers=eval_customers,
        k=SOURCE_K["repurchase"],
        lookback_weeks=None,
        ordering="production",
    )
    source_dfs["repurchase"] = repurchase_df

    # Product code
    product_code_df = product_code_mod.generate(
        history=history,
        cutoff_week=cutoff_week,
        customers=eval_customers,
        k=SOURCE_K["product_code"],
        lookback_weeks=PRODUCT_CODE_LOOKBACK,
    )
    source_dfs["product_code"] = product_code_df

    # Copurchase
    copurchase_df = copurchase_mod.generate(
        history=history,
        cutoff_week=cutoff_week,
        customers=eval_customers,
        k=SOURCE_K["copurchase"],
        lookback_weeks=COPURCHASE_LOOKBACK,
        customer_lookback_weeks=COPURCHASE_CUSTOMER_LOOKBACK,
        min_count=MIN_COPURCHASE_COUNT,
    )
    source_dfs["copurchase"] = copurchase_df

    # Global last week
    pop_last_week_df = popularity_mod.generate_global_last_week(
        history=history,
        cutoff_week=cutoff_week,
        customers=eval_customers,
        k=SOURCE_K["popularity_last_week"],
    )
    source_dfs["popularity_last_week"] = pop_last_week_df

    # Global decayed
    pop_decayed_df = popularity_mod.generate_global_decayed(
        history=history,
        cutoff_week=cutoff_week,
        customers=eval_customers,
        k=SOURCE_K["popularity_decayed"],
    )
    source_dfs["popularity_decayed"] = pop_decayed_df

    # Segment popular
    seg_popular_df = popularity_mod.generate_segment_popular(
        history=history,
        cutoff_week=cutoff_week,
        customers=eval_customers,
        k=SOURCE_K["segment_popular"],
    )
    source_dfs["segment_popular"] = seg_popular_df

    merged = merge_candidates(
        source_dfs=source_dfs,
        customers=eval_customers,
        n=MERGE_N,
        priority=MERGE_PRIORITY,
    )
    return merged


def _compute_features_inner(
    history_lf: pl.LazyFrame,
    ground_truth: dict[int, set[int]],
    eval_customers: list[int],
    articles_df: pl.LazyFrame,
    customers_df: pl.LazyFrame,
    fold_week: int,
    neg_sample_rate: float = 1.0,
    seed: int = 42,
) -> pl.DataFrame:
    """Core feature computation given pre-loaded data frames (no IO).

    Exposed separately from build_fold_features() to allow injection of
    synthetic data in tests. The caller is responsible for ensuring
    history_lf contains only rows with week_idx < fold_week.
    """
    # Generate candidates
    merged = _generate_candidates(
        history=history_lf,
        cutoff_week=fold_week,
        eval_customers=eval_customers,
    )

    merged_with_cand_features = compute_candidate_features(
        merged_candidates=merged,
        sources=MERGE_PRIORITY,
    )

    customer_feats = compute_customer_features(
        history=history_lf,
        cutoff_week=fold_week,
        customers_df=customers_df,
        anchor_epoch_days=ANCHOR_EPOCH_DAYS,
    )

    article_feats = compute_article_features(
        history=history_lf,
        cutoff_week=fold_week,
        articles_df=articles_df,
        customers_df=customers_df,
        anchor_epoch_days=ANCHOR_EPOCH_DAYS,
    )

    candidates_pairs = merged_with_cand_features.select(["customer_idx", "article_idx"])
    interaction_feats = compute_interaction_features(
        history=history_lf,
        candidates=candidates_pairs,
        cutoff_week=fold_week,
        article_features=article_feats,
        customer_features=customer_feats,
        articles_df=articles_df,
        anchor_epoch_days=ANCHOR_EPOCH_DAYS,
    )

    art_join_cols = [c for c in article_feats.columns if not c.startswith("_")]
    cust_join_cols = [c for c in customer_feats.columns if not c.startswith("_")]

    result = (
        merged_with_cand_features
        .join(customer_feats.select(cust_join_cols), on="customer_idx", how="left")
        .join(article_feats.select(art_join_cols), on="article_idx", how="left")
        .join(interaction_feats, on=["customer_idx", "article_idx"], how="left")
    )

    # Add labels
    gt_rows: list[dict] = [
        {"customer_idx": c, "article_idx": a}
        for c, arts in ground_truth.items()
        for a in arts
    ]
    if gt_rows:
        gt_df = pl.DataFrame(gt_rows, schema={"customer_idx": pl.Int32, "article_idx": pl.Int32})
        gt_df = gt_df.with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    else:
        gt_df = pl.DataFrame(
            schema={"customer_idx": pl.Int32, "article_idx": pl.Int32, "label": pl.Int8}
        )

    result = (
        result
        .join(gt_df, on=["customer_idx", "article_idx"], how="left")
        .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    )

    # Downsample negatives
    if neg_sample_rate < 1.0:
        result = _downsample_negatives(result, neg_sample_rate, seed)

    # Enforce column order and dtypes
    feature_names = [spec.name for spec in FEATURE_LIST]
    ordered_cols = ["customer_idx", "article_idx", "label"] + feature_names

    for spec in FEATURE_LIST:
        if spec.name not in result.columns:
            result = result.with_columns(pl.lit(None, dtype=spec.dtype).alias(spec.name))

    result = result.select(ordered_cols)

    cast_exprs = [
        pl.col("customer_idx").cast(pl.Int32),
        pl.col("article_idx").cast(pl.Int32),
        pl.col("label").cast(pl.Int8),
    ]
    for spec in FEATURE_LIST:
        cast_exprs.append(pl.col(spec.name).cast(spec.dtype))

    return result.with_columns(cast_exprs)


def _downsample_negatives(
    df: pl.DataFrame,
    neg_sample_rate: float,
    seed: int,
) -> pl.DataFrame:
    """Keep all positives and randomly sample neg_sample_rate of negatives (deterministic)."""
    positives = df.filter(pl.col("label") == 1)
    negatives = df.filter(pl.col("label") == 0)

    # Use polars struct hash (xxhash-based, uniform UInt64 output) divided by 2^64
    # to get a [0, 1) uniform pseudo-random value per row.
    _U64_MAX = float(2**64)
    sampled_negatives = (
        negatives
        .with_columns(
            (
                pl.struct(["customer_idx", "article_idx"])
                .hash(seed=seed)
                .cast(pl.Float64)
                / _U64_MAX
            ).alias("_rand")
        )
        .filter(pl.col("_rand") < neg_sample_rate)
        .drop("_rand")
    )

    return pl.concat([positives, sampled_negatives], how="vertical")


def build_fold_features(
    fold_week: int,
    neg_sample_rate: float = 0.2,
    seed: int = 42,
) -> pl.DataFrame:
    """Build the full feature matrix for a single fold or the holdout week.

    Parameters
    ----------
    fold_week:
        One of VALIDATION_WEEKS (100-103) or HOLDOUT_WEEK (104).
    neg_sample_rate:
        Fraction of negative examples to retain for training folds 100-102.
        All positives are always kept.
    seed:
        Random seed for negative downsampling.

    Returns
    -------
    pl.DataFrame
        Columns: customer_idx (Int32), article_idx (Int32), label (Int8),
        then all 65 FEATURE_LIST columns in order.
        Written to data/processed/features/fold_{fold_week}.parquet.
    """
    if fold_week == HOLDOUT_WEEK:
        history, ground_truth, eval_customers = build_holdout_fold()
    else:
        history, ground_truth, eval_customers = build_fold(fold_week)

    hist_df = history.collect()
    history_lf = hist_df.lazy()
    articles_df = load_articles()
    customers_df = load_customers()

    if fold_week == HOLDOUT_WEEK:
        # Holdout: _compute_features_inner is called without downsampling,
        # then labels are replaced with placeholder -1 and written separately.
        result = _compute_features_inner(
            history_lf, ground_truth, eval_customers,
            articles_df, customers_df, fold_week,
            neg_sample_rate=1.0, seed=seed,
        )
        # Overwrite label column with -1 placeholder and write labels separately.
        labels_df = result.select(["customer_idx", "article_idx", "label"])
        labels_path = PROCESSED_DIR / "features" / "labels_104.parquet"
        labels_df.write_parquet(labels_path, compression="zstd")
        result = result.with_columns(pl.lit(-1, dtype=pl.Int8).alias("label"))
    else:
        sample_rate = neg_sample_rate if fold_week in [100, 101, 102] else 1.0
        result = _compute_features_inner(
            history_lf, ground_truth, eval_customers,
            articles_df, customers_df, fold_week,
            neg_sample_rate=sample_rate, seed=seed,
        )

    features_dir = PROCESSED_DIR / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    out_path = features_dir / f"fold_{fold_week}.parquet"
    result.write_parquet(out_path, compression="zstd")

    return result
