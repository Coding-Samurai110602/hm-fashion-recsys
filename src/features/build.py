"""Main feature build pipeline for a single fold or holdout week."""
from __future__ import annotations

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
from src.candidates.base import OUTPUT_DTYPES
from src.candidates.merge import merge_candidates

from src.features.candidate import compute_candidate_features
from src.features.customer import compute_customer_features
from src.features.article import compute_article_features
from src.features.interaction import compute_interaction_features
from src.features.registry import FEATURE_LIST


# ─────────────────────────────────────────────────────────────────────────────
# Legacy vectorised candidate generation (no precomputed state)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_candidates(
    history: pl.LazyFrame,
    cutoff_week: int,
    eval_customers: list[int],
) -> pl.DataFrame:
    """Run all candidate sources and merge into a single candidate set."""
    source_dfs: dict[str, pl.DataFrame] = {}

    repurchase_df = repurchase_mod.generate(
        history=history,
        cutoff_week=cutoff_week,
        customers=eval_customers,
        k=SOURCE_K["repurchase"],
        lookback_weeks=None,
        ordering="production",
    )
    source_dfs["repurchase"] = repurchase_df

    product_code_df = product_code_mod.generate(
        history=history,
        cutoff_week=cutoff_week,
        customers=eval_customers,
        k=SOURCE_K["product_code"],
        lookback_weeks=PRODUCT_CODE_LOOKBACK,
    )
    source_dfs["product_code"] = product_code_df

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

    pop_last_week_df = popularity_mod.generate_global_last_week(
        history=history,
        cutoff_week=cutoff_week,
        customers=eval_customers,
        k=SOURCE_K["popularity_last_week"],
    )
    source_dfs["popularity_last_week"] = pop_last_week_df

    pop_decayed_df = popularity_mod.generate_global_decayed(
        history=history,
        cutoff_week=cutoff_week,
        customers=eval_customers,
        k=SOURCE_K["popularity_decayed"],
    )
    source_dfs["popularity_decayed"] = pop_decayed_df

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


# ─────────────────────────────────────────────────────────────────────────────
# State-aware candidate generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pop_from_state(
    top_k: pl.DataFrame,
    eval_customers: list[int],
    source_name: str,
) -> pl.DataFrame:
    """Cross-join precomputed top-k popularity list with eval customers."""
    customers_df = pl.DataFrame({"customer_idx": pl.Series(eval_customers, dtype=pl.Int32)})
    result = customers_df.join(top_k, how="cross")
    return (
        result
        .with_columns(pl.lit(source_name).cast(pl.Utf8).alias("source"))
        .select([
            pl.col("customer_idx").cast(OUTPUT_DTYPES["customer_idx"]),
            pl.col("article_idx").cast(OUTPUT_DTYPES["article_idx"]),
            pl.col("source").cast(OUTPUT_DTYPES["source"]),
            pl.col("score").cast(OUTPUT_DTYPES["score"]),
            pl.col("source_rank").cast(OUTPUT_DTYPES["source_rank"]),
        ])
    )


def _seg_pop_from_state(state, eval_customers: list[int]) -> pl.DataFrame:
    """Segment popular candidates from precomputed per-bucket lists."""
    eval_buckets = state.all_cust_buckets.filter(
        pl.col("customer_idx").is_in(eval_customers)
    )
    if eval_buckets.is_empty() or not state.seg_pop_by_bucket:
        return pl.DataFrame(schema={
            "customer_idx": pl.Int32, "article_idx": pl.Int32,
            "source": pl.Utf8, "score": pl.Float32, "source_rank": pl.Int16,
        })

    parts = []
    for bucket, top_k in state.seg_pop_by_bucket.items():
        bucket_customers = eval_buckets.filter(pl.col("age_bucket") == bucket)
        if bucket_customers.is_empty():
            continue
        joined = bucket_customers.join(top_k, how="cross").drop("age_bucket")
        parts.append(joined)

    if not parts:
        return pl.DataFrame(schema={
            "customer_idx": pl.Int32, "article_idx": pl.Int32,
            "source": pl.Utf8, "score": pl.Float32, "source_rank": pl.Int16,
        })

    result = pl.concat(parts, how="vertical")
    # Deduplicate keeping highest score (customer may match multiple buckets)
    result = (
        result
        .sort(["customer_idx", "score", "article_idx"], descending=[False, True, False])
        .unique(subset=["customer_idx", "article_idx"], keep="first", maintain_order=True)
    )
    result = result.sort(
        ["customer_idx", "score", "article_idx"], descending=[False, True, False]
    ).with_columns(
        pl.cum_count("article_idx").over("customer_idx").cast(pl.Int16).alias("source_rank")
    ).filter(pl.col("source_rank") <= SOURCE_K["segment_popular"])

    return (
        result
        .with_columns(pl.lit("segment_popular").cast(pl.Utf8).alias("source"))
        .select([
            pl.col("customer_idx").cast(OUTPUT_DTYPES["customer_idx"]),
            pl.col("article_idx").cast(OUTPUT_DTYPES["article_idx"]),
            pl.col("source").cast(OUTPUT_DTYPES["source"]),
            pl.col("score").cast(OUTPUT_DTYPES["score"]),
            pl.col("source_rank").cast(OUTPUT_DTYPES["source_rank"]),
        ])
    )


def _product_code_from_state(
    history_lf: pl.LazyFrame,
    state,
    eval_customers: list[int],
) -> pl.DataFrame:
    """Product-code candidates using state.article_product_codes and state.prev_week_sales."""
    customer_filter = eval_customers

    base_lf = history_lf
    if len(customer_filter) > 50000:
        base_lf = base_lf.filter(pl.col("customer_idx").is_in(customer_filter))
    else:
        base_lf = base_lf.filter(pl.col("customer_idx").is_in(customer_filter))

    all_history_pairs = (
        base_lf
        .select(["customer_idx", "article_idx"])
        .unique(subset=["customer_idx", "article_idx"])
    )

    recent_lf = base_lf.filter(
        pl.col("week_idx") >= state.as_of_week - PRODUCT_CODE_LOOKBACK
    )
    recent_pairs = (
        recent_lf
        .select(["customer_idx", "article_idx"])
        .unique(subset=["customer_idx", "article_idx"])
    )

    articles_lf_pc = state.article_product_codes.lazy()

    recent_with_code = recent_pairs.join(articles_lf_pc, on="article_idx", how="left")

    customer_product_codes = (
        recent_with_code
        .select(["customer_idx", "product_code"])
        .unique(subset=["customer_idx", "product_code"])
    )
    candidate_articles = customer_product_codes.join(
        articles_lf_pc, on="product_code", how="left"
    ).select(["customer_idx", "article_idx"])

    all_history_collected = all_history_pairs.collect()
    candidate_collected = candidate_articles.collect()

    candidates = candidate_collected.join(
        all_history_collected,
        on=["customer_idx", "article_idx"],
        how="anti",
    )

    candidates = candidates.join(
        state.prev_week_sales, on="article_idx", how="left"
    ).with_columns(
        pl.col("sales_count").fill_null(0).cast(pl.Float32).alias("score")
    )

    candidates = (
        candidates
        .sort(["customer_idx", "score", "article_idx"], descending=[False, True, False])
        .unique(subset=["customer_idx", "article_idx"], keep="first", maintain_order=True)
    )

    k = SOURCE_K["product_code"]
    candidates = candidates.sort(
        ["customer_idx", "score", "article_idx"],
        descending=[False, True, False],
    ).with_columns(
        pl.cum_count("article_idx").over("customer_idx").cast(pl.Int16).alias("source_rank")
    ).filter(pl.col("source_rank") <= k)

    if candidates.is_empty():
        return pl.DataFrame(schema={
            "customer_idx": pl.Int32, "article_idx": pl.Int32,
            "source": pl.Utf8, "score": pl.Float32, "source_rank": pl.Int16,
        })

    return (
        candidates
        .with_columns(pl.lit("product_code").cast(pl.Utf8).alias("source"))
        .select([
            pl.col("customer_idx").cast(OUTPUT_DTYPES["customer_idx"]),
            pl.col("article_idx").cast(OUTPUT_DTYPES["article_idx"]),
            pl.col("source").cast(OUTPUT_DTYPES["source"]),
            pl.col("score").cast(OUTPUT_DTYPES["score"]),
            pl.col("source_rank").cast(OUTPUT_DTYPES["source_rank"]),
        ])
    )


def _copurchase_from_state(
    history_lf: pl.LazyFrame,
    state,
    eval_customers: list[int],
) -> pl.DataFrame:
    """Copurchase candidates via precomputed symmetric matrix (Polars, Float64 intermediate).

    Scores are computed as SUM(cosine_score) accumulated in Float64 then cast to Float32.
    Float64 accumulation makes the result independent of summation order, giving identical
    scores to the DuckDB DOUBLE SUM path used in the original batch build.
    """
    customer_filter = eval_customers

    customer_recent_df = (
        history_lf
        .filter(
            (pl.col("week_idx") >= state.as_of_week - COPURCHASE_CUSTOMER_LOOKBACK)
            & (pl.col("week_idx") < state.as_of_week)
        )
        .filter(pl.col("customer_idx").is_in(customer_filter))
        .select(["customer_idx", "article_idx"])
        .unique(subset=["customer_idx", "article_idx"])
        .collect()
    )

    all_history_df = (
        history_lf
        .filter(pl.col("customer_idx").is_in(customer_filter))
        .select(["customer_idx", "article_idx"])
        .unique(subset=["customer_idx", "article_idx"])
        .collect()
    )

    if customer_recent_df.is_empty() or state.copurchase_symmetric.is_empty():
        return pl.DataFrame(schema={
            "customer_idx": pl.Int32, "article_idx": pl.Int32,
            "source": pl.Utf8, "score": pl.Float32, "source_rank": pl.Int16,
        })

    scores = (
        customer_recent_df
        .join(
            state.copurchase_symmetric,
            left_on="article_idx",
            right_on="seed_article",
            how="inner",
        )
        # Drop the seed article_idx column; keep candidate_article as the recommendation target
        .drop("article_idx")
        .rename({"candidate_article": "article_idx", "cosine_score": "_cop_score"})
        .group_by(["customer_idx", "article_idx"])
        .agg(
            pl.col("_cop_score").sum().cast(pl.Float32).alias("score")
        )
    )

    filtered = scores.join(all_history_df, on=["customer_idx", "article_idx"], how="anti")

    if filtered.is_empty():
        return pl.DataFrame(schema={
            "customer_idx": pl.Int32, "article_idx": pl.Int32,
            "source": pl.Utf8, "score": pl.Float32, "source_rank": pl.Int16,
        })

    k = SOURCE_K["copurchase"]
    result = (
        filtered
        .sort(["customer_idx", "score", "article_idx"], descending=[False, True, False])
        .with_columns(
            pl.cum_count("article_idx").over("customer_idx").cast(pl.Int16).alias("source_rank")
        )
        .filter(pl.col("source_rank") <= k)
        .with_columns(pl.lit("copurchase").cast(pl.Utf8).alias("source"))
        .select([
            pl.col("customer_idx").cast(OUTPUT_DTYPES["customer_idx"]),
            pl.col("article_idx").cast(OUTPUT_DTYPES["article_idx"]),
            pl.col("source").cast(OUTPUT_DTYPES["source"]),
            pl.col("score").cast(OUTPUT_DTYPES["score"]),
            pl.col("source_rank").cast(OUTPUT_DTYPES["source_rank"]),
        ])
    )

    return result


def _generate_candidates_with_state(
    history_lf: pl.LazyFrame,
    state,
    eval_customers: list[int],
) -> pl.DataFrame:
    """Generate candidates using precomputed state for expensive global computations.

    Replaces _generate_candidates() in the state-aware path. Reuses the precomputed
    article_product_codes, prev_week_sales, copurchase_symmetric, and popularity top-k
    lists from state, avoiding repeated expensive scans of the full transaction history.
    """
    source_dfs: dict[str, pl.DataFrame] = {}

    # Repurchase: filter history_lf to eval_customers (fast when history is in-memory)
    source_dfs["repurchase"] = repurchase_mod.generate(
        history=history_lf,
        cutoff_week=state.as_of_week,
        customers=eval_customers,
        k=SOURCE_K["repurchase"],
        lookback_weeks=None,
        ordering="production",
    )

    # Product code: uses precomputed article_product_codes and prev_week_sales
    source_dfs["product_code"] = _product_code_from_state(history_lf, state, eval_customers)

    # Copurchase: uses precomputed symmetric matrix (no DuckDB pair-matrix rebuild)
    source_dfs["copurchase"] = _copurchase_from_state(history_lf, state, eval_customers)

    # Popularity: cross-join precomputed top-k lists with eval_customers (no history scan)
    source_dfs["popularity_last_week"] = _pop_from_state(
        state.pop_last_week_top, eval_customers, "popularity_last_week"
    )
    source_dfs["popularity_decayed"] = _pop_from_state(
        state.pop_decayed_top, eval_customers, "popularity_decayed"
    )

    # Segment popular: look up each customer's bucket in precomputed per-bucket lists
    source_dfs["segment_popular"] = _seg_pop_from_state(state, eval_customers)

    return merge_candidates(
        source_dfs=source_dfs,
        customers=eval_customers,
        n=MERGE_N,
        priority=MERGE_PRIORITY,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core feature computation (state-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_features_inner(
    history_lf: pl.LazyFrame,
    ground_truth: dict[int, set[int]],
    eval_customers: list[int],
    articles_df: pl.LazyFrame,
    customers_df: pl.LazyFrame,
    fold_week: int,
    neg_sample_rate: float = 1.0,
    seed: int = 42,
    _state=None,
) -> pl.DataFrame:
    """Core feature computation given pre-loaded data frames (no IO).

    Exposed separately from build_fold_features() to allow injection of
    synthetic data in tests. The caller is responsible for ensuring
    history_lf contains only rows with week_idx < fold_week.

    Parameters
    ----------
    _state:
        Optional RecommenderState. If provided, expensive customer-independent
        computations (article features, popularity scans, copurchase matrix) are
        read from state rather than recomputed. batch and single-customer paths use
        the same state, so results are guaranteed identical.
        If None, state is built internally from history_lf (first-time or test path).
    """
    from src.model.state import build_state, RecommenderState

    # ── Build state internally if not provided ─────────────────────────────
    if _state is None:
        history_df_collected = history_lf.collect()
        _state = build_state(fold_week, history_df_collected, articles_df, customers_df)

    # ── Pre-filter history to eval_customers ─────────────────────────────
    # Customer and interaction features aggregate per customer_idx; the result
    # is independent of other customers' rows.  Filtering avoids O(31.8M) scans
    # per customer for single-customer inference while producing identical values.
    history_cust = history_lf.filter(pl.col("customer_idx").is_in(eval_customers))

    # ── Generate candidates ────────────────────────────────────────────────
    merged = _generate_candidates_with_state(history_cust, _state, eval_customers)

    merged_with_cand_features = compute_candidate_features(
        merged_candidates=merged,
        sources=MERGE_PRIORITY,
    )

    # ── Customer features (computed from history filtered to eval_customers) ─
    customer_feats = compute_customer_features(
        history=history_cust,
        cutoff_week=fold_week,
        customers_df=_state.customers_df.lazy(),
        anchor_epoch_days=ANCHOR_EPOCH_DAYS,
    )

    # ── Article features (precomputed in state — no rescan) ────────────────
    article_feats = _state.article_feats

    candidates_pairs = merged_with_cand_features.select(["customer_idx", "article_idx"])
    interaction_feats = compute_interaction_features(
        history=history_cust,
        candidates=candidates_pairs,
        cutoff_week=fold_week,
        article_features=article_feats,
        customer_features=customer_feats,
        articles_df=_state.articles_lf,
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

    # ── Scale-invariant popularity share features (use state denominators) ─
    result = result.with_columns(
        (pl.col("popularity_last_week_score") / float(_state.total_tx_last_week))
        .cast(pl.Float32).alias("popularity_last_week_share")
    )

    result = result.with_columns(
        (pl.col("popularity_decayed_score") / float(_state.total_decayed_tx))
        .cast(pl.Float32).alias("popularity_decayed_share")
    )

    # segment_popular_share: normalise by per-bucket total from state
    eval_cust_buckets = _state.all_cust_buckets.filter(
        pl.col("customer_idx").is_in(eval_customers)
    )
    result = (
        result
        .join(eval_cust_buckets.select(["customer_idx", "age_bucket"]),
              on="customer_idx", how="left")
        .join(_state.bucket_totals, on="age_bucket", how="left")
        .with_columns(
            (pl.col("segment_popular_score") / pl.col("_bucket_total"))
            .cast(pl.Float32).alias("segment_popular_share")
        )
        .drop(["age_bucket", "_bucket_total"])
    )

    # ── Labels ────────────────────────────────────────────────────────────
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

    # ── Downsample negatives ───────────────────────────────────────────────
    if neg_sample_rate < 1.0:
        result = _downsample_negatives(result, neg_sample_rate, seed)

    # ── Enforce column order and dtypes ────────────────────────────────────
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

    return result.with_columns(cast_exprs).sort(["customer_idx", "article_idx"])


def _downsample_negatives(
    df: pl.DataFrame,
    neg_sample_rate: float,
    seed: int,
) -> pl.DataFrame:
    """Keep all positives and randomly sample neg_sample_rate of negatives (deterministic)."""
    positives = df.filter(pl.col("label") == 1)
    negatives = df.filter(pl.col("label") == 0)

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


def build_fold_101_full(seed: int = 42) -> pl.DataFrame:
    """Build fold 101 without negative downsampling for unbiased rolling-origin evaluation."""
    history, ground_truth, eval_customers = build_fold(101)
    hist_df = history.collect()
    history_lf = hist_df.lazy()
    articles_df = load_articles()
    customers_df = load_customers()

    result = _compute_features_inner(
        history_lf, ground_truth, eval_customers,
        articles_df, customers_df, fold_week=101,
        neg_sample_rate=1.0, seed=seed,
    )

    features_dir = PROCESSED_DIR / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    out_path = features_dir / "fold_101_full.parquet"
    result.write_parquet(out_path, compression="zstd")
    print(f"  fold_101_full: {len(result):,} rows, positives={int((result['label']==1).sum()):,}, "
          f"{out_path.stat().st_size/1e6:.1f}MB")
    return result


def build_fold_102_full(seed: int = 42) -> pl.DataFrame:
    """Build fold 102 without negative downsampling for unbiased tuning validation."""
    history, ground_truth, eval_customers = build_fold(102)
    hist_df = history.collect()
    history_lf = hist_df.lazy()
    articles_df = load_articles()
    customers_df = load_customers()

    result = _compute_features_inner(
        history_lf, ground_truth, eval_customers,
        articles_df, customers_df, fold_week=102,
        neg_sample_rate=1.0, seed=seed,
    )

    features_dir = PROCESSED_DIR / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    out_path = features_dir / "fold_102_full.parquet"
    result.write_parquet(out_path, compression="zstd")
    print(f"  fold_102_full: {len(result):,} rows, positives={int((result['label']==1).sum()):,}, "
          f"{out_path.stat().st_size/1e6:.1f}MB")
    return result


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
    seed:
        Random seed for negative downsampling.
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
        result = _compute_features_inner(
            history_lf, ground_truth, eval_customers,
            articles_df, customers_df, fold_week,
            neg_sample_rate=1.0, seed=seed,
        )
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
