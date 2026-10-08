"""Candidate feature computation from merged candidate output."""
import polars as pl


def compute_candidate_features(
    merged_candidates: pl.DataFrame,
    sources: list[str],
) -> pl.DataFrame:
    """Enrich merged candidates with per-source indicator and count features.

    Parameters
    ----------
    merged_candidates:
        Output of merge_candidates(). Must contain customer_idx, article_idx,
        final_rank, and for each source in sources: {source}_rank (Int16) and
        {source}_score (Float32) columns (may be null if source did not produce
        the pair).
    sources:
        Ordered list of source names (e.g. MERGE_PRIORITY).

    Returns
    -------
    pl.DataFrame
        Input DataFrame enriched with:
        - in_{source} (Int8): 1 if {source}_rank is not null, else 0
        - n_sources (Int8): sum of all in_{source} flags
        All {source}_rank and {source}_score columns ensured present (null
        if source did not produce the pair).
    """
    result = merged_candidates.clone()

    # Ensure all source rank/score columns are present; add as null if missing
    for source in sources:
        rank_col = f"{source}_rank"
        score_col = f"{source}_score"
        if rank_col not in result.columns:
            result = result.with_columns(
                pl.lit(None, dtype=pl.Int16).alias(rank_col)
            )
        if score_col not in result.columns:
            result = result.with_columns(
                pl.lit(None, dtype=pl.Float32).alias(score_col)
            )

    # Build in_{source} indicator columns
    in_cols = []
    for source in sources:
        rank_col = f"{source}_rank"
        in_col = f"in_{source}"
        result = result.with_columns(
            pl.when(pl.col(rank_col).is_not_null())
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias(in_col)
        )
        in_cols.append(in_col)

    # n_sources = sum of all in_{source} flags
    n_sources_expr = pl.lit(0, dtype=pl.Int32)
    for col in in_cols:
        n_sources_expr = n_sources_expr + pl.col(col).cast(pl.Int32)

    result = result.with_columns(
        n_sources_expr.cast(pl.Int8).alias("n_sources")
    )

    return result
