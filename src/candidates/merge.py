"""Merge multiple candidate sources into a single ranked candidate set per customer."""
import polars as pl

from src.config import MERGE_PRIORITY


def merge_candidates(
    source_dfs: dict[str, pl.DataFrame],
    customers: list[int],
    n: int,
    priority: list[str] | None = None,
) -> pl.DataFrame:
    """Combine per-source candidates into a deduplicated, priority-filled set of n per customer."""
    if priority is None:
        priority = MERGE_PRIORITY

    # Build priority lookup: source_name -> integer (lower = higher priority)
    priority_map: dict[str, int] = {src: idx for idx, src in enumerate(priority)}

    # Assign priority integer to each source DataFrame and concatenate
    tagged: list[pl.DataFrame] = []
    for source_name, df in source_dfs.items():
        prio = priority_map.get(source_name, len(priority))
        tagged.append(
            df.with_columns(
                pl.lit(prio).cast(pl.Int32).alias("_priority")
            )
        )

    if not tagged:
        # Return an empty frame with the expected schema
        return _empty_output(priority)

    combined = pl.concat(tagged, how="diagonal")

    # Sort globally: customer ASC, priority ASC (lower = better), source_rank ASC.
    # Within each source, source_rank already encodes the source's own ordering rules
    # (e.g., recency_only or production for repurchase; score-descending for popularity).
    # Sorting on source_rank preserves that ordering rather than re-sorting by raw score.
    combined = combined.sort(
        ["customer_idx", "_priority", "source_rank"],
        descending=[False, False, False],
    )

    # Deduplicate: keep first occurrence of each (customer_idx, article_idx) pair
    deduped = combined.unique(
        subset=["customer_idx", "article_idx"],
        keep="first",
        maintain_order=True,
    )

    # Limit to n candidates per customer
    # Compute row number within customer and filter
    deduped = deduped.with_columns(
        pl.cum_count("article_idx")
        .over("customer_idx")
        .cast(pl.Int32)
        .alias("_row_num")
    ).filter(pl.col("_row_num") <= n)

    # Record which source first contributed each candidate
    deduped = deduped.rename({"source": "final_source"})

    # Compute final_rank: row number within customer after dedup (already sorted)
    deduped = deduped.with_columns(
        pl.col("_row_num").cast(pl.Int32).alias("final_rank")
    )

    # Pivot: attach per-source score and rank columns
    # For each source, left-join the original source DataFrame on (customer_idx, article_idx)
    result = deduped.select(
        ["customer_idx", "article_idx", "final_source", "final_rank"]
    )

    for source_name, df in source_dfs.items():
        score_col = f"{source_name}_score"
        rank_col = f"{source_name}_rank"
        source_slim = (
            df
            .select(["customer_idx", "article_idx", "score", "source_rank"])
            .rename({"score": score_col, "source_rank": rank_col})
            .with_columns(
                pl.col(score_col).cast(pl.Float32),
                pl.col(rank_col).cast(pl.Int16),
            )
        )
        result = result.join(
            source_slim, on=["customer_idx", "article_idx"], how="left"
        )

    # Ensure correct dtypes for base columns
    result = result.with_columns(
        pl.col("customer_idx").cast(pl.Int32),
        pl.col("article_idx").cast(pl.Int32),
        pl.col("final_source").cast(pl.Utf8),
        pl.col("final_rank").cast(pl.Int32),
    )

    return result


def _empty_output(priority: list[str]) -> pl.DataFrame:
    """Return an empty merged output DataFrame with the correct schema."""
    schema: dict[str, pl.DataType] = {
        "customer_idx": pl.Int32,
        "article_idx": pl.Int32,
        "final_source": pl.Utf8,
        "final_rank": pl.Int32,
    }
    for src in priority:
        schema[f"{src}_score"] = pl.Float32
        schema[f"{src}_rank"] = pl.Int16
    return pl.DataFrame(schema=schema)
