"""Shared schema constants and validation for all candidate sources."""
import polars as pl

OUTPUT_COLUMNS: list[str] = [
    "customer_idx",
    "article_idx",
    "source",
    "score",
    "source_rank",
]

OUTPUT_DTYPES: dict[str, pl.DataType] = {
    "customer_idx": pl.Int32,
    "article_idx": pl.Int32,
    "source": pl.Utf8,
    "score": pl.Float32,
    "source_rank": pl.Int16,
}


def validate_candidates(df: pl.DataFrame, k: int, source_name: str) -> None:
    """Assert that df conforms to the candidate output schema and per-customer constraints."""
    # All required columns present
    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    assert not missing, (
        f"[{source_name}] Missing columns: {missing}"
    )

    # Correct dtypes
    for col, expected in OUTPUT_DTYPES.items():
        actual = df.schema[col]
        assert actual == expected, (
            f"[{source_name}] Column '{col}' has dtype {actual}, expected {expected}"
        )

    # No (customer_idx, article_idx) duplicates
    n_rows = len(df)
    n_unique = df.select(["customer_idx", "article_idx"]).n_unique()
    assert n_rows == n_unique, (
        f"[{source_name}] Found {n_rows - n_unique} duplicate (customer_idx, article_idx) pairs"
    )

    # Max source_rank per customer <= k
    if n_rows > 0:
        max_rank = (
            df.group_by("customer_idx")
            .agg(pl.col("source_rank").max().alias("max_rank"))
            .select("max_rank")
            .max()
            .item()
        )
        assert max_rank <= k, (
            f"[{source_name}] Max source_rank {max_rank} exceeds k={k}"
        )
