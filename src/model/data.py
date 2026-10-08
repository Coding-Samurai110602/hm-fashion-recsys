"""Load feature fold parquets and prepare LightGBM arrays."""
from __future__ import annotations

import numpy as np
import polars as pl

from src.config import HOLDOUT_WEEK, PROCESSED_DIR
from src.features.registry import FEATURE_LIST

_FEATURES_DIR = PROCESSED_DIR / "features"

FEATURE_NAMES: list[str] = [spec.name for spec in FEATURE_LIST]

# Integer-coded categorical columns to declare as categorical in LightGBM.
# Only ordinal-encoded columns with non-negative integer values qualify.
# Customer flag columns (c_FN, c_Active, c_club_member_status, c_fashion_news_frequency)
# use -1 as a null sentinel, which is not a valid category index, so they are excluded.
CATEGORICAL_FEATURES: list[str] = [
    "a_product_group_name",
    "a_index_group_name",
    "a_garment_group_name",
    "a_colour_group_name",
    "a_section_name",
]


def load_fold(fold_week: int) -> pl.DataFrame:
    """Load a feature parquet for a validation fold or holdout."""
    path = _FEATURES_DIR / f"fold_{fold_week}.parquet"
    return pl.read_parquet(path)


def prepare_dataset(
    df: pl.DataFrame,
    drop_zero_positive_groups: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a fold DataFrame to (X, y, groups) for LightGBM.

    Rows are sorted by customer_idx so groups are contiguous. All feature
    values are cast to float32 (nulls → NaN, which LightGBM handles as
    missing). Group sizes correspond to the sorted customer order.

    Parameters
    ----------
    df:
        Feature fold DataFrame.
    drop_zero_positive_groups:
        If True, remove entire customers that have zero positive labels.
        Only meaningful for training folds.

    Returns
    -------
    X : np.ndarray (n_rows, n_features), float32
    y : np.ndarray (n_rows,), int32  — 0/1 labels
    groups : np.ndarray (n_groups,), int32 — group sizes in customer order
    """
    df = df.sort("customer_idx")

    if drop_zero_positive_groups:
        has_pos = (
            df.group_by("customer_idx")
            .agg((pl.col("label") == 1).any().alias("has_pos"))
            .filter(pl.col("has_pos"))
            .select("customer_idx")
        )
        df = df.join(has_pos, on="customer_idx", how="inner").sort("customer_idx")

    # Assert contiguity: each customer's rows must be contiguous before we build groups.
    # After sort("customer_idx") this is always true, but fail loudly if something
    # upstream breaks the invariant (e.g., a join that corrupts row order).
    _cidx = df["customer_idx"].to_numpy()
    if len(_cidx) > 1:
        assert (_cidx[:-1] <= _cidx[1:]).all(), (
            "prepare_dataset: customer_idx is not monotonically non-decreasing after sort — "
            "groups would be wrong. Ensure input DataFrame has no corrupted customer_idx values."
        )

    # Cast all features to Float32 (nulls become NaN, safe for LightGBM)
    X = (
        df.select(FEATURE_NAMES)
        .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
        .to_numpy(allow_copy=True)
    )
    y = df["label"].to_numpy(allow_copy=True).astype(np.int32)

    # Group sizes: number of candidates per customer in sorted order
    groups = (
        df.group_by("customer_idx", maintain_order=True)
        .agg(pl.len().alias("sz"))
        ["sz"]
        .to_numpy(allow_copy=True)
        .astype(np.int32)
    )

    return X, y, groups


def concat_folds(fold_weeks: list[int]) -> pl.DataFrame:
    """Load and vertically concatenate multiple fold DataFrames."""
    frames = [load_fold(w) for w in fold_weeks]
    return pl.concat(frames, how="vertical")


def count_groups_dropped(df: pl.DataFrame) -> dict:
    """Count customers and rows removed by drop_zero_positive_groups."""
    total_groups = df["customer_idx"].n_unique()
    total_rows = len(df)
    df_sorted = df.sort("customer_idx")
    has_pos = (
        df_sorted.group_by("customer_idx")
        .agg((pl.col("label") == 1).any().alias("has_pos"))
        .filter(pl.col("has_pos"))
        .select("customer_idx")
    )
    kept_rows = len(df_sorted.join(has_pos, on="customer_idx", how="inner"))
    return {
        "total_groups": total_groups,
        "kept_groups": int(has_pos.height),
        "dropped_groups": total_groups - int(has_pos.height),
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "dropped_rows": total_rows - kept_rows,
    }
