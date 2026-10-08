"""Regression test: univariate MAP@12 of final_rank on fold 103.

Ranking each customer's 200 candidates by final_rank ASC reproduces the
Session 2 heuristic MAP@12 = 0.024675.  This test guards against a repeat
of the denominator bug (averaging only over customers with ≥1 hit rather
than all eval customers).

Requires data/processed/features/fold_103.parquet to exist.
"""
import pytest
import polars as pl
from pathlib import Path

from src.metrics import map_at_k
from src.time_split import build_fold
from src.config import PROCESSED_DIR

FOLD_103_PATH = PROCESSED_DIR / "features" / "fold_103.parquet"

_HEURISTIC_MAP12_FOLD103 = 0.024675292792042714  # from reports/candidates/fold_103_results.json


@pytest.mark.skipif(
    not FOLD_103_PATH.exists(),
    reason="fold_103.parquet not present — run scripts/build_features.py first",
)
def test_univariate_final_rank_map12_matches_heuristic():
    """Univariate MAP@12 of final_rank (ascending) on fold 103 equals heuristic MAP@12.

    The heuristic orders candidates by [final_rank ASC, article_idx ASC] and
    takes the first 12.  Ranking the same candidates in univariate fashion by
    final_rank ASC must reproduce the same MAP@12 = 0.024675 to 4 decimal places.

    If this test fails with a much higher value (e.g. 0.091), the denominator
    in the univariate MAP@12 computation is dividing by customers-with-hits
    rather than all eval customers.
    """
    df = pl.read_parquet(FOLD_103_PATH)

    _, gt_dict, eval_customers = build_fold(103)

    # Replicate the heuristic: sort by final_rank ASC, then article_idx ASC
    df_sorted = df.sort(["customer_idx", "final_rank", "article_idx"])

    # Row number within each customer group (1-indexed, preserves sort order)
    df_ranked = df_sorted.with_columns(
        pl.col("article_idx").cum_count().over("customer_idx").alias("_rnum")
    )

    top12 = df_ranked.filter(pl.col("_rnum") <= 12)

    preds_raw = (
        top12.group_by("customer_idx")
        .agg(pl.col("article_idx").alias("articles"))
    )

    predictions: dict[int, list[int]] = {
        row["customer_idx"]: row["articles"]
        for row in preds_raw.iter_rows(named=True)
    }
    for c in eval_customers:
        if c not in predictions:
            predictions[c] = []

    result = map_at_k(predictions, gt_dict, k=12)

    assert abs(result - _HEURISTIC_MAP12_FOLD103) < 1e-4, (
        f"Univariate final_rank MAP@12 = {result:.6f}; "
        f"expected {_HEURISTIC_MAP12_FOLD103:.6f} (heuristic). "
        "Denominator must be ALL eval customers, not only those with hits."
    )
