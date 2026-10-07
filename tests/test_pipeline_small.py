"""Integration test: full pipeline on a 1% customer sample (seed 42), fold 103.

Asserts: completes, coverage = 100%, recall in [0, 1], schema correct.
"""
import random

import polars as pl
import pytest

from src.config import MERGE_N, MERGE_PRIORITY, RECALL_NS, SOURCE_K
from src.time_split import build_fold
from src.candidates.repurchase import generate as repurchase_gen
from src.candidates.product_code import generate as product_code_gen
from src.candidates.popularity import (
    generate_global_last_week,
    generate_global_decayed,
    generate_segment_popular,
)
from src.candidates.copurchase import generate as copurchase_gen
from src.candidates.merge import merge_candidates
from src.evaluate import evaluate_candidates

FOLD_WEEK = 103
SEED = 42
SAMPLE_FRACTION = 0.01


@pytest.fixture(scope="module")
def pipeline_result():
    """Run the full pipeline on a 1% sample. Shared across all tests in this module."""
    history, ground_truth, eval_customers = build_fold(FOLD_WEEK)

    rng = random.Random(SEED)
    n_sample = max(1, int(len(eval_customers) * SAMPLE_FRACTION))
    sampled_customers = sorted(rng.sample(eval_customers, n_sample))
    sampled_gt = {c: ground_truth[c] for c in sampled_customers}

    source_dfs = {}

    source_dfs["repurchase"] = repurchase_gen(
        history, FOLD_WEEK, sampled_customers, SOURCE_K["repurchase"]
    )
    source_dfs["product_code"] = product_code_gen(
        history, FOLD_WEEK, sampled_customers, SOURCE_K["product_code"]
    )
    source_dfs["popularity_last_week"] = generate_global_last_week(
        history, FOLD_WEEK, sampled_customers, SOURCE_K["popularity_last_week"]
    )
    source_dfs["popularity_decayed"] = generate_global_decayed(
        history, FOLD_WEEK, sampled_customers, SOURCE_K["popularity_decayed"]
    )
    source_dfs["segment_popular"] = generate_segment_popular(
        history, FOLD_WEEK, sampled_customers, SOURCE_K["segment_popular"]
    )
    source_dfs["copurchase"] = copurchase_gen(
        history, FOLD_WEEK, sampled_customers, SOURCE_K["copurchase"]
    )

    merged = merge_candidates(source_dfs, sampled_customers, MERGE_N, MERGE_PRIORITY)
    metrics = evaluate_candidates(merged, sampled_gt, RECALL_NS)

    return merged, sampled_gt, sampled_customers, source_dfs, metrics


def test_pipeline_completes(pipeline_result):
    """Pipeline runs without exception; merged DataFrame is non-empty."""
    merged, _, _, _, _ = pipeline_result
    assert len(merged) > 0, "Merged candidates DataFrame is empty"


def test_coverage_100_percent(pipeline_result):
    """Every eval customer in the sample has at least 1 candidate after merge."""
    merged, _, sampled_customers, _, _ = pipeline_result
    cust_with_cands = set(merged["customer_idx"].unique().to_list())
    missing = set(sampled_customers) - cust_with_cands
    assert len(missing) == 0, (
        f"{len(missing)} customers have no candidates: {list(missing)[:5]}"
    )


def test_recall_in_range(pipeline_result):
    """All recall@N values are in [0.0, 1.0]."""
    _, _, _, _, metrics = pipeline_result
    for key, val in metrics.items():
        if key.startswith("recall@"):
            assert 0.0 <= val <= 1.0, f"{key} = {val} is out of [0, 1]"


def test_merged_schema(pipeline_result):
    """Merged DataFrame has required columns with correct base dtypes."""
    merged, _, _, _, _ = pipeline_result
    assert "customer_idx" in merged.columns
    assert "article_idx" in merged.columns
    assert "final_source" in merged.columns
    assert "final_rank" in merged.columns
    assert merged.schema["customer_idx"] == pl.Int32
    assert merged.schema["article_idx"] == pl.Int32


def test_no_duplicate_pairs_in_merged(pipeline_result):
    """Merged output has no duplicate (customer_idx, article_idx) pairs."""
    merged, _, _, _, _ = pipeline_result
    n = len(merged)
    n_unique = merged.select(["customer_idx", "article_idx"]).n_unique()
    assert n == n_unique, f"{n - n_unique} duplicate pairs in merged output"


def test_max_candidates_per_customer(pipeline_result):
    """Each customer has at most MERGE_N candidates in the merged output."""
    merged, _, _, _, _ = pipeline_result
    max_n = (
        merged.group_by("customer_idx")
        .agg(pl.len().alias("n_cands"))
        ["n_cands"]
        .max()
    )
    assert max_n <= MERGE_N, f"Customer has {max_n} candidates > MERGE_N={MERGE_N}"


def test_source_columns_present(pipeline_result):
    """Each source contributes a score and rank column in the merged output."""
    merged, _, _, source_dfs, _ = pipeline_result
    for src in source_dfs:
        score_col = f"{src}_score"
        rank_col = f"{src}_rank"
        assert score_col in merged.columns, f"Missing {score_col}"
        assert rank_col in merged.columns, f"Missing {rank_col}"
