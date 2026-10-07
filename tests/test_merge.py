"""Tests for merge_candidates: source_rank ordering, priority, and deduplication.

Three synthetic scenarios that the merge fix must satisfy:
  (a) Within a source, final order follows source_rank, not raw score.
  (b) Across sources, MERGE_PRIORITY determines fill order regardless of score.
  (c) Deduplicate: same article in two sources keeps the higher-priority occurrence.
"""
import polars as pl

from src.candidates.merge import merge_candidates


def _make_source(
    customers: list[int],
    articles: list[int],
    scores: list[float],
    source_ranks: list[int],
    name: str,
) -> pl.DataFrame:
    return pl.DataFrame({
        "customer_idx": pl.Series(customers, dtype=pl.Int32),
        "article_idx": pl.Series(articles, dtype=pl.Int32),
        "source": pl.Series([name] * len(customers), dtype=pl.Utf8),
        "score": pl.Series(scores, dtype=pl.Float32),
        "source_rank": pl.Series(source_ranks, dtype=pl.Int16),
    })


def _order(merged: pl.DataFrame, customer: int) -> list[int]:
    return (
        merged.filter(pl.col("customer_idx") == customer)
        .sort("final_rank")["article_idx"]
        .to_list()
    )


def test_merge_follows_source_rank_not_score():
    """Score order disagrees with source_rank: merged output must follow source_rank.

    Article 10 has score=100 (higher) but source_rank=2 (lower priority within source).
    Article 20 has score=50  (lower)  but source_rank=1 (higher priority within source).
    Correct merge: article 20 first (source_rank=1), then article 10 (source_rank=2).
    Bug (old sort on score DESC): article 10 first.
    """
    src = _make_source(
        customers=[1, 1],
        articles=[10, 20],
        scores=[100.0, 50.0],
        source_ranks=[2, 1],
        name="repurchase",
    )
    merged = merge_candidates({"repurchase": src}, customers=[1], n=5)
    assert _order(merged, 1) == [20, 10], (
        f"Expected source_rank order [20,10], got {_order(merged, 1)}. "
        "merge must respect source_rank, not re-sort by score."
    )


def test_merge_priority_across_sources():
    """Higher-priority source articles come before lower-priority articles regardless of score.

    Source A (priority 0): articles 10, 20 with low scores.
    Source B (priority 1): articles 30, 40 with very high scores.
    Correct: A's articles precede B's articles.
    """
    src_a = _make_source([1, 1], [10, 20], [1.0, 1.0], [1, 2], "A")
    src_b = _make_source([1, 1], [30, 40], [999.0, 998.0], [1, 2], "B")
    merged = merge_candidates(
        {"A": src_a, "B": src_b},
        customers=[1], n=10,
        priority=["A", "B"],
    )
    order = _order(merged, 1)
    assert order.index(10) < order.index(30), f"A's art 10 should precede B's art 30; order={order}"
    assert order.index(20) < order.index(40), f"A's art 20 should precede B's art 40; order={order}"


def test_merge_dedup_keeps_higher_priority_source():
    """When article appears in both sources, keep the higher-priority (first) occurrence.

    Article 10 appears in source A (priority 0, source_rank=1) and B (priority 1, source_rank=1).
    After dedup: article 10 appears once, attributed to source A.
    Article 20 appears only in B and is retained after dedup.
    """
    src_a = _make_source([1], [10], [1.0], [1], "A")
    src_b = _make_source([1, 1], [10, 20], [999.0, 1.0], [1, 2], "B")
    merged = merge_candidates(
        {"A": src_a, "B": src_b},
        customers=[1], n=10,
        priority=["A", "B"],
    )
    rows_10 = merged.filter(pl.col("article_idx") == 10)
    assert len(rows_10) == 1, f"Article 10 must appear once; got {len(rows_10)}"
    assert rows_10["final_source"][0] == "A", (
        f"Article 10 must be from source A (higher priority); "
        f"got '{rows_10['final_source'][0]}'"
    )
    # Article 20 must still be present
    assert len(merged.filter(pl.col("article_idx") == 20)) == 1, "Article 20 must be retained"
