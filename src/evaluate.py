"""Candidate evaluation: recall, coverage, overlap, segment breakdowns.

All recall computations use polars joins (no Python loops over millions of rows).
"""
from __future__ import annotations

import polars as pl

from src.config import RECALL_NS, SEGMENT_LABELS
from src.metrics import map_at_k


def _gt_to_df(ground_truth: dict[int, set[int]]) -> pl.DataFrame:
    """Convert ground_truth dict to a polars DataFrame of (customer_idx, article_idx)."""
    customers = []
    articles = []
    for c, articles_set in ground_truth.items():
        for a in articles_set:
            customers.append(c)
            articles.append(a)
    return pl.DataFrame({
        "customer_idx": pl.Series(customers, dtype=pl.Int32),
        "article_idx": pl.Series(articles, dtype=pl.Int32),
    })


def _top_n_candidates(merged_df: pl.DataFrame, n: int) -> pl.DataFrame:
    """Return the top-n candidates per customer (sorted by final_rank)."""
    return merged_df.filter(pl.col("final_rank") <= n).select(["customer_idx", "article_idx"])


def candidate_recall_at_n(
    merged_df: pl.DataFrame,
    ground_truth: dict[int, set[int]],
    n: int,
) -> float:
    """Micro recall@N via polars join: fraction of (customer, article) GT pairs in top-N candidates."""
    if not ground_truth or len(merged_df) == 0:
        return 0.0
    gt_df = _gt_to_df(ground_truth)
    top_n = _top_n_candidates(merged_df, n)
    hits = gt_df.join(top_n, on=["customer_idx", "article_idx"], how="inner")
    return len(hits) / len(gt_df)


def evaluate_candidates(
    merged_df: pl.DataFrame,
    ground_truth: dict[int, set[int]],
    ns: list[int] | None = None,
) -> dict:
    """Compute recall@N, coverage, and mean candidates per customer."""
    if ns is None:
        ns = RECALL_NS

    eval_customers = set(ground_truth.keys())
    n_eval = len(eval_customers)
    if n_eval == 0:
        return {"coverage": 0.0, "mean_cands_per_customer": 0.0, "n_eval_customers": 0,
                **{f"recall@{n}": 0.0 for n in ns}}

    cust_with_cands = set(merged_df["customer_idx"].unique().to_list())
    coverage = len(eval_customers & cust_with_cands) / n_eval

    cands_per_cust_mean = (
        merged_df
        .filter(pl.col("customer_idx").is_in(list(eval_customers)))
        .group_by("customer_idx")
        .agg(pl.len().alias("n_cands"))
        ["n_cands"]
        .mean()
    )

    # Precompute gt_df once and reuse for all N values
    gt_df = _gt_to_df(ground_truth)
    total_gt = len(gt_df)

    recalls = {}
    for n in ns:
        top_n = _top_n_candidates(merged_df, n)
        hits = gt_df.join(top_n, on=["customer_idx", "article_idx"], how="inner")
        recalls[f"recall@{n}"] = len(hits) / total_gt if total_gt > 0 else 0.0

    return {
        "coverage": coverage,
        "mean_cands_per_customer": float(cands_per_cust_mean) if cands_per_cust_mean is not None else 0.0,
        "n_eval_customers": n_eval,
        **recalls,
    }


def source_recall_alone(
    source_df: pl.DataFrame,
    ground_truth: dict[int, set[int]],
    source_name: str,
    ns: list[int] | None = None,
) -> dict:
    """Evaluate a single source in isolation."""
    if ns is None:
        ns = RECALL_NS

    eval_customers = set(ground_truth.keys())
    cust_with_cands = set(source_df["customer_idx"].unique().to_list())
    coverage = len(eval_customers & cust_with_cands) / len(eval_customers) if eval_customers else 0.0

    gt_df = _gt_to_df(ground_truth)
    total_gt = len(gt_df)
    # Treat source as a ranked list: source_rank determines position
    # Rename source_rank to final_rank for reuse with _top_n_candidates
    source_as_merged = source_df.rename({"source_rank": "final_rank"})

    recalls = {}
    for n in ns:
        top_n = _top_n_candidates(source_as_merged, n)
        hits = gt_df.join(top_n, on=["customer_idx", "article_idx"], how="inner")
        recalls[f"recall@{n}"] = len(hits) / total_gt if total_gt > 0 else 0.0

    return {"source": source_name, "coverage": coverage, **recalls}


def incremental_recall(
    source_dfs: dict[str, pl.DataFrame],
    ground_truth: dict[int, set[int]],
    priority: list[str],
    n: int,
    ns: list[int] | None = None,
) -> list[dict]:
    """Add sources one at a time and measure marginal recall gain."""
    from src.candidates.merge import merge_candidates

    if ns is None:
        ns = RECALL_NS

    gt_df = _gt_to_df(ground_truth)
    total_gt = len(gt_df)
    customers = sorted(ground_truth.keys())

    results = []
    cumulative: dict[str, pl.DataFrame] = {}
    prev_recalls: dict[str, float] = {f"recall@{nval}": 0.0 for nval in ns}

    for src in priority:
        if src not in source_dfs:
            continue
        cumulative[src] = source_dfs[src]
        merged = merge_candidates(cumulative, customers, n, priority)
        entry = {"source_added": src}
        for nval in ns:
            top_n = _top_n_candidates(merged, nval)
            hits = gt_df.join(top_n, on=["customer_idx", "article_idx"], how="inner")
            r = len(hits) / total_gt if total_gt > 0 else 0.0
            entry[f"recall@{nval}"] = r
            entry[f"gain@{nval}"] = r - prev_recalls[f"recall@{nval}"]
        prev_recalls = {f"recall@{nval}": entry[f"recall@{nval}"] for nval in ns}
        results.append(entry)

    return results


def pairwise_jaccard(
    source_dfs: dict[str, pl.DataFrame],
    max_rank: int = 200,
) -> pl.DataFrame:
    """Compute pairwise Jaccard similarity of candidate (customer_idx, article_idx) pairs."""
    sources = list(source_dfs.keys())
    # Build set of (customer_idx, article_idx) pairs per source using polars
    source_pairs: dict[str, set[tuple[int, int]]] = {}
    for src, df in source_dfs.items():
        filtered = df.filter(pl.col("source_rank") <= max_rank)
        source_pairs[src] = set(
            zip(
                filtered["customer_idx"].to_list(),
                filtered["article_idx"].to_list(),
            )
        )

    rows = []
    for i, s1 in enumerate(sources):
        for j, s2 in enumerate(sources):
            if j <= i:
                continue
            p1, p2 = source_pairs[s1], source_pairs[s2]
            inter = len(p1 & p2)
            union = len(p1 | p2)
            rows.append({
                "source_a": s1,
                "source_b": s2,
                "jaccard": inter / union if union > 0 else 0.0,
                "intersection": inter,
                "union": union,
            })
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"source_a": pl.Utf8, "source_b": pl.Utf8, "jaccard": pl.Float64,
                "intersection": pl.Int64, "union": pl.Int64}
    )


def customer_segments(
    history: pl.LazyFrame,
    cutoff_week: int,
    eval_customers: list[int],
) -> pl.DataFrame:
    """Return customer_idx -> segment label based on prior purchase count."""
    prior_counts = (
        history
        .filter(
            pl.col("customer_idx").is_in(eval_customers)
            & (pl.col("week_idx") < cutoff_week)
        )
        .group_by("customer_idx")
        .agg(pl.len().alias("prior_count"))
        .collect()
    )

    all_customers_df = pl.DataFrame(
        {"customer_idx": pl.Series(eval_customers, dtype=pl.Int32)}
    )
    return (
        all_customers_df
        .join(prior_counts, on="customer_idx", how="left")
        .with_columns(pl.col("prior_count").fill_null(0))
        .with_columns(
            pl.when(pl.col("prior_count") == 0)
            .then(pl.lit("0"))
            .when(pl.col("prior_count") <= 4)
            .then(pl.lit("1-4"))
            .when(pl.col("prior_count") <= 19)
            .then(pl.lit("5-19"))
            .otherwise(pl.lit("20+"))
            .alias("segment")
        )
    )


def recall_by_segment(
    merged_df: pl.DataFrame,
    ground_truth: dict[int, set[int]],
    segments_df: pl.DataFrame,
    ns: list[int] | None = None,
) -> pl.DataFrame:
    """Compute recall@N broken down by customer segment."""
    if ns is None:
        ns = RECALL_NS

    gt_df = _gt_to_df(ground_truth)

    rows = []
    for seg in ["0", "1-4", "5-19", "20+"]:
        seg_customers = set(
            segments_df.filter(pl.col("segment") == seg)["customer_idx"].to_list()
        )
        seg_gt_df = gt_df.filter(pl.col("customer_idx").is_in(list(seg_customers)))
        total_seg_gt = len(seg_gt_df)
        seg_merged = merged_df.filter(pl.col("customer_idx").is_in(list(seg_customers)))

        row: dict = {"segment": seg, "n_customers": len(seg_customers)}
        for n in ns:
            if total_seg_gt == 0:
                row[f"recall@{n}"] = float("nan")
            else:
                top_n = _top_n_candidates(seg_merged, n)
                hits = seg_gt_df.join(top_n, on=["customer_idx", "article_idx"], how="inner")
                row[f"recall@{n}"] = len(hits) / total_seg_gt
        rows.append(row)

    return pl.DataFrame(rows)


def heuristic_map_at_12(
    source_dfs: dict[str, pl.DataFrame],
    ground_truth: dict[int, set[int]],
    priority: list[str],
    n: int = 200,
) -> float:
    """Rank merged candidates by priority + score, compute MAP@12."""
    from src.candidates.merge import merge_candidates

    customers = sorted(ground_truth.keys())
    merged = merge_candidates(source_dfs, customers, n, priority)

    # Build predictions as polars: top-12 per customer
    top12 = (
        merged
        .filter(pl.col("final_rank") <= 12)
        .sort(["customer_idx", "final_rank"])
        .group_by("customer_idx")
        .agg(pl.col("article_idx").alias("articles"))
    )

    predictions: dict[int, list[int]] = {
        row["customer_idx"]: row["articles"]
        for row in top12.iter_rows(named=True)
    }

    return map_at_k(predictions, ground_truth, k=12)


def oracle_map_at_12(
    merged_df: pl.DataFrame,
    ground_truth: dict[int, set[int]],
) -> dict:
    """Oracle MAP@12: GT items ranked first in the candidate pool.

    Upper bound for any ranker working within this candidate set.
    Also returns candidate precision = mean fraction of candidates that are GT items.

    Returns keys: oracle_map12, candidate_precision_mean, total_hits,
                  total_candidates, n_customers.
    """
    top_all = (
        merged_df
        .sort(["customer_idx", "final_rank"])
        .group_by("customer_idx")
        .agg(pl.col("article_idx").alias("articles"))
    )
    cands_dict: dict[int, list[int]] = {
        row["customer_idx"]: row["articles"]
        for row in top_all.iter_rows(named=True)
    }

    total_oracle_ap = 0.0
    total_precision = 0.0
    total_hits = 0
    total_cands = 0
    n_customers = len(ground_truth)

    for cust, gt in ground_truth.items():
        cands = cands_dict.get(cust, [])
        hits = [a for a in cands if a in gt]
        n_hits = len(hits)
        n_cands = len(cands)
        h = min(n_hits, 12)
        ap_oracle = h / min(len(gt), 12) if gt else 0.0
        total_oracle_ap += ap_oracle
        total_precision += n_hits / n_cands if n_cands > 0 else 0.0
        total_hits += n_hits
        total_cands += n_cands

    return {
        "oracle_map12": total_oracle_ap / n_customers if n_customers > 0 else 0.0,
        "candidate_precision_mean": total_precision / n_customers if n_customers > 0 else 0.0,
        "total_hits": total_hits,
        "total_candidates": total_cands,
        "n_customers": n_customers,
    }
