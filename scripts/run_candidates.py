"""CLI: generate candidates for validation folds and write results.

Usage:
    python scripts/run_candidates.py --folds 100 101 102 103 --n 20 50 100 200
    python scripts/run_candidates.py --folds 103 --scale   # all 1.37M customers
"""
import sys
from pathlib import Path
# Allow `python scripts/run_candidates.py` without PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import resource
import time
from pathlib import Path

import polars as pl

from src.config import (
    CANDIDATES_DIR,
    CANDIDATES_PARQUET_DIR,
    FIGURES_CANDIDATES_DIR,
    MERGE_N,
    MERGE_PRIORITY,
    MIN_COPURCHASE_COUNT,
    RECALL_NS,
    SOURCE_K,
    VALIDATION_WEEKS,
    HOLDOUT_WEEK,
)
from src.data_io import load_transactions
from src.time_split import build_fold, build_holdout_fold
from src.candidates.repurchase import generate as repurchase_gen
from src.candidates.product_code import generate as product_code_gen
from src.candidates.popularity import (
    generate_global_last_week,
    generate_global_decayed,
    generate_segment_popular,
)
from src.candidates.copurchase import generate as copurchase_gen
from src.candidates.merge import merge_candidates
from src.evaluate import (
    evaluate_candidates,
    incremental_recall,
    oracle_map_at_12,
    source_recall_alone,
    customer_segments,
    recall_by_segment,
    heuristic_map_at_12,
)
from src.metrics import map_at_k

# Ensure output directories exist
CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATES_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)


def _rss_mb() -> float:
    """Current process RSS in MB. On macOS ru_maxrss is in bytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def _candidate_count_stats(merged: pl.DataFrame, n_budget: int) -> dict:
    """Compute per-customer candidate count statistics."""
    counts = (
        merged.group_by("customer_idx")
        .agg(pl.len().alias("n_cands"))
        ["n_cands"]
    )
    n_at_budget = int((counts == n_budget).sum())
    n_total = len(counts)
    return {
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "p10": float(counts.quantile(0.10, interpolation="nearest")),
        "p90": float(counts.quantile(0.90, interpolation="nearest")),
        "pct_at_budget": round(100.0 * n_at_budget / n_total, 2) if n_total > 0 else 0.0,
        "budget": n_budget,
    }


def run_fold(
    fold_week: int,
    ns: list[int],
    scale: bool = False,
    verbose: bool = True,
) -> dict:
    """Run full candidate pipeline for one fold. Returns metrics dict."""
    t_fold_start = time.time()
    rss_start_mb = _rss_mb()
    log = lambda msg: print(f"  [{fold_week}] {msg}") if verbose else None

    if fold_week == HOLDOUT_WEEK:
        history, ground_truth, eval_customers = build_holdout_fold()
    else:
        history, ground_truth, eval_customers = build_fold(fold_week)

    log(f"Fold loaded: {len(eval_customers)} eval customers, {len(ground_truth)} GT")

    if scale:
        # Use all customers in the full transaction history (not just eval_customers)
        all_customers = (
            load_transactions()
            .filter(pl.col("week_idx") < fold_week)
            .select("customer_idx")
            .unique()
            .collect()
            ["customer_idx"]
            .to_list()
        )
        customers = sorted(all_customers)
        log(f"Scale mode: {len(customers):,} customers")
    else:
        customers = eval_customers

    # -----------------------------------------------------------------------
    # Generate candidates from each source
    # -----------------------------------------------------------------------
    log("Generating repurchase candidates...")
    t0 = time.time()
    rep_df = repurchase_gen(history, fold_week, customers, SOURCE_K["repurchase"])
    log(f"  repurchase: {len(rep_df):,} rows, {rep_df['customer_idx'].n_unique():,} customers ({time.time()-t0:.1f}s)")

    log("Generating product_code candidates...")
    t0 = time.time()
    pc_df = product_code_gen(history, fold_week, customers, SOURCE_K["product_code"])
    log(f"  product_code: {len(pc_df):,} rows, {pc_df['customer_idx'].n_unique():,} customers ({time.time()-t0:.1f}s)")

    log("Generating popularity_last_week candidates...")
    t0 = time.time()
    pop_lw_df = generate_global_last_week(history, fold_week, customers, SOURCE_K["popularity_last_week"])
    log(f"  popularity_last_week: {len(pop_lw_df):,} rows ({time.time()-t0:.1f}s)")

    log("Generating popularity_decayed candidates...")
    t0 = time.time()
    pop_dec_df = generate_global_decayed(history, fold_week, customers, SOURCE_K["popularity_decayed"])
    log(f"  popularity_decayed: {len(pop_dec_df):,} rows ({time.time()-t0:.1f}s)")

    log("Generating segment_popular candidates...")
    t0 = time.time()
    seg_df = generate_segment_popular(history, fold_week, customers, SOURCE_K["segment_popular"])
    log(f"  segment_popular: {len(seg_df):,} rows ({time.time()-t0:.1f}s)")

    log("Generating copurchase candidates...")
    t0 = time.time()
    cop_df = copurchase_gen(history, fold_week, customers, SOURCE_K["copurchase"], min_count=MIN_COPURCHASE_COUNT)
    log(f"  copurchase: {len(cop_df):,} rows, {cop_df['customer_idx'].n_unique():,} customers ({time.time()-t0:.1f}s)")

    source_dfs = {
        "repurchase": rep_df,
        "product_code": pc_df,
        "popularity_last_week": pop_lw_df,
        "popularity_decayed": pop_dec_df,
        "segment_popular": seg_df,
        "copurchase": cop_df,
    }

    # -----------------------------------------------------------------------
    # Merge
    # -----------------------------------------------------------------------
    log("Merging candidates...")
    t0 = time.time()
    merged = merge_candidates(source_dfs, customers, MERGE_N, MERGE_PRIORITY)
    log(f"  merged: {len(merged):,} rows, {merged['customer_idx'].n_unique():,} customers ({time.time()-t0:.1f}s)")

    # -----------------------------------------------------------------------
    # Evaluate (only for eval_customers, not the full scale set)
    # -----------------------------------------------------------------------
    metrics = {}
    if not scale:
        log("Evaluating...")
        merged_eval = merged.filter(pl.col("customer_idx").is_in(eval_customers))

        # Per-source recall alone
        source_recalls = {}
        for src_name, src_df in source_dfs.items():
            src_eval = src_df.filter(pl.col("customer_idx").is_in(eval_customers))
            source_recalls[src_name] = source_recall_alone(src_eval, ground_truth, src_name, ns)

        # Merged recall
        merged_metrics = evaluate_candidates(merged_eval, ground_truth, ns)

        # Incremental recall
        incremental = incremental_recall(
            {k: v.filter(pl.col("customer_idx").is_in(eval_customers)) for k, v in source_dfs.items()},
            ground_truth,
            MERGE_PRIORITY,
            MERGE_N,
            ns,
        )

        # Segment breakdown
        segs = customer_segments(history, fold_week, eval_customers)
        seg_recall = recall_by_segment(merged_eval, ground_truth, segs, ns)

        # Heuristic MAP@12
        heuristic_map = heuristic_map_at_12(
            {k: v.filter(pl.col("customer_idx").is_in(eval_customers)) for k, v in source_dfs.items()},
            ground_truth,
            MERGE_PRIORITY,
            MERGE_N,
        )

        # Oracle MAP@12 and candidate precision (reproducible here; no separate one-off script needed)
        oracle_metrics = oracle_map_at_12(merged_eval, ground_truth)

        # Pairwise Jaccard overlap between source candidate sets
        from src.evaluate import pairwise_jaccard as _pairwise_jaccard
        jaccard_df = _pairwise_jaccard(
            {k: v.filter(pl.col("customer_idx").is_in(eval_customers)) for k, v in source_dfs.items()},
            max_rank=50,
        )

        cand_stats = _candidate_count_stats(merged_eval, MERGE_N)
        metrics = {
            "fold_week": fold_week,
            "n_eval_customers": len(eval_customers),
            "source_recalls": source_recalls,
            "merged": merged_metrics,
            "candidate_counts": cand_stats,
            "incremental": incremental,
            "segment_recall": seg_recall.to_dicts(),
            "jaccard": jaccard_df.to_dicts(),
            "heuristic_map12": heuristic_map,
            "oracle_map12": oracle_metrics["oracle_map12"],
            "candidate_precision_mean": oracle_metrics["candidate_precision_mean"],
            "rss_delta_mb": round(_rss_mb() - rss_start_mb, 1),
            "runtime_s": time.time() - t_fold_start,
        }
        log(f"recall@200={merged_metrics.get('recall@200', 0):.4f}, MAP@12={heuristic_map:.6f}, "
            f"oracle@12={oracle_metrics['oracle_map12']:.4f}, "
            f"mean_cands={cand_stats['mean']:.1f}, pct_at_budget={cand_stats['pct_at_budget']:.1f}%, "
            f"runtime={metrics['runtime_s']:.1f}s")

    # -----------------------------------------------------------------------
    # Write scale-test parquet (slim schema: drop large string columns to save disk)
    # -----------------------------------------------------------------------
    if scale:
        out_path = CANDIDATES_PARQUET_DIR / f"candidates_week{fold_week}.parquet"
        log(f"Writing scale parquet -> {out_path}")
        slim_cols = ["customer_idx", "article_idx", "final_source", "final_rank"]
        merged.select(slim_cols).write_parquet(out_path, compression="zstd")
        file_size_mb = out_path.stat().st_size / 1024 / 1024
        cand_stats = _candidate_count_stats(merged, MERGE_N)
        rss_mb = _rss_mb()
        log(f"  Written: {len(merged):,} rows, {file_size_mb:.1f} MB, "
            f"mean_cands={cand_stats['mean']:.1f}, pct_at_budget={cand_stats['pct_at_budget']:.1f}%, "
            f"rss={rss_mb:.0f} MB, runtime {time.time()-t_fold_start:.1f}s")
        metrics = {
            "fold_week": fold_week,
            "n_customers": merged["customer_idx"].n_unique(),
            "n_rows": len(merged),
            "file_size_mb": file_size_mb,
            "candidate_counts": cand_stats,
            "peak_rss_mb": round(rss_mb, 1),
            "note_memory": "peak_rss_mb is process RSS (resource.getrusage); captures Polars/DuckDB native memory unlike tracemalloc",
            "runtime_s": time.time() - t_fold_start,
        }

    return metrics, merged, source_dfs, ground_truth, eval_customers


def regression_check(verbose: bool = True) -> dict:
    """Reproduce Baseline B MAP@12 = 0.024457 on holdout week 104.

    Recency-only pipeline:
    - repurchase(recency_only, k=12): last_date DESC, article_idx ASC — matches EDA Baseline B
    - popularity_last_week(k=12): prev-week top-12 — matches EDA Baseline A fill
    - merge n=12 → MAP@12

    Production pipeline:
    - repurchase(production, k=50): last_date DESC, purchase_count DESC, article_idx ASC
    - same popularity fill
    - merge n=12 → MAP@12 (now genuinely different from recency_only after the merge fix)

    Also computes before/after-merge ordering comparison for the record.
    """
    history, ground_truth, eval_customers = build_holdout_fold()

    # --- Recency-only (EDA-matching) pipeline ---
    rep_recency_k12 = repurchase_gen(
        history, HOLDOUT_WEEK, eval_customers, k=12, ordering="recency_only"
    )
    pop_fill = generate_global_last_week(
        history, HOLDOUT_WEEK, eval_customers, k=12
    )
    merged_recency = merge_candidates(
        {"repurchase": rep_recency_k12, "popularity_last_week": pop_fill},
        eval_customers, n=12,
        priority=["repurchase", "popularity_last_week"],
    )

    def _preds(merged_df: pl.DataFrame) -> dict[int, list[int]]:
        return {
            row["customer_idx"]: row["articles"]
            for row in (
                merged_df
                .filter(pl.col("final_rank") <= 12)
                .sort(["customer_idx", "final_rank"])
                .group_by("customer_idx")
                .agg(pl.col("article_idx").alias("articles"))
                .iter_rows(named=True)
            )
        }

    def _ranked_source_dict(df: pl.DataFrame) -> dict[int, list[int]]:
        return {
            row["customer_idx"]: row["articles"]
            for row in (
                df.sort(["customer_idx", "source_rank"])
                .group_by("customer_idx")
                .agg(pl.col("article_idx").alias("articles"))
                .iter_rows(named=True)
            )
        }

    predictions_recency = _preds(merged_recency)
    measured_map = map_at_k(predictions_recency, ground_truth, k=12)

    # --- Production ordering pipeline ---
    rep_prod_k12 = repurchase_gen(
        history, HOLDOUT_WEEK, eval_customers, k=12, ordering="production"
    )
    rep_prod_k50 = repurchase_gen(
        history, HOLDOUT_WEEK, eval_customers, k=SOURCE_K["repurchase"],
        ordering="production",
    )
    merged_prod = merge_candidates(
        {"repurchase": rep_prod_k50, "popularity_last_week": pop_fill},
        eval_customers, n=12,
        priority=["repurchase", "popularity_last_week"],
    )
    predictions_prod = _preds(merged_prod)
    prod_map = map_at_k(predictions_prod, ground_truth, k=12)

    # --- Before-merge ordering comparison (k=12 lists) ---
    recency_dict_k12 = _ranked_source_dict(rep_recency_k12)
    prod_dict_k12 = _ranked_source_dict(rep_prod_k12)
    all_custs_before = set(recency_dict_k12.keys()) | set(prod_dict_k12.keys())
    n_lists_differ_before = 0
    n_sets_differ_before = 0
    n_gt_affected_before = 0
    for cust in all_custs_before:
        r = recency_dict_k12.get(cust, [])
        p = prod_dict_k12.get(cust, [])
        if r != p:
            n_lists_differ_before += 1
            if set(r) != set(p):
                n_sets_differ_before += 1
                gt = ground_truth.get(cust, set())
                if gt & set(r).symmetric_difference(set(p)):
                    n_gt_affected_before += 1
    pct_lists_differ = 100.0 * n_lists_differ_before / len(all_custs_before) if all_custs_before else 0.0
    pct_sets_differ = 100.0 * n_sets_differ_before / len(all_custs_before) if all_custs_before else 0.0

    # --- After-merge ordering comparison (n=12 predictions) ---
    all_custs_after = set(predictions_recency.keys()) | set(predictions_prod.keys())
    n_preds_differ_after = sum(
        1 for c in all_custs_after
        if predictions_recency.get(c, []) != predictions_prod.get(c, [])
    )

    expected_map = 0.024457
    tolerance = 0.0001
    match = abs(measured_map - expected_map) < tolerance

    if verbose:
        print(f"\nRegression check (src pipeline, recency_only, week {HOLDOUT_WEEK}):")
        print(f"  Expected MAP@12 (EDA Baseline B) = {expected_map:.6f}")
        print(f"  Measured MAP@12 (recency_only)   = {measured_map:.6f}")
        print(f"  Delta                             = {measured_map - expected_map:+.6f}")
        print(f"  Match (tol {tolerance}): {'YES' if match else 'NO (INVESTIGATE!)'}")
        print(f"\n  [INFO] Production ordering MAP@12 (week {HOLDOUT_WEEK}) = {prod_map:.6f}")
        print(f"  [INFO] Difference (prod - recency_only) = {prod_map - measured_map:+.6f}")
        print(f"\n  Ordering comparison (before merge, k=12):")
        print(f"    Customers with repurchase history: {len(all_custs_before):,}")
        print(f"    Lists differ (order or set): {n_lists_differ_before:,} ({pct_lists_differ:.1f}%)")
        print(f"    Sets differ:                 {n_sets_differ_before:,} ({pct_sets_differ:.2f}%)")
        print(f"    Of set-diffs involving a GT item: {n_gt_affected_before:,}")
        print(f"\n  After-merge predictions differ: {n_preds_differ_after:,} of {len(all_custs_after):,}")

    return {
        "expected_map12": expected_map,
        "measured_map12_recency_only": measured_map,
        "match": match,
        "delta_recency_only": measured_map - expected_map,
        "production_ordering_map12": prod_map,
        "difference_prod_minus_recency": prod_map - measured_map,
        "before_merge_k12": {
            "n_customers": len(all_custs_before),
            "n_lists_differ": n_lists_differ_before,
            "pct_lists_differ": round(pct_lists_differ, 2),
            "n_sets_differ": n_sets_differ_before,
            "pct_sets_differ": round(pct_sets_differ, 2),
            "n_gt_affected": n_gt_affected_before,
        },
        "after_merge_n12": {
            "n_customers": len(all_custs_after),
            "n_predictions_differ": n_preds_differ_after,
        },
        "method": "src pipeline: repurchase(recency_only,k=12) + popularity_last_week(k=12), merge n=12",
    }


def run_scale_chunked(fold_week: int, chunk_size: int = 200_000, verbose: bool = True) -> dict:
    """Generate candidates for all customers in fold history, written in chunks to avoid OOM.

    Strategy:
      - Personalized sources (repurchase, product_code, copurchase): generated once for all
        customers; these are modest in size (~21M + 13M + 8.6M rows).
      - Popularity sources: the naive cross-join creates 135M rows per source for 1.35M
        customers. Instead we compute the top-k list once and expand in batches of chunk_size.
      - Each batch is merged and written to a temp parquet; temp files are scanned and
        concatenated into the final output at the end.
    """
    import tempfile, shutil
    log = lambda msg: print(f"  [scale-{fold_week}] {msg}") if verbose else None
    t0 = time.time()

    history, _, _ = build_fold(fold_week)

    all_customers_series = (
        load_transactions()
        .filter(pl.col("week_idx") < fold_week)
        .select("customer_idx").unique()
        .collect()["customer_idx"]
    )
    all_customers = sorted(all_customers_series.to_list())
    n_total = len(all_customers)
    log(f"{n_total:,} customers in history")

    # ------------------------------------------------------------------
    # Step 1: Personalized sources for ALL customers (fit in RAM)
    # ------------------------------------------------------------------
    log("Generating repurchase (all customers)...")
    rep_all = repurchase_gen(history, fold_week, all_customers, SOURCE_K["repurchase"])
    log(f"  {len(rep_all):,} rows")

    log("Generating product_code (all customers)...")
    pc_all = product_code_gen(history, fold_week, all_customers, SOURCE_K["product_code"])
    log(f"  {len(pc_all):,} rows")

    log("Generating copurchase (all customers)...")
    cop_all = copurchase_gen(history, fold_week, all_customers, SOURCE_K["copurchase"], min_count=MIN_COPURCHASE_COUNT)
    log(f"  {len(cop_all):,} rows")

    # ------------------------------------------------------------------
    # Step 2: Popularity top-k lists (no customer expansion yet)
    # ------------------------------------------------------------------
    log("Computing popularity top-k lists...")
    from src.config import ANCHOR_EPOCH_DAYS

    pop_lw_topk = (
        history
        .filter(pl.col("week_idx") == fold_week - 1)
        .group_by("article_idx")
        .agg(pl.len().cast(pl.Float32).alias("score"))
        .sort(["score", "article_idx"], descending=[True, False])
        .head(SOURCE_K["popularity_last_week"])
        .collect()
        .with_columns((pl.int_range(pl.len()) + 1).cast(pl.Int16).alias("source_rank"))
    )

    cutoff_epoch = ANCHOR_EPOCH_DAYS + fold_week * 7
    pop_dec_topk = (
        history
        .filter((pl.col("week_idx") >= fold_week - 4) & (pl.col("week_idx") < fold_week))
        .with_columns(
            (1.0 / (1.0 + (pl.lit(cutoff_epoch) - pl.col("t_dat").cast(pl.Int32)).cast(pl.Float64))).alias("dw")
        )
        .group_by("article_idx")
        .agg(pl.col("dw").sum().cast(pl.Float32).alias("score"))
        .sort(["score", "article_idx"], descending=[True, False])
        .head(SOURCE_K["popularity_decayed"])
        .collect()
        .with_columns((pl.int_range(pl.len()) + 1).cast(pl.Int16).alias("source_rank"))
    )
    log(f"  pop_lw: {len(pop_lw_topk)} articles, pop_dec: {len(pop_dec_topk)} articles")

    # Segment popular top-k per age bucket
    from src.data_io import get_customer_age_buckets
    cust_buckets = get_customer_age_buckets()
    seg_topk = (
        history
        .filter((pl.col("week_idx") >= fold_week - 2) & (pl.col("week_idx") < fold_week))
        .join(cust_buckets.lazy().select(["customer_idx", "age_bucket"]), on="customer_idx", how="inner")
        .group_by(["age_bucket", "article_idx"])
        .agg(pl.len().cast(pl.Float32).alias("score"))
        .sort(["age_bucket", "score", "article_idx"], descending=[False, True, False])
        .collect()
        .with_columns(
            pl.cum_count("article_idx").over("age_bucket").cast(pl.Int16).alias("source_rank")
        )
        .filter(pl.col("source_rank") <= SOURCE_K["segment_popular"])
    )
    log(f"  seg_topk: {len(seg_topk)} (age_bucket × article) pairs")

    def _expand_pop(topk: pl.DataFrame, batch_customers: list[int], src_name: str) -> pl.DataFrame:
        cdf = pl.DataFrame({"customer_idx": pl.Series(batch_customers, dtype=pl.Int32)})
        return (
            cdf.join(topk, how="cross")
            .with_columns(pl.lit(src_name).cast(pl.Utf8).alias("source"))
            .select(["customer_idx", "article_idx", "source", "score", "source_rank"])
        )

    def _expand_seg(seg_topk: pl.DataFrame, batch_customers: list[int], cust_buckets: pl.DataFrame) -> pl.DataFrame:
        batch_df = pl.DataFrame({"customer_idx": pl.Series(batch_customers, dtype=pl.Int32)})
        batch_buckets = batch_df.join(cust_buckets, on="customer_idx", how="left").with_columns(
            pl.col("age_bucket").fill_null("missing")
        )
        result = (
            batch_buckets.join(seg_topk, on="age_bucket", how="inner")
            .select(["customer_idx", "article_idx", "score", "source_rank"])
            .with_columns(pl.lit("segment_popular").cast(pl.Utf8).alias("source"))
        )
        # Re-rank per customer after potential dup articles across buckets
        result = (
            result.sort(["customer_idx", "score", "article_idx"], descending=[False, True, False])
            .unique(subset=["customer_idx", "article_idx"], keep="first", maintain_order=True)
            .with_columns(
                pl.cum_count("article_idx").over("customer_idx").cast(pl.Int16).alias("source_rank")
            )
            .filter(pl.col("source_rank") <= SOURCE_K["segment_popular"])
        )
        return result.select(["customer_idx", "article_idx", "source", "score", "source_rank"])

    # ------------------------------------------------------------------
    # Step 3: Process customers in chunks
    # ------------------------------------------------------------------
    tmp_dir = Path(tempfile.mkdtemp(prefix="hm_scale_"))
    log(f"Writing chunks to {tmp_dir}")
    total_rows = 0
    slim_cols = ["customer_idx", "article_idx", "final_source", "final_rank"]

    chunks = [all_customers[i:i+chunk_size] for i in range(0, n_total, chunk_size)]
    for ci, batch in enumerate(chunks):
        batch_set = set(batch)
        rep_b   = rep_all.filter(pl.col("customer_idx").is_in(batch))
        pc_b    = pc_all.filter(pl.col("customer_idx").is_in(batch))
        cop_b   = cop_all.filter(pl.col("customer_idx").is_in(batch))
        pop_lw_b = _expand_pop(pop_lw_topk, batch, "popularity_last_week")
        pop_dec_b = _expand_pop(pop_dec_topk, batch, "popularity_decayed")
        seg_b = _expand_seg(seg_topk, batch, cust_buckets)

        source_dfs_b = {
            "repurchase": rep_b, "product_code": pc_b, "copurchase": cop_b,
            "popularity_last_week": pop_lw_b, "popularity_decayed": pop_dec_b,
            "segment_popular": seg_b,
        }
        merged_b = merge_candidates(source_dfs_b, batch, MERGE_N, MERGE_PRIORITY)
        chunk_path = tmp_dir / f"chunk_{ci:04d}.parquet"
        merged_b.select(slim_cols).write_parquet(chunk_path, compression="zstd")
        total_rows += len(merged_b)
        if (ci + 1) % 2 == 0 or ci == len(chunks) - 1:
            log(f"  chunk {ci+1}/{len(chunks)}: {len(batch):,} customers, {len(merged_b):,} rows (total {total_rows:,})")

    # ------------------------------------------------------------------
    # Step 4: Concatenate chunks into final parquet
    # ------------------------------------------------------------------
    out_path = CANDIDATES_PARQUET_DIR / f"candidates_week{fold_week}.parquet"
    log(f"Concatenating {len(chunks)} chunks -> {out_path}")
    pl.scan_parquet(str(tmp_dir / "chunk_*.parquet")).collect().write_parquet(out_path, compression="zstd")
    shutil.rmtree(tmp_dir)

    file_size_mb = out_path.stat().st_size / 1024 / 1024
    runtime = time.time() - t0
    peak_rss_mb = round(_rss_mb(), 1)

    # Compute candidate count stats from the final parquet
    final_df = pl.scan_parquet(out_path).collect()
    cand_stats = _candidate_count_stats(final_df, MERGE_N)
    log(f"Done: {total_rows:,} rows, {n_total:,} customers, {file_size_mb:.1f} MB, "
        f"mean_cands={cand_stats['mean']:.1f}, pct_at_budget={cand_stats['pct_at_budget']:.1f}%, "
        f"rss={peak_rss_mb:.0f} MB, {runtime:.1f}s")

    return {
        "fold_week": fold_week,
        "n_customers": n_total,
        "n_rows": total_rows,
        "file_size_mb": file_size_mb,
        "candidate_counts": cand_stats,
        "peak_rss_mb": peak_rss_mb,
        "note_memory": "peak_rss_mb is process RSS (resource.getrusage); captures Polars/DuckDB native memory unlike tracemalloc",
        "chunk_size": chunk_size,
        "n_chunks": len(chunks),
        "runtime_s": runtime,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate candidates.")
    parser.add_argument(
        "--folds", type=int, nargs="+", default=VALIDATION_WEEKS,
        help="Validation fold weeks to run (default: 100 101 102 103)"
    )
    parser.add_argument(
        "--n", type=int, nargs="+", default=RECALL_NS,
        help="N values for recall@N (default: 20 50 100 200)"
    )
    parser.add_argument("--scale", action="store_true",
                        help="Run scale test (all 1.37M customers) for the last fold")
    parser.add_argument("--regression", action="store_true",
                        help="Run regression check on holdout week 104")
    args = parser.parse_args()

    print(f"Running candidate pipeline for folds {args.folds}, N={args.n}")
    print(f"Source k values: {SOURCE_K}")
    print(f"Merge budget: {MERGE_N} per customer\n")

    all_metrics = []

    for fold_week in args.folds:
        print(f"\n{'='*60}")
        print(f"Fold week {fold_week}")
        print('='*60)
        metrics, merged, source_dfs, ground_truth, eval_customers = run_fold(
            fold_week, args.n, scale=False, verbose=True
        )
        all_metrics.append(metrics)

        # Save per-fold CSV
        fold_path = CANDIDATES_DIR / f"fold_{fold_week}_results.json"
        with open(fold_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Saved -> {fold_path}")

    # -----------------------------------------------------------------------
    # Summary across folds
    # -----------------------------------------------------------------------
    if len(all_metrics) > 1:
        import statistics
        summary = {"folds": args.folds, "n_values": args.n, "per_source": {}, "merged": {}, "heuristic_map12": {}}

        # Mean + std for each recall metric
        for n in args.n:
            key = f"recall@{n}"
            merged_vals = [m["merged"][key] for m in all_metrics]
            summary["merged"][key] = {
                "mean": statistics.mean(merged_vals),
                "std": statistics.stdev(merged_vals) if len(merged_vals) > 1 else 0.0,
                "per_fold": merged_vals,
            }

        for src in MERGE_PRIORITY:
            summary["per_source"][src] = {}
            for n in args.n:
                key = f"recall@{n}"
                src_vals = [m["source_recalls"][src][key] for m in all_metrics if src in m.get("source_recalls", {})]
                if src_vals:
                    summary["per_source"][src][key] = {
                        "mean": statistics.mean(src_vals),
                        "std": statistics.stdev(src_vals) if len(src_vals) > 1 else 0.0,
                    }

        map_vals = [m["heuristic_map12"] for m in all_metrics]
        summary["heuristic_map12"] = {
            "mean": statistics.mean(map_vals),
            "std": statistics.stdev(map_vals) if len(map_vals) > 1 else 0.0,
            "per_fold": map_vals,
        }

        oracle_vals = [m["oracle_map12"] for m in all_metrics]
        cand_prec_vals = [m["candidate_precision_mean"] for m in all_metrics]
        summary["oracle_map12"] = {
            "mean": statistics.mean(oracle_vals),
            "std": statistics.stdev(oracle_vals) if len(oracle_vals) > 1 else 0.0,
            "per_fold": oracle_vals,
            "note": "Upper bound MAP@12 achievable by a perfect ranker within the 200-candidate pool.",
        }
        summary["candidate_precision"] = {
            "mean": statistics.mean(cand_prec_vals),
            "per_fold": cand_prec_vals,
            "note": "Mean fraction of candidates that are GT items per customer.",
        }

        summary_path = CANDIDATES_DIR / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved -> {summary_path}")

        print("\n--- Summary ---")
        for n in args.n:
            key = f"recall@{n}"
            v = summary["merged"][key]
            print(f"  Merged recall@{n}: {v['mean']:.4f} ± {v['std']:.4f}")
        print(f"  Heuristic MAP@12: {summary['heuristic_map12']['mean']:.6f} ± {summary['heuristic_map12']['std']:.6f}")

    # -----------------------------------------------------------------------
    # Scale test (chunked to avoid OOM from popularity cross-join)
    # -----------------------------------------------------------------------
    if args.scale:
        scale_fold = args.folds[-1]
        print(f"\n{'='*60}")
        print(f"Scale test (chunked): all customers, fold {scale_fold}")
        print('='*60)
        scale_metrics = run_scale_chunked(scale_fold, chunk_size=200_000, verbose=True)
        print(f"  Peak RSS: {scale_metrics['peak_rss_mb']:.1f} MB (process RSS via resource.getrusage)")

        scale_path = CANDIDATES_DIR / "scale_test.json"
        with open(scale_path, "w") as f:
            json.dump(scale_metrics, f, indent=2, default=str)
        print(f"  Saved -> {scale_path}")

    # -----------------------------------------------------------------------
    # Regression check
    # -----------------------------------------------------------------------
    if args.regression:
        reg = regression_check(verbose=True)
        reg_path = CANDIDATES_DIR / "regression_check.json"
        with open(reg_path, "w") as f:
            json.dump(reg, f, indent=2)
        print(f"  Saved -> {reg_path}")


if __name__ == "__main__":
    main()
