"""Part 1: Robustness, metric suite, and segment analysis.

Usage:
    python scripts/run_evaluation.py               # full run
    python scripts/run_evaluation.py --part A      # rolling origin only
    python scripts/run_evaluation.py --part B      # metric suite only
    python scripts/run_evaluation.py --part C      # segment analysis only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import numpy as np
import polars as pl

from src.config import PROCESSED_DIR, REPORTS_DIR
from src.data_io import load_articles, load_customers, load_transactions
from src.metrics import map_at_k, per_customer_ap
from src.model.data import FEATURE_NAMES, concat_folds, prepare_dataset
from src.model.evaluate import bootstrap_paired_diff, score_fold
from src.model.train import DEFAULT_PARAMS, SEED, train_final
from src.time_split import build_fold
from src.evaluation.suite import run_metric_suite
from src.evaluation.segments import compute_extended_segments, run_segment_analysis

EVAL_DIR = REPORTS_DIR / "evaluation"
FIG_DIR = REPORTS_DIR / "figures" / "evaluation"
FEATURES_DIR = PROCESSED_DIR / "features"
MODELS_DIR = Path(__file__).parent.parent / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: heuristic and Baseline B from a full feature parquet
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_from_parquet(path: Path) -> dict[int, list[int]]:
    """Read heuristic top-12 from a full feature parquet (final_rank <= 12)."""
    df = pl.read_parquet(path)
    top12 = (
        df.filter(pl.col("final_rank") <= 12)
        .sort(["customer_idx", "final_rank"])
        .group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_idx").alias("articles"))
    )
    return {r["customer_idx"]: list(r["articles"]) for r in top12.iter_rows(named=True)}


def _baseline_b_from_parquet(path: Path) -> dict[int, list[int]]:
    """Baseline B: recency-ordered repurchase + popularity fill from a feature parquet."""
    df = pl.read_parquet(path)
    pop_rank_fill = df["popularity_last_week_rank"].fill_null(999_999).cast(pl.Int32)
    df_b = df.with_columns([
        pl.when(pl.col("in_repurchase") == 1)
        .then(pl.lit(0, dtype=pl.Int8))
        .otherwise(pl.lit(1, dtype=pl.Int8))
        .alias("_grp"),
        pl.when(pl.col("in_repurchase") == 1)
        .then(-pl.col("repurchase_score").cast(pl.Float64))
        .otherwise(pop_rank_fill.cast(pl.Float64))
        .alias("_sort_key"),
    ]).sort(
        ["customer_idx", "_grp", "_sort_key", "article_idx"],
        descending=[False, False, False, False],
    ).with_columns(
        pl.int_range(pl.len(), dtype=pl.Int32).over("customer_idx").alias("_rn")
    ).filter(pl.col("_rn") < 12)
    return {
        r["customer_idx"]: list(r["articles"])
        for r in df_b
        .group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_idx").alias("articles"))
        .iter_rows(named=True)
    }


def _load_best_params() -> tuple[dict, int]:
    """Load tuned params and inner_rounds from reports/ranker/tuning.json."""
    with open(REPORTS_DIR / "ranker" / "tuning.json") as f:
        t = json.load(f)
    best_params = DEFAULT_PARAMS.copy()
    best_params.update(t["best_params"])
    inner_rounds = t["best_lgb_round_inner"]
    return best_params, inner_rounds


def _train_rolling_model(
    fold_trains: list[int],
    best_params: dict,
    inner_rounds: int,
) -> tuple[lgb.Booster, int]:
    """Train a model with fold_trains as training folds, scaled rounds."""
    booster, final_rounds, ratio, _ = train_final(best_params, inner_rounds, fold_train=fold_trains)
    return booster, final_rounds


def _eval_fold_full(
    fold_week: int,
    booster: lgb.Booster,
    best_params: dict,
    inner_rounds: int,
    n_boot: int = 1000,
) -> dict:
    """Evaluate a single fold: ranker vs heuristic vs Baseline B."""
    full_path = FEATURES_DIR / f"fold_{fold_week}_full.parquet"
    if not full_path.exists():
        raise FileNotFoundError(f"{full_path} not found")

    df_full = pl.read_parquet(full_path)
    _, ground_truth, _ = build_fold(fold_week)

    preds_ranker, _ = score_fold(booster, df_full, k=12, sanity_check=False)
    preds_heuristic = _heuristic_from_parquet(full_path)
    preds_bb = _baseline_b_from_parquet(full_path)

    ap_r = per_customer_ap(preds_ranker, ground_truth, 12)
    ap_h = per_customer_ap(preds_heuristic, ground_truth, 12)
    ap_bb = per_customer_ap(preds_bb, ground_truth, 12)

    boot_vs_heuristic = bootstrap_paired_diff(ap_r, ap_h, n=n_boot, seed=SEED)
    boot_vs_bb = bootstrap_paired_diff(ap_r, ap_bb, n=n_boot, seed=SEED)

    return {
        "fold": fold_week,
        "n_eval_customers": len(ground_truth),
        "ranker_map@12": map_at_k(preds_ranker, ground_truth, 12),
        "heuristic_map@12": map_at_k(preds_heuristic, ground_truth, 12),
        "baseline_b_map@12": map_at_k(preds_bb, ground_truth, 12),
        "boot_vs_heuristic": boot_vs_heuristic,
        "boot_vs_baseline_b": boot_vs_bb,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Part A: Rolling Origin
# ─────────────────────────────────────────────────────────────────────────────

def part_a(n_boot: int = 1000) -> dict:
    print("\n" + "=" * 60)
    print("PART A — Robustness Across Weeks (Rolling Origin)")
    print("=" * 60)

    best_params, inner_rounds = _load_best_params()
    print(f"  Loaded tuned params  inner_rounds={inner_rounds}")

    # Build fold_101_full if missing
    fold_101_full_path = FEATURES_DIR / "fold_101_full.parquet"
    if not fold_101_full_path.exists():
        print("  Building fold_101_full.parquet ...")
        from src.features.build import build_fold_101_full
        build_fold_101_full()
    else:
        print("  fold_101_full.parquet already exists — skipping build")

    results = []

    # ── Fold 101: train on fold 100 only ─────────────────────────────────────
    print("\n  [Fold 101] Training on fold 100 only ...")
    t0 = time.time()
    booster_101, rounds_101 = _train_rolling_model([100], best_params, inner_rounds)
    print(f"    final_rounds={rounds_101}, elapsed={time.time()-t0:.0f}s")
    res_101 = _eval_fold_full(101, booster_101, best_params, inner_rounds, n_boot)
    res_101["train_folds"] = [100]
    res_101["final_rounds"] = rounds_101
    res_101["note"] = "unbiased"
    results.append(res_101)
    print(f"    ranker={res_101['ranker_map@12']:.6f}  heuristic={res_101['heuristic_map@12']:.6f}  "
          f"CI=[{res_101['boot_vs_heuristic']['ci_lo']:+.6f}, {res_101['boot_vs_heuristic']['ci_hi']:+.6f}]")

    # ── Fold 102: train on folds 100+101 ──────────────────────────────────────
    print("\n  [Fold 102] Training on folds 100+101 (OPTIMISTIC: fold 102 was tuning fold) ...")
    t0 = time.time()
    booster_102, rounds_102 = _train_rolling_model([100, 101], best_params, inner_rounds)
    print(f"    final_rounds={rounds_102}, elapsed={time.time()-t0:.0f}s")
    res_102 = _eval_fold_full(102, booster_102, best_params, inner_rounds, n_boot)
    res_102["train_folds"] = [100, 101]
    res_102["final_rounds"] = rounds_102
    res_102["note"] = "optimistic: fold 102 was the tuning fold; hyperparameters saw this fold's distribution"
    results.append(res_102)
    print(f"    ranker={res_102['ranker_map@12']:.6f}  heuristic={res_102['heuristic_map@12']:.6f}  "
          f"CI=[{res_102['boot_vs_heuristic']['ci_lo']:+.6f}, {res_102['boot_vs_heuristic']['ci_hi']:+.6f}]")

    # ── Fold 103: existing final model ────────────────────────────────────────
    print("\n  [Fold 103] Loading existing final model (train 100-102) ...")
    booster_103 = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_ranker.txt"))
    df_103 = pl.read_parquet(FEATURES_DIR / "fold_103.parquet")
    _, gt_103, _ = build_fold(103)
    preds_r_103, _ = score_fold(booster_103, df_103, k=12, sanity_check=False)
    preds_h_103 = _heuristic_from_parquet(FEATURES_DIR / "fold_103.parquet")
    preds_bb_103 = _baseline_b_from_parquet(FEATURES_DIR / "fold_103.parquet")
    ap_r103 = per_customer_ap(preds_r_103, gt_103, 12)
    ap_h103 = per_customer_ap(preds_h_103, gt_103, 12)
    ap_bb103 = per_customer_ap(preds_bb_103, gt_103, 12)
    boot_h103 = bootstrap_paired_diff(ap_r103, ap_h103, n=n_boot, seed=SEED)
    boot_bb103 = bootstrap_paired_diff(ap_r103, ap_bb103, n=n_boot, seed=SEED)
    # Verify fold 103 MAP@12 matches eval_fold103.json to 6 decimals
    map103 = map_at_k(preds_r_103, gt_103, 12)
    expected_103 = 0.035832
    diff_103 = abs(map103 - expected_103)
    assert diff_103 < 1e-4, (
        f"Fold 103 MAP@12={map103:.6f} diverges from eval_fold103.json {expected_103:.6f} "
        f"by {diff_103:.2e}"
    )
    res_103 = {
        "fold": 103,
        "n_eval_customers": len(gt_103),
        "train_folds": [100, 101, 102],
        "final_rounds": 421,
        "note": "final model; eval fold is cleanly held out from tuning",
        "ranker_map@12": map103,
        "heuristic_map@12": map_at_k(preds_h_103, gt_103, 12),
        "baseline_b_map@12": map_at_k(preds_bb_103, gt_103, 12),
        "boot_vs_heuristic": boot_h103,
        "boot_vs_baseline_b": boot_bb103,
    }
    results.append(res_103)
    print(f"    ranker={res_103['ranker_map@12']:.6f}  heuristic={res_103['heuristic_map@12']:.6f}  "
          f"CI=[{boot_h103['ci_lo']:+.6f}, {boot_h103['ci_hi']:+.6f}]")
    print(f"    Fold 103 MAP@12 sanity check: expected {expected_103:.6f}, got {map103:.6f} (diff={diff_103:.2e}) ✓")

    # Summary
    lifts = [r["boot_vs_heuristic"]["mean_diff"] for r in results]
    mean_lift = float(np.mean(lifts))
    all_ci_exclude_zero = all(r["boot_vs_heuristic"]["ci_excludes_zero"] for r in results)
    output = {
        "folds": results,
        "mean_lift_vs_heuristic": mean_lift,
        "lift_holds_every_week": all_ci_exclude_zero,
        "inner_rounds": inner_rounds,
        "scale_rule": "final_rounds = round(inner_rounds * n_rows_final / n_rows_inner)",
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DIR / "rolling_origin.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Mean lift vs heuristic: {mean_lift:+.6f}")
    print(f"  Lift holds every week (all CIs exclude 0): {all_ci_exclude_zero}")
    print(f"  Saved → {EVAL_DIR / 'rolling_origin.json'}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Part B: Metric Suite (fold 103)
# ─────────────────────────────────────────────────────────────────────────────

def _build_article_pop_ranks(tx_lf: pl.LazyFrame, cutoff_week: int) -> dict[int, int]:
    """Rank articles by total sales before cutoff (rank 1 = most popular)."""
    pop_df = (
        tx_lf
        .filter(pl.col("week_idx") < cutoff_week)
        .group_by("article_idx")
        .agg(pl.len().alias("sales_count"))
        .sort("sales_count", descending=True)
        .with_columns(
            pl.int_range(1, pl.len() + 1, dtype=pl.Int32).alias("pop_rank")
        )
        .collect()
    )
    return {r["article_idx"]: r["pop_rank"] for r in pop_df.iter_rows(named=True)}


def _build_customer_history(
    tx_lf: pl.LazyFrame,
    cutoff_week: int,
    eval_customers: list[int],
) -> dict[int, set[int]]:
    """Dict customer_idx -> set of article_idx bought before cutoff."""
    hist = (
        tx_lf
        .filter(
            (pl.col("week_idx") < cutoff_week)
            & (pl.col("customer_idx").is_in(eval_customers))
        )
        .select(["customer_idx", "article_idx"])
        .unique()
        .collect()
    )
    result: dict[int, set[int]] = {}
    for row in hist.iter_rows():
        c, a = row
        if c not in result:
            result[c] = set()
        result[c].add(a)
    return result


def _build_articles_before_cutoff(tx_lf: pl.LazyFrame, cutoff_week: int) -> set[int]:
    """Set of all articles with at least one transaction before cutoff."""
    return set(
        tx_lf
        .filter(pl.col("week_idx") < cutoff_week)
        .select("article_idx")
        .unique()
        .collect()["article_idx"]
        .to_list()
    )


def _build_sold_articles_history(tx_lf: pl.LazyFrame, cutoff_week: int) -> set[int]:
    """Same as articles_before_cutoff — used as coverage denominator."""
    return _build_articles_before_cutoff(tx_lf, cutoff_week)


def part_b(n_boot: int = 1000) -> dict:
    print("\n" + "=" * 60)
    print("PART B — Metric Suite (fold 103, ranker vs heuristic)")
    print("=" * 60)

    tx_lf = load_transactions()
    _, gt_103, eval_custs = build_fold(103)

    print("  Building supporting data structures ...")
    article_pop_ranks = _build_article_pop_ranks(tx_lf, 103)
    customer_history = _build_customer_history(tx_lf, 103, eval_custs)
    articles_before = _build_articles_before_cutoff(tx_lf, 103)
    sold_articles = articles_before  # same set for coverage denominator

    print(f"  Article pop ranks built: {len(article_pop_ranks):,} articles ranked")
    print(f"  Customer history built: {len(customer_history):,} eval customers with history")
    print(f"  Articles before cutoff: {len(articles_before):,}")

    # Load models and score fold 103
    booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_ranker.txt"))
    df_103 = pl.read_parquet(FEATURES_DIR / "fold_103.parquet")
    preds_ranker, _ = score_fold(booster, df_103, k=12, sanity_check=False)
    preds_heuristic = _heuristic_from_parquet(FEATURES_DIR / "fold_103.parquet")

    print("\n  Computing ranker metric suite ...")
    suite_ranker = run_metric_suite(
        preds_ranker, gt_103, article_pop_ranks, customer_history,
        articles_before, sold_articles, n_boot=n_boot, seed=SEED,
    )
    print("\n  Computing heuristic metric suite ...")
    suite_heuristic = run_metric_suite(
        preds_heuristic, gt_103, article_pop_ranks, customer_history,
        articles_before, sold_articles, n_boot=n_boot, seed=SEED,
    )

    # Print table
    def _fmt(d, key):
        v = d.get(key, {})
        if isinstance(v, dict):
            return f"{v.get('mean', float('nan')):.4f} [{v.get('ci_lo', float('nan')):.4f}, {v.get('ci_hi', float('nan')):.4f}]"
        return f"{v:.4f}"

    print("\n  === Metric Suite Table ===")
    print(f"  {'Metric':<35s}  {'Ranker':>30s}  {'Heuristic':>30s}")
    for metric_key, label in [
        ("map@12_ci", "MAP@12"),
        ("hit_rate@12_ci", "Hit Rate@12"),
    ]:
        r_acc = suite_ranker["accuracy"]
        h_acc = suite_heuristic["accuracy"]
        print(f"  {label:<35s}  {_fmt(r_acc, metric_key):>30s}  {_fmt(h_acc, metric_key):>30s}")
    for metric_key, label in [
        ("catalog_coverage_ci", "Catalog Coverage"),
        ("popularity_bias_ci", "Pop Bias (mean rank)"),
        ("repeat_share_ci", "Repeat Share"),
        ("novelty_share_ci", "Novelty Share"),
    ]:
        r_ba = suite_ranker["beyond_accuracy"]
        h_ba = suite_heuristic["beyond_accuracy"]
        print(f"  {label:<35s}  {_fmt(r_ba, metric_key):>30s}  {_fmt(h_ba, metric_key):>30s}")

    output = {
        "fold": 103,
        "ranker": suite_ranker,
        "heuristic": suite_heuristic,
    }
    with open(EVAL_DIR / "metric_suite.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {EVAL_DIR / 'metric_suite.json'}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Part C: Segment Analysis
# ─────────────────────────────────────────────────────────────────────────────

def part_c(n_boot: int = 1000) -> dict:
    print("\n" + "=" * 60)
    print("PART C — Segment Analysis (fold 103, Holm-corrected)")
    print("=" * 60)

    tx_lf = load_transactions()
    customers_lf = load_customers()
    _, gt_103, eval_custs = build_fold(103)

    booster = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_ranker.txt"))
    df_103 = pl.read_parquet(FEATURES_DIR / "fold_103.parquet")
    preds_ranker, _ = score_fold(booster, df_103, k=12, sanity_check=False)
    preds_heuristic = _heuristic_from_parquet(FEATURES_DIR / "fold_103.parquet")

    print("  Computing segment definitions ...")
    segments = compute_extended_segments(tx_lf, customers_lf, 103, eval_custs, gt_103)
    for family, seg_dict in segments.items():
        sizes = {k: len(v) for k, v in seg_dict.items()}
        print(f"  {family}: {sizes}")

    print("\n  Running segment analysis with Holm correction ...")
    seg_results = run_segment_analysis(
        preds_ranker, preds_heuristic, gt_103, segments,
        k=12, n_boot=n_boot, seed=SEED,
    )

    # Print results
    flagged = [r for r in seg_results if r.get("flagged", False)]
    print(f"\n  {'Family':<20s} {'Segment':<20s} {'n':>7s} {'Ranker':>10s} {'Heuristic':>10s} {'p_adj':>8s} {'Sig':>5s} {'Flag':>5s}")
    for r in seg_results:
        print(f"  {r['family']:<20s} {r['segment']:<20s} {r['n']:>7,d} "
              f"{r['map@12_ranker']:>10.6f} {r['map@12_heuristic']:>10.6f} "
              f"{r.get('p_adj', float('nan')):>8.4f} {str(r.get('significant_adj', '?')):>5s} "
              f"{str(r.get('flagged', False)):>5s}")

    if flagged:
        print(f"\n  *** {len(flagged)} segment(s) where ranker does NOT clearly beat heuristic:")
        for r in flagged:
            print(f"      {r['family']} / {r['segment']}  n={r['n']}  "
                  f"CI=[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]")
    else:
        print("\n  All segments: ranker clearly beats heuristic (all CIs exclude 0).")

    output = {
        "fold": 103,
        "n_segments_tested": len(seg_results),
        "n_flagged": len(flagged),
        "results": seg_results,
    }
    with open(EVAL_DIR / "segment_analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {EVAL_DIR / 'segment_analysis.json'}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["A", "B", "C"], default=None,
                        help="Run only one part (default: all)")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    parts = [args.part] if args.part else ["A", "B", "C"]

    if "A" in parts:
        part_a(n_boot=args.n_boot)
    if "B" in parts:
        part_b(n_boot=args.n_boot)
    if "C" in parts:
        part_c(n_boot=args.n_boot)

    print(f"\nTotal runtime: {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
