"""CLI: build point-in-time feature tables for all folds.

Usage:
    python scripts/build_features.py --folds 100 101 102 103 104
    python scripts/build_features.py --folds 103          # single fold (determinism check)
"""
import sys
import json
import time
import resource
import hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import polars as pl

from src.config import (
    HOLDOUT_WEEK,
    MERGE_PRIORITY,
    PROCESSED_DIR,
    REPORTS_DIR,
    VALIDATION_WEEKS,
)
from src.features.build import build_fold_features
from src.features.registry import FEATURE_LIST

FEATURES_DIR = PROCESSED_DIR / "features"
REPORTS_FEATURES_DIR = REPORTS_DIR / "features"


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def _parquet_hash(path: Path) -> str:
    """SHA256 of sorted parquet bytes for determinism check."""
    df = pl.read_parquet(path)
    df_sorted = df.sort(["customer_idx", "article_idx"])
    import io
    buf = io.BytesIO()
    df_sorted.write_ipc(buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _check_label_consistency(
    fold_week: int,
    df: pl.DataFrame,
    session2_recall: float,
    tol: float = 1e-4,
) -> dict:
    """Verify positives (before downsampling) match Session 2 recall at full list.

    For folds 100-103, the number of positives / ground-truth pairs must
    equal the Session 2 recall@200 (full list = 200 candidates).
    """
    if fold_week == HOLDOUT_WEEK:
        return {"skipped": "holdout week - labels not analysed this session"}

    # If downsampling was applied, we can't directly measure from the stored frame.
    # The check is done inside build_features_with_counts() before downsampling.
    return {"note": "label consistency checked in build run; see build_summary.json"}


def _check_null_rates(df: pl.DataFrame, fold_week: int) -> dict:
    """Flag any feature that is 100% null or constant (excluding keys and label)."""
    feature_cols = [spec.name for spec in FEATURE_LIST]
    flags = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        col_data = df[col]
        null_rate = col_data.is_null().mean()
        flags[col] = {"null_rate": round(float(null_rate), 4)}
        if null_rate == 1.0:
            flags[col]["flag"] = "ALL_NULL"
        elif col_data.drop_nulls().n_unique() == 1:
            flags[col]["flag"] = "CONSTANT"
    return flags


def _load_fold_n_eval_customers(fold_week: int) -> int:
    """Read n_eval_customers for a validation fold from its candidate JSON."""
    json_path = Path(__file__).parent.parent / "reports" / "candidates" / f"fold_{fold_week}_results.json"
    if not json_path.exists():
        return -1
    with open(json_path) as fh:
        d = json.load(fh)
    return int(d.get("n_eval_customers", -1))


def build_fold_with_metrics(
    fold_week: int,
    neg_sample_rate: float,
    seed: int,
) -> dict:
    """Build one fold and return metrics dict."""
    rss_start = _rss_mb()
    t0 = time.time()

    # Report expected row count before downsampling (= n_eval_customers × budget=200)
    n_eval = _load_fold_n_eval_customers(fold_week)
    n_expected_before_ds = n_eval * 200 if n_eval > 0 and fold_week in [100, 101, 102] else -1
    if n_expected_before_ds > 0:
        print(f"  Expected rows before downsampling: {n_eval:,} eval customers × 200 = {n_expected_before_ds:,}")

    # Build (internally handles downsampling and writes parquet)
    df = build_fold_features(fold_week, neg_sample_rate=neg_sample_rate, seed=seed)

    runtime = time.time() - t0
    peak_rss = _rss_mb()

    out_path = FEATURES_DIR / f"fold_{fold_week}.parquet"
    file_size_mb = _file_size_mb(out_path)

    n_rows = len(df)
    n_positives = int((df["label"] == 1).sum()) if fold_week != HOLDOUT_WEEK else -1
    label_rate = n_positives / n_rows if n_rows > 0 and fold_week != HOLDOUT_WEEK else -1.0

    null_flags = _check_null_rates(df, fold_week)

    return {
        "fold_week": fold_week,
        "n_eval_customers": n_eval,
        "n_rows_before_downsampling": n_expected_before_ds,
        "n_rows": n_rows,
        "n_positives": n_positives,
        "label_rate": round(label_rate, 6) if label_rate >= 0 else -1,
        "n_columns": len(df.columns),
        "file_size_mb": round(file_size_mb, 2),
        "runtime_s": round(runtime, 1),
        "peak_rss_mb": round(peak_rss, 1),
        "rss_delta_mb": round(peak_rss - rss_start, 1),
        "downsampling": neg_sample_rate if fold_week in [100, 101, 102] else "none",
        "null_rates": null_flags,
    }


def _read_expected_recall(fold_week: int) -> float:
    """Read recall@200 from the regenerated candidate fold JSON (k=200)."""
    json_path = Path(__file__).parent.parent / "reports" / "candidates" / f"fold_{fold_week}_results.json"
    with open(json_path) as fh:
        d = json.load(fh)
    return float(d["merged"]["recall@200"])


def run_label_consistency_check(fold_week: int) -> dict:
    """Check that positives / total_gt_items == recall@200 from candidate JSON (to 4 decimals).

    recall@200 = n_positives_in_pool / total_gt_items_across_customers.
    Expected recall is read from reports/candidates/fold_{w}_results.json (not hardcoded),
    so the check automatically reflects the active candidate configuration (k=200).
    """
    if fold_week not in list(range(100, 104)):
        return {"fold_week": fold_week, "skipped": "not a validation fold"}

    out_path = FEATURES_DIR / f"fold_{fold_week}.parquet"
    if not out_path.exists():
        return {"fold_week": fold_week, "skipped": "parquet not found"}

    df = pl.read_parquet(out_path)

    # Positives are still correct even in downsampled training folds (we only remove negatives)
    n_positives = int((df["label"] == 1).sum())

    # Compute total GT items from the ground truth via build_fold
    from src.time_split import build_fold
    _, ground_truth, _ = build_fold(fold_week)
    total_gt_items = sum(len(v) for v in ground_truth.values())

    measured_recall = n_positives / total_gt_items if total_gt_items > 0 else 0.0
    expected_recall = _read_expected_recall(fold_week)
    diff = abs(measured_recall - expected_recall)
    match = diff < 1e-4

    return {
        "fold_week": fold_week,
        "n_positives_in_candidates": n_positives,
        "total_gt_items": total_gt_items,
        "expected_recall_200": expected_recall,
        "measured_recall_200": round(measured_recall, 6),
        "diff": round(diff, 6),
        "match_4_decimals": match,
    }


def main():
    parser = argparse.ArgumentParser(description="Build point-in-time feature tables.")
    parser.add_argument(
        "--folds", type=int, nargs="+", default=list(VALIDATION_WEEKS) + [HOLDOUT_WEEK],
        help="Fold weeks to build (default: 100 101 102 103 104)"
    )
    parser.add_argument("--neg_sample_rate", type=float, default=0.2,
                        help="Negative downsampling rate for training folds (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for downsampling (default: 42)")
    parser.add_argument("--determinism", action="store_true",
                        help="Run fold 103 twice and compare hashes")
    args = parser.parse_args()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building features for folds {args.folds}")
    print(f"neg_sample_rate={args.neg_sample_rate}, seed={args.seed}")
    print(f"FEATURE_LIST: {len(FEATURE_LIST)} features\n")

    build_summary: list[dict] = []
    for fold_week in args.folds:
        print(f"\n{'='*60}")
        print(f"Fold {fold_week}")
        print("="*60)
        metrics = build_fold_with_metrics(fold_week, args.neg_sample_rate, args.seed)
        build_summary.append(metrics)

        print(f"  rows={metrics['n_rows']:,}  positives={metrics['n_positives']:,}  "
              f"label_rate={metrics['label_rate']:.4%}  "
              f"size={metrics['file_size_mb']:.1f}MB  "
              f"runtime={metrics['runtime_s']:.0f}s  "
              f"peak_rss={metrics['peak_rss_mb']:.0f}MB")

        flagged = [k for k, v in metrics["null_rates"].items() if "flag" in v]
        if flagged:
            print(f"  [WARN] Flagged features: {flagged}")

    # ------------------------------------------------------------------
    # Label consistency checks
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Label consistency checks")
    print("="*60)
    label_checks = []
    for fold_week in args.folds:
        if fold_week == HOLDOUT_WEEK:
            continue
        check = run_label_consistency_check(fold_week)
        label_checks.append(check)
        status = "OK" if check.get("match_4_decimals") else "MISMATCH"
        print(f"  Fold {fold_week}: expected={check.get('expected_recall_200', 'N/A'):.4f}  "
              f"measured={check.get('measured_recall_200', 'N/A'):.4f}  "
              f"diff={check.get('diff', 'N/A'):.6f}  [{status}]")

    # ------------------------------------------------------------------
    # Determinism check (optional)
    # ------------------------------------------------------------------
    determinism_result = {}
    if args.determinism:
        print("\n" + "="*60)
        print("Determinism check (fold 103, built twice)")
        print("="*60)
        path_103 = FEATURES_DIR / "fold_103.parquet"
        if path_103.exists():
            hash1 = _parquet_hash(path_103)
            print(f"  Build 1 hash: {hash1[:16]}...")
            build_fold_features(103, neg_sample_rate=args.neg_sample_rate, seed=args.seed)
            hash2 = _parquet_hash(path_103)
            print(f"  Build 2 hash: {hash2[:16]}...")
            match = hash1 == hash2
            determinism_result = {"hash1": hash1, "hash2": hash2, "match": match}
            print(f"  Deterministic: {'YES' if match else 'NO (investigate!)'}")

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary = {
        "neg_sample_rate": args.neg_sample_rate,
        "seed": args.seed,
        "n_features": len(FEATURE_LIST),
        "feature_groups": {
            g: sum(1 for f in FEATURE_LIST if f.group == g)
            for g in ["candidate", "customer", "article", "interaction"]
        },
        "per_fold": build_summary,
        "label_consistency": label_checks,
        "determinism": determinism_result,
    }

    summary_path = REPORTS_FEATURES_DIR / "build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBuild summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
