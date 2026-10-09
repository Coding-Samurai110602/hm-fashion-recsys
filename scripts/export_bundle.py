"""Export the serving bundle to artifacts/bundle_week104/.

Bundle contents (all SHA-256 verified via manifest.json):
  model.txt                    : Primary week-104 model (lgbm_ranker_primary_104.txt)
  state/article_feats.parquet  : Article features as of week 104
  state/pop_last_week.parquet  : Top-k last-week popularity list
  state/pop_decayed.parquet    : Top-k decayed popularity list
  state/seg_pop/<bucket>.parquet: Per-age-bucket top-k segment lists
  state/copurchase.parquet     : Symmetric co-purchase cosine matrix
  state/prev_week_sales.parquet: Article sales in week 103
  state/article_product_codes.parquet: (article_idx, product_code)
  state/age_buckets.parquet    : (customer_idx, age_bucket) all customers
  state/bucket_totals.parquet  : Segment normalization denominators
  state/customer_history.parquet: Transaction history up to week 104
  state/customers.parquet      : Customer demographics including customer_id hex string
  state/article_meta.parquet   : Display metadata
  state/gt_week104.parquet     : Week-104 ground truth (display only, clearly labelled)
  state/ab_arrays.parquet      : Per-customer hit@12/AP@12 (ranker + heuristic, week 104)
  reports/*.json               : Snapshotted evaluation report JSONs (self-contained for Docker)
  config.json                  : Feature list + dtypes + reason templates + share denominators
  manifest.json                : SHA-256 hashes + git commit + model metrics + created_at

Why week 104: the demo shows each customer's recommendations next to what they actually
bought that week (week-104 ground truth stored in the bundle for display only).

Usage:
    caffeinate -i python -u scripts/export_bundle.py 2>&1
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import polars as pl

ROOT = Path(__file__).parent.parent
BUNDLE_DIR = ROOT / "artifacts" / "bundle_week104"
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

AS_OF_WEEK = 104


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _export_ab_arrays(state_dir: Path, bundle_dir: Path, processed: Path) -> None:
    """Compute per-customer hit@12/AP@12 (ranker + heuristic) on week-104 eval set."""
    import numpy as np
    from src.features.registry import FEATURE_LIST

    feat_cols = [spec.name for spec in FEATURE_LIST]

    # Load fold_104 features (labels=-1 placeholder) + join actual labels
    print("  Loading fold_104 features...", flush=True)
    fold104 = pl.read_parquet(processed / "features" / "fold_104.parquet")
    labels104 = pl.read_parquet(processed / "features" / "labels_104.parquet").select(
        ["customer_idx", "article_idx", "label"]
    )
    # Replace placeholder labels with actual labels
    fold104 = fold104.drop("label").join(
        labels104, on=["customer_idx", "article_idx"], how="left"
    ).with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    print(f"  fold_104: {len(fold104):,} rows, {fold104['label'].sum():,} positives", flush=True)

    # Load primary model
    import lightgbm as lgb
    primary_model = lgb.Booster(model_file=str(bundle_dir / "model.txt"))

    # Run ranker inference
    print("  Running ranker inference...", flush=True)
    X = fold104.select(feat_cols).to_numpy(allow_copy=True)
    scores = primary_model.predict(X)

    # Add scores to dataframe
    fold104 = fold104.with_columns(
        pl.Series("ranker_score", scores, dtype=pl.Float32)
    )

    # Compute per-customer AP@12 for ranker and heuristic (final_rank)
    print("  Computing AP@12 per customer...", flush=True)

    def _compute_ap12_per_customer(df: pl.DataFrame, score_col: str, ascending: bool) -> pl.DataFrame:
        """Return DataFrame[customer_idx, ap12, hit12]."""
        # Sort within customer by score
        direction = ascending  # True = ascending (final_rank: low is good)
        sorted_df = df.sort([score_col], descending=not direction).with_columns(
            pl.col("label").cast(pl.Int32)
        )

        results = []
        for cust_idx, group in sorted_df.group_by("customer_idx"):
            labels_arr = group.head(12)["label"].to_numpy()
            n_gt = int(group["label"].sum())
            if n_gt == 0:
                results.append({"customer_idx": int(cust_idx[0]), "ap12": 0.0, "hit12": 0})
                continue
            n_hit = int(labels_arr.sum())
            if n_hit == 0:
                results.append({"customer_idx": int(cust_idx[0]), "ap12": 0.0, "hit12": 0})
                continue
            # AP@12
            prec_sum = 0.0
            n_hits_so_far = 0
            for k_i, lbl in enumerate(labels_arr, 1):
                if lbl == 1:
                    n_hits_so_far += 1
                    prec_sum += n_hits_so_far / k_i
            ap = prec_sum / min(n_gt, 12)
            results.append({
                "customer_idx": int(cust_idx[0]),
                "ap12": float(ap),
                "hit12": int(n_hit > 0),
            })
        return pl.DataFrame(results, schema={"customer_idx": pl.Int32, "ap12": pl.Float32, "hit12": pl.Int8})

    ranker_stats = _compute_ap12_per_customer(fold104, "ranker_score", ascending=False)
    heuristic_stats = _compute_ap12_per_customer(fold104, "final_rank", ascending=True)

    # Join and save
    ab_arrays = ranker_stats.rename({"ap12": "ranker_ap12", "hit12": "ranker_hit12"}).join(
        heuristic_stats.rename({"ap12": "heuristic_ap12", "hit12": "heuristic_hit12"}),
        on="customer_idx", how="inner",
    )
    ab_arrays.write_parquet(state_dir / "ab_arrays.parquet", compression="zstd")
    n_cust = len(ab_arrays)
    ranker_map = float(ab_arrays["ranker_ap12"].mean())
    heuristic_map = float(ab_arrays["heuristic_ap12"].mean())
    print(f"  ab_arrays: {n_cust:,} customers, ranker MAP@12={ranker_map:.6f}, "
          f"heuristic MAP@12={heuristic_map:.6f}", flush=True)


def _snapshot_reports(bundle_dir: Path, reports_dir: Path) -> None:
    """Copy evaluation report JSONs into bundle/reports/ for Docker self-containedness."""
    reports_out = bundle_dir / "reports"
    reports_out.mkdir(exist_ok=True)

    sources = {
        "holdout_results.json": reports_dir / "evaluation" / "holdout_results.json",
        "ab_test.json":         reports_dir / "evaluation" / "ab_test.json",
        "metric_suite.json":    reports_dir / "evaluation" / "metric_suite.json",
        "segment_analysis.json": reports_dir / "evaluation" / "segment_analysis.json",
        "rolling_origin.json":  reports_dir / "evaluation" / "rolling_origin.json",
        "eval_fold103.json":    reports_dir / "ranker" / "eval_fold103.json",
        "ablations.json":       reports_dir / "ranker" / "ablations.json",
        "shap_summary.json":    reports_dir / "explain" / "shap_summary.json",
    }
    copied = 0
    for dest_name, src_path in sources.items():
        if src_path.exists():
            (reports_out / dest_name).write_bytes(src_path.read_bytes())
            copied += 1
        else:
            print(f"  WARNING: report not found, skipping: {src_path}", flush=True)
    print(f"  reports: {copied}/{len(sources)} JSONs snapshotted", flush=True)


def main():
    t0 = time.time()
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    state_dir = BUNDLE_DIR / "state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "seg_pop").mkdir(exist_ok=True)

    print(f"[export_bundle] Exporting to {BUNDLE_DIR}", flush=True)

    # ── 1. Load transactions and build state ─────────────────────────────────
    print("[1] Loading transactions and building state (as_of_week=104)...", flush=True)
    from src.config import ANCHOR_EPOCH_DAYS, ANCHOR_DATE
    from src.model.state import build_state

    transactions_lf = pl.scan_parquet(PROCESSED / "transactions_train.parquet")
    history_df = (
        transactions_lf
        .with_columns(
            ((pl.col("t_dat").cast(pl.Int32) - ANCHOR_EPOCH_DAYS) // 7)
            .cast(pl.Int16).alias("week_idx")
        )
        .filter(pl.col("week_idx") < AS_OF_WEEK)
        .collect()
    )
    print(f"[1] History: {len(history_df):,} rows (week_idx < {AS_OF_WEEK})", flush=True)

    articles_lf = pl.scan_parquet(PROCESSED / "articles.parquet")
    customers_lf = pl.scan_parquet(PROCESSED / "customers.parquet")

    state = build_state(AS_OF_WEEK, history_df, articles_lf, customers_lf)
    print(f"[1] State built in {state.build_time_s:.1f}s, RSS={state.rss_mb:.0f}MB", flush=True)

    # ── 2. Copy model file ─────────────────────────────────────────────────
    print("[2] Copying model...", flush=True)
    src_model = MODELS_DIR / "lgbm_ranker_primary_104.txt"
    dst_model = BUNDLE_DIR / "model.txt"
    dst_model.write_bytes(src_model.read_bytes())

    # ── 3. Save state parquets ─────────────────────────────────────────────
    print("[3] Writing state parquets...", flush=True)

    state.article_feats.write_parquet(state_dir / "article_feats.parquet", compression="zstd")
    print(f"  article_feats: {len(state.article_feats):,} rows", flush=True)

    state.pop_last_week_top.write_parquet(state_dir / "pop_last_week.parquet", compression="zstd")
    print(f"  pop_last_week: {len(state.pop_last_week_top):,} rows", flush=True)

    state.pop_decayed_top.write_parquet(state_dir / "pop_decayed.parquet", compression="zstd")
    print(f"  pop_decayed: {len(state.pop_decayed_top):,} rows", flush=True)

    state.copurchase_symmetric.write_parquet(state_dir / "copurchase.parquet", compression="zstd")
    print(f"  copurchase: {len(state.copurchase_symmetric):,} pairs", flush=True)

    state.prev_week_sales.write_parquet(state_dir / "prev_week_sales.parquet", compression="zstd")
    state.article_product_codes.write_parquet(state_dir / "article_product_codes.parquet", compression="zstd")
    state.all_cust_buckets.write_parquet(state_dir / "age_buckets.parquet", compression="zstd")
    state.bucket_totals.write_parquet(state_dir / "bucket_totals.parquet", compression="zstd")

    # Customers table (for feature computation + customer_id→idx lookup for API)
    customers_slim = customers_lf.select([
        "customer_idx", "customer_id", "age", "FN", "Active",
        "club_member_status", "fashion_news_frequency",
    ]).collect()
    customers_slim.write_parquet(state_dir / "customers.parquet", compression="zstd")
    print(f"  customers: {len(customers_slim):,} rows (includes customer_id hex string)", flush=True)

    # Segment popular — per-bucket
    for bucket, df_bkt in state.seg_pop_by_bucket.items():
        safe_name = bucket.replace("/", "_").replace("<", "lt").replace("+", "plus")
        df_bkt.write_parquet(state_dir / "seg_pop" / f"{safe_name}.parquet", compression="zstd")
    print(f"  seg_pop: {len(state.seg_pop_by_bucket)} buckets", flush=True)

    # Customer history (trimmed to week 104 columns only for bundle)
    history_slim = history_df.select([
        "customer_idx", "article_idx", "t_dat", "price", "sales_channel_id", "week_idx"
    ])
    history_slim.write_parquet(state_dir / "customer_history.parquet", compression="zstd")
    print(f"  customer_history: {len(history_slim):,} rows", flush=True)

    # Article display metadata
    article_meta = articles_lf.select([
        "article_idx", "article_id", "prod_name", "product_type_name",
        "colour_group_name", "department_name",
    ]).collect()
    article_meta.write_parquet(state_dir / "article_meta.parquet", compression="zstd")
    print(f"  article_meta: {len(article_meta):,} rows", flush=True)

    # Raw articles table (needed by compute_interaction_features and compute_article_features)
    # Includes product_code, product_group_name, garment_group_name, etc.
    articles_raw = articles_lf.collect()
    articles_raw.write_parquet(state_dir / "articles_raw.parquet", compression="zstd")
    print(f"  articles_raw: {len(articles_raw):,} rows", flush=True)

    # Week-104 ground truth (display only)
    from src.time_split import build_holdout_fold
    _history_lf, ground_truth, _eval_custs = build_holdout_fold()
    rows = []
    for cust, articles in ground_truth.items():
        for art in articles:
            rows.append({"customer_idx": int(cust), "article_idx": int(art)})
    gt_df = pl.DataFrame(rows, schema={"customer_idx": pl.Int32, "article_idx": pl.Int32})
    gt_df.write_parquet(state_dir / "gt_week104.parquet", compression="zstd")
    print(f"  gt_week104 (display only): {len(gt_df):,} rows", flush=True)

    # ── 3b. Per-customer hit@12/AP@12 arrays (week 104, ranker + heuristic) ──
    print("[3b] Computing per-customer hit@12/AP@12 arrays (week 104)...", flush=True)
    _export_ab_arrays(state_dir, BUNDLE_DIR, PROCESSED)

    # ── 3c. Snapshot evaluation report JSONs for self-contained Docker image ──
    print("[3c] Snapshotting evaluation report JSONs...", flush=True)
    _snapshot_reports(BUNDLE_DIR, REPORTS_DIR)

    # ── 4. Config JSON ────────────────────────────────────────────────────
    print("[4] Writing config.json...", flush=True)
    from src.features.registry import FEATURE_LIST
    from src.explain.reasons import _REASON_REGISTRY

    config = {
        "as_of_week": AS_OF_WEEK,
        "feature_names": [spec.name for spec in FEATURE_LIST],
        "feature_dtypes": {spec.name: str(spec.dtype) for spec in FEATURE_LIST},
        "feature_groups": {spec.name: spec.group for spec in FEATURE_LIST},
        "reason_templates": list(_REASON_REGISTRY.keys()),
        "total_tx_last_week": state.total_tx_last_week,
        "total_decayed_tx": state.total_decayed_tx,
    }
    (BUNDLE_DIR / "config.json").write_text(json.dumps(config, indent=2))

    # ── 5. Load model metrics from holdout_results.json ───────────────────
    print("[5] Loading model metrics...", flush=True)
    holdout_path = REPORTS_DIR / "evaluation" / "holdout_results.json"
    if holdout_path.exists():
        h = json.loads(holdout_path.read_text())
        metrics_summary = {
            "primary_map@12": h["models"]["primary"]["map@12"],
            "primary_lift_vs_heuristic_pct": h["primary_lift_vs_heuristic_pct"],
            "primary_lift_vs_baseline_b_pct": h["primary_lift_vs_baseline_b_pct"],
            "n_eval_customers": h["n_eval_customers"],
            "holdout_week": h["holdout_week"],
        }
    else:
        metrics_summary = {}

    # ── 6. Manifest (SHA-256 all files except manifest itself) ────────────
    print("[6] Computing SHA-256 hashes and writing manifest...", flush=True)
    file_hashes: dict[str, str] = {}
    for fpath in sorted(BUNDLE_DIR.rglob("*")):
        if fpath.is_file() and fpath.name != "manifest.json":
            rel = str(fpath.relative_to(BUNDLE_DIR))
            file_hashes[rel] = _sha256(fpath)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "as_of_week": AS_OF_WEEK,
        "model_file": "model.txt",
        "feature_count": len(FEATURE_LIST),
        "model_metrics_summary": metrics_summary,
        "file_hashes": file_hashes,
    }
    (BUNDLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ── 7. Report ─────────────────────────────────────────────────────────
    total_size = sum(
        f.stat().st_size for f in BUNDLE_DIR.rglob("*") if f.is_file()
    )
    print(f"\n[export_bundle] Bundle written to {BUNDLE_DIR}")
    print(f"  Files: {len(file_hashes):,}")
    print(f"  Total size: {total_size/1e6:.1f} MB")
    print(f"  SHA-256 hashes: {len(file_hashes)}")
    print(f"  Runtime: {time.time()-t0:.1f}s")
    print(f"  Git commit: {manifest['git_commit']}")
    if metrics_summary:
        print(f"  Primary MAP@12 (week 104): {metrics_summary.get('primary_map@12', 'N/A'):.6f}")


if __name__ == "__main__":
    main()
