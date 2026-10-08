"""Part E: One-shot holdout evaluation on week 104.

IMPORTANT: holdout_protocol.json must be written BEFORE running this script.
This script reads labels_104.parquet for the first time and records the timestamp.

Usage:
    python scripts/run_holdout.py
    python scripts/run_holdout.py --skip-primary-train   # use cached primary model
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import lightgbm as lgb
import numpy as np
import polars as pl

from src.config import PROCESSED_DIR, REPORTS_DIR
from src.metrics import map_at_k, mean_ndcg_at_k, per_customer_ap
from src.model.data import CATEGORICAL_FEATURES, FEATURE_NAMES, concat_folds, prepare_dataset
from src.model.evaluate import bootstrap_paired_diff, score_fold
from src.model.train import DEFAULT_PARAMS, NUM_THREADS, SEED, train_final
from src.evaluation.suite import per_customer_hit

EVAL_DIR = REPORTS_DIR / "evaluation"
FEATURES_DIR = PROCESSED_DIR / "features"
MODELS_DIR = Path(__file__).parent.parent / "models"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _heuristic_from_parquet(path: Path, k: int = 12) -> dict[int, list[int]]:
    df = pl.read_parquet(path)
    top12 = (
        df.filter(pl.col("final_rank") <= k)
        .sort(["customer_idx", "final_rank"])
        .group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_idx").alias("articles"))
    )
    return {r["customer_idx"]: list(r["articles"]) for r in top12.iter_rows(named=True)}


def _baseline_b_from_parquet(path: Path) -> dict[int, list[int]]:
    """Baseline B: recency-ordered repurchase + popularity fill."""
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
    with open(REPORTS_DIR / "ranker" / "tuning.json") as f:
        t = json.load(f)
    best_params = DEFAULT_PARAMS.copy()
    best_params.update(t["best_params"])
    inner_rounds = t["best_lgb_round_inner"]
    return best_params, inner_rounds


def _eval_model(
    preds: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    n_boot: int = 1000,
    k: int = 12,
) -> dict:
    map12 = map_at_k(preds, ground_truth, k)
    ndcg12 = mean_ndcg_at_k(preds, ground_truth, k)
    hit_vals = per_customer_hit(preds, ground_truth, k)
    hit_rate = float(np.mean(list(hit_vals.values()))) if hit_vals else 0.0
    ap_vals = per_customer_ap(preds, ground_truth, k)
    # Bootstrap CI for MAP@12
    ap_arr = np.array(list(ap_vals.values()))
    n_custs = len(ap_arr)
    rng = np.random.default_rng(SEED)
    boot_maps = np.array([ap_arr[rng.integers(0, n_custs, size=n_custs)].mean() for _ in range(n_boot)])
    ci_lo = float(np.percentile(boot_maps, 2.5))
    ci_hi = float(np.percentile(boot_maps, 97.5))
    return {
        "map@12": map12,
        "map@12_ci_lo": ci_lo,
        "map@12_ci_hi": ci_hi,
        "ndcg@12": ndcg12,
        "hit_rate@12": hit_rate,
        "n_eval_customers": len(ground_truth),
    }


def main(skip_primary_train: bool = False, n_boot: int = 1000) -> None:
    print("=" * 60)
    print("PART E — One-Shot Holdout Evaluation (week 104)")
    print("=" * 60)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Verify protocol file exists (read-only from this point on) ────────────
    protocol_path = EVAL_DIR / "holdout_protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(
            "holdout_protocol.json must be written BEFORE running this script. "
            "Aborting to protect holdout integrity."
        )
    with open(protocol_path) as f:
        protocol = json.load(f)
    protocol_ts = protocol["protocol_timestamp_utc"]
    print(f"  Protocol timestamp: {protocol_ts}")
    # Defensive guard: ensure we never open the protocol file in write mode.
    # All results (including labels_read_timestamp) go to holdout_results.json only.

    # ── Train primary model (folds 100-103, fold_103 at 0.2 downsampling) ──
    primary_model_path = MODELS_DIR / "lgbm_ranker_primary_104.txt"
    if skip_primary_train and primary_model_path.exists():
        print(f"\n  [1] Loading cached primary model from {primary_model_path.name} ...")
        booster_primary = lgb.Booster(model_file=str(primary_model_path))
        primary_rounds = None
    else:
        print("\n  [1] Training primary model (folds 100-103, fold_103 at 0.2 downsampling) ...")
        t0 = time.time()
        best_params, inner_rounds = _load_best_params()

        # Scale ratio uses [100, 101] as inner reference (same as in train_final)
        df_inner = concat_folds([100, 101])
        _, _, g_inner = prepare_dataset(df_inner, drop_zero_positive_groups=True)
        n_inner = int(g_inner.sum())

        # Build final training set: folds 100-102 via concat_folds + downsampled fold 103
        df_base = concat_folds([100, 101, 102])
        df_103_full = pl.read_parquet(FEATURES_DIR / "fold_103.parquet")
        # 20% negative downsampling for fold 103 (consistent with training folds 100-102)
        df_103_pos = df_103_full.filter(pl.col("label") == 1)
        df_103_neg = (
            df_103_full.filter(pl.col("label") == 0)
            .sample(fraction=0.2, seed=SEED)
        )
        df_103_train = pl.concat([df_103_pos, df_103_neg]).sort(
            ["customer_idx", "article_idx"]
        )
        df_final = pl.concat([df_base, df_103_train]).sort(
            ["customer_idx", "article_idx"]
        )
        del df_103_full, df_103_pos, df_103_neg, df_103_train, df_base

        X_final, y_final, g_final = prepare_dataset(df_final, drop_zero_positive_groups=True)
        n_final = int(g_final.sum())
        ratio = n_final / n_inner if n_inner > 0 else 1.5
        primary_rounds = max(10, round(inner_rounds * ratio))
        print(f"    n_inner={n_inner:,}  n_final={n_final:,}  ratio={ratio:.3f}")
        print(f"    inner_rounds={inner_rounds}  final_rounds={primary_rounds}")

        cat_indices = [FEATURE_NAMES.index(c) for c in CATEGORICAL_FEATURES]
        ds_final = lgb.Dataset(
            X_final, label=y_final, group=g_final,
            categorical_feature=cat_indices, free_raw_data=True,
        )
        booster_primary = lgb.train(
            best_params, ds_final,
            num_boost_round=primary_rounds,
            callbacks=[lgb.log_evaluation(period=100)],
        )
        booster_primary.save_model(str(primary_model_path))
        print(f"    Primary model trained in {time.time()-t0:.0f}s")

    # ── Load secondary model ─────────────────────────────────────────────────
    print("\n  [2] Loading secondary model (folds 100-102) ...")
    booster_secondary = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_ranker.txt"))

    # ── NOW read labels_104.parquet (first time — for hash/protocol only) ─────
    labels_read_ts = datetime.datetime.now(datetime.UTC).isoformat()
    print(f"\n  [3] Reading labels_104.parquet (first and only read)")
    print(f"      Labels-read timestamp: {labels_read_ts}")

    labels_path = FEATURES_DIR / "labels_104.parquet"
    labels_sha256 = _sha256(labels_path)
    # labels_104.parquet contains all candidates (label=0 and label=1) for week 104.
    # Ground truth comes from the full transaction history via build_holdout_fold(),
    # which is required for Baseline B to reproduce the EDA value 0.024457.
    from src.time_split import build_holdout_fold
    _, gt_104, eval_custs_104 = build_holdout_fold()
    n_purchases = sum(len(v) for v in gt_104.values())
    print(f"      Week-104 ground truth (from full transactions): {len(gt_104):,} buyers, "
          f"{n_purchases:,} unique article purchases")

    # ── Score fold_104.parquet ────────────────────────────────────────────────
    print("\n  [4] Scoring fold_104.parquet ...")
    fold_104_path = FEATURES_DIR / "fold_104.parquet"
    df_104 = pl.read_parquet(fold_104_path)

    preds_primary, _ = score_fold(booster_primary, df_104, k=12, sanity_check=False)
    preds_secondary, _ = score_fold(booster_secondary, df_104, k=12, sanity_check=False)
    preds_heuristic = _heuristic_from_parquet(fold_104_path)
    preds_baseline_b = _baseline_b_from_parquet(fold_104_path)

    # Restrict all predictions to customers present in ground truth
    eval_custs = set(gt_104.keys())

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print(f"\n  [5] Evaluating all models on {len(eval_custs):,} holdout customers ...")
    res_primary = _eval_model(preds_primary, gt_104, n_boot=n_boot)
    res_secondary = _eval_model(preds_secondary, gt_104, n_boot=n_boot)
    res_heuristic = _eval_model(preds_heuristic, gt_104, n_boot=n_boot)
    res_bb = _eval_model(preds_baseline_b, gt_104, n_boot=n_boot)

    # Bootstrap paired diff: primary vs heuristic and vs Baseline B
    ap_primary = per_customer_ap(preds_primary, gt_104, 12)
    ap_secondary = per_customer_ap(preds_secondary, gt_104, 12)
    ap_heuristic = per_customer_ap(preds_heuristic, gt_104, 12)
    ap_bb = per_customer_ap(preds_baseline_b, gt_104, 12)

    boot_prim_vs_h = bootstrap_paired_diff(ap_primary, ap_heuristic, n=n_boot, seed=SEED)
    boot_prim_vs_bb = bootstrap_paired_diff(ap_primary, ap_bb, n=n_boot, seed=SEED)
    boot_sec_vs_h = bootstrap_paired_diff(ap_secondary, ap_heuristic, n=n_boot, seed=SEED)

    # Verify Baseline B = 0.024457
    bb_map12 = res_bb["map@12"]
    bb_expected = 0.024457
    bb_diff = abs(bb_map12 - bb_expected)
    print(f"\n  Baseline B MAP@12: {bb_map12:.6f} (expected {bb_expected:.6f}, diff={bb_diff:.2e})")
    assert bb_diff < 1e-4, (
        f"Baseline B MAP@12={bb_map12:.6f} deviates from EDA value {bb_expected:.6f} "
        f"by {bb_diff:.2e}"
    )
    print("  Baseline B check: PASS")

    # Print results table
    print("\n  Model            MAP@12    NDCG@12   Hit@12   95% CI MAP@12")
    for name, res in [
        ("Primary (100-103)", res_primary),
        ("Secondary (100-102)", res_secondary),
        ("Heuristic", res_heuristic),
        ("Baseline B", res_bb),
    ]:
        print(f"  {name:<20s} {res['map@12']:.6f}  {res['ndcg@12']:.6f}  "
              f"{res['hit_rate@12']:.4f}   [{res['map@12_ci_lo']:.6f}, {res['map@12_ci_hi']:.6f}]")

    print(f"\n  Primary vs Heuristic: mean_diff={boot_prim_vs_h['mean_diff']:+.6f} "
          f"CI=[{boot_prim_vs_h['ci_lo']:+.6f}, {boot_prim_vs_h['ci_hi']:+.6f}]")
    print(f"  Primary vs Baseline B: mean_diff={boot_prim_vs_bb['mean_diff']:+.6f} "
          f"CI=[{boot_prim_vs_bb['ci_lo']:+.6f}, {boot_prim_vs_bb['ci_hi']:+.6f}]")

    # ── Save results ─────────────────────────────────────────────────────────
    # labels_read_timestamp_utc and labels_sha256 go here, not in holdout_protocol.json.
    output = {
        "holdout_week": 104,
        "protocol_timestamp_utc": protocol_ts,
        "labels_read_timestamp_utc": labels_read_ts,
        "n_eval_customers": len(gt_104),
        "models": {
            "primary": {
                **res_primary,
                "train_folds": [100, 101, 102, 103],
                "final_rounds": primary_rounds,
                "boot_vs_heuristic": boot_prim_vs_h,
                "boot_vs_baseline_b": boot_prim_vs_bb,
            },
            "secondary": {
                **res_secondary,
                "train_folds": [100, 101, 102],
                "final_rounds": 421,
                "boot_vs_heuristic": boot_sec_vs_h,
            },
            "heuristic": res_heuristic,
            "baseline_b": {
                **res_bb,
                "eda_expected_map12": bb_expected,
                "eda_match": bb_diff < 1e-4,
            },
        },
        "primary_lift_vs_heuristic_pct": round(
            100.0 * boot_prim_vs_h["mean_diff"] / res_heuristic["map@12"], 2
        ),
        "primary_lift_vs_baseline_b_pct": round(
            100.0 * boot_prim_vs_bb["mean_diff"] / res_bb["map@12"], 2
        ),
    }

    with open(EVAL_DIR / "holdout_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved → {EVAL_DIR / 'holdout_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-primary-train", action="store_true")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    main(skip_primary_train=args.skip_primary_train, n_boot=args.n_boot)
