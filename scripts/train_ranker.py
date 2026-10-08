"""CLI: train LightGBM LambdaRank ranker, tune, evaluate on fold 103.

Usage:
    # Full run with tuning:
    python scripts/train_ranker.py

    # Skip tuning (load saved best params from reports/ranker/tuning.json):
    python scripts/train_ranker.py --skip-tuning

    # Skip tuning and use only default params:
    python scripts/train_ranker.py --skip-tuning --use-defaults
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
import polars as pl

from src.config import HOLDOUT_WEEK, MERGE_PRIORITY, PROCESSED_DIR, REPORTS_DIR, SEGMENT_LABELS, VALIDATION_WEEKS
from src.metrics import map_at_k, per_customer_ap
from src.model.data import (
    FEATURE_NAMES,
    concat_folds,
    count_groups_dropped,
    load_fold,
    prepare_dataset,
)
from src.model.evaluate import (
    bootstrap_paired_diff,
    compute_full_metrics,
    eval_baseline_b_fold103,
    eval_heuristic_fold103,
    score_fold,
    segment_map,
    segment_map_with_bootstrap,
)
from src.model.train import (
    DEFAULT_PARAMS,
    SEED,
    _build_lgb_dataset,
    get_feature_importance,
    train_final,
    train_inner,
)
from src.time_split import build_fold
from src.evaluate import customer_segments

RANKER_DIR = REPORTS_DIR / "ranker"
MODELS_DIR = Path(__file__).parent.parent / "models"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def run_part_a_log() -> None:
    """Log Part A decision to console (parquets already rebuilt by build_features.py)."""
    print("\n" + "=" * 60)
    print("PART A — Popularity Feature Fix")
    print("=" * 60)

    # Load PSI results computed when feature tables were rebuilt
    psi_path = REPORTS_DIR / "features" / "psi_results.json"
    if psi_path.exists():
        with open(psi_path) as f:
            psi_data = json.load(f)
        high_psi = [(r["feature"], r["psi"]) for r in psi_data.get("psi_fold100_vs_fold103", []) if r["psi"] > 0.2]
        print(f"  High-PSI features (>0.2) from Session 3: {len(high_psi)}")
        for feat, psi in high_psi[:5]:
            print(f"    {feat:<40s}  PSI={psi:.4f}")

    print("\n  New features added: popularity_last_week_share, popularity_decayed_share, segment_popular_share")
    print("  Decision: KEEP all raw score columns.")
    print("  Rationale: scalar division does not change distribution shape (PSI unchanged).")
    print("  Shares provide bounded [0,1] representation; model can learn both jointly.")
    print("  Feature count: 65 → 68")


def _log_groups_dropped() -> None:
    print("\n" + "=" * 60)
    print("PART B — Groups with zero positives (training folds 100-102)")
    print("=" * 60)
    total_dropped_groups = 0
    total_dropped_rows = 0
    for fw in [100, 101, 102]:
        df = load_fold(fw)
        stats = count_groups_dropped(df)
        print(f"  Fold {fw}: dropped {stats['dropped_groups']:,} groups / "
              f"{stats['dropped_rows']:,} rows "
              f"(kept {stats['kept_groups']:,}/{stats['total_groups']:,} groups)")
        total_dropped_groups += stats["dropped_groups"]
        total_dropped_rows += stats["dropped_rows"]
    print(f"  Total dropped across folds 100-102: {total_dropped_groups:,} groups, "
          f"{total_dropped_rows:,} rows")
    return total_dropped_groups, total_dropped_rows


def run_tuning(n_trials: int = 30) -> dict:
    """Optuna search: train 100+101, validate on fold_102_full (no downsampling).

    Validation uses fold_102_full.parquet (all candidates, no negative sampling) so
    MAP@12 reflects all fold-102 eval customers — a downsampled fold would inflate
    MAP@12 by ~3-4× and favour wrong hyperparameters.

    Alignment fix: X_v is computed from df_val_sorted and scores are attached to
    the SAME df_val_sorted frame (no re-sort between predict and assignment).
    Early stopping is based on LightGBM's internal ndcg@12 on fold_102_full.
    best_iteration from early stopping is recorded per trial.
    """
    import lightgbm as lgb
    from src.model.evaluate import score_fold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    _FEATURES_DIR = PROCESSED_DIR / "features"
    fold_102_full_path = _FEATURES_DIR / "fold_102_full.parquet"
    if not fold_102_full_path.exists():
        raise FileNotFoundError(
            f"{fold_102_full_path} not found. Run: "
            "python -c 'from src.features.build import build_fold_102_full; build_fold_102_full()'"
        )

    df_train = concat_folds([100, 101])
    df_val_full = pl.read_parquet(fold_102_full_path)  # full fold 102, no downsampling

    X_tr, y_tr, g_tr = prepare_dataset(df_train, drop_zero_positive_groups=True)
    # prepare_dataset sorts by customer_idx; fold_102_full is pre-sorted, so this is a no-op sort
    X_val, y_val, g_val = prepare_dataset(df_val_full, drop_zero_positive_groups=False)

    cat_indices = [FEATURE_NAMES.index(c) for c in
                   ["a_product_group_name", "a_index_group_name",
                    "a_garment_group_name", "a_colour_group_name", "a_section_name"]]

    # LGB datasets: early stopping uses fold_102_full NDCG@12
    ds_train = lgb.Dataset(X_tr, label=y_tr, group=g_tr,
                           categorical_feature=cat_indices, free_raw_data=False)
    ds_val = lgb.Dataset(X_val, label=y_val, group=g_val,
                         categorical_feature=cat_indices, reference=ds_train, free_raw_data=False)

    _, ground_truth_102, _ = build_fold(102)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [12],
            "lambdarank_truncation_level": trial.suggest_int("lambdarank_truncation_level", 5, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
            "n_jobs": 4,
            "verbose": -1,
            "deterministic": True,
            "force_row_wise": True,
            "seed": SEED,
            "data_random_seed": SEED,
            "feature_fraction_seed": SEED,
            "bagging_seed": SEED,
        }

        booster = lgb.train(
            params,
            ds_train,
            num_boost_round=500,
            valid_sets=[ds_val],
            valid_names=["val_102_full"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        # MAP@12 on fold_102_full via score_fold (sorts internally; no re-sort after predict)
        preds_102, _ = score_fold(booster, df_val_full, k=12, sanity_check=False)
        score_val = map_at_k(preds_102, ground_truth_102, k=12)
        # Record best_iteration from early stopping (never hardcode rounds)
        trial.set_user_attr("best_lgb_round", booster.best_iteration)
        return score_val

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Trial 0: default params (baseline for comparison)
    default_trial_params = {
        "lambdarank_truncation_level": DEFAULT_PARAMS["lambdarank_truncation_level"],
        "learning_rate": DEFAULT_PARAMS["learning_rate"],
        "num_leaves": DEFAULT_PARAMS["num_leaves"],
        "min_data_in_leaf": DEFAULT_PARAMS["min_data_in_leaf"],
        "feature_fraction": DEFAULT_PARAMS["feature_fraction"],
        "bagging_fraction": DEFAULT_PARAMS["bagging_fraction"],
        "bagging_freq": DEFAULT_PARAMS["bagging_freq"],
        "lambda_l2": DEFAULT_PARAMS["lambda_l2"],
    }
    study.enqueue_trial(default_trial_params)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    default_score = study.trials[0].value
    best_score = study.best_value
    best_params = study.best_params.copy()
    # best_lgb_round from early stopping for the best trial (never hardcoded)
    best_round = study.best_trial.user_attrs.get("best_lgb_round", 500)

    trials_data = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "best_lgb_round": t.user_attrs.get("best_lgb_round"),
        }
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]

    result = {
        "val_fold": "fold_102_full (no downsampling)",
        "default_params_map12_fold102_full": default_score,
        "best_map12_fold102_full": best_score,
        "tuning_gain": best_score - default_score,
        "best_params": best_params,
        "best_lgb_round_inner": best_round,
        "n_trials": n_trials,
        "trials": trials_data,
    }

    RANKER_DIR.mkdir(parents=True, exist_ok=True)
    with open(RANKER_DIR / "tuning.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def _oracle_map12_fold103() -> float:
    """Read oracle MAP@12 for fold 103 from reports/candidates/summary.json."""
    summary_path = REPORTS_DIR / "candidates" / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    return summary["oracle_map12"]["per_fold"][3]


def run_evaluation(booster, fold_103_df: pl.DataFrame) -> dict:
    """Score fold 103 and compute all metrics. Returns the full eval dict."""
    _, ground_truth_103, _ = build_fold(103)

    # Sanity checks: all recommendations from candidate list, exactly 12, finite scores
    predictions_ranker, _ = score_fold(booster, fold_103_df, k=12, sanity_check=True)

    # Heuristic predictions (final_rank ≤ 12 in fold_103 parquet)
    predictions_heuristic = eval_heuristic_fold103()

    # Baseline B predictions (recency-ordered repurchase + popularity fill)
    print("  Computing Baseline B predictions from fold_103.parquet ...")
    predictions_baseline_b = eval_baseline_b_fold103()

    # Per-customer APs
    ap_ranker = per_customer_ap(predictions_ranker, ground_truth_103)
    ap_heuristic = per_customer_ap(predictions_heuristic, ground_truth_103)
    ap_baseline_b = per_customer_ap(predictions_baseline_b, ground_truth_103)

    # Aggregate metrics
    metrics_ranker = compute_full_metrics(predictions_ranker, ground_truth_103)
    map_heuristic = map_at_k(predictions_heuristic, ground_truth_103, k=12)
    map_baseline_b = map_at_k(predictions_baseline_b, ground_truth_103, k=12)

    # Bootstrap: ranker minus heuristic (1,000 resamples, seed 42)
    boot_vs_heuristic = bootstrap_paired_diff(ap_ranker, ap_heuristic, n=1000, seed=SEED)
    # Bootstrap: ranker minus Baseline B
    boot_vs_baseline_b = bootstrap_paired_diff(ap_ranker, ap_baseline_b, n=1000, seed=SEED)

    # Segment breakdown with per-segment bootstrap CIs
    from src.data_io import load_transactions
    tx_lf = load_transactions()
    segs = customer_segments(tx_lf, 103, list(ground_truth_103.keys()))
    seg_results = segment_map_with_bootstrap(
        predictions_ranker, predictions_heuristic, ground_truth_103, segs,
        k=12, n_boot=1000, seed=SEED,
    )

    return {
        "fold": 103,
        "n_eval_customers": len(ground_truth_103),
        "ranker": metrics_ranker,
        "heuristic": {
            "map@12": map_heuristic,
            "note": "final_rank <= 12 from fold_103.parquet",
        },
        "baseline_b": {
            "map@12": map_baseline_b,
            "note": "recency-ordered repurchase + popularity fill from fold_103 feature columns",
        },
        "oracle": {
            "map@12": _oracle_map12_fold103(),
            "note": "from reports/candidates/summary.json k=200, fold 103 per_fold[3]",
        },
        "bootstrap_ranker_vs_heuristic": boot_vs_heuristic,
        "bootstrap_ranker_vs_baseline_b": boot_vs_baseline_b,
        "segment_breakdown": seg_results,
    }


def run_ablations(best_params: dict, final_rounds: int) -> dict:
    """Ablation: drop one feature group at a time, evaluate on fold 103.

    Uses in-place NaN masking + restore to avoid full array copies:
    peak memory = X_full (1GB) + X_val_full (3.9GB) + small column backups,
    instead of 2×X_full + 2×X_val_full (~10 GB).
    """
    import resource
    import lightgbm as lgb
    from src.features.registry import FEATURE_LIST

    _, ground_truth_103, _ = build_fold(103)
    df_val = load_fold(103)

    groups = ["candidate", "customer", "article", "interaction"]
    group_features = {
        g: [i for i, spec in enumerate(FEATURE_LIST) if spec.group == g]
        for g in groups
    }

    cat_indices = [FEATURE_NAMES.index(c) for c in
                   ["a_product_group_name", "a_index_group_name",
                    "a_garment_group_name", "a_colour_group_name", "a_section_name"]]

    # Build Float32 arrays once; all ablations reuse these arrays in-place.
    df_train_full = concat_folds([100, 101, 102])
    X_full, y_full, g_full = prepare_dataset(df_train_full, drop_zero_positive_groups=True)
    del df_train_full

    df_val_sorted = df_val.sort("customer_idx")
    X_val_full = (
        df_val_sorted.select(FEATURE_NAMES)
        .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
        .to_numpy(allow_copy=True)
    )

    # Full model (re-train to get a within-ablation baseline with identical seeds)
    ds_full = lgb.Dataset(X_full, label=y_full, group=g_full,
                          categorical_feature=cat_indices, free_raw_data=True)
    booster_full = lgb.train(
        best_params,
        ds_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    scores_full = booster_full.predict(X_val_full)
    df_scored_full = df_val_sorted.with_columns(
        pl.Series("_score", scores_full, dtype=pl.Float64)
    )
    preds_full = _extract_top12(df_scored_full)
    ap_full = per_customer_ap(preds_full, ground_truth_103)
    map_full = map_at_k(preds_full, ground_truth_103)
    del booster_full, ds_full

    rss_mb_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

    ablation_results = []
    for grp in groups:
        feat_indices = group_features[grp]

        # Backup and zero out columns in-place (avoids full array copy)
        bkup_train = X_full[:, feat_indices].copy()
        bkup_val = X_val_full[:, feat_indices].copy()
        X_full[:, feat_indices] = np.nan
        X_val_full[:, feat_indices] = np.nan

        ds_ablate = lgb.Dataset(X_full, label=y_full, group=g_full,
                                categorical_feature=cat_indices, free_raw_data=True)
        booster_ablate = lgb.train(
            best_params,
            ds_ablate,
            num_boost_round=final_rounds,
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        scores_ablate = booster_ablate.predict(X_val_full)

        # Restore before scoring (scores already captured above)
        X_full[:, feat_indices] = bkup_train
        X_val_full[:, feat_indices] = bkup_val
        del bkup_train, bkup_val, ds_ablate

        df_scored_ab = df_val_sorted.with_columns(
            pl.Series("_score", scores_ablate, dtype=pl.Float64)
        )
        preds_ablate = _extract_top12(df_scored_ab)
        ap_ablate = per_customer_ap(preds_ablate, ground_truth_103)
        map_ablate = map_at_k(preds_ablate, ground_truth_103)
        delta_map = map_ablate - map_full

        boot = bootstrap_paired_diff(ap_ablate, ap_full, n=1000, seed=SEED)
        ablation_results.append({
            "group_dropped": grp,
            "n_features_dropped": len(feat_indices),
            "map@12": map_ablate,
            "delta_map@12": delta_map,
            "bootstrap": boot,
        })
        print(f"  Ablate {grp}: MAP@12={map_ablate:.6f}  delta={delta_map:+.6f}  "
              f"CI=[{boot['ci_lo']:+.6f}, {boot['ci_hi']:+.6f}]  "
              f"CI excl 0: {boot['ci_excludes_zero']}")
        del booster_ablate

    rss_mb_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

    return {
        "full_model_map@12": map_full,
        "ablations": ablation_results,
        "peak_rss_mb": rss_mb_peak,
        "rss_mb_before_ablations": rss_mb_start,
    }


def _extract_top12(df_scored: pl.DataFrame) -> dict[int, list[int]]:
    """Extract top-12 predictions from a scored DataFrame (stable tiebreaker: article_idx ASC)."""
    top12 = (
        df_scored
        .sort(["customer_idx", "_score", "article_idx"], descending=[False, True, False])
        .with_columns(
            (pl.int_range(pl.len(), dtype=pl.Int32).over("customer_idx")).alias("_rn")
        )
        .filter(pl.col("_rn") < 12)
    )
    return {
        row["customer_idx"]: list(row["articles"])
        for row in top12.group_by("customer_idx", maintain_order=True)
        .agg(pl.col("article_idx").alias("articles"))
        .iter_rows(named=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Load best params from reports/ranker/tuning.json")
    parser.add_argument("--use-defaults", action="store_true",
                        help="Use default params (ignores any saved tuning)")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--skip-ablations", action="store_true",
                        help="Skip Part E ablations (run in Part 2)")
    parser.add_argument("--skip-latency", action="store_true",
                        help="Skip Part F inference latency benchmark (run in Part 2)")
    args = parser.parse_args()

    RANKER_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ---- Part A log ----
    run_part_a_log()

    # ---- Part B: groups dropped ----
    dropped_groups, dropped_rows = _log_groups_dropped()

    # ---- Part C: Tuning ----
    print("\n" + "=" * 60)
    print("PART C — Hyperparameter Tuning")
    print("=" * 60)

    tuning_path = RANKER_DIR / "tuning.json"

    if args.use_defaults:
        best_params = DEFAULT_PARAMS.copy()
        inner_rounds = 500
        default_score = None
        best_score = None
        print("  Using default params (--use-defaults)")
    elif args.skip_tuning and tuning_path.exists():
        with open(tuning_path) as f:
            tuning_data = json.load(f)
        best_params_raw = tuning_data["best_params"]
        inner_rounds = tuning_data["best_lgb_round_inner"]
        # Support both old and new key names
        default_score = tuning_data.get(
            "default_params_map12_fold102_full",
            tuning_data.get("default_params_map12_fold102"),
        )
        best_score = tuning_data.get(
            "best_map12_fold102_full",
            tuning_data.get("best_map12_fold102"),
        )
        best_params = DEFAULT_PARAMS.copy()
        best_params.update(best_params_raw)
        print(f"  Loaded tuning results from {tuning_path}")
        if default_score is not None:
            print(f"  Default MAP@12 (fold_102_full): {default_score:.6f}")
        if best_score is not None:
            print(f"  Best MAP@12   (fold_102_full): {best_score:.6f}  "
                  f"(gain={best_score - default_score:+.6f})")
    else:
        print(f"  Running Optuna ({args.n_trials} trials, val=fold_102_full) ...")
        tuning_result = run_tuning(n_trials=args.n_trials)
        inner_rounds = tuning_result["best_lgb_round_inner"]
        default_score = tuning_result["default_params_map12_fold102_full"]
        best_score = tuning_result["best_map12_fold102_full"]
        best_params = DEFAULT_PARAMS.copy()
        best_params.update(tuning_result["best_params"])
        print(f"  Default MAP@12 (fold_102_full): {default_score:.6f}")
        print(f"  Best MAP@12   (fold_102_full): {best_score:.6f}  "
              f"(gain={best_score - default_score:+.6f})")
        print(f"  Best inner-loop rounds (from early stopping): {inner_rounds}")

    # ---- Final model ----
    print("\n" + "=" * 60)
    print("PART C/D — Training Final Model (folds 100-102)")
    print("=" * 60)

    final_booster, final_rounds, scale_ratio, scale_rule = train_final(best_params, inner_rounds)
    print(f"  Inner rounds: {inner_rounds} → Final rounds: {final_rounds} "
          f"(ratio={scale_ratio:.3f}; rule: {scale_rule})")

    model_path = MODELS_DIR / "lgbm_ranker.txt"
    final_booster.save_model(str(model_path))
    print(f"  Model saved → {model_path}")

    # ---- Part D: Evaluation on fold 103 ----
    print("\n" + "=" * 60)
    print("PART D — Evaluation on Fold 103")
    print("=" * 60)

    fold_103_df = load_fold(103)
    eval_result = run_evaluation(final_booster, fold_103_df)
    m = eval_result["ranker"]
    h = eval_result["heuristic"]
    bb = eval_result["baseline_b"]
    oc = eval_result["oracle"]
    boot = eval_result["bootstrap_ranker_vs_heuristic"]

    boot_bb = eval_result["bootstrap_ranker_vs_baseline_b"]
    print(f"\n  Comparison table (fold 103, n={eval_result['n_eval_customers']:,}):")
    print(f"  {'Model':<30s}  {'MAP@12':>10s}  {'NDCG@12':>10s}  {'recall@12':>10s}  {'prec@12':>10s}")
    print(f"  {'Ranker (LightGBM LambdaRank)':<30s}  {m['map@12']:>10.6f}  {m.get('ndcg@12', float('nan')):>10.6f}  {m.get('recall@12', float('nan')):>10.6f}  {m.get('precision@12', float('nan')):>10.6f}")
    print(f"  {'Heuristic':<30s}  {h['map@12']:>10.6f}")
    print(f"  {'Baseline B (recency + pop fill)':<30s}  {bb['map@12']:>10.6f}")
    print(f"  {'Oracle (k=200 perfect rank)':<30s}  {oc['map@12']:>10.6f}")

    print(f"\n  Bootstrap ranker vs heuristic (1,000 resamples, seed 42):")
    print(f"    mean diff={boot['mean_diff']:+.6f}  "
          f"95% CI=[{boot['ci_lo']:+.6f}, {boot['ci_hi']:+.6f}]  "
          f"CI excludes zero: {boot['ci_excludes_zero']}")
    print(f"    relative lift: {boot['relative_lift']:+.2%}")
    print(f"\n  Bootstrap ranker vs Baseline B:")
    print(f"    mean diff={boot_bb['mean_diff']:+.6f}  "
          f"95% CI=[{boot_bb['ci_lo']:+.6f}, {boot_bb['ci_hi']:+.6f}]  "
          f"CI excludes zero: {boot_bb['ci_excludes_zero']}")
    print(f"    relative lift: {boot_bb['relative_lift']:+.2%}")
    print(f"\n  Segment breakdown (ranker vs heuristic):")
    for seg in eval_result.get("segment_breakdown", []):
        seg_boot = seg.get("bootstrap_ranker_vs_heuristic", {})
        print(f"    seg={seg['segment']:5s} n={seg['n_customers']:6,}  "
              f"ranker={seg.get('map@12_ranker', float('nan')):.6f}  "
              f"heuristic={seg.get('map@12_heuristic', float('nan')):.6f}  "
              f"CI=[{seg_boot.get('ci_lo', float('nan')):+.6f}, {seg_boot.get('ci_hi', float('nan')):+.6f}]")

    RANKER_DIR.mkdir(parents=True, exist_ok=True)
    eval_out = RANKER_DIR / "eval_fold103.json"
    with open(eval_out, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"\n  Eval saved → {eval_out}")

    # ---- Part E: Ablations (Part 2) ----
    if args.skip_ablations:
        print("\n[SKIP] Part E ablations deferred to Part 2 (--skip-ablations)")
        ablation_result = None
    else:
        print("\n" + "=" * 60)
        print("PART E — Feature Group Ablations")
        print("=" * 60)
        ablation_result = run_ablations(best_params, final_rounds)
        ablation_result["gain_importance_top25"] = get_feature_importance(final_booster, "gain")[:25]
        ablation_out = RANKER_DIR / "ablations.json"
        with open(ablation_out, "w") as f:
            json.dump(ablation_result, f, indent=2)
        print(f"\n  Ablations saved → {ablation_out}")

    # ---- Part F: Inference latency (Part 2) ----
    lat = None
    if args.skip_latency:
        print("\n[SKIP] Part F latency benchmark deferred to Part 2 (--skip-latency)")
    else:
        print("\n" + "=" * 60)
        print("PART F — Inference Latency")
        print("=" * 60)
        from src.model.inference import Recommender
        reco = Recommender(model_path=model_path)
        rng = np.random.default_rng(SEED)
        _, ground_truth_103, eval_custs = build_fold(103)
        bench_customers = rng.choice(eval_custs, size=min(100, len(eval_custs)), replace=False).tolist()
        # 5 warm-up calls (state is pre-built during Recommender.build_state)
        lat = reco.latency_benchmark(bench_customers, as_of_week=103, warmup=5)
        print(f"  p50={lat['p50_ms']:.1f}ms  p95={lat['p95_ms']:.1f}ms  "
              f"n={lat['n_calls']}  RSS={lat['rss_mb']:.0f}MB")

    # ---- Model card ----
    card = {
        "feature_list": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "params": best_params,
        "training_folds": [100, 101, 102],
        "inner_loop_folds": [100, 101],
        "val_fold_tuning": "fold_102_full (no downsampling)",
        "inner_rounds": inner_rounds,
        "final_rounds": final_rounds,
        "scale_rule": scale_rule,
        "scale_ratio": scale_ratio,
        "default_map12_fold102_full": default_score,
        "tuned_map12_fold102_full": best_score,
        "tuning_gain": (best_score - default_score) if (best_score is not None and default_score is not None) else None,
        "tuned_better_than_default": (best_score > default_score) if (best_score is not None and default_score is not None) else None,
        "fold103_metrics": eval_result["ranker"],
        "bootstrap_vs_heuristic": eval_result["bootstrap_ranker_vs_heuristic"],
        "bootstrap_vs_baseline_b": eval_result["bootstrap_ranker_vs_baseline_b"],
        "segment_breakdown": eval_result["segment_breakdown"],
        "inference_latency": lat,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "groups_dropped_training": {
            "n_groups": dropped_groups,
            "n_rows": dropped_rows,
        },
    }
    card_path = MODELS_DIR / "model_card.json"
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"\n  Model card saved → {card_path}")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Total runtime: {total_time/60:.1f}min")
    print(f"MAP@12 fold 103: {eval_result['ranker']['map@12']:.6f}")
    print(f"Bootstrap CI excludes zero vs heuristic: {eval_result['bootstrap_ranker_vs_heuristic']['ci_excludes_zero']}")
    print(f"Bootstrap CI excludes zero vs Baseline B: {eval_result['bootstrap_ranker_vs_baseline_b']['ci_excludes_zero']}")
    if default_score is not None and best_score is not None:
        tuning_helps = best_score > default_score
        print(f"Tuning helped: {tuning_helps} "
              f"(default={default_score:.6f}, tuned={best_score:.6f}, "
              f"gain={best_score-default_score:+.6f})")


if __name__ == "__main__":
    main()
