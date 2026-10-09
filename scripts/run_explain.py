"""SHAP explainability pipeline for the H&M recommendation system.

Parts:
  A. Compute SHAP on 5,000 fold-103 eval customers (seed 42).
     Additivity check: max error reported.
  B. Global explanations: mean |SHAP| per feature (all candidates and top-12 rows).
     Group-level SHAP vs ablation deltas. Dependence plots for top 5 features.
     Segment comparison: cold-start vs 20+ buyers.
  C. Local explanations: within-list contrast, reason codes, faithfulness check,
     5 worked examples saved to reports/explain/examples.json.

Usage:
    caffeinate -i python -u scripts/run_explain.py 2>&1
    caffeinate -i python -u scripts/run_explain.py --from-cache 2>&1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

matplotlib.use("Agg")

ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "models" / "lgbm_ranker.txt"
FOLD103_PATH = ROOT / "data" / "processed" / "features" / "fold_103.parquet"
REPORTS_DIR = ROOT / "reports" / "explain"
FIGURES_DIR = ROOT / "reports" / "figures" / "explain"
ABL_PATH = ROOT / "reports" / "ranker" / "ablations.json"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

N_CUSTOMERS = 5_000
SEED = 42


_CACHE_FILES = {
    "shap_vals": "shap_vals_all.npy",
    "expected_val": "expected_val_all.npy",
    "scores": "scores_all.npy",
    "top12_mask": "top12_mask.npy",
    "df_sample": "df_sample.parquet",
}


def _cache_exists() -> bool:
    return all((REPORTS_DIR / fname).exists() for fname in _CACHE_FILES.values())


def _load_and_verify_cache(booster) -> tuple:
    """Load the 5 cached SHAP files and verify integrity before use.

    Checks:
    - Row count matches between shap_vals, expected_val, scores, top12_mask, df_sample.
    - Customer count matches N_CUSTOMERS.
    - Scores from cache match booster.predict(X) within 1e-6 (computed fresh).

    Returns (df_sample, shap_vals, expected_val, scores).
    Raises ValueError on any mismatch.
    """
    from src.model.data import FEATURE_NAMES

    print("[A] --from-cache: loading pre-computed SHAP arrays...", flush=True)
    shap_vals = np.load(str(REPORTS_DIR / "shap_vals_all.npy"))
    expected_val = np.load(str(REPORTS_DIR / "expected_val_all.npy"))
    scores_cached = np.load(str(REPORTS_DIR / "scores_all.npy"))
    top12_mask = np.load(str(REPORTS_DIR / "top12_mask.npy"))
    df_sample = pl.read_parquet(str(REPORTS_DIR / "df_sample.parquet"))

    n_rows = len(df_sample)
    n_custs = df_sample["customer_idx"].n_unique()

    # Row count consistency
    for name, arr in [("shap_vals", shap_vals), ("expected_val", expected_val),
                      ("scores", scores_cached), ("top12_mask", top12_mask)]:
        if len(arr) != n_rows:
            raise ValueError(f"Cache mismatch: {name} has {len(arr)} rows but df_sample has {n_rows}")

    # Customer count
    if n_custs != N_CUSTOMERS:
        raise ValueError(f"Cache mismatch: n_customers={n_custs} but expected {N_CUSTOMERS}")

    # Recompute scores fresh and compare within 1e-6
    X_check = (
        df_sample.select(FEATURE_NAMES)
        .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
        .to_numpy(allow_copy=True)
    ).astype(np.float32)
    scores_fresh = booster.predict(X_check).astype(np.float32)
    max_score_diff = float(np.max(np.abs(scores_cached.astype(np.float64) - scores_fresh.astype(np.float64))))
    if max_score_diff > 1e-6:
        raise ValueError(f"Cache mismatch: scores differ by {max_score_diff:.2e} (tol=1e-6)")

    print(f"[A] Cache verified: {n_rows:,} rows, {n_custs:,} customers, "
          f"max score diff={max_score_diff:.2e}", flush=True)

    return df_sample, shap_vals, expected_val, scores_cached


def main():
    parser = argparse.ArgumentParser(description="SHAP explainability pipeline")
    parser.add_argument(
        "--from-cache", action="store_true",
        help="Load pre-computed SHAP arrays instead of recomputing TreeSHAP."
    )
    args = parser.parse_args()

    t0_global = time.time()
    print(f"[run_explain] Loading model from {MODEL_PATH}", flush=True)
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    from src.explain.shap_values import (
        compute_within_list_contrast,
        get_top12_mask,
        group_shap,
        load_fold103_sample,
        mean_abs_shap_by_feature,
        mean_abs_contrast_shap_by_feature,
    )
    from src.explain.reasons import (
        REASON_FEATURES,
        faithfulness_check,
        generate_reasons,
    )
    from src.features.registry import FEATURE_LIST
    from src.model.data import FEATURE_NAMES

    feature_groups = {spec.name: spec.group for spec in FEATURE_LIST}

    # ── Part A: SHAP computation (or load from cache) ─────────────────────────
    if args.from_cache:
        if not _cache_exists():
            raise FileNotFoundError(
                "Cache files not found in reports/explain/. "
                "Run without --from-cache first to compute SHAP."
            )
        t0 = time.time()
        df_sample, shap_vals, expected_val, scores = _load_and_verify_cache(booster)
        n_rows = len(df_sample)
        n_custs = df_sample["customer_idx"].n_unique()
        print(f"[A] Loaded {n_rows:,} rows for {n_custs:,} customers in {time.time()-t0:.1f}s", flush=True)
    else:
        print(f"\n[A] Loading fold_103 and sampling {N_CUSTOMERS} customers (seed {SEED})", flush=True)
        t0 = time.time()
        df_sample, shap_vals, expected_val, scores = load_fold103_sample(
            str(FOLD103_PATH), booster, n_customers=N_CUSTOMERS, seed=SEED
        )
        n_rows = len(df_sample)
        n_custs = df_sample["customer_idx"].n_unique()
        print(f"[A] Loaded {n_rows:,} rows for {n_custs:,} customers in {time.time()-t0:.1f}s", flush=True)

    # Build X_full from df_sample (already in memory)
    X_full = (
        df_sample.select(FEATURE_NAMES)
        .with_columns([pl.col(n).cast(pl.Float32) for n in FEATURE_NAMES])
        .to_numpy(allow_copy=True)
    ).astype(np.float32)

    # Additivity: use already-computed shap_vals + expected_val (no re-compute)
    print("[A] Checking additivity from pre-computed SHAP arrays...", flush=True)
    sv64 = shap_vals.astype(np.float64)
    ev64 = expected_val.astype(np.float64)
    s64 = scores.astype(np.float64)
    errors_all = np.abs(sv64.sum(axis=1) + ev64 - s64)
    add_result_all = {
        "max_error": float(errors_all.max()),
        "mean_error": float(errors_all.mean()),
        "n_rows": int(n_rows),
        "tol": 1e-5,
        "passed": bool(errors_all.max() <= 1e-5),
    }
    print(f"[A] Additivity (all candidates): max_error={add_result_all['max_error']:.2e}  "
          f"passed={add_result_all['passed']}", flush=True)

    # Top-12 mask
    top12_mask = get_top12_mask(df_sample, scores)
    df_top12 = df_sample.filter(pl.Series(top12_mask))
    shap_top12 = shap_vals[top12_mask]
    scores_top12 = scores[top12_mask]
    print(f"[A] Top-12 rows: {top12_mask.sum():,}", flush=True)

    # Additivity on top-12 rows (also from pre-computed arrays)
    sv12_64 = shap_top12.astype(np.float64)
    ev12_64 = expected_val[top12_mask].astype(np.float64)
    s12_64 = scores_top12.astype(np.float64)
    errors_top12 = np.abs(sv12_64.sum(axis=1) + ev12_64 - s12_64)
    add_result_top12 = {
        "max_error": float(errors_top12.max()),
        "mean_error": float(errors_top12.mean()),
        "n_rows": int(top12_mask.sum()),
        "tol": 1e-5,
        "passed": bool(errors_top12.max() <= 1e-5),
    }
    print(f"[A] Additivity (top-12 rows): max_error={add_result_top12['max_error']:.2e}  "
          f"passed={add_result_top12['passed']}", flush=True)

    # Save SHAP arrays as numpy files for notebook (skip if loaded from cache)
    if not args.from_cache:
        np.save(str(REPORTS_DIR / "shap_vals_all.npy"), shap_vals)
        np.save(str(REPORTS_DIR / "expected_val_all.npy"), expected_val)
        np.save(str(REPORTS_DIR / "scores_all.npy"), scores)
        np.save(str(REPORTS_DIR / "top12_mask.npy"), top12_mask)
        df_sample.write_parquet(str(REPORTS_DIR / "df_sample.parquet"))
        print("[A] SHAP arrays and sample parquet saved.", flush=True)
    else:
        print("[A] Using cached SHAP arrays (not re-saving).", flush=True)

    # ── Compute within-list contrast (needed for both Part B and Part C) ──────
    print("[A] Computing within-list contrast for all candidates...", flush=True)
    customer_indices = df_sample["customer_idx"].to_numpy()
    contrast_all = compute_within_list_contrast(shap_vals, customer_indices)
    contrast_top12 = contrast_all[top12_mask]
    print(f"[A] Contrast computed: shape={contrast_all.shape}", flush=True)

    # ── Part B: Global explanations ───────────────────────────────────────────
    print("\n[B] Global explanations", flush=True)

    # Raw mean |SHAP| per feature
    feat_imp_all = mean_abs_shap_by_feature(shap_vals, FEATURE_NAMES)
    feat_imp_top12 = mean_abs_shap_by_feature(shap_top12, FEATURE_NAMES)

    # Ranking-relevant: mean |contrast SHAP| per feature
    # Customer-level features shift all of a customer's scores together → low contrast
    feat_contrast_all = mean_abs_contrast_shap_by_feature(contrast_all, FEATURE_NAMES)
    feat_contrast_top12 = mean_abs_contrast_shap_by_feature(contrast_top12, FEATURE_NAMES)

    top25_all = list(feat_imp_all.items())[:25]
    top25_top12 = list(feat_imp_top12.items())[:25]
    top25_contrast_all = list(feat_contrast_all.items())[:25]
    top25_contrast_top12 = list(feat_contrast_top12.items())[:25]
    top10_all = [f for f, _ in top25_all[:10]]
    top10_top12 = [f for f, _ in top25_top12[:10]]

    # Load gain importance from ablations.json
    abl = json.loads(ABL_PATH.read_text())
    gain_top10 = [x["feature"] for x in abl["gain_importance_top25"][:10]]

    print("[B] Top-10 features by mean |SHAP| vs mean |contrast SHAP| (all candidates):", flush=True)
    print(f"  {'Rank':>4s}  {'Feature':<42s}  {'|SHAP|':>10s}  {'|contrast|':>12s}  {'Gain rank':>10s}",
          flush=True)
    contrast_all_map = dict(top25_contrast_all)
    for i, (f, v) in enumerate(top25_all[:10]):
        cv = contrast_all_map.get(f, 0.0)
        gr = gain_top10.index(f) + 1 if f in gain_top10 else ">10"
        marker = "" if f in gain_top10 else "  ← ranked lower by gain"
        print(f"  {i+1:2d}.  {f:<42s}  {v:>10.6f}  {cv:>12.6f}  {str(gr):>10s}{marker}",
              flush=True)

    print("[B] Top-10 features by mean |SHAP| (top-12 rows):", flush=True)
    contrast_top12_map = dict(top25_contrast_top12)
    for i, (f, v) in enumerate(top25_top12[:10]):
        cv = contrast_top12_map.get(f, 0.0)
        gr = gain_top10.index(f) + 1 if f in gain_top10 else ">10"
        marker = "" if f in gain_top10 else "  ← ranked lower by gain"
        print(f"  {i+1:2d}. {f}: |SHAP|={v:.6f}  |contrast|={cv:.6f}  gain_rank={gr}{marker}",
              flush=True)

    # Build comparison: which features agree/disagree between SHAP and gain
    shap_rank = {f: i+1 for i, (f, _) in enumerate(top25_all[:10])}
    gain_rank_map = {f: i+1 for i, f in enumerate(gain_top10)}
    all_feats = set(list(shap_rank.keys()) + list(gain_rank_map.keys()))
    disagreements = []
    for f in all_feats:
        sr = shap_rank.get(f, ">10")
        gr = gain_rank_map.get(f, ">10")
        if sr != gr:
            disagreements.append({"feature": f, "shap_rank": sr, "gain_rank": gr})

    # Group SHAP vs ablation deltas
    grp_shap_all = group_shap(shap_vals, FEATURE_NAMES, feature_groups)
    grp_shap_top12 = group_shap(shap_top12, FEATURE_NAMES, feature_groups)
    grp_contrast_all = group_shap(contrast_all, FEATURE_NAMES, feature_groups)
    abl_deltas = {a["group_dropped"]: a["delta_map@12"] for a in abl["ablations"]}
    print("[B] Group SHAP vs contrast SHAP (all candidates):", flush=True)
    for grp, sv in grp_shap_all.items():
        cv = grp_contrast_all.get(grp, 0.0)
        abl_d = abl_deltas.get(grp, float("nan"))
        print(f"  {grp}: |SHAP|={sv:.6f}  |contrast|={cv:.6f}  abl_delta={abl_d:+.6f}",
              flush=True)

    # ── Dependence plots for top 5 features ────────────────────────────────
    print("[B] Generating dependence plots for top 5 features...", flush=True)
    top5_feats = [f for f, _ in top25_all[:5]]
    feat_idx = {n: i for i, n in enumerate(FEATURE_NAMES)}

    # Choose a coloring feature for each — use the highest-correlated non-self feature
    for i, feat in enumerate(top5_feats):
        fi = feat_idx[feat]
        fv = X_full[:, fi].astype(float)
        sv = shap_vals[:, fi].astype(float)

        # Pick coloring feature: find feature most correlated with SHAP of this feature
        corrs = []
        for j, fn in enumerate(FEATURE_NAMES):
            if j == fi:
                corrs.append(-999.0)
                continue
            col = X_full[:, j].astype(float)
            mask = ~(np.isnan(fv) | np.isnan(sv) | np.isnan(col))
            if mask.sum() < 20:
                corrs.append(0.0)
                continue
            c = np.corrcoef(sv[mask], col[mask])[0, 1]
            corrs.append(float(c) if not np.isnan(c) else 0.0)
        color_feat_idx = int(np.argmax(np.abs(corrs)))
        color_feat = FEATURE_NAMES[color_feat_idx]
        color_vals = X_full[:, color_feat_idx].astype(float)

        # Subsample for plot performance
        plot_rng = np.random.default_rng(42 + i)
        plot_n = min(3000, len(fv))
        pidx = plot_rng.choice(len(fv), size=plot_n, replace=False)
        valid = ~(np.isnan(fv[pidx]) | np.isnan(sv[pidx]))

        fig, ax = plt.subplots(figsize=(7, 4))
        sc = ax.scatter(
            fv[pidx][valid], sv[pidx][valid],
            c=color_vals[pidx][valid], cmap="viridis",
            alpha=0.3, s=6, rasterized=True
        )
        plt.colorbar(sc, ax=ax, label=color_feat)
        ax.set_xlabel(feat)
        ax.set_ylabel(f"SHAP value for {feat}")
        ax.set_title(f"Dependence: {feat}\n(colored by {color_feat})")
        plt.tight_layout()
        fname = FIGURES_DIR / f"shap_dep_{i+1:02d}_{feat}.png"
        plt.savefig(str(fname), dpi=100)
        plt.close()
        print(f"[B] Saved dependence plot: {fname.name}", flush=True)

    # ── Global importance bar chart ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    for ax, data, title in zip(
        axes,
        [top25_all[:15], top25_top12[:15]],
        ["All candidates", "Top-12 recommended rows"]
    ):
        feats = [f for f, _ in data]
        vals = [v for _, v in data]
        ax.barh(feats[::-1], vals[::-1])
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(f"Global feature importance ({title})")
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "shap_global_importance.png"), dpi=100)
    plt.close()
    print("[B] Saved global importance plot.", flush=True)

    # ── Segment comparison: cold-start vs 20+ buyers ────────────────────────
    print("[B] Segment SHAP comparison (cold-start vs 20+ buyers)...", flush=True)
    cold_mask = (df_sample["c_n_purchases"].fill_null(0) == 0).to_numpy()
    heavy_mask = (df_sample["c_n_purchases"].fill_null(0) >= 20).to_numpy()
    n_cold = int(cold_mask.sum())
    n_heavy = int(heavy_mask.sum())
    print(f"[B] Segment counts: cold-start={n_cold:,}, 20+ buyers={n_heavy:,}", flush=True)

    if n_cold > 0:
        seg_imp_cold = mean_abs_shap_by_feature(shap_vals[cold_mask], FEATURE_NAMES)
    else:
        seg_imp_cold = {}
    if n_heavy > 0:
        seg_imp_heavy = mean_abs_shap_by_feature(shap_vals[heavy_mask], FEATURE_NAMES)
    else:
        seg_imp_heavy = {}

    # Segment comparison plot: top-10 features for each segment
    seg_feats_cold = [f for f, _ in list(seg_imp_cold.items())[:10]]
    seg_feats_heavy = [f for f, _ in list(seg_imp_heavy.items())[:10]]
    all_seg_feats = list(dict.fromkeys(seg_feats_cold + seg_feats_heavy))

    if seg_imp_cold and seg_imp_heavy:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(all_seg_feats))
        w = 0.35
        cold_vals = [seg_imp_cold.get(f, 0.0) for f in all_seg_feats]
        heavy_vals = [seg_imp_heavy.get(f, 0.0) for f in all_seg_feats]
        ax.bar(x - w/2, cold_vals, w, label=f"Cold-start (n={n_cold:,})", alpha=0.8)
        ax.bar(x + w/2, heavy_vals, w, label=f"20+ buyers (n={n_heavy:,})", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(all_seg_feats, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean |SHAP|")
        ax.set_title("Mean |SHAP| by segment: cold-start vs 20+ buyers")
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(FIGURES_DIR / "shap_segment_comparison.png"), dpi=100)
        plt.close()
        print("[B] Saved segment comparison plot.", flush=True)

    # ── Part C: Local explanations ────────────────────────────────────────────
    # Reuse contrast_all computed in Part A/B (no re-computation)
    print("\n[C] Using pre-computed within-list contrast...", flush=True)
    contrast = contrast_all
    print(f"[C] Contrast shape: {contrast.shape}", flush=True)

    # Faithfulness check on 1,000 sampled recommended rows (top-12 mask)
    print("[C] Faithfulness check (1,000 recommended rows, seed 42)...", flush=True)
    top12_idx = np.where(top12_mask)[0]
    faith_n = min(1000, len(top12_idx))
    faith_rng = np.random.default_rng(42)
    faith_sel = faith_rng.choice(top12_idx, size=faith_n, replace=False)
    X_faith = X_full[faith_sel]
    contrast_faith = contrast[faith_sel]

    faith_result = faithfulness_check(booster, X_faith, contrast_faith, FEATURE_NAMES, seed=42)
    print(f"[C] Faithfulness: top_mean_drop={faith_result['top_mean_drop']:.6f}  "
          f"random_mean_drop={faith_result['random_mean_drop']:.6f}  "
          f"mean_diff={faith_result['mean_diff']:+.6f}  "
          f"p_value={faith_result['p_value']}  passed={faith_result['passed']}", flush=True)

    # Generate reasons for all top-12 recommended rows and check for conflicts / duplicate themes
    print("[C] Generating reason codes for all top-12 rows + checking conflicts...", flush=True)
    top12_rows_for_reasons = [(int(i), contrast[i], X_full[i]) for i in top12_idx]
    reasons_list = []
    n_conflicts = 0
    n_duplicate_themes = 0
    for row_idx, cont, fv in top12_rows_for_reasons:
        reasons = generate_reasons(cont, fv, FEATURE_NAMES, top_k=3)
        reasons_list.append(reasons)
        # Conflict: both exact-article and product-code present in the same item's reasons
        has_exact = any("exact item" in r for r in reasons)
        has_pc = any("colour or size" in r for r in reasons)
        if has_exact and has_pc:
            n_conflicts += 1
        # Duplicate theme: two reasons from the same theme
        from src.explain.reasons import _THEME, _REASON_REGISTRY
        feat_idx_map = {n: i for i, n in enumerate(FEATURE_NAMES)}
        used_themes: list[str] = []
        for feat, template_fn in _REASON_REGISTRY.items():
            if feat not in feat_idx_map:
                continue
                fidx = feat_idx_map[feat]
            fidx = feat_idx_map[feat]
            if fidx >= len(cont) or float(cont[fidx]) <= 0.0:
                continue
            fval = fv[fidx]
            try:
                fval_py = float(fval)
                fval_py = None if np.isnan(fval_py) else fval_py
            except (ValueError, TypeError):
                fval_py = None
            r = template_fn(fval_py)
            if r is not None and any(r in reason for reason in reasons):
                used_themes.append(_THEME.get(feat, "other"))
        theme_counts: dict[str, int] = {}
        for t in used_themes:
            theme_counts[t] = theme_counts.get(t, 0) + 1
        if any(c > 1 for c in theme_counts.values()):
            n_duplicate_themes += 1

    n_with_reasons = sum(1 for r in reasons_list if r)
    print(f"[C] Rows with ≥1 reason: {n_with_reasons:,} / {len(top12_idx):,} "
          f"({100*n_with_reasons/max(1,len(top12_idx)):.1f}%)", flush=True)
    print(f"[C] Conflict check (exact+product-code same item): {n_conflicts} conflicts "
          f"(must be 0)", flush=True)
    print(f"[C] Duplicate-theme check: {n_duplicate_themes} rows with 2+ reasons from same theme "
          f"(must be 0)", flush=True)
    if n_conflicts > 0 or n_duplicate_themes > 0:
        raise RuntimeError(
            f"Reason quality check FAILED: conflicts={n_conflicts}, "
            f"duplicate_themes={n_duplicate_themes}"
        )

    # ── 5 worked examples ────────────────────────────────────────────────────
    print("[C] Generating 5 worked examples...", flush=True)
    # Choose 5 customers from different segments
    segments = {
        "cold-start": df_sample.filter(pl.col("c_n_purchases").fill_null(0) == 0)["customer_idx"].unique().to_list(),
        "light_buyer_1-4": df_sample.filter(
            (pl.col("c_n_purchases").fill_null(0) >= 1) &
            (pl.col("c_n_purchases").fill_null(0) <= 4)
        )["customer_idx"].unique().to_list(),
        "medium_buyer_5-19": df_sample.filter(
            (pl.col("c_n_purchases").fill_null(0) >= 5) &
            (pl.col("c_n_purchases").fill_null(0) <= 19)
        )["customer_idx"].unique().to_list(),
        "heavy_buyer_20+": df_sample.filter(
            pl.col("c_n_purchases").fill_null(0) >= 20
        )["customer_idx"].unique().to_list(),
    }

    example_customers = []
    ex_rng = np.random.default_rng(42)
    for seg_name, custs in segments.items():
        if custs:
            chosen = int(ex_rng.choice(custs))
            example_customers.append((seg_name, chosen))
        if len(example_customers) >= 4:
            break
    # Add one more from any segment
    all_custs = df_sample["customer_idx"].unique().to_list()
    seen = {c for _, c in example_customers}
    remaining = [c for c in all_custs if c not in seen]
    if remaining:
        example_customers.append(("additional", int(ex_rng.choice(remaining))))

    examples = []
    articles_df = pl.read_parquet(ROOT / "data" / "processed" / "articles.parquet")

    # Precompute customer → row range (sorted df_sample by customer_idx)
    custs_arr = df_sample["customer_idx"].to_numpy()
    uniq_custs, cust_starts = np.unique(custs_arr, return_index=True)
    cust_ends = np.append(cust_starts[1:], len(custs_arr))
    cust_to_range: dict[int, tuple[int, int]] = {
        int(c): (int(s), int(e))
        for c, s, e in zip(uniq_custs, cust_starts, cust_ends)
    }

    for seg_name, cust_idx in example_customers:
        cust_range = cust_to_range.get(int(cust_idx))
        if cust_range is None:
            continue
        s_idx, e_idx = cust_range
        cust_row_indices_arr = np.arange(s_idx, e_idx)
        cust_rows = df_sample.slice(s_idx, e_idx - s_idx)

        cust_scores = scores[cust_row_indices_arr]
        cust_shap = shap_vals[cust_row_indices_arr]
        cust_X = X_full[cust_row_indices_arr]
        cust_contrast = contrast[cust_row_indices_arr]

        # history summary
        n_hist = int(cust_rows["c_n_purchases"].fill_null(0)[0])
        age = cust_rows["c_age"].fill_null(float("nan"))[0]
        age_str = f"{float(age):.0f}" if not np.isnan(float(age)) else "unknown"

        # top-12 recommendations
        top_k = min(12, len(cust_scores))
        top_local = np.argsort(-cust_scores)[:top_k]
        article_list = cust_rows["article_idx"].to_list()  # Python ints
        recs = []
        for local_i in top_local:
            local_i = int(local_i)  # numpy.int64 → Python int
            art_idx = article_list[local_i]
            score = float(cust_scores[local_i])
            cont = cust_contrast[local_i]
            fv = cust_X[local_i]
            reasons = generate_reasons(cont, fv, FEATURE_NAMES, top_k=3)

            # get article display info
            art_row = articles_df.filter(pl.col("article_idx") == art_idx)
            if len(art_row) > 0:
                art_name = str(art_row["prod_name"][0])
                art_type = str(art_row["product_type_name"][0])
                art_colour = str(art_row["colour_group_name"][0])
            else:
                art_name = art_type = art_colour = "unknown"

            recs.append({
                "article_idx": art_idx,
                "prod_name": art_name,
                "product_type": art_type,
                "colour": art_colour,
                "score": round(score, 6),
                "reasons": reasons,
            })

        examples.append({
            "segment": seg_name,
            "customer_idx": cust_idx,
            "history_summary": {
                "n_purchases": n_hist,
                "age": age_str,
            },
            "recommendations": recs,
        })
        print(f"[C] Example ({seg_name}, customer {cust_idx}): {len(recs)} recs, "
              f"first reasons: {recs[0]['reasons'] if recs else []}", flush=True)

    examples_path = REPORTS_DIR / "examples.json"
    examples_path.write_text(json.dumps(examples, indent=2))
    print(f"[C] 5 worked examples saved to {examples_path}", flush=True)

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "n_customers_sampled": n_custs,
        "n_rows_sampled": n_rows,
        "n_top12_rows": int(top12_mask.sum()),
        "additivity": {
            "all_candidates": add_result_all,
            "top12_rows": add_result_top12,
        },
        "top10_shap_all_candidates": [
            {"feature": f, "mean_abs_shap": round(v, 8)} for f, v in top25_all[:10]
        ],
        "top10_shap_top12_rows": [
            {"feature": f, "mean_abs_shap": round(v, 8)} for f, v in top25_top12[:10]
        ],
        "top10_contrast_shap_all_candidates": [
            {"feature": f, "mean_abs_contrast_shap": round(v, 8)}
            for f, v in top25_contrast_all[:10]
        ],
        "top10_contrast_shap_top12_rows": [
            {"feature": f, "mean_abs_contrast_shap": round(v, 8)}
            for f, v in top25_contrast_top12[:10]
        ],
        "top10_gain_importance": gain_top10,
        "shap_vs_gain_disagreements": disagreements,
        "group_shap_all_candidates": {g: round(v, 8) for g, v in grp_shap_all.items()},
        "group_contrast_shap_all_candidates": {
            g: round(v, 8) for g, v in grp_contrast_all.items()
        },
        "group_shap_top12": {g: round(v, 8) for g, v in grp_shap_top12.items()},
        "ablation_deltas": {g: round(v, 8) for g, v in abl_deltas.items()},
        "segment_comparison": {
            "cold_start": {
                "n": n_cold,
                "top10": [{"feature": f, "mean_abs_shap": round(v, 8)}
                          for f, v in list(seg_imp_cold.items())[:10]],
            },
            "heavy_buyer_20plus": {
                "n": n_heavy,
                "top10": [{"feature": f, "mean_abs_shap": round(v, 8)}
                          for f, v in list(seg_imp_heavy.items())[:10]],
            },
        },
        "faithfulness": faith_result,
        "reasons_coverage": {
            "n_top12_rows": len(top12_idx),
            "n_with_reasons": n_with_reasons,
            "pct_with_reasons": round(100.0 * n_with_reasons / max(1, len(top12_idx)), 2),
        },
        "reason_quality_check": {
            "n_conflicts": n_conflicts,
            "n_duplicate_themes": n_duplicate_themes,
            "passed": n_conflicts == 0 and n_duplicate_themes == 0,
        },
        "runtime_s": round(time.time() - t0_global, 1),
    }

    summary_path = REPORTS_DIR / "shap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[run_explain] Summary saved to {summary_path}", flush=True)
    print(f"[run_explain] Total runtime: {time.time()-t0_global:.1f}s", flush=True)

    print("\n=== PART 1 COMPLETE ===")
    print(f"Additivity max error (all): {add_result_all['max_error']:.2e}  passed={add_result_all['passed']}")
    print(f"Additivity max error (top-12): {add_result_top12['max_error']:.2e}  passed={add_result_top12['passed']}")
    print("\nTop-10 by mean |SHAP| vs mean |contrast SHAP| (all candidates):")
    print(f"  {'Feature':<42s} {'SHAP rank':>10s} {'Contrast rank':>14s} {'Gain rank':>10s}")
    contrast_rank_map = {f: i+1 for i, (f, _) in enumerate(top25_contrast_all[:10])}
    for i, (f, _) in enumerate(top25_all[:10]):
        gr = gain_rank_map.get(f, ">10")
        cr = contrast_rank_map.get(f, ">10")
        print(f"  {f:<42s} {i+1:>10d} {str(cr):>14s} {str(gr):>10s}")
    print(f"\nTop-10 by mean |contrast SHAP| (all candidates, ranking-relevant view):")
    for i, (f, v) in enumerate(top25_contrast_all[:10]):
        gr = gain_rank_map.get(f, ">10")
        marker = "" if f in gain_top10 else "  ← ranked lower by gain"
        print(f"  {i+1:2d}. {f}: {v:.6f}  gain_rank={gr}{marker}")
    print(f"\nGroup SHAP vs contrast SHAP vs ablation deltas:")
    for grp in ["candidate", "customer", "article", "interaction"]:
        sv = grp_shap_all.get(grp, 0.0)
        cv = grp_contrast_all.get(grp, 0.0)
        abl_d = abl_deltas.get(grp, float("nan"))
        print(f"  {grp}: |SHAP|={sv:.6f}  |contrast|={cv:.6f}  ablation_delta={abl_d:+.6f}")
    print(f"\nReason quality (60k recommended rows):")
    print(f"  conflicts={n_conflicts}  duplicate_themes={n_duplicate_themes}  passed={n_conflicts==0 and n_duplicate_themes==0}")
    print(f"\nFaithfulness: top_drop={faith_result['top_mean_drop']:.4f}  "
          f"rand_drop={faith_result['random_mean_drop']:.4f}  "
          f"p={faith_result['p_value']}  passed={faith_result['passed']}")
    print("\nWorked example (first):")
    ex = examples[0]
    print(f"  Segment: {ex['segment']}, customer {ex['customer_idx']}, "
          f"n_purchases={ex['history_summary']['n_purchases']}")
    for rec in ex["recommendations"][:3]:
        print(f"    {rec['prod_name'][:40]} | {rec['colour']}")
        for r in rec["reasons"]:
            print(f"      → {r}")


if __name__ == "__main__":
    main()
