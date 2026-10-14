"""Load pre-computed insight reports from the bundle (primary) or disk (fallback)."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("api")
_REPORTS_DIR = Path("reports")

# Bundle-level report cache populated at startup when bundle is loaded.
_bundle_reports: dict[str, dict] = {}


def set_bundle_reports(reports: dict[str, dict]) -> None:
    """Called at startup from deps.load_bundle_at_startup to populate bundle report cache."""
    _bundle_reports.clear()
    _bundle_reports.update(reports)


@lru_cache(maxsize=32)
def _load_json_disk(rel_path: str) -> dict:
    """Fallback: read from reports/ on disk."""
    p = _REPORTS_DIR / rel_path
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _get(bundle_key: str, disk_path: str) -> dict:
    """Return report from bundle if available, else fall back to disk."""
    if bundle_key in _bundle_reports:
        return _bundle_reports[bundle_key]
    return _load_json_disk(disk_path)


def get_summary() -> dict:
    holdout = _get("holdout_results", "evaluation/holdout_results.json")
    eval103 = _get("eval_fold103", "ranker/eval_fold103.json")
    ci = eval103.get("bootstrap_ranker_vs_heuristic", {})
    primary = holdout.get("models", {}).get("primary", {})
    heuristic = holdout.get("models", {}).get("heuristic", {})
    baseline_b = holdout.get("models", {}).get("baseline_b", {})
    boot_vs_h = primary.get("boot_vs_heuristic", {})
    boot_vs_b = primary.get("boot_vs_baseline_b", {})
    return {
        "holdout_week": holdout.get("holdout_week", 104),
        "n_eval_customers": holdout.get("n_eval_customers", 0),
        "holdout_map12_ranker": primary.get("map@12"),
        "holdout_map12_heuristic": heuristic.get("map@12"),
        "holdout_lift_pct": holdout.get("primary_lift_vs_heuristic_pct"),
        "holdout_ci_lo": boot_vs_h.get("ci_lo"),
        "holdout_ci_hi": boot_vs_h.get("ci_hi"),
        "holdout_map12_baseline_b": baseline_b.get("map@12"),
        "holdout_lift_vs_baseline_b_pct": holdout.get("primary_lift_vs_baseline_b_pct"),
        "holdout_baseline_b_ci_lo": boot_vs_b.get("ci_lo"),
        "holdout_baseline_b_ci_hi": boot_vs_b.get("ci_hi"),
        "fold103_ranker_map12": eval103.get("ranker", {}).get("map@12"),
        "fold103_heuristic_map12": eval103.get("heuristic", {}).get("map@12"),
        "fold103_ci_lo": ci.get("ci_lo"),
        "fold103_ci_hi": ci.get("ci_hi"),
        "oracle_map12_fold103": eval103.get("oracle", {}).get("map@12"),
    }


def get_weekly_lift() -> list:
    data = _get("rolling_origin", "evaluation/rolling_origin.json")
    result = []
    for f in data.get("folds", []):
        ranker = f.get("ranker_map@12") or f.get("ranker_map12")
        heuristic = f.get("heuristic_map@12") or f.get("heuristic_map12")
        lift = (
            round((ranker - heuristic) / heuristic * 100, 2)
            if ranker is not None and heuristic
            else None
        )
        entry: dict = {
            "fold": f.get("fold"),
            "ranker_map12": ranker,
            "heuristic_map12": heuristic,
            "lift_pct": lift,
        }
        if "optimistic" in str(f.get("note", "")).lower():
            entry["optimistic"] = True
        result.append(entry)
    return result


def get_segments() -> list:
    data = _get("segment_analysis", "evaluation/segment_analysis.json")
    # real bundle uses "results" key; synthetic bundle used "segments"
    rows = data.get("results", data.get("segments", []))
    output = []
    for r in rows:
        ranker = r.get("map@12_ranker") or r.get("ranker_map12")
        heuristic = r.get("map@12_heuristic") or r.get("heuristic_map12")
        lift = (
            round((ranker - heuristic) / heuristic * 100, 2)
            if ranker is not None and heuristic
            else None
        )
        output.append(
            {
                "family": r.get("family"),
                "segment": r.get("segment"),
                "n": r.get("n"),
                "ranker_map12": ranker,
                "heuristic_map12": heuristic,
                "ci_lo": r.get("ci_lo"),
                "ci_hi": r.get("ci_hi"),
                "p_ttest_adj": r.get("p_ttest_adj"),
                "lift_pct": lift,
            }
        )
    return output


def get_ablations() -> list:
    data = _get("ablations", "ranker/ablations.json")
    result = []
    for a in data.get("ablations", []):
        boot = a.get("bootstrap", {})
        result.append(
            {
                "group": a.get("group_dropped"),
                "n_features": a.get("n_features_dropped"),
                "map12_delta": a.get("delta_map@12"),
                "ci_lo": boot.get("ci_lo"),
                "ci_hi": boot.get("ci_hi"),
                "ci_excludes_zero": boot.get("ci_excludes_zero"),
            }
        )
    return result


def get_trends(bundle) -> dict:
    """Compute live trend data from bundle state — no calendar dates in response."""
    import polars as pl

    # ── trending_now: top 20 from pop_last_week × article_meta ─────────────
    trending_now = []
    if bundle.pop_last_week is not None and bundle.article_meta is not None:
        joined = bundle.pop_last_week.join(bundle.article_meta, on="article_idx", how="inner")
        if "source_rank" in joined.columns:
            joined = joined.sort("source_rank")
        for row in joined.head(20).iter_rows(named=True):
            art_id = str(row.get("article_id", ""))
            trending_now.append(
                {
                    "article_idx": int(row.get("article_idx", 0)),
                    "article_id": art_id,
                    "prod_name": str(row.get("prod_name", "")),
                    "product_group": str(row.get("product_type_name", "")),
                    "colour": str(row.get("colour_group_name", "")),
                    "image_url": f"/api/v1/articles/{art_id}/image",
                    "sales_last_week": int(row.get("source_rank", 0)),
                    "trend_ratio": float(row.get("score", 0.0)),
                }
            )

    # ── trending_by_age: top 10 per age bucket from seg_pop × article_meta ─
    trending_by_age: dict[str, list[dict]] = {}
    if bundle.seg_pop_by_bucket and bundle.article_meta is not None:
        for bucket, seg_df in bundle.seg_pop_by_bucket.items():
            if bucket == "missing":
                continue
            joined = seg_df.join(bundle.article_meta, on="article_idx", how="inner")
            if "source_rank" in joined.columns:
                joined = joined.sort("source_rank")
            bucket_items = []
            for row in joined.head(10).iter_rows(named=True):
                art_id = str(row.get("article_id", ""))
                bucket_items.append(
                    {
                        "article_idx": int(row.get("article_idx", 0)),
                        "article_id": art_id,
                        "prod_name": str(row.get("prod_name", "")),
                        "product_group": str(row.get("product_type_name", "")),
                        "colour": str(row.get("colour_group_name", "")),
                        "image_url": f"/api/v1/articles/{art_id}/image",
                        "score": float(row.get("score", 0.0)),
                    }
                )
            trending_by_age[bucket] = bucket_items

    # ── top_product_groups: weighted by sales_last_week (pop score) ─────────
    top_product_groups: list[dict] = []
    if bundle.pop_last_week is not None and bundle.article_meta is not None:
        joined = bundle.pop_last_week.join(bundle.article_meta, on="article_idx", how="inner")
        if "product_type_name" in joined.columns and "score" in joined.columns:
            group_agg = (
                joined.group_by("product_type_name")
                .agg(
                    pl.col("score").sum().alias("total_score"),
                    pl.len().alias("count"),
                )
                .sort("total_score", descending=True)
            )
            total_score = float(group_agg["total_score"].sum()) or 1.0
            for row in group_agg.head(10).iter_rows(named=True):
                top_product_groups.append(
                    {
                        "product_group": str(row["product_type_name"]),
                        "share": round(float(row["total_score"]) / total_score, 4),
                        "count": int(row["count"]),
                    }
                )

    # ── popularity_decay: age buckets from article_feats ────────────────────
    # Buckets: <14 days, 14-60 days, 60-180 days, 180+ days
    decay_labels = [
        ("Just launched", 14),
        ("Recently launched", 60),
        ("Established", 180),
        ("Mature", None),
    ]
    popularity_decay: list[dict] = []

    if (
        bundle.article_feats is not None
        and "a_days_since_first_sale" in bundle.article_feats.columns
        and "a_sales_1w" in bundle.article_feats.columns
    ):
        feats = bundle.article_feats.select(["a_days_since_first_sale", "a_sales_1w"]).drop_nulls()
        total_sales = float(feats["a_sales_1w"].sum()) or 1.0

        prev_max = 0
        for label, age_max in decay_labels:
            if age_max is not None:
                bucket_df = feats.filter(
                    (pl.col("a_days_since_first_sale") >= prev_max)
                    & (pl.col("a_days_since_first_sale") < age_max)
                )
            else:
                bucket_df = feats.filter(pl.col("a_days_since_first_sale") >= prev_max)

            bucket_sales = float(bucket_df["a_sales_1w"].sum())
            popularity_decay.append(
                {
                    "age_label": label,
                    "age_days_max": age_max,
                    "share": round(bucket_sales / total_sales, 4),
                }
            )
            if age_max is not None:
                prev_max = age_max
    else:
        # Fallback: equal shares
        for label, age_max in decay_labels:
            popularity_decay.append(
                {
                    "age_label": label,
                    "age_days_max": age_max,
                    "share": 0.25,
                }
            )

    return {
        "trending_now": trending_now,
        "trending_by_age": trending_by_age,
        "top_product_groups": top_product_groups,
        "popularity_decay": popularity_decay,
    }


def get_importance() -> list:
    raw = _get("shap_summary", "explain/shap_summary.json")
    shap_rows: list = raw.get("top10_shap_all_candidates", [])
    contrast_rows: list = raw.get("top10_contrast_shap_all_candidates", [])
    gain_list: list = raw.get("top10_gain_importance", [])
    gain_rank = {feat: i + 1 for i, feat in enumerate(gain_list)}
    contrast_rank = {row.get("feature", ""): i + 1 for i, row in enumerate(contrast_rows)}
    contrast_shap = {row.get("feature", ""): row.get("mean_abs_contrast_shap") for row in contrast_rows}
    result = []
    for i, row in enumerate(shap_rows):
        feat = row.get("feature", "")
        result.append(
            {
                "rank_shap": i + 1,
                "feature": feat,
                "mean_abs_shap": row.get("mean_abs_shap"),
                "rank_contrast": contrast_rank.get(feat),
                "mean_abs_contrast_shap": contrast_shap.get(feat),
                "rank_gain": gain_rank.get(feat),
            }
        )
    return result
