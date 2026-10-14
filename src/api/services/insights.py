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
