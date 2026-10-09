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
    return {
        "holdout_week": holdout.get("holdout_week", 104),
        "n_eval_customers": holdout.get("n_eval_customers", 0),
        "primary_map12": holdout.get("models", {}).get("primary", {}).get("map@12"),
        "primary_lift_vs_heuristic_pct": holdout.get("primary_lift_vs_heuristic_pct"),
        "primary_lift_vs_baseline_b_pct": holdout.get("primary_lift_vs_baseline_b_pct"),
        "ranker_map12_fold103": eval103.get("ranker", {}).get("map@12"),
        "heuristic_map12_fold103": eval103.get("heuristic", {}).get("map@12"),
        "oracle_map12_fold103": eval103.get("oracle", {}).get("map@12"),
        "bootstrap_ci_vs_heuristic": eval103.get(
            "bootstrap_ranker_vs_heuristic", {}
        ).get("ci_95"),
    }


def get_weekly_lift() -> dict:
    return _get("rolling_origin", "evaluation/rolling_origin.json")


def get_segments() -> dict:
    return _get("segment_analysis", "evaluation/segment_analysis.json")


def get_ablations() -> dict:
    return _get("ablations", "ranker/ablations.json")


def get_importance() -> dict:
    raw = _get("shap_summary", "explain/shap_summary.json")
    ablations = _get("ablations", "ranker/ablations.json")
    return {
        "top10_gain_importance": raw.get("top10_gain_importance", []),
        "top10_shap_all_candidates": raw.get("top10_shap_all_candidates", []),
        "top10_contrast_shap_all_candidates": raw.get(
            "top10_contrast_shap_all_candidates", []
        ),
        "group_shap_all_candidates": raw.get("group_shap_all_candidates", {}),
        "group_contrast_shap_all_candidates": raw.get(
            "group_contrast_shap_all_candidates", {}
        ),
        "gain_importance_top25": ablations.get("gain_importance_top25", []),
    }
