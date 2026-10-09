"""Reason code registry and within-list contrast SHAP for the recommendation frontend.

Within-list contrast design
----------------------------
LambdaRank scores are only meaningful relative to the other items in a customer's
list. A plain SHAP value says "this feature pushed the score up/down from the global
average", but the user wants to know "why does this item rank above the other items
this customer sees?" We therefore define the explanation of item i as:

    contrast_i = SHAP_i  −  mean(SHAP over all candidates for this customer)

Only features with positive contrast contribution may become reasons:
the item must be *above* the customer's within-list average for that feature
to count as a reason it ranks well.

Theme-based deduplication
--------------------------
At most one reason per theme (repurchase, category, popularity_trend, demographic_fit,
price_fit, co_purchase). Within each theme the feature with the highest positive contrast
is selected; then the top 3 themes by contrast value are returned.

Exact-item / product-code conflict rule
-----------------------------------------
If i_days_since_bought_article is eligible (positive contrast, valid value), the
product-code reason for the same item is suppressed — both say "you bought this";
the exact-item reason is more specific and always wins.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.explain.shap_values import compute_within_list_contrast  # re-exported for callers
from src.model.data import FEATURE_NAMES

# ---------------------------------------------------------------------------
# Theme assignment: one reason per theme in the final output
# ---------------------------------------------------------------------------

_THEME: dict[str, str] = {
    "i_days_since_bought_article": "repurchase",
    "i_days_since_bought_product_code": "repurchase",
    "a_trend_ratio": "popularity_trend",
    "i_age_gap": "demographic_fit",
    "i_customer_share_product_group": "category",
    "i_customer_share_garment_group": "category",
    "a_repurchase_rate": "popularity_trend",
    "i_bought_same_product_code": "co_purchase",
}


# ---------------------------------------------------------------------------
# Reason template registry
# ---------------------------------------------------------------------------
# Maps feature_name → callable(feature_value) → str | None
# Returning None means the feature has no sensible explanation for this value
# (e.g. null, or zero when zero has no consumer-facing meaning).
# Only features with a registered template are eligible as reason codes.
# ---------------------------------------------------------------------------

def _fmt_days(days: float | None, label: str) -> str | None:
    if days is None or np.isnan(days):
        return None
    d = int(round(days))
    if d <= 0:
        return None
    unit = "day" if d == 1 else "days"
    return f"{label} {d} {unit} ago"


def _fmt_trend(ratio: float | None) -> str | None:
    if ratio is None or np.isnan(ratio):
        return None
    pct = int(round((ratio - 1.0) * 100))
    if pct < 5:
        return None
    return f"Trending: {pct}% more sales this week than its 4-week average"


def _fmt_share(share: float | None, label: str) -> str | None:
    """Category-share reason with tiered thresholds.

    share >= 25%  : "a category you buy often"
    10% <= share < 25% : "a category you've bought from before"
    share < 10%   : no reason (return None)
    """
    if share is None or np.isnan(share):
        return None
    if share >= 0.25:
        pct = int(round(share * 100))
        return f"In a {label} you buy often ({pct}% of your purchases)"
    if share >= 0.10:
        pct = int(round(share * 100))
        return f"In a {label} you've bought from before ({pct}% of your purchases)"
    return None


def _fmt_age_gap(gap: float | None) -> str | None:
    if gap is None or np.isnan(gap):
        return None
    if gap > 10:
        return None
    return "Popular with customers your age"


def _fmt_repurchase_rate(rate: float | None) -> str | None:
    if rate is None or np.isnan(rate):
        return None
    pct = int(round(rate * 100))
    if pct < 10:
        return None
    return f"Shoppers who bought this often come back ({pct}% repurchase rate)"


_REASON_REGISTRY: dict[str, Any] = {
    "i_days_since_bought_article": lambda v: _fmt_days(v, "You bought this exact item"),
    "i_days_since_bought_product_code": lambda v: _fmt_days(
        v, "You bought another colour or size of this product"
    ),
    "a_trend_ratio": _fmt_trend,
    "i_age_gap": _fmt_age_gap,
    "i_customer_share_product_group": lambda v: _fmt_share(v, "product category"),
    "i_customer_share_garment_group": lambda v: _fmt_share(v, "garment category"),
    "a_repurchase_rate": _fmt_repurchase_rate,
    "i_bought_same_product_code": lambda v: (
        None if (v is None or np.isnan(float(v)) or int(v) <= 0)
        else "Often bought together with something you bought recently"
    ),
}

REASON_FEATURES: list[str] = list(_REASON_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Reason code generation
# ---------------------------------------------------------------------------

def generate_reasons(
    contrast: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    top_k: int = 3,
) -> list[str]:
    """Generate top-k reason strings for one candidate row.

    Applies three rules in order:
    1. Only features with positive contrast and a registered template are eligible.
    2. If i_days_since_bought_article is eligible, suppress
       i_days_since_bought_product_code (exact-item reason wins).
    3. At most one reason per theme; within each theme the highest-contrast feature
       wins. Returns top top_k themes sorted by contrast value.

    Parameters
    ----------
    contrast : np.ndarray, shape (n_features,); within-list contrast for this row.
    feature_values : np.ndarray, shape (n_features,); raw feature values for this row.
    feature_names : list[str]; canonical feature name order.
    top_k : int; maximum number of reasons (one per theme).

    Returns
    -------
    List of reason strings (up to top_k). May be shorter if few features qualify.
    """
    feat_idx = {n: i for i, n in enumerate(feature_names)}

    # Check whether the exact-article feature is eligible
    article_feat = "i_days_since_bought_article"
    article_eligible = False
    if article_feat in feat_idx:
        aidx = feat_idx[article_feat]
        if aidx < len(contrast) and float(contrast[aidx]) > 0.0:
            fval = feature_values[aidx]
            try:
                fval_py = float(fval)
                if not np.isnan(fval_py) and int(round(fval_py)) > 0:
                    article_eligible = True
            except (ValueError, TypeError):
                pass

    # Collect (contrast, reason) tuples per theme
    eligible_by_theme: dict[str, list[tuple[float, str]]] = {}

    for feat, template_fn in _REASON_REGISTRY.items():
        if feat not in feat_idx:
            continue
        idx = feat_idx[feat]
        if idx >= len(contrast):
            continue

        # Suppress product-code when exact-article is eligible
        if feat == "i_days_since_bought_product_code" and article_eligible:
            continue

        cval = float(contrast[idx])
        if cval <= 0.0:
            continue

        fval = feature_values[idx]
        try:
            fval_py = float(fval)
            if np.isnan(fval_py):
                fval_py = None
        except (ValueError, TypeError):
            fval_py = None

        reason = template_fn(fval_py)
        if reason is not None:
            theme = _THEME.get(feat, "other")
            eligible_by_theme.setdefault(theme, []).append((cval, reason))

    # Within each theme pick highest-contrast reason; then sort themes by contrast
    theme_best: list[tuple[float, str]] = []
    for candidates in eligible_by_theme.values():
        candidates.sort(key=lambda x: x[0], reverse=True)
        theme_best.append(candidates[0])

    theme_best.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in theme_best[:top_k]]


# ---------------------------------------------------------------------------
# Faithfulness check
# ---------------------------------------------------------------------------

def faithfulness_check(
    booster,
    X: np.ndarray,
    contrast: np.ndarray,
    feature_names: list[str],
    seed: int = 42,
) -> dict:
    """Set top-contrast feature to NaN and compare score drop vs a random feature.

    For each row, identifies the feature with the highest positive contrast,
    sets it to NaN (missing), re-scores. Same for a random (non-top) feature.
    Faithfulness: top feature must cause a larger mean score drop.

    Uses a paired sign test (scipy) and reports means.

    Parameters
    ----------
    booster : lgb.Booster
    X : np.ndarray, shape (n_rows, n_features); float32
    contrast : np.ndarray, shape (n_rows, n_features); float32
    feature_names : list[str]
    seed : int

    Returns
    -------
    dict with top_mean_drop, random_mean_drop, mean_diff, p_value, passed (bool).
    """
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(seed)
    n_rows, n_feats = X.shape

    base_scores = booster.predict(X).astype(np.float64)

    top_drops = np.empty(n_rows, dtype=np.float64)
    rand_drops = np.empty(n_rows, dtype=np.float64)

    feat_idx = {n: i for i, n in enumerate(feature_names)}
    eligible_indices = [feat_idx[f] for f in REASON_FEATURES if f in feat_idx]

    for row_i in range(n_rows):
        # find top positive-contrast feature among eligible reason features
        row_contrast = contrast[row_i]
        pos_mask = row_contrast[eligible_indices] > 0
        if pos_mask.any():
            cands = [eligible_indices[j] for j, m in enumerate(pos_mask) if m]
            top_feat = cands[int(np.argmax(row_contrast[cands]))]
        else:
            # fall back: highest contrast among all features
            top_feat = int(np.argmax(row_contrast))

        # choose a random feature different from top_feat
        other_feats = [j for j in range(n_feats) if j != top_feat]
        rand_feat = int(rng.choice(other_feats))

        # ablate top feature
        row_x = X[row_i:row_i + 1].copy()
        orig_top = row_x[0, top_feat]
        row_x[0, top_feat] = np.nan
        score_top_ablated = float(booster.predict(row_x)[0])
        top_drops[row_i] = base_scores[row_i] - score_top_ablated
        row_x[0, top_feat] = orig_top

        # ablate random feature
        orig_rand = row_x[0, rand_feat]
        row_x[0, rand_feat] = np.nan
        score_rand_ablated = float(booster.predict(row_x)[0])
        rand_drops[row_i] = base_scores[row_i] - score_rand_ablated
        row_x[0, rand_feat] = orig_rand

    diff = top_drops - rand_drops
    # Wilcoxon signed-rank test: H0 = top feature drop same as random
    if len(diff) >= 10 and diff.std() > 0:
        try:
            stat, p_val = wilcoxon(diff, alternative="greater")
        except Exception:
            stat, p_val = float("nan"), float("nan")
    else:
        stat, p_val = float("nan"), float("nan")

    return {
        "top_mean_drop": float(top_drops.mean()),
        "random_mean_drop": float(rand_drops.mean()),
        "mean_diff": float(diff.mean()),
        "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
        "p_value": float(p_val) if not np.isnan(p_val) else None,
        "n_rows": n_rows,
        "passed": bool(top_drops.mean() > rand_drops.mean()),
    }
