"""Metric suite beyond MAP@k: hit rate, catalog coverage, popularity bias, repeat/novelty share."""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Point-estimate metrics
# ─────────────────────────────────────────────────────────────────────────────

def per_customer_hit(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 12,
) -> dict[int, float]:
    """Binary hit indicator per customer (1.0 if ≥1 recommended item is relevant)."""
    return {
        c: float(any(a in ground_truth.get(c, set()) for a in recs[:k]))
        for c, recs in predictions.items()
        if c in ground_truth
    }


def compute_hit_rate(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 12,
) -> float:
    """Fraction of eval customers with ≥1 purchased item in their top-k."""
    hits = per_customer_hit(predictions, ground_truth, k)
    return float(np.mean(list(hits.values()))) if hits else 0.0


def compute_catalog_coverage(
    predictions: dict[int, list[int]],
    sold_articles: set[int],
) -> float:
    """Unique articles recommended / unique articles sold in history."""
    recommended: set[int] = set()
    for arts in predictions.values():
        recommended.update(arts)
    return len(recommended) / len(sold_articles) if sold_articles else 0.0


def per_customer_pop_bias(
    predictions: dict[int, list[int]],
    article_pop_ranks: dict[int, int],
    ground_truth: dict[int, set[int]] | None = None,
) -> dict[int, float]:
    """Mean popularity rank of each customer's recommendations.

    Lower rank = more popular. Only includes articles with known ranks.
    If ground_truth is provided, restricts to eval customers.
    """
    custs = set(ground_truth.keys()) if ground_truth is not None else set(predictions.keys())
    result = {}
    for c, arts in predictions.items():
        if c not in custs:
            continue
        ranks = [article_pop_ranks[a] for a in arts if a in article_pop_ranks]
        result[c] = float(np.mean(ranks)) if ranks else float("nan")
    return result


def compute_popularity_bias(
    predictions: dict[int, list[int]],
    article_pop_ranks: dict[int, int],
    ground_truth: dict[int, set[int]] | None = None,
) -> float:
    """Mean popularity rank of recommended items averaged over eval customers."""
    vals = per_customer_pop_bias(predictions, article_pop_ranks, ground_truth)
    finite = [v for v in vals.values() if not np.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")


def per_customer_repeat_share(
    predictions: dict[int, list[int]],
    customer_history: dict[int, set[int]],
    ground_truth: dict[int, set[int]] | None = None,
) -> dict[int, float]:
    """Fraction of each customer's recommendations that they bought before."""
    custs = set(ground_truth.keys()) if ground_truth is not None else set(predictions.keys())
    result = {}
    for c, arts in predictions.items():
        if c not in custs or not arts:
            continue
        hist = customer_history.get(c, set())
        result[c] = sum(1 for a in arts if a in hist) / len(arts)
    return result


def compute_repeat_share(
    predictions: dict[int, list[int]],
    customer_history: dict[int, set[int]],
    ground_truth: dict[int, set[int]] | None = None,
) -> float:
    """Mean fraction of recommended items the customer bought before."""
    vals = per_customer_repeat_share(predictions, customer_history, ground_truth)
    return float(np.mean(list(vals.values()))) if vals else 0.0


def per_customer_novelty_share(
    predictions: dict[int, list[int]],
    articles_before_cutoff: set[int],
    ground_truth: dict[int, set[int]] | None = None,
) -> dict[int, float]:
    """Fraction of each customer's recommendations that were never sold before cutoff."""
    custs = set(ground_truth.keys()) if ground_truth is not None else set(predictions.keys())
    result = {}
    for c, arts in predictions.items():
        if c not in custs or not arts:
            continue
        result[c] = sum(1 for a in arts if a not in articles_before_cutoff) / len(arts)
    return result


def compute_novelty_share(
    predictions: dict[int, list[int]],
    articles_before_cutoff: set[int],
    ground_truth: dict[int, set[int]] | None = None,
) -> float:
    """Mean fraction of recommendations never sold before the cutoff."""
    vals = per_customer_novelty_share(predictions, articles_before_cutoff, ground_truth)
    return float(np.mean(list(vals.values()))) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI (over customers)
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    per_customer_values: dict[int, float],
    n: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap 95% CI for the mean of per-customer values (resample customers)."""
    vals = np.array([v for v in per_customer_values.values() if not np.isnan(v)], dtype=float)
    n_custs = len(vals)
    if n_custs == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_customers": 0}

    rng = np.random.default_rng(seed)
    boot_means = np.array([vals[rng.integers(0, n_custs, size=n_custs)].mean() for _ in range(n)])

    return {
        "mean": float(vals.mean()),
        "ci_lo": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        "n_customers": n_custs,
    }


def bootstrap_coverage_ci(
    predictions: dict[int, list[int]],
    sold_articles: set[int],
    n: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI for catalog coverage by resampling customers."""
    custs = list(predictions.keys())
    n_custs = len(custs)
    custs_arr = np.array(custs, dtype=np.int64)

    rng = np.random.default_rng(seed)
    n_sold = len(sold_articles)
    sold_list = list(sold_articles)

    # Precompute per-customer article sets for fast sampling
    cust_to_idx = {c: i for i, c in enumerate(custs)}
    per_cust_arts = [set(predictions[c]) for c in custs]

    boot_covs = np.zeros(n)
    for i in range(n):
        idx = rng.integers(0, n_custs, size=n_custs)
        sampled_arts: set[int] = set()
        for j in idx:
            sampled_arts.update(per_cust_arts[j])
        boot_covs[i] = len(sampled_arts) / n_sold if n_sold > 0 else 0.0

    point = compute_catalog_coverage(predictions, sold_articles)
    return {
        "mean": float(point),
        "ci_lo": float(np.percentile(boot_covs, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(boot_covs, 100 * (1 - alpha / 2))),
        "n_customers": n_custs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full metric suite
# ─────────────────────────────────────────────────────────────────────────────

def run_metric_suite(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    article_pop_ranks: dict[int, int],
    customer_history: dict[int, set[int]],
    articles_before_cutoff: set[int],
    sold_articles: set[int],
    n_boot: int = 1000,
    seed: int = 42,
    k: int = 12,
) -> dict:
    """Compute accuracy + beyond-accuracy metrics with bootstrap 95% CIs.

    Parameters
    ----------
    predictions: top-k predictions per customer
    ground_truth: target-week purchases per customer
    article_pop_ranks: article_idx -> rank (1=most popular) from training history
    customer_history: customer_idx -> set of article_idx bought before cutoff
    articles_before_cutoff: set of articles with any transaction before cutoff
    sold_articles: set of all articles sold in training history (for coverage denominator)
    """
    from src.metrics import map_at_k, mean_ndcg_at_k, mean_recall_user_at_k, mean_precision_at_k
    from src.model.evaluate import bootstrap_paired_diff
    from src.metrics import per_customer_ap

    ap_vals = per_customer_ap(predictions, ground_truth, k)

    # Accuracy metrics (bootstrap via resampling per-customer APs)
    map12 = map_at_k(predictions, ground_truth, k)
    ndcg12 = mean_ndcg_at_k(predictions, ground_truth, k)
    recall12 = mean_recall_user_at_k(predictions, ground_truth, k)
    prec12 = mean_precision_at_k(predictions, ground_truth, k)

    ap_ci = bootstrap_ci(ap_vals, n=n_boot, seed=seed)

    # Hit rate@12
    hit_vals = per_customer_hit(predictions, ground_truth, k)
    hit_ci = bootstrap_ci(hit_vals, n=n_boot, seed=seed)

    # Popularity bias
    pop_vals = per_customer_pop_bias(predictions, article_pop_ranks, ground_truth)
    pop_ci = bootstrap_ci(pop_vals, n=n_boot, seed=seed)

    # Repeat share
    rep_vals = per_customer_repeat_share(predictions, customer_history, ground_truth)
    rep_ci = bootstrap_ci(rep_vals, n=n_boot, seed=seed)

    # Novelty share
    nov_vals = per_customer_novelty_share(predictions, articles_before_cutoff, ground_truth)
    nov_ci = bootstrap_ci(nov_vals, n=n_boot, seed=seed)

    # Catalog coverage (special bootstrap)
    # Filter predictions to eval customers
    eval_preds = {c: predictions[c] for c in ground_truth if c in predictions}
    cov_ci = bootstrap_coverage_ci(eval_preds, sold_articles, n=n_boot, seed=seed)

    return {
        "n_eval_customers": len(ground_truth),
        "k": k,
        "accuracy": {
            "map@12": map12,
            "map@12_ci": ap_ci,
            "ndcg@12": ndcg12,
            "recall@12": recall12,
            "precision@12": prec12,
            "hit_rate@12": hit_ci["mean"],
            "hit_rate@12_ci": hit_ci,
        },
        "beyond_accuracy": {
            "catalog_coverage": cov_ci["mean"],
            "catalog_coverage_ci": cov_ci,
            "popularity_bias_mean_rank": pop_ci["mean"],
            "popularity_bias_ci": pop_ci,
            "repeat_share": rep_ci["mean"],
            "repeat_share_ci": rep_ci,
            "novelty_share": nov_ci["mean"],
            "novelty_share_ci": nov_ci,
        },
    }
