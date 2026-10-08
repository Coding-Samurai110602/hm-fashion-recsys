"""Evaluation metrics: MAP@k, recall@k, precision@k."""
from __future__ import annotations


def average_precision_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Average precision at k for a single user.

    Duplicate predictions do not count as additional hits (tracked via seen set).
    Returns 0.0 for empty relevant or empty recommended.
    """
    if not relevant or not recommended:
        return 0.0

    seen: set[int] = set()
    hits = 0
    score = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant and item not in seen:
            hits += 1
            score += hits / (i + 1)
        seen.add(item)

    return score / min(len(relevant), k)


def map_at_k(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 12,
) -> float:
    """Mean average precision at k over all users in ground_truth.

    Users in ground_truth with no prediction get AP=0.
    """
    if not ground_truth:
        return 0.0

    total = 0.0
    for customer_idx, relevant in ground_truth.items():
        recs = predictions.get(customer_idx, [])
        total += average_precision_at_k(recs, relevant, k)

    return total / len(ground_truth)


def recall_at_k(
    candidates: list[int] | set[int],
    relevant: set[int],
    k: int,
) -> float:
    """Recall at k for a single user.

    candidates is treated as an ordered list; only first k are considered.
    Returns 0.0 for empty relevant; 1.0 if all relevant items are found.
    """
    if not relevant:
        return 0.0
    if not candidates:
        return 0.0

    top_k: set[int]
    if isinstance(candidates, set):
        top_k = candidates  # sets have no ordering; use all
    else:
        top_k = set(list(candidates)[:k])

    return len(top_k & relevant) / len(relevant)


def mean_recall_at_k(
    candidates_per_customer: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int,
) -> float:
    """Macro mean recall@k across all evaluation customers."""
    if not ground_truth:
        return 0.0
    scores = [
        recall_at_k(candidates_per_customer.get(c, []), gt, k)
        for c, gt in ground_truth.items()
    ]
    return sum(scores) / len(scores)


def micro_recall_at_k(
    candidates_per_customer: dict[int, list[int] | set[int]],
    ground_truth: dict[int, set[int]],
    k: int,
) -> float:
    """Micro recall@k: fraction of (customer, article) ground-truth pairs recovered.

    This is the primary metric for candidate evaluation.
    """
    if not ground_truth:
        return 0.0

    total_gt = 0
    total_hit = 0
    for c, gt in ground_truth.items():
        cands = candidates_per_customer.get(c, [])
        if isinstance(cands, set):
            top_k_set = cands
        else:
            top_k_set = set(list(cands)[:k])
        total_gt += len(gt)
        total_hit += len(top_k_set & gt)

    return total_hit / total_gt if total_gt > 0 else 0.0


def ndcg_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k for a single user.

    Binary relevance (1 if hit, 0 otherwise). Ideal DCG = sum of 1/log2(i+2)
    for i in range(min(|relevant|, k)).
    """
    import math
    if not relevant or not recommended:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def mean_ndcg_at_k(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 12,
) -> float:
    """Mean NDCG@k over all users in ground_truth."""
    if not ground_truth:
        return 0.0
    total = sum(
        ndcg_at_k(predictions.get(c, []), gt, k)
        for c, gt in ground_truth.items()
    )
    return total / len(ground_truth)


def mean_precision_at_k(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 12,
) -> float:
    """Mean precision@k over all users in ground_truth."""
    if not ground_truth:
        return 0.0
    total = sum(
        precision_at_k(predictions.get(c, []), gt, k)
        for c, gt in ground_truth.items()
    )
    return total / len(ground_truth)


def mean_recall_user_at_k(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 12,
) -> float:
    """Mean per-user recall@k over all users in ground_truth."""
    if not ground_truth:
        return 0.0
    total = sum(
        recall_at_k(predictions.get(c, []), gt, k)
        for c, gt in ground_truth.items()
    )
    return total / len(ground_truth)


def per_customer_ap(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 12,
) -> dict[int, float]:
    """Per-customer AP@k. Includes all ground_truth customers (AP=0 if no prediction)."""
    return {
        c: average_precision_at_k(predictions.get(c, []), gt, k)
        for c, gt in ground_truth.items()
    }


def precision_at_k(
    recommended: list[int],
    relevant: set[int],
    k: int,
) -> float:
    """Precision at k for a single user.

    Returns 0.0 for empty recommended or k=0.
    """
    if not recommended or k == 0:
        return 0.0
    top_k = recommended[:k]
    return sum(1 for item in top_k if item in relevant) / k
