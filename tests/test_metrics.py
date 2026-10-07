"""Unit tests for metrics.py — including all 5 MAP@12 cases from EDA."""
import pytest
from src.metrics import (
    average_precision_at_k,
    map_at_k,
    micro_recall_at_k,
    mean_recall_at_k,
    precision_at_k,
    recall_at_k,
)

# ---------------------------------------------------------------------------
# MAP@12 unit tests (ported from EDA Section 11)
# ---------------------------------------------------------------------------

def test_ap_perfect():
    """All 12 predictions correct, all in order → AP = 1.0."""
    recs = list(range(1, 13))
    rel = set(range(1, 13))
    assert average_precision_at_k(recs, rel, 12) == pytest.approx(1.0)


def test_ap_no_hit():
    """No overlap between predictions and relevant → AP = 0.0."""
    recs = list(range(13, 25))
    rel = set(range(1, 4))
    assert average_precision_at_k(recs, rel, 12) == 0.0


def test_ap_partial_alternating():
    """Hits at positions 1, 3, 5 with 3 relevant items.

    AP = (1/1 + 2/3 + 3/5) / min(3,12) = (1 + 0.6667 + 0.6) / 3 ≈ 0.7556
    """
    recs = [1, 99, 2, 98, 3, 97, 96, 95, 94, 93, 92, 91]
    rel = {1, 2, 3}
    expected = (1.0 / 1 + 2.0 / 3 + 3.0 / 5) / 3
    assert average_precision_at_k(recs, rel, 12) == pytest.approx(expected)


def test_ap_duplicate_prediction():
    """Duplicate item in prediction list must not count as second hit.

    rec = [1, 1, 2, 3], relevant = {1, 2}
    Position 1: hit 1 → hits=1, score += 1/1
    Position 2: 1 already seen → skip
    Position 3: hit 2 → hits=2, score += 2/3
    AP = (1 + 2/3) / min(2,4) = (5/3) / 2 = 5/6
    """
    recs = [1, 1, 2, 3]
    rel = {1, 2}
    expected = (1.0 / 1 + 2.0 / 3) / 2
    assert average_precision_at_k(recs, rel, 4) == pytest.approx(expected)


def test_map_multi_user():
    """MAP aggregates correctly over multiple users."""
    preds = {
        0: [1, 2, 3],
        1: [10, 11, 12],
        2: [5, 6, 7],
    }
    gt = {
        0: {1, 2, 3},
        1: {99},        # no hit → AP=0
        2: {5},         # hit at pos 1 → AP=1.0
    }
    # AP user 0 = 1.0, user 1 = 0.0, user 2 = 1.0 → MAP = 2/3
    assert map_at_k(preds, gt, k=12) == pytest.approx(2.0 / 3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_ap_empty_relevant():
    assert average_precision_at_k([1, 2, 3], set(), 12) == 0.0


def test_ap_empty_predictions():
    assert average_precision_at_k([], {1, 2}, 12) == 0.0


def test_map_empty_ground_truth():
    assert map_at_k({0: [1, 2]}, {}, k=12) == 0.0


def test_map_user_with_no_prediction():
    """User in ground_truth but not in predictions → AP=0, still counted in mean."""
    preds = {0: [1]}
    gt = {0: {1}, 1: {2}}
    # user 0: AP=1.0, user 1: AP=0.0 → MAP = 0.5
    assert map_at_k(preds, gt, k=12) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Recall@k unit tests
# ---------------------------------------------------------------------------

def test_recall_full_coverage():
    """All 3 relevant items in top-5 candidates."""
    assert recall_at_k([1, 2, 3, 4, 5], {1, 2, 3}, k=5) == pytest.approx(1.0)


def test_recall_partial():
    """2 of 3 relevant in top-5; recall = 2/3."""
    assert recall_at_k([1, 2, 99, 98, 97], {1, 2, 3}, k=5) == pytest.approx(2.0 / 3)


def test_recall_k_cutoff():
    """Item at position > k is not counted even if relevant."""
    # relevant item 3 is at position 6; recall@5 should not count it
    assert recall_at_k([1, 2, 99, 98, 97, 3], {1, 2, 3}, k=5) == pytest.approx(2.0 / 3)


def test_recall_empty_relevant():
    assert recall_at_k([1, 2, 3], set(), k=5) == 0.0


def test_recall_empty_candidates():
    assert recall_at_k([], {1, 2}, k=5) == 0.0


def test_recall_single_item_hit():
    assert recall_at_k([42], {42}, k=1) == pytest.approx(1.0)


def test_recall_single_item_miss():
    assert recall_at_k([1], {42}, k=1) == 0.0


def test_micro_recall():
    """Micro recall: sum of hits / sum of |gt|."""
    cands = {0: [1, 2, 3], 1: [4, 5]}
    gt = {0: {1, 2, 99}, 1: {4, 88}}
    # user 0: 2 hits out of 3 gt; user 1: 1 hit out of 2 gt
    # micro = (2+1) / (3+2) = 3/5
    assert micro_recall_at_k(cands, gt, k=200) == pytest.approx(3.0 / 5)


def test_micro_recall_empty_gt():
    assert micro_recall_at_k({0: [1]}, {}, k=12) == 0.0


# ---------------------------------------------------------------------------
# Precision@k unit tests
# ---------------------------------------------------------------------------

def test_precision_perfect():
    """All k predictions are relevant."""
    assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)


def test_precision_partial():
    """1 of 3 predictions is relevant → precision@3 = 1/3."""
    assert precision_at_k([1, 99, 98], {1}, k=3) == pytest.approx(1.0 / 3)


def test_precision_no_hit():
    assert precision_at_k([99, 98, 97], {1, 2, 3}, k=3) == 0.0


def test_precision_k_cutoff():
    """Precision@2: items beyond k are not counted; only first 2 [99, 98] count → 0.0."""
    assert precision_at_k([99, 98, 1, 2], {1, 2, 3}, k=2) == pytest.approx(0.0)


def test_precision_empty_recommendations():
    assert precision_at_k([], {1, 2}, k=12) == 0.0


def test_precision_empty_relevant():
    """Relevant is empty; precision counts non-matching → 0."""
    assert precision_at_k([1, 2, 3], set(), k=3) == 0.0
