"""Tests for time_split.py: date correctness, leakage guards, fold structure."""
import datetime
import pytest
from src.config import HOLDOUT_WEEK, VALIDATION_WEEKS
from src.time_split import week_to_dates, build_fold


# ---------------------------------------------------------------------------
# Date math — no data loading required
# ---------------------------------------------------------------------------

def test_anchor_is_wednesday():
    """ANCHOR_DATE 2018-09-19 must be a Wednesday (weekday 2)."""
    anchor = datetime.date(2018, 9, 19)
    assert anchor.weekday() == 2, f"Expected Wednesday (2), got {anchor.weekday()}"


@pytest.mark.parametrize("w", VALIDATION_WEEKS + [HOLDOUT_WEEK])
def test_week_starts_on_wednesday(w):
    """Every week_idx starts on a Wednesday."""
    start_str, end_str = week_to_dates(w)
    start = datetime.date.fromisoformat(start_str)
    assert start.weekday() == 2, f"Week {w} start {start_str} is not Wednesday"


@pytest.mark.parametrize("w", VALIDATION_WEEKS + [HOLDOUT_WEEK])
def test_week_has_exactly_7_days(w):
    """Each week spans exactly 7 calendar days (Wed to Tue inclusive)."""
    start_str, end_str = week_to_dates(w)
    start = datetime.date.fromisoformat(start_str)
    end = datetime.date.fromisoformat(end_str)
    delta = (end - start).days + 1
    assert delta == 7, f"Week {w} spans {delta} days, expected 7"


@pytest.mark.parametrize("w", VALIDATION_WEEKS + [HOLDOUT_WEEK])
def test_week_ends_on_tuesday(w):
    """Each week must end on a Tuesday (weekday 1)."""
    _, end_str = week_to_dates(w)
    end = datetime.date.fromisoformat(end_str)
    assert end.weekday() == 1, f"Week {w} end {end_str} is not Tuesday"


def test_holdout_week_exact_dates():
    """Week 104 = 2020-09-16 (Wed) to 2020-09-22 (Tue)."""
    start, end = week_to_dates(HOLDOUT_WEEK)
    assert start == "2020-09-16", f"Expected 2020-09-16, got {start}"
    assert end == "2020-09-22", f"Expected 2020-09-22, got {end}"


def test_holdout_not_in_validation_weeks():
    """HOLDOUT_WEEK must never appear in VALIDATION_WEEKS."""
    assert HOLDOUT_WEEK not in VALIDATION_WEEKS


def test_validation_weeks_are_consecutive():
    """Validation weeks must be [100, 101, 102, 103]."""
    assert VALIDATION_WEEKS == [100, 101, 102, 103]


def test_validation_weeks_before_holdout():
    """All validation weeks must precede HOLDOUT_WEEK."""
    assert all(w < HOLDOUT_WEEK for w in VALIDATION_WEEKS)


# ---------------------------------------------------------------------------
# build_fold safety guard — no data loading required
# ---------------------------------------------------------------------------

def test_build_fold_raises_on_holdout():
    """build_fold must raise ValueError for HOLDOUT_WEEK."""
    with pytest.raises(ValueError, match="holdout"):
        build_fold(HOLDOUT_WEEK)


# ---------------------------------------------------------------------------
# Fold structure tests — require parquet data
# Uses fold 103 (closest to holdout; smallest history difference)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fold_103():
    """Load fold 103 once per test session."""
    history, ground_truth, eval_customers = build_fold(103)
    return history, ground_truth, eval_customers


@pytest.mark.requires_data
def test_fold_ground_truth_nonempty(fold_103):
    """Fold 103 ground truth must contain at least one customer."""
    _, ground_truth, eval_customers = fold_103
    assert len(ground_truth) > 0
    assert len(eval_customers) > 0


@pytest.mark.requires_data
def test_fold_eval_customers_match_ground_truth(fold_103):
    """Every eval_customer must appear in ground_truth and vice versa."""
    _, ground_truth, eval_customers = fold_103
    assert set(eval_customers) == set(ground_truth.keys())


@pytest.mark.requires_data
def test_fold_ground_truth_sets_nonempty(fold_103):
    """Every customer in ground_truth must have at least one article."""
    _, ground_truth, _ = fold_103
    for customer_idx, articles in ground_truth.items():
        assert len(articles) >= 1, f"customer {customer_idx} has empty ground truth"


@pytest.mark.requires_data
def test_fold_history_max_week_before_target(fold_103):
    """History max week_idx must be strictly less than 103."""
    import polars as pl
    history, _, _ = fold_103
    max_week = (
        history
        .select(pl.col("week_idx").max())
        .collect()
        .item()
    )
    assert max_week < 103, f"History contains week {max_week} >= 103 (leakage!)"


@pytest.mark.requires_data
def test_fold_history_no_target_week_dates(fold_103):
    """History must contain no dates from week 103 (2020-09-02 to 2020-09-08)."""
    import polars as pl
    from src.time_split import week_to_dates
    history, _, _ = fold_103
    start_str, end_str = week_to_dates(103)
    rows_in_target = (
        history
        .filter(
            (pl.col("t_dat") >= pl.lit(start_str).str.to_date())
            & (pl.col("t_dat") <= pl.lit(end_str).str.to_date())
        )
        .collect()
        .height
    )
    assert rows_in_target == 0, f"{rows_in_target} target-week rows leaked into history"


@pytest.mark.requires_data
def test_fold_eval_customers_sorted(fold_103):
    """eval_customers must be sorted ascending."""
    _, _, eval_customers = fold_103
    assert eval_customers == sorted(eval_customers)
