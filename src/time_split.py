"""Time-based validation splits; no leakage from target week."""
import polars as pl
from src.config import ANCHOR_EPOCH_DAYS, HOLDOUT_WEEK
from src.data_io import load_transactions


def week_to_dates(week_idx: int) -> tuple[str, str]:
    """Return (start_date_iso, end_date_iso) for a given week_idx."""
    import datetime
    anchor = datetime.date(2018, 9, 19)
    start = anchor + datetime.timedelta(weeks=week_idx)
    end = start + datetime.timedelta(days=6)
    return start.isoformat(), end.isoformat()


def build_fold(
    target_week: int,
) -> tuple[pl.LazyFrame, dict[int, set[int]], list[int]]:
    """Build a single validation fold.

    Parameters
    ----------
    target_week : int
        The week whose buyers are the evaluation set. Must not be HOLDOUT_WEEK.

    Returns
    -------
    history : pl.LazyFrame
        All transactions with week_idx < target_week. Caller must filter further
        as needed; no target-week data is included.
    ground_truth : dict[int, set[int]]
        customer_idx -> set of unique article_idx bought in target_week.
    eval_customers : list[int]
        Sorted list of customer_idx with >= 1 purchase in target_week.
    """
    if target_week == HOLDOUT_WEEK:
        raise ValueError(
            f"target_week={target_week} is the holdout week. "
            "Use it only for the regression check in Part D."
        )

    txns = load_transactions()

    # History: strictly before target week
    history = txns.filter(pl.col("week_idx") < target_week)

    # Ground truth: unique articles per customer in target week
    target_rows = (
        txns.filter(pl.col("week_idx") == target_week)
        .select(["customer_idx", "article_idx"])
        .unique()
        .collect()
    )

    ground_truth: dict[int, set[int]] = {}
    for row in target_rows.iter_rows():
        c, a = row
        if c not in ground_truth:
            ground_truth[c] = set()
        ground_truth[c].add(a)

    eval_customers = sorted(ground_truth.keys())
    return history, ground_truth, eval_customers


def build_holdout_fold() -> tuple[pl.LazyFrame, dict[int, set[int]], list[int]]:
    """Build the holdout fold (week 104). Only for regression check.

    Returns the same structure as build_fold but uses week 104 as target.
    """
    txns = load_transactions()
    history = txns.filter(pl.col("week_idx") < HOLDOUT_WEEK)

    target_rows = (
        txns.filter(pl.col("week_idx") == HOLDOUT_WEEK)
        .select(["customer_idx", "article_idx"])
        .unique()
        .collect()
    )

    ground_truth: dict[int, set[int]] = {}
    for row in target_rows.iter_rows():
        c, a = row
        if c not in ground_truth:
            ground_truth[c] = set()
        ground_truth[c].add(a)

    eval_customers = sorted(ground_truth.keys())
    return history, ground_truth, eval_customers
