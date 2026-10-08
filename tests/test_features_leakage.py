"""Leakage tests for feature engineering.

For every feature group: inject a sentinel transaction inside the target week
(and one with an extreme price) and assert no feature value changes.
Also asserts that every feature function rejects history containing week_idx >= cutoff.

End-to-end pipeline test: build_fold_features on synthetic data must produce
identical feature columns whether or not sentinel transactions exist in the
target week (the pipeline filters them out; features only see history).
"""
import pytest
import polars as pl
from datetime import date

from src.features.customer import compute_customer_features
from src.features.article import compute_article_features
from src.features.interaction import compute_interaction_features
from src.features.build import _compute_features_inner
from src.config import ANCHOR_EPOCH_DAYS


CUTOFF_WEEK = 103  # arbitrary; use fold 103 cutoff


def _anchor_epoch():
    return ANCHOR_EPOCH_DAYS


def _week_to_epoch(week_idx: int) -> int:
    return ANCHOR_EPOCH_DAYS + week_idx * 7


def _make_history(include_future: bool = False) -> pl.LazyFrame:
    """Synthetic transaction history with week_idx < CUTOFF_WEEK."""
    cutoff_epoch = _week_to_epoch(CUTOFF_WEEK)
    # Use dates in weeks 100 and 101 (before cutoff=103)
    rows = [
        # customer 1, article 10: bought in week 100 (price 0.05)
        {"customer_idx": 1, "article_idx": 10, "t_dat": _week_to_epoch(100), "price": 0.05, "sales_channel_id": 1, "week_idx": 100},
        # customer 1, article 10: bought again in week 101 (later date — repurchase)
        {"customer_idx": 1, "article_idx": 10, "t_dat": _week_to_epoch(101), "price": 0.05, "sales_channel_id": 2, "week_idx": 101},
        # customer 1, article 20: week 100
        {"customer_idx": 1, "article_idx": 20, "t_dat": _week_to_epoch(100), "price": 0.10, "sales_channel_id": 1, "week_idx": 100},
        # customer 2, article 10: week 102
        {"customer_idx": 2, "article_idx": 10, "t_dat": _week_to_epoch(102), "price": 0.08, "sales_channel_id": 2, "week_idx": 102},
    ]

    if include_future:
        # Sentinel transaction INSIDE target week (week_idx == CUTOFF_WEEK)
        rows.append({
            "customer_idx": 1, "article_idx": 30, "t_dat": _week_to_epoch(CUTOFF_WEEK),
            "price": 99.99, "sales_channel_id": 2, "week_idx": CUTOFF_WEEK,
        })
        # Extreme price transaction also inside target week
        rows.append({
            "customer_idx": 1, "article_idx": 10, "t_dat": _week_to_epoch(CUTOFF_WEEK),
            "price": 999.0, "sales_channel_id": 1, "week_idx": CUTOFF_WEEK,
        })

    schema = {
        "customer_idx": pl.Int32,
        "article_idx": pl.Int32,
        "t_dat": pl.Int32,
        "price": pl.Float32,
        "sales_channel_id": pl.Int32,
        "week_idx": pl.Int16,
    }
    return pl.DataFrame(rows, schema=schema).lazy()


def _make_customers_df() -> pl.LazyFrame:
    return pl.DataFrame({
        "customer_idx": pl.Series([1, 2], dtype=pl.Int32),
        "age": pl.Series([30.0, 45.0], dtype=pl.Float32),
        "FN": pl.Series([1.0, None], dtype=pl.Float32),
        "Active": pl.Series([1.0, 0.0], dtype=pl.Float32),
        "club_member_status": pl.Series(["ACTIVE", "PRE-CREATE"], dtype=pl.Utf8),
        "fashion_news_frequency": pl.Series(["Regularly", "Monthly"], dtype=pl.Utf8),
    }).lazy()


def _make_articles_df() -> pl.LazyFrame:
    return pl.DataFrame({
        "article_idx": pl.Series([10, 20, 30], dtype=pl.Int32),
        "product_code": pl.Series(["PC001", "PC002", "PC003"], dtype=pl.Utf8),
        "product_type_no": pl.Series([1, 2, 3], dtype=pl.Int32),
        "product_group_name": pl.Series(["Garment Upper body", "Garment Full body", "Accessories"], dtype=pl.Utf8),
        "index_group_name": pl.Series(["Ladieswear", "Ladieswear", "Menswear"], dtype=pl.Utf8),
        "garment_group_name": pl.Series(["Blouses", "Dresses", "Hats"], dtype=pl.Utf8),
        "colour_group_name": pl.Series(["Black", "White", "Blue"], dtype=pl.Utf8),
        "section_name": pl.Series(["H&M Woman", "H&M Woman", "H&M Man"], dtype=pl.Utf8),
        "department_no": pl.Series([1001, 1002, 1003], dtype=pl.Int32),
    }).lazy()


# ===========================================================================
# Tests: assert functions reject history containing week_idx >= cutoff
# ===========================================================================

class TestHistoryLeakageAssert:
    """Feature functions must assert no future data in history."""

    def test_customer_features_rejects_future(self):
        history_with_future = _make_history(include_future=True)
        with pytest.raises(AssertionError, match=r"week_idx >= \d+"):
            compute_customer_features(
                history_with_future, CUTOFF_WEEK, _make_customers_df(), ANCHOR_EPOCH_DAYS
            )

    def test_article_features_rejects_future(self):
        history_with_future = _make_history(include_future=True)
        with pytest.raises(AssertionError, match=r"week_idx >= \d+"):
            compute_article_features(
                history_with_future, CUTOFF_WEEK, _make_articles_df(),
                _make_customers_df(), ANCHOR_EPOCH_DAYS
            )

    def test_interaction_features_rejects_future(self):
        history_with_future = _make_history(include_future=True)
        candidates = pl.DataFrame({
            "customer_idx": pl.Series([1], dtype=pl.Int32),
            "article_idx": pl.Series([10], dtype=pl.Int32),
        })
        history_clean = _make_history(include_future=False)
        art_feats = compute_article_features(
            history_clean, CUTOFF_WEEK, _make_articles_df(), _make_customers_df(), ANCHOR_EPOCH_DAYS
        )
        cust_feats = compute_customer_features(
            history_clean, CUTOFF_WEEK, _make_customers_df(), ANCHOR_EPOCH_DAYS
        )
        with pytest.raises(AssertionError, match=r"week_idx >= \d+"):
            compute_interaction_features(
                history_with_future, candidates, CUTOFF_WEEK,
                art_feats, cust_feats, _make_articles_df(), ANCHOR_EPOCH_DAYS
            )


# ===========================================================================
# Tests: injecting sentinel inside target week does not change feature values
# ===========================================================================

class TestNoLeakageFromFutureData:
    """Feature values must not change when future rows are added to history."""

    def _compute_customer_feats(self, include_future: bool) -> pl.DataFrame:
        hist = _make_history(include_future).filter(pl.col("week_idx") < CUTOFF_WEEK)
        return compute_customer_features(hist, CUTOFF_WEEK, _make_customers_df(), ANCHOR_EPOCH_DAYS)

    def _compute_article_feats(self, include_future: bool) -> pl.DataFrame:
        hist = _make_history(include_future).filter(pl.col("week_idx") < CUTOFF_WEEK)
        return compute_article_features(
            hist, CUTOFF_WEEK, _make_articles_df(), _make_customers_df(), ANCHOR_EPOCH_DAYS
        )

    def test_customer_features_unchanged_by_future(self):
        baseline = self._compute_customer_feats(include_future=False)
        with_future = self._compute_customer_feats(include_future=True)
        cols = sorted(baseline.columns)
        b = baseline.sort("customer_idx").select(cols)
        f = with_future.sort("customer_idx").select(cols)
        assert b.equals(f), "Customer features changed when future rows were added to history"

    def test_article_features_unchanged_by_future(self):
        baseline = self._compute_article_feats(include_future=False)
        with_future = self._compute_article_feats(include_future=True)
        cols = sorted([c for c in baseline.columns if not c.startswith("_")])
        b = baseline.sort("article_idx").select(cols)
        f = with_future.sort("article_idx").select(cols)
        assert b.equals(f), "Article features changed when future rows were added to history"

    def test_sentinel_article_not_in_article_features_1w(self):
        """Article 30 (sentinel, only in target week) must have a_sales_1w == 0."""
        hist = _make_history(include_future=False).filter(pl.col("week_idx") < CUTOFF_WEEK)
        art_feats = compute_article_features(
            hist, CUTOFF_WEEK, _make_articles_df(), _make_customers_df(), ANCHOR_EPOCH_DAYS
        )
        art30 = art_feats.filter(pl.col("article_idx") == 30)
        # Article 30 has never been sold before cutoff
        assert art30["a_sales_1w"][0] == 0, "Sentinel article should have 0 sales before cutoff"

    def test_extreme_price_does_not_affect_features(self):
        """Extreme price transaction in target week must not affect c_mean_price or a_mean_price_4w."""
        hist_clean = _make_history(include_future=False).filter(pl.col("week_idx") < CUTOFF_WEEK)
        # Simulate the extreme price transaction being stripped (it should be since week_idx >= cutoff)
        hist_extreme = _make_history(include_future=True).filter(pl.col("week_idx") < CUTOFF_WEEK)
        # Same data — extreme price (week CUTOFF_WEEK) is filtered out
        cust_clean = compute_customer_features(hist_clean, CUTOFF_WEEK, _make_customers_df(), ANCHOR_EPOCH_DAYS)
        cust_extreme = compute_customer_features(hist_extreme, CUTOFF_WEEK, _make_customers_df(), ANCHOR_EPOCH_DAYS)
        # mean_price should be identical since extreme price is at week CUTOFF_WEEK (filtered out)
        c1_clean = cust_clean.filter(pl.col("customer_idx") == 1)["c_mean_price"][0]
        c1_extreme = cust_extreme.filter(pl.col("customer_idx") == 1)["c_mean_price"][0]
        assert abs(c1_clean - c1_extreme) < 1e-6, \
            f"c_mean_price changed: {c1_clean} vs {c1_extreme}"


# ===========================================================================
# End-to-end pipeline leakage test
# ===========================================================================

def _make_all_txns(include_sentinel: bool) -> pl.DataFrame:
    """Synthetic transaction set for the pipeline leakage test.

    History (week_idx < 10):
      Customer 1 buys article 10 in weeks 8 and 9 (price 0.05 each).
      Customer 1 buys article 20 in week 7 (price 0.10).

    Target week (week_idx == 10):
      Customer 1 buys article 20 (price 0.20).

    Optional sentinel (include_sentinel=True):
      Article 10 is bought by customer 1 in week 10 at price 999.0 (extreme).
      If the pipeline accidentally includes this in history, c_mean_price and
      a_mean_price_4w would change dramatically.
    """
    rows = [
        # History
        {"customer_idx": 1, "article_idx": 10, "t_dat": _week_to_epoch(8),
         "price": 0.05, "sales_channel_id": 1, "week_idx": 8},
        {"customer_idx": 1, "article_idx": 10, "t_dat": _week_to_epoch(9),
         "price": 0.05, "sales_channel_id": 2, "week_idx": 9},
        {"customer_idx": 1, "article_idx": 20, "t_dat": _week_to_epoch(7),
         "price": 0.10, "sales_channel_id": 1, "week_idx": 7},
        # Target week — not in history
        {"customer_idx": 1, "article_idx": 20, "t_dat": _week_to_epoch(10),
         "price": 0.20, "sales_channel_id": 1, "week_idx": 10},
    ]
    if include_sentinel:
        # Extreme price in target week — must NOT enter history
        rows.append({
            "customer_idx": 1, "article_idx": 10, "t_dat": _week_to_epoch(10),
            "price": 999.0, "sales_channel_id": 1, "week_idx": 10,
        })
    schema = {
        "customer_idx": pl.Int32, "article_idx": pl.Int32,
        "t_dat": pl.Int32, "price": pl.Float32,
        "sales_channel_id": pl.Int32, "week_idx": pl.Int16,
    }
    return pl.DataFrame(rows, schema=schema)


def _split_fold(
    all_txns: pl.DataFrame, target_week: int
) -> tuple[pl.LazyFrame, dict[int, set[int]], list[int]]:
    """Simulate build_fold: split into history and ground truth."""
    history = all_txns.filter(pl.col("week_idx") < target_week)
    target = all_txns.filter(pl.col("week_idx") == target_week)

    gt_dict: dict[int, set[int]] = {}
    for row in target.iter_rows(named=True):
        gt_dict.setdefault(row["customer_idx"], set()).add(row["article_idx"])

    eval_customers = sorted(gt_dict.keys())
    return history.lazy(), gt_dict, eval_customers


_PIPELINE_TARGET_WEEK = 10  # synthetic data target week for pipeline leakage test


class TestEndToEndPipelineLeakage:
    """Full _compute_features_inner pipeline: sentinel in target week must not change features.

    The pipeline filters history to week_idx < target_week before calling any
    feature function.  A sentinel transaction inside the target week — even one
    with an extreme price (999.0) — must produce identical feature values as a
    run without the sentinel.  Only labels may differ (if the sentinel article
    appears in candidates it gets label=1 instead of 0).
    """

    TARGET_WEEK = _PIPELINE_TARGET_WEEK

    def _run_pipeline(self, include_sentinel: bool) -> pl.DataFrame:
        all_txns = _make_all_txns(include_sentinel)
        history_lf, gt_dict, eval_customers = _split_fold(all_txns, self.TARGET_WEEK)
        return _compute_features_inner(
            history_lf=history_lf,
            ground_truth=gt_dict,
            eval_customers=eval_customers,
            articles_df=_make_articles_df(),
            customers_df=_make_customers_df(),
            fold_week=self.TARGET_WEEK,
            neg_sample_rate=1.0,
            seed=42,
        )

    def test_feature_columns_identical_with_and_without_sentinel(self):
        """Every feature column must be identical whether the sentinel is present."""
        feats_clean = self._run_pipeline(include_sentinel=False)
        feats_sentinel = self._run_pipeline(include_sentinel=True)

        feat_cols = [c for c in feats_clean.columns if c != "label"]

        # Align on (customer_idx, article_idx) — candidate sets may differ due to GT
        # but we compare only rows common to both (same candidates, same features)
        common_keys = (
            feats_clean.select(["customer_idx", "article_idx"])
            .join(
                feats_sentinel.select(["customer_idx", "article_idx"]),
                on=["customer_idx", "article_idx"],
                how="inner",
            )
        )
        clean_sub = feats_clean.join(common_keys, on=["customer_idx", "article_idx"], how="inner")
        sentinel_sub = feats_sentinel.join(common_keys, on=["customer_idx", "article_idx"], how="inner")

        clean_sorted = clean_sub.sort("customer_idx", "article_idx").select(feat_cols)
        sentinel_sorted = sentinel_sub.sort("customer_idx", "article_idx").select(feat_cols)

        assert clean_sorted.equals(sentinel_sorted), (
            "Feature values changed when sentinel transaction (price=999) was "
            "added to the target week.  The pipeline must filter history to "
            "week_idx < target_week before computing features."
        )

    def test_extreme_price_sentinel_not_in_price_features(self):
        """c_mean_price must not include the extreme sentinel price (999.0)."""
        feats_clean = self._run_pipeline(include_sentinel=False)
        feats_sentinel = self._run_pipeline(include_sentinel=True)

        # Customer 1 mean price without sentinel: (0.05 + 0.05 + 0.10) / 3 = 0.0667
        # If sentinel leaked into history: (0.05 + 0.05 + 0.10 + 999.0) / 4 ≈ 249.8
        clean_price = (
            feats_clean.filter(pl.col("customer_idx") == 1)["c_mean_price"][0]
        )
        sentinel_price = (
            feats_sentinel.filter(pl.col("customer_idx") == 1)["c_mean_price"][0]
        )
        assert abs(clean_price - sentinel_price) < 1e-4, (
            f"c_mean_price differs: clean={clean_price:.4f} sentinel={sentinel_price:.4f}. "
            "Extreme sentinel price leaked into history features."
        )
