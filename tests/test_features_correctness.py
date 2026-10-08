"""Correctness tests for feature engineering.

Hand-computed synthetic examples for key features.
"""
import pytest
import polars as pl
import math

from src.features.customer import compute_customer_features
from src.features.article import compute_article_features
from src.features.interaction import compute_interaction_features
from src.features.candidate import compute_candidate_features
from src.config import ANCHOR_EPOCH_DAYS

CUTOFF_WEEK = 10  # small week index for clear arithmetic
CUTOFF_EPOCH = ANCHOR_EPOCH_DAYS + CUTOFF_WEEK * 7  # = 17793 + 70 = 17863


def _w2e(week_idx: int) -> int:
    return ANCHOR_EPOCH_DAYS + week_idx * 7


# ---------------------------------------------------------------------------
# Synthetic history: carefully constructed for hand verification
# ---------------------------------------------------------------------------
# Transactions (all before cutoff_week=10, i.e. week_idx < 10):
#   Cust 1, Art 10:  week 8 (price 0.10), week 9 (price 0.20)  — repurchase
#   Cust 1, Art 20:  week 7 (price 0.30)
#   Cust 1, Art 10:  week 9 (price 0.20)  [already counted above]
#   Cust 2, Art 10:  week 5 (price 0.15), week 6 (price 0.15)  — repurchase
#   Cust 3, Art 10:  week 9 (price 0.10)  — no repurchase

HISTORY_ROWS = [
    # cust 1, art 10: 2 purchases on different weeks -> repurchase
    {"customer_idx": 1, "article_idx": 10, "t_dat": _w2e(8), "price": 0.10, "sales_channel_id": 1, "week_idx": 8},
    {"customer_idx": 1, "article_idx": 10, "t_dat": _w2e(9), "price": 0.20, "sales_channel_id": 2, "week_idx": 9},
    # cust 1, art 20: 1 purchase
    {"customer_idx": 1, "article_idx": 20, "t_dat": _w2e(7), "price": 0.30, "sales_channel_id": 1, "week_idx": 7},
    # cust 2, art 10: 2 purchases on different weeks -> repurchase
    {"customer_idx": 2, "article_idx": 10, "t_dat": _w2e(5), "price": 0.15, "sales_channel_id": 2, "week_idx": 5},
    {"customer_idx": 2, "article_idx": 10, "t_dat": _w2e(6), "price": 0.15, "sales_channel_id": 2, "week_idx": 6},
    # cust 3, art 10: 1 purchase -> no repurchase
    {"customer_idx": 3, "article_idx": 10, "t_dat": _w2e(9), "price": 0.10, "sales_channel_id": 1, "week_idx": 9},
]

HISTORY_SCHEMA = {
    "customer_idx": pl.Int32, "article_idx": pl.Int32,
    "t_dat": pl.Int32, "price": pl.Float32, "sales_channel_id": pl.Int32, "week_idx": pl.Int16,
}

# Customers: ages for buyer age computations
CUSTOMERS_ROWS = [
    {"customer_idx": 1, "age": 30.0, "FN": 1.0, "Active": 1.0,
     "club_member_status": "ACTIVE", "fashion_news_frequency": "Regularly"},
    {"customer_idx": 2, "age": 40.0, "FN": None, "Active": 0.0,
     "club_member_status": "PRE-CREATE", "fashion_news_frequency": "Monthly"},
    {"customer_idx": 3, "age": 50.0, "FN": 0.0, "Active": None,
     "club_member_status": None, "fashion_news_frequency": None},
]
CUSTOMERS_SCHEMA = {
    "customer_idx": pl.Int32, "age": pl.Float64, "FN": pl.Float64, "Active": pl.Float64,
    "club_member_status": pl.Utf8, "fashion_news_frequency": pl.Utf8,
}

# Articles
ARTICLES_ROWS = [
    {"article_idx": 10, "product_code": "PC001", "product_type_no": 1,
     "product_group_name": "Garment Upper body", "index_group_name": "Ladieswear",
     "garment_group_name": "Blouses", "colour_group_name": "Black",
     "section_name": "H&M Woman", "department_no": 1001},
    {"article_idx": 20, "product_code": "PC002", "product_type_no": 2,
     "product_group_name": "Garment Full body", "index_group_name": "Ladieswear",
     "garment_group_name": "Dresses", "colour_group_name": "White",
     "section_name": "H&M Woman", "department_no": 1002},
]
ARTICLES_SCHEMA = {
    "article_idx": pl.Int32, "product_code": pl.Utf8, "product_type_no": pl.Int32,
    "product_group_name": pl.Utf8, "index_group_name": pl.Utf8, "garment_group_name": pl.Utf8,
    "colour_group_name": pl.Utf8, "section_name": pl.Utf8, "department_no": pl.Int32,
}


def _history() -> pl.LazyFrame:
    return pl.DataFrame(HISTORY_ROWS, schema=HISTORY_SCHEMA).lazy()


def _customers() -> pl.LazyFrame:
    return pl.DataFrame(CUSTOMERS_ROWS, schema=CUSTOMERS_SCHEMA).lazy()


def _articles() -> pl.LazyFrame:
    return pl.DataFrame(ARTICLES_ROWS, schema=ARTICLES_SCHEMA).lazy()


# ---------------------------------------------------------------------------
# Customer feature correctness
# ---------------------------------------------------------------------------

class TestCustomerFeatureCorrectness:

    @pytest.fixture(scope="class")
    def cust_feats(self):
        return compute_customer_features(_history(), CUTOFF_WEEK, _customers(), ANCHOR_EPOCH_DAYS)

    def test_n_purchases_total(self, cust_feats):
        # Cust 1: 3 transactions (art10 x2, art20 x1)
        c1 = cust_feats.filter(pl.col("customer_idx") == 1)["c_n_purchases"][0]
        assert c1 == 3, f"Expected 3, got {c1}"

    def test_n_purchases_1w(self, cust_feats):
        # Last 1 week = week 9 only (week_idx >= 9 AND < 10)
        # Cust 1: art10 in week 9 -> 1 purchase; Cust 3: art10 in week 9 -> 1 purchase
        c1 = cust_feats.filter(pl.col("customer_idx") == 1)["c_n_purchases_1w"][0]
        c2 = cust_feats.filter(pl.col("customer_idx") == 2)["c_n_purchases_1w"][0]
        assert c1 == 1, f"Cust 1 last-1w: expected 1, got {c1}"
        assert c2 == 0, f"Cust 2 last-1w: expected 0, got {c2}"

    def test_days_since_last_purchase(self, cust_feats):
        # Cust 1 last purchase: week 9, epoch = _w2e(9)
        # days = CUTOFF_EPOCH - _w2e(9) = (17793 + 70) - (17793 + 63) = 7
        c1 = cust_feats.filter(pl.col("customer_idx") == 1)["c_days_since_last_purchase"][0]
        assert abs(c1 - 7.0) < 1e-4, f"Expected 7.0, got {c1}"
        # Cust 2 last purchase: week 6, epoch = _w2e(6), days = 70 - 42 = 28
        c2 = cust_feats.filter(pl.col("customer_idx") == 2)["c_days_since_last_purchase"][0]
        assert abs(c2 - 28.0) < 1e-4, f"Expected 28.0, got {c2}"

    def test_n_unique_articles(self, cust_feats):
        # Cust 1: articles 10 and 20 -> 2 unique
        c1 = cust_feats.filter(pl.col("customer_idx") == 1)["c_n_unique_articles"][0]
        assert c1 == 2, f"Expected 2, got {c1}"

    def test_online_share(self, cust_feats):
        # Cust 1: 3 transactions, 1 is channel 2 (week 9, art10) -> 1/3
        c1 = cust_feats.filter(pl.col("customer_idx") == 1)["c_online_share"][0]
        assert abs(c1 - 1/3) < 1e-5, f"Expected {1/3:.6f}, got {c1}"
        # Cust 2: 2 transactions, both channel 2 -> 1.0
        c2 = cust_feats.filter(pl.col("customer_idx") == 2)["c_online_share"][0]
        assert abs(c2 - 1.0) < 1e-5, f"Expected 1.0, got {c2}"

    def test_categorical_encoding(self, cust_feats):
        # club_member_status: ACTIVE->1, PRE-CREATE->0, null->-1
        c1 = cust_feats.filter(pl.col("customer_idx") == 1)["c_club_member_status"][0]
        c2 = cust_feats.filter(pl.col("customer_idx") == 2)["c_club_member_status"][0]
        c3 = cust_feats.filter(pl.col("customer_idx") == 3)["c_club_member_status"][0]
        assert c1 == 1, f"ACTIVE should be 1, got {c1}"
        assert c2 == 0, f"PRE-CREATE should be 0, got {c2}"
        assert c3 == -1, f"null should be -1, got {c3}"

    def test_fn_encoding(self, cust_feats):
        # FN: 1.0->1, None->-1
        c1 = cust_feats.filter(pl.col("customer_idx") == 1)["c_FN"][0]
        c2 = cust_feats.filter(pl.col("customer_idx") == 2)["c_FN"][0]
        assert c1 == 1, f"FN=1.0 should encode to 1, got {c1}"
        assert c2 == -1, f"FN=None should encode to -1, got {c2}"


# ---------------------------------------------------------------------------
# Article feature correctness
# ---------------------------------------------------------------------------

class TestArticleFeatureCorrectness:

    @pytest.fixture(scope="class")
    def art_feats(self):
        return compute_article_features(
            _history(), CUTOFF_WEEK, _articles(), _customers(), ANCHOR_EPOCH_DAYS
        )

    def test_sales_1w(self, art_feats):
        # Last 1 week: week_idx >= 9 and < 10
        # Art 10: cust1 (week9) + cust3 (week9) = 2 sales
        a10 = art_feats.filter(pl.col("article_idx") == 10)["a_sales_1w"][0]
        assert a10 == 2, f"Art 10 sales_1w: expected 2, got {a10}"
        # Art 20: cust1 (week7) = not in week 9 -> 0
        a20 = art_feats.filter(pl.col("article_idx") == 20)["a_sales_1w"][0]
        assert a20 == 0, f"Art 20 sales_1w: expected 0, got {a20}"

    def test_sales_4w(self, art_feats):
        # Last 4 weeks: week_idx >= 6 and < 10
        # Art 10: week6(cust2) + week8(cust1) + week9(cust1) + week9(cust3) = 4 sales
        a10 = art_feats.filter(pl.col("article_idx") == 10)["a_sales_4w"][0]
        assert a10 == 4, f"Art 10 sales_4w: expected 4, got {a10}"

    def test_trend_ratio(self, art_feats):
        # trend_ratio = sales_1w / (sales_4w / 4)
        # Art 10: 2 / (4/4) = 2 / 1.0 = 2.0
        a10 = art_feats.filter(pl.col("article_idx") == 10)["a_trend_ratio"][0]
        assert abs(a10 - 2.0) < 1e-5, f"Art 10 trend_ratio: expected 2.0, got {a10}"

    def test_days_since_last_sale(self, art_feats):
        # Art 10 last sale: week 9, epoch = _w2e(9)
        # days = CUTOFF_EPOCH - _w2e(9) = 7
        a10 = art_feats.filter(pl.col("article_idx") == 10)["a_days_since_last_sale"][0]
        assert abs(a10 - 7.0) < 1e-4, f"Art 10 days_since_last_sale: expected 7, got {a10}"

    def test_repurchase_rate(self, art_feats):
        # Art 10:
        #   Cust 1: bought weeks 8 and 9 (2 distinct dates) -> repurchaser
        #   Cust 2: bought weeks 5 and 6 (2 distinct dates) -> repurchaser
        #   Cust 3: bought week 9 only (1 distinct date) -> not repurchaser
        # Rate = 2 repurchasers / 3 total buyers = 2/3
        a10 = art_feats.filter(pl.col("article_idx") == 10)["a_repurchase_rate"][0]
        expected = 2.0 / 3.0
        assert abs(a10 - expected) < 1e-5, f"Art 10 repurchase_rate: expected {expected:.6f}, got {a10}"

    def test_price_ratio(self, art_feats):
        # a_price_vs_own_history = last-week mean price / all-history mean price
        # Art 10:
        #   last-week (week 9): cust1=0.20, cust3=0.10 -> mean = 0.15
        #   all-history: 0.10 + 0.20 + 0.15 + 0.15 + 0.10 = 0.70 / 5 = 0.14
        #   ratio = 0.15 / 0.14
        a10 = art_feats.filter(pl.col("article_idx") == 10)["a_price_vs_own_history"][0]
        expected = 0.15 / 0.14
        assert abs(a10 - expected) < 1e-3, f"Art 10 price_vs_own_history: expected {expected:.4f}, got {a10}"

    def test_mean_buyer_age_12w(self, art_feats):
        # Last 12 weeks: all transactions (week_idx in range [cutoff-12, cutoff) = [-2, 10))
        # All weeks 5-9 are within 12 weeks of week 10
        # Art 10 buyers: cust1(age30), cust2(age40), cust3(age50) -> mean = (30+40+50)/3 = 40.0
        a10 = art_feats.filter(pl.col("article_idx") == 10)["a_mean_buyer_age_12w"][0]
        # Note: each customer counted once per transaction (not distinct)
        # Transactions of art10: cust1(w8), cust1(w9), cust2(w5), cust2(w6), cust3(w9) -> 5 rows
        # Ages: 30, 30, 40, 40, 50 -> mean = 38.0
        expected_mean = (30.0 + 30.0 + 40.0 + 40.0 + 50.0) / 5.0
        assert abs(a10 - expected_mean) < 1e-4, f"Art 10 mean_buyer_age: expected {expected_mean}, got {a10}"


# ---------------------------------------------------------------------------
# Interaction feature correctness
# ---------------------------------------------------------------------------

class TestInteractionFeatureCorrectness:

    @pytest.fixture(scope="class")
    def all_feats(self):
        hist = _history()
        art_feats = compute_article_features(hist, CUTOFF_WEEK, _articles(), _customers(), ANCHOR_EPOCH_DAYS)
        cust_feats = compute_customer_features(hist, CUTOFF_WEEK, _customers(), ANCHOR_EPOCH_DAYS)
        candidates = pl.DataFrame({
            "customer_idx": pl.Series([1, 1, 2], dtype=pl.Int32),
            "article_idx": pl.Series([10, 20, 10], dtype=pl.Int32),
        })
        int_feats = compute_interaction_features(
            hist, candidates, CUTOFF_WEEK, art_feats, cust_feats, _articles(), ANCHOR_EPOCH_DAYS
        )
        return cust_feats, art_feats, int_feats

    def test_times_bought(self, all_feats):
        _, _, int_feats = all_feats
        # Cust1, Art10: 2 times
        c1a10 = int_feats.filter(
            (pl.col("customer_idx") == 1) & (pl.col("article_idx") == 10)
        )["i_times_bought"][0]
        assert c1a10 == 2, f"Expected 2, got {c1a10}"
        # Cust2, Art10: 2 times
        c2a10 = int_feats.filter(
            (pl.col("customer_idx") == 2) & (pl.col("article_idx") == 10)
        )["i_times_bought"][0]
        assert c2a10 == 2, f"Expected 2, got {c2a10}"

    def test_days_since_bought_article(self, all_feats):
        _, _, int_feats = all_feats
        # Cust1, Art10: last bought week 9, days from cutoff = 7
        val = int_feats.filter(
            (pl.col("customer_idx") == 1) & (pl.col("article_idx") == 10)
        )["i_days_since_bought_article"][0]
        assert abs(val - 7.0) < 1e-4, f"Expected 7.0, got {val}"

    def test_bought_same_product_code(self, all_feats):
        _, _, int_feats = all_feats
        # Cust1, Art10 (product_code=PC001): cust1 has 2 purchases of art10 (same PC001) = 2
        # PC001 only maps to art10
        val = int_feats.filter(
            (pl.col("customer_idx") == 1) & (pl.col("article_idx") == 10)
        )["i_bought_same_product_code"][0]
        assert val == 2, f"Expected 2, got {val}"

    def test_days_since_bought_product_code(self, all_feats):
        _, _, int_feats = all_feats
        # Cust1, Art10 (PC001): last purchase of PC001 is week9 -> days = 7
        val = int_feats.filter(
            (pl.col("customer_idx") == 1) & (pl.col("article_idx") == 10)
        )["i_days_since_bought_product_code"][0]
        assert abs(val - 7.0) < 1e-4, f"Expected 7.0, got {val}"

    def test_price_ratio(self, all_feats):
        cust_feats, art_feats, int_feats = all_feats
        # i_price_ratio = a_mean_price_4w / c_mean_price
        # Art10 mean_price_4w (last 4 weeks = week 6-9):
        #   art10 in [6-9]: cust2(w6,0.15) + cust1(w8,0.10) + cust1(w9,0.20) + cust3(w9,0.10)
        #   mean = (0.15 + 0.10 + 0.20 + 0.10) / 4 = 0.1375
        # Cust1 mean_price (all history): (0.10 + 0.20 + 0.30) / 3 = 0.20
        # ratio = 0.1375 / 0.20 = 0.6875
        val = int_feats.filter(
            (pl.col("customer_idx") == 1) & (pl.col("article_idx") == 10)
        )["i_price_ratio"][0]
        expected = 0.1375 / 0.20
        assert abs(val - expected) < 1e-3, f"Expected {expected:.4f}, got {val}"

    def test_n_sources(self):
        """n_sources = count of sources that produced the pair."""
        merged = pl.DataFrame({
            "customer_idx": pl.Series([1, 1], dtype=pl.Int32),
            "article_idx": pl.Series([10, 20], dtype=pl.Int32),
            "final_source": pl.Series(["repurchase", "popularity_last_week"], dtype=pl.Utf8),
            "final_rank": pl.Series([1, 2], dtype=pl.Int32),
            "repurchase_rank": pl.Series([1, None], dtype=pl.Int16),
            "repurchase_score": pl.Series([1.0, None], dtype=pl.Float32),
            "product_code_rank": pl.Series([None, None], dtype=pl.Int16),
            "product_code_score": pl.Series([None, None], dtype=pl.Float32),
            "copurchase_rank": pl.Series([None, None], dtype=pl.Int16),
            "copurchase_score": pl.Series([None, None], dtype=pl.Float32),
            "popularity_last_week_rank": pl.Series([5, 1], dtype=pl.Int16),
            "popularity_last_week_score": pl.Series([100.0, 200.0], dtype=pl.Float32),
            "popularity_decayed_rank": pl.Series([None, None], dtype=pl.Int16),
            "popularity_decayed_score": pl.Series([None, None], dtype=pl.Float32),
            "segment_popular_rank": pl.Series([None, None], dtype=pl.Int16),
            "segment_popular_score": pl.Series([None, None], dtype=pl.Float32),
        })
        from src.config import MERGE_PRIORITY
        result = compute_candidate_features(merged, MERGE_PRIORITY)

        # Row 0 (art10): repurchase + popularity_last_week = 2 sources
        n0 = result.filter(pl.col("article_idx") == 10)["n_sources"][0]
        assert n0 == 2, f"Expected 2 sources for art10, got {n0}"

        # Row 1 (art20): only popularity_last_week = 1 source
        n1 = result.filter(pl.col("article_idx") == 20)["n_sources"][0]
        assert n1 == 1, f"Expected 1 source for art20, got {n1}"

        # in_repurchase for art10: 1; for art20: 0
        in_rep_10 = result.filter(pl.col("article_idx") == 10)["in_repurchase"][0]
        in_rep_20 = result.filter(pl.col("article_idx") == 20)["in_repurchase"][0]
        assert in_rep_10 == 1, f"Expected in_repurchase=1 for art10, got {in_rep_10}"
        assert in_rep_20 == 0, f"Expected in_repurchase=0 for art20, got {in_rep_20}"
