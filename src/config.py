"""Central constants for the H&M recommendation system."""
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
CANDIDATES_DIR = REPORTS_DIR / "candidates"
FIGURES_CANDIDATES_DIR = REPORTS_DIR / "figures" / "candidates"
CANDIDATES_PARQUET_DIR = PROCESSED_DIR / "candidates"

# ---------------------------------------------------------------------------
# Week indexing:  week_idx = (t_dat - ANCHOR_DATE).days // 7
# Anchor = Wed 2018-09-19; weeks run Wed-Tue.
# Week 0 has only 6 days of data (excluded from analysis).
# Week 104 = 2020-09-16 to 2020-09-22 = final test week (holdout).
# ---------------------------------------------------------------------------
ANCHOR_DATE = date(2018, 9, 19)
ANCHOR_EPOCH_DAYS: int = (ANCHOR_DATE - date(1970, 1, 1)).days  # 17793

HOLDOUT_WEEK: int = 104
VALIDATION_WEEKS: list[int] = [100, 101, 102, 103]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
MAP_K: int = 12
RECALL_NS: list[int] = [20, 50, 100, 200]

# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
SEED: int = 42

# Repurchase lookback windows (in weeks; None = all history)
REPURCHASE_WINDOWS: list[int | None] = [None, 12, 4]

# Product-code: recent purchase window (weeks) before cutoff
PRODUCT_CODE_LOOKBACK: int = 12

# Popularity time windows
POPULARITY_LAST_WEEK: int = 1       # global_last_week  (k=200; see reports/candidates/part_a_budget_experiment.json)
POPULARITY_DECAY_WEEKS: int = 4     # global_decayed window
POPULARITY_DECAY_HALFLIFE: float = 7.0  # exponential decay half-life (days; unused — inverse-time decay selected
SEGMENT_POPULAR_WEEKS: int = 2      # segment_popular window

# Copurchase
COPURCHASE_LOOKBACK: int = 8        # weeks for building co-purchase matrix
COPURCHASE_CUSTOMER_LOOKBACK: int = 4  # weeks of customer purchases to aggregate over
MIN_COPURCHASE_COUNT: int = 3       # minimum pair count; evaluated {3,5,10} on folds 100-103: 3 best (0.0248 vs 0.0233 vs 0.0184 recall@50)

# Age buckets for segment_popular (matches "age_bucket" column)
AGE_BUCKETS: list[str] = ["<25", "25-34", "35-44", "45-54", "55+", "missing"]

# Customer segments by prior purchase count
SEGMENT_BINS: list[int] = [0, 1, 5, 20]  # edges → 0, 1-4, 5-19, 20+
SEGMENT_LABELS: list[str] = ["0", "1-4", "5-19", "20+"]

# Per-source candidate limits
SOURCE_K: dict[str, int] = {
    "repurchase": 50,
    "product_code": 50,
    "copurchase": 50,
    "popularity_last_week": 200,
    "popularity_decayed": 100,
    "segment_popular": 50,
}

# Final merge budget per customer
MERGE_N: int = 200

# Merge allocation priority (order determines fill order)
MERGE_PRIORITY: list[str] = [
    "repurchase",
    "product_code",
    "copurchase",
    "popularity_last_week",
    "popularity_decayed",
    "segment_popular",
]
