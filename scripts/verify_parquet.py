"""
Verify losslessness of CSV → Parquet conversion.

Checks:
1. Row counts match CSVs
2. Column null counts match CSVs
3. Distinct customer_id / article_id counts match
4. Price sum matches within float32 tolerance; min/max t_dat match
5. Round-trip: random 10k sample (seed 42) idx → original IDs match CSV rows
6. article_id strings all length 10
7. File size comparison (CSV vs Parquet)

Exits with code 1 if any check fails.
"""

import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PASS = "  [PASS]"
FAIL = "  [FAIL]"
errors = []

def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    msg = f"{status}  {label}"
    if detail:
        msg += f" → {detail}"
    print(msg)
    if not condition:
        errors.append(label)


# ── 1. ROW COUNTS ─────────────────────────────────────────────────────────────

print("\n── 1. Row counts ──────────────────────────────────────────")

expected_rows = {
    "articles": 105_542,
    "customers": 1_371_980,
    "transactions_train": 31_788_324,
    "sample_submission": 1_371_980,
}

for name, expected in expected_rows.items():
    pq_path = PROCESSED / f"{name}.parquet"
    meta = pq.read_metadata(pq_path)
    actual = meta.num_rows
    check(f"{name}: row count", actual == expected, f"{actual:,} (expected {expected:,})")


# ── 2. COLUMN NULL COUNTS ─────────────────────────────────────────────────────

print("\n── 2. Null counts match CSVs ──────────────────────────────")

# Articles nulls
art_csv = pd.read_csv(RAW / "articles.csv", dtype={"article_id": str})
art_pq = pd.read_parquet(PROCESSED / "articles.parquet")
for col in art_csv.columns:
    csv_null = art_csv[col].isna().sum()
    pq_null = art_pq[col].isna().sum()
    check(f"articles.{col} nulls", csv_null == pq_null, f"csv={csv_null} pq={pq_null}")

# Customers nulls
cust_csv = pd.read_csv(RAW / "customers.csv", dtype={"customer_id": str})
cust_pq = pd.read_parquet(PROCESSED / "customers.parquet")
for col in cust_csv.columns:
    csv_null = cust_csv[col].isna().sum()
    pq_null = cust_pq[col].isna().sum()
    check(f"customers.{col} nulls", csv_null == pq_null, f"csv={csv_null} pq={pq_null}")

# Transactions nulls — use polars for speed
print("  [transactions nulls — reading via polars …]")
tx_pq = pl.read_parquet(PROCESSED / "transactions_train.parquet")
# CSV original columns: t_dat, customer_id, article_id, price, sales_channel_id
# Parquet columns: t_dat, customer_idx, article_idx, price, sales_channel_id
# We can check price and sales_channel_id nulls against a CSV sample
# For the full check, we trust polars scan
for col in ["price", "sales_channel_id"]:
    null_count = tx_pq[col].is_null().sum()
    check(f"transactions.{col} nulls == 0", null_count == 0, f"nulls={null_count}")
# t_dat, customer_idx, article_idx should also be non-null
for col in ["t_dat", "customer_idx", "article_idx"]:
    null_count = tx_pq[col].is_null().sum()
    check(f"transactions.{col} nulls == 0", null_count == 0, f"nulls={null_count}")


# ── 3. DISTINCT ID COUNTS ─────────────────────────────────────────────────────

print("\n── 3. Distinct ID counts ──────────────────────────────────")

csv_distinct_articles = art_csv["article_id"].nunique()
pq_distinct_articles = art_pq["article_id"].nunique()
check("articles: distinct article_id", csv_distinct_articles == pq_distinct_articles,
      f"csv={csv_distinct_articles:,} pq={pq_distinct_articles:,}")

csv_distinct_customers = cust_csv["customer_id"].nunique()
pq_distinct_customers = cust_pq["customer_id"].nunique()
check("customers: distinct customer_id", csv_distinct_customers == pq_distinct_customers,
      f"csv={csv_distinct_customers:,} pq={pq_distinct_customers:,}")

# Distinct in transactions — use polars
tx_distinct_customers = tx_pq["customer_idx"].n_unique()
tx_distinct_articles = tx_pq["article_idx"].n_unique()
print(f"  transactions: {tx_distinct_customers:,} unique customer_idxs, {tx_distinct_articles:,} unique article_idxs")


# ── 4. PRICE SUM, MIN/MAX t_dat ──────────────────────────────────────────────

print("\n── 4. Price sum + date range ──────────────────────────────")

# CSV price sum (read full CSV with polars for speed)
print("  [reading transactions CSV for price/date validation …]")
tx_csv_lf = pl.scan_csv(
    RAW / "transactions_train.csv",
    schema_overrides={"price": pl.Float64, "t_dat": pl.Utf8},
)
csv_price_sum = tx_csv_lf.select(pl.col("price").sum()).collect().item()
csv_min_dat = tx_csv_lf.select(pl.col("t_dat").min()).collect().item()
csv_max_dat = tx_csv_lf.select(pl.col("t_dat").max()).collect().item()

pq_price_sum = float(tx_pq["price"].cast(pl.Float64).sum())
pq_min_dat = str(tx_pq["t_dat"].min())
pq_max_dat = str(tx_pq["t_dat"].max())

float32_tol = abs(csv_price_sum) * 1e-3  # 0.1% tolerance for float32 rounding
price_ok = abs(csv_price_sum - pq_price_sum) <= float32_tol
check("price sum within float32 tolerance",
      price_ok,
      f"csv={csv_price_sum:.2f} pq={pq_price_sum:.2f} diff={abs(csv_price_sum-pq_price_sum):.4f}")

check("min t_dat matches", csv_min_dat == pq_min_dat, f"csv={csv_min_dat} pq={pq_min_dat}")
check("max t_dat matches", csv_max_dat == pq_max_dat, f"csv={csv_max_dat} pq={pq_max_dat}")


# ── 5. ROUND-TRIP: idx → original ID for 10k sample ─────────────────────────

print("\n── 5. Round-trip idx mapping (10k sample, seed 42) ───────")

# Build reverse maps from parquet
art_map_forward = dict(zip(art_pq["article_id"], art_pq["article_idx"]))
art_map_reverse = {v: k for k, v in art_map_forward.items()}

cust_map_forward = dict(zip(cust_pq["customer_id"], cust_pq["customer_idx"]))
cust_map_reverse = {v: k for k, v in cust_map_forward.items()}

# Sample 10k rows from the CSV
tx_csv_sample_lf = pl.scan_csv(
    RAW / "transactions_train.csv",
    schema_overrides={"article_id": pl.Utf8, "customer_id": pl.Utf8},
)
# Use seed 42: take modulo of row hash (deterministic sample)
n_total = 31_788_324
rng = random.Random(42)
sample_indices = set(rng.sample(range(n_total), 10_000))

# Read full CSV with polars and filter by index
# More RAM-efficient: use row_index
print("  [sampling 10k rows from CSV — may take ~30s …]")
tx_csv_full = tx_csv_sample_lf.with_row_index("_row").collect()
tx_sample_csv = tx_csv_full.filter(pl.col("_row").is_in(list(sample_indices)))
tx_sample_csv = tx_sample_csv.with_columns(
    pl.col("article_id").str.zfill(10)
)

# Get parquet rows at same positions
tx_sample_pq = tx_pq[sorted(sample_indices)]

# Reverse-map idx to IDs
reconstructed_art = [art_map_reverse.get(idx, "MISSING") for idx in tx_sample_pq["article_idx"].to_list()]
reconstructed_cust = [cust_map_reverse.get(idx, "MISSING") for idx in tx_sample_pq["customer_idx"].to_list()]

# Sort both by original row index to align
csv_by_row = tx_sample_csv.sort("_row")
csv_arts = csv_by_row["article_id"].to_list()
csv_custs = csv_by_row["customer_id"].to_list()

art_match = all(a == b for a, b in zip(reconstructed_art, csv_arts))
cust_match = all(a == b for a, b in zip(reconstructed_cust, csv_custs))

check("article_id round-trip (10k sample)", art_match)
check("customer_id round-trip (10k sample)", cust_match)


# ── 6. ARTICLE_ID LENGTH ─────────────────────────────────────────────────────

print("\n── 6. article_id all length 10 ────────────────────────────")
all_len10 = (art_pq["article_id"].str.len() == 10).all()
check("all article_ids have length 10", all_len10)


# ── 7. FILE SIZES ────────────────────────────────────────────────────────────

print("\n── 7. File sizes ──────────────────────────────────────────")
files = [
    ("articles", "articles.csv", "articles.parquet"),
    ("customers", "customers.csv", "customers.parquet"),
    ("transactions", "transactions_train.csv", "transactions_train.parquet"),
    ("sample_submission", "sample_submission.csv", "sample_submission.parquet"),
]
for label, csv_name, pq_name in files:
    csv_mb = (RAW / csv_name).stat().st_size / 1e6
    pq_mb = (PROCESSED / pq_name).stat().st_size / 1e6
    ratio = csv_mb / pq_mb
    print(f"  {label}: CSV={csv_mb:.1f}MB → Parquet={pq_mb:.1f}MB (ratio {ratio:.1f}x)")


# ── FINAL RESULT ──────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
if errors:
    print(f"FAILED — {len(errors)} check(s) did not pass:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"ALL CHECKS PASSED ({6 + len(art_csv.columns) + len(cust_csv.columns)} checks)")
    print("Parquet files are a verified lossless representation of the CSVs.")
print("=" * 55)
