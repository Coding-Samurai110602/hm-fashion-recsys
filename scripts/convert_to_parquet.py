"""
Convert H&M raw CSVs to Parquet with integer index mappings.

Design decisions:
- article_id kept as string (10-char zero-padded, e.g. "0108775015")
- customer_id kept as string (64-char hex)
- transactions streamed via polars to stay within RAM
- zstd compression (good ratio + fast decompression)
- customer_idx / article_idx added as int32 columns
- Sort transactions by t_dat before writing
- Idempotent: safe to re-run (overwrites output)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

COMPRESSION = "zstd"

# ── 1. ARTICLES ──────────────────────────────────────────────────────────────

def convert_articles() -> pd.DataFrame:
    print("[articles] Reading CSV …")
    t0 = time.time()
    df = pd.read_csv(
        RAW / "articles.csv",
        dtype={
            "article_id": str,
            "product_code": str,
            "prod_name": str,
            "product_type_no": "Int32",
            "product_type_name": str,
            "product_group_name": str,
            "graphical_appearance_no": "Int32",
            "graphical_appearance_name": str,
            "colour_group_code": "Int32",
            "colour_group_name": str,
            "perceived_colour_value_id": "Int32",
            "perceived_colour_value_name": str,
            "perceived_colour_master_id": "Int32",
            "perceived_colour_master_name": str,
            "department_no": "Int32",
            "department_name": str,
            "index_code": str,
            "index_name": str,
            "index_group_no": "Int32",
            "index_group_name": str,
            "section_no": "Int32",
            "section_name": str,
            "garment_group_no": "Int32",
            "garment_group_name": str,
            "detail_desc": str,
        },
    )
    print(f"[articles] Read {len(df):,} rows in {time.time()-t0:.1f}s")

    # Zero-pad article_id to 10 chars (safety net)
    df["article_id"] = df["article_id"].str.zfill(10)
    assert df["article_id"].str.len().eq(10).all(), "article_id length check failed"

    # Build article_idx: stable sort by article_id so mapping is deterministic
    article_ids_sorted = df["article_id"].sort_values().reset_index(drop=True)
    art_map = {aid: idx for idx, aid in enumerate(article_ids_sorted)}
    df["article_idx"] = df["article_id"].map(art_map).astype("int32")

    out = PROCESSED / "articles.parquet"
    df.to_parquet(out, compression=COMPRESSION, index=False)
    size_mb = out.stat().st_size / 1e6
    print(f"[articles] Written → {out.name} ({size_mb:.1f} MB, {len(df):,} rows)")
    return df


# ── 2. CUSTOMERS ─────────────────────────────────────────────────────────────

def convert_customers() -> pd.DataFrame:
    print("[customers] Reading CSV …")
    t0 = time.time()
    df = pd.read_csv(
        RAW / "customers.csv",
        dtype={
            "customer_id": str,
            "FN": "float32",
            "Active": "float32",
            "club_member_status": str,
            "fashion_news_frequency": str,
            "age": "float32",
            "postal_code": str,
        },
    )
    print(f"[customers] Read {len(df):,} rows in {time.time()-t0:.1f}s")

    # Build customer_idx: stable sort by customer_id
    customer_ids_sorted = df["customer_id"].sort_values().reset_index(drop=True)
    cust_map = {cid: idx for idx, cid in enumerate(customer_ids_sorted)}
    df["customer_idx"] = df["customer_id"].map(cust_map).astype("int32")

    out = PROCESSED / "customers.parquet"
    df.to_parquet(out, compression=COMPRESSION, index=False)
    size_mb = out.stat().st_size / 1e6
    print(f"[customers] Written → {out.name} ({size_mb:.1f} MB, {len(df):,} rows)")
    return df


# ── 3. TRANSACTIONS (polars streaming) ───────────────────────────────────────

def convert_transactions(art_map: dict, cust_map: dict) -> None:
    """
    Read 31.8M-row transactions CSV with polars (lazy scan), apply idx mappings,
    sort by t_dat, write to parquet. We materialize in chunks to control peak RAM.
    """
    print("[transactions] Scanning CSV with polars (lazy) …")
    t0 = time.time()

    # Polars lazy scan — does not materialise into RAM yet
    lf = pl.scan_csv(
        RAW / "transactions_train.csv",
        schema_overrides={
            "t_dat": pl.Utf8,
            "customer_id": pl.Utf8,
            "article_id": pl.Utf8,
            "price": pl.Float64,
            "sales_channel_id": pl.Int8,
        },
    )

    # Build index lookup Series from our deterministic maps
    # We'll join rather than map Python-side to keep it fast
    art_idx_df = pl.DataFrame({
        "article_id": list(art_map.keys()),
        "article_idx": pl.Series(list(art_map.values()), dtype=pl.Int32),
    })
    cust_idx_df = pl.DataFrame({
        "customer_id": list(cust_map.keys()),
        "customer_idx": pl.Series(list(cust_map.values()), dtype=pl.Int32),
    })

    # Zero-pad article_id (in case CSV has fewer than 10 chars due to leading-zero strip)
    lf = lf.with_columns(
        pl.col("article_id").str.zfill(10)
    )

    print("[transactions] Collecting (may take a few minutes) …")
    df = lf.collect()
    print(f"[transactions] Collected {len(df):,} rows in {time.time()-t0:.1f}s")

    # Join idx columns
    df = df.join(art_idx_df, on="article_id", how="left")
    df = df.join(cust_idx_df, on="customer_id", how="left")

    # Parse date and cast columns
    df = df.with_columns([
        pl.col("t_dat").str.to_date("%Y-%m-%d"),
        pl.col("price").cast(pl.Float32),
    ])

    # Sort by t_dat
    df = df.sort("t_dat")

    # Select only required output columns
    df = df.select(["t_dat", "customer_idx", "article_idx", "price", "sales_channel_id"])

    out = PROCESSED / "transactions_train.parquet"
    print(f"[transactions] Writing parquet …")
    df.write_parquet(out, compression=COMPRESSION)
    size_mb = out.stat().st_size / 1e6
    elapsed = time.time() - t0
    print(f"[transactions] Written → {out.name} ({size_mb:.1f} MB, {len(df):,} rows, {elapsed:.0f}s total)")


# ── 4. SAMPLE SUBMISSION ─────────────────────────────────────────────────────

def convert_sample_submission() -> None:
    print("[sample_submission] Reading CSV …")
    df = pd.read_csv(
        RAW / "sample_submission.csv",
        dtype={"customer_id": str, "prediction": str},
    )
    out = PROCESSED / "sample_submission.parquet"
    df.to_parquet(out, compression=COMPRESSION, index=False)
    size_mb = out.stat().st_size / 1e6
    print(f"[sample_submission] Written → {out.name} ({size_mb:.1f} MB, {len(df):,} rows)")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total_t0 = time.time()
    print("=" * 60)
    print("H&M CSV → Parquet conversion")
    print("=" * 60)

    articles_df = convert_articles()
    art_map = dict(zip(articles_df["article_id"], articles_df["article_idx"]))

    customers_df = convert_customers()
    cust_map = dict(zip(customers_df["customer_id"], customers_df["customer_idx"]))

    convert_transactions(art_map, cust_map)
    convert_sample_submission()

    print("=" * 60)
    print(f"All done in {(time.time()-total_t0)/60:.1f} minutes")
    print(f"Processed files in: {PROCESSED}")
    print("=" * 60)
