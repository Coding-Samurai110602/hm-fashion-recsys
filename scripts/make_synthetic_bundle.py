"""Create a small fake bundle for CI testing.

Produces a bundle with the same file layout, schemas, and manifest as the real
artifacts/bundle_week104/ but using 300 synthetic articles and 200 synthetic
customers. Trains a tiny LightGBM model on synthetic features. No H&M data.

Usage:
    python scripts/make_synthetic_bundle.py [--out-dir <path>]

Default output: artifacts/bundle_synthetic/
Deterministic: seed 42.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb

# ─── Constants ─────────────────────────────────────────────────────────────────
SEED = 42
N_ARTICLES = 300
N_CUSTOMERS = 200
AS_OF_WEEK = 104
ANCHOR_DATE = date(2018, 9, 19)

AGE_BUCKETS = ["<25", "25-34", "35-44", "45-54", "55+", "missing"]
PRODUCT_TYPES = ["Trousers", "Sweater", "Blouse", "Dress", "Jacket", "Shoes", "Bag"]
GARMENT_GROUPS = ["Accessories", "Garment Upper body", "Garment Lower body",
                  "Garment Full body", "Shoes"]
PRODUCT_GROUPS = ["Ladieswear", "Menswear", "Divided", "Sport"]
SECTIONS = ["Womens Everyday Basics", "Mens Basics", "H&M+", "Sport"]
COLOURS = ["Black", "White", "Blue", "Grey", "Red", "Green", "Pink"]
DEPARTMENTS = ["Jeans", "Jersey", "Knitwear", "Outerwear", "Swimwear"]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _week_to_date(week_idx: int) -> date:
    return ANCHOR_DATE + timedelta(weeks=week_idx)


def make_articles(rng: np.random.Generator) -> pl.DataFrame:
    article_idxs = list(range(N_ARTICLES))
    article_ids = [f"{10_000_000 + i:010d}" for i in article_idxs]
    product_codes = [100_000 + i // 3 for i in article_idxs]
    return pl.DataFrame({
        "article_idx": pl.Series(article_idxs, dtype=pl.Int32),
        "article_id": article_ids,
        "product_code": pl.Series(product_codes, dtype=pl.Int32),
        "prod_name": [f"Article {i}" for i in article_idxs],
        "product_type_no": pl.Series(rng.integers(1, 50, N_ARTICLES), dtype=pl.Int16),
        "product_type_name": [PRODUCT_TYPES[i % len(PRODUCT_TYPES)] for i in article_idxs],
        "product_group_name": [PRODUCT_GROUPS[i % len(PRODUCT_GROUPS)] for i in article_idxs],
        "graphical_appearance_no": pl.Series(rng.integers(1, 10, N_ARTICLES), dtype=pl.Int16),
        "graphical_appearance_name": ["Solid" for _ in article_idxs],
        "colour_group_code": pl.Series(rng.integers(1, 10, N_ARTICLES), dtype=pl.Int16),
        "colour_group_name": [COLOURS[i % len(COLOURS)] for i in article_idxs],
        "perceived_colour_value_id": pl.Series(rng.integers(1, 5, N_ARTICLES), dtype=pl.Int8),
        "perceived_colour_value_name": ["Medium" for _ in article_idxs],
        "perceived_colour_master_id": pl.Series(rng.integers(1, 5, N_ARTICLES), dtype=pl.Int8),
        "perceived_colour_master_name": [COLOURS[i % len(COLOURS)] for i in article_idxs],
        "department_no": pl.Series(rng.integers(1, 20, N_ARTICLES), dtype=pl.Int16),
        "department_name": [DEPARTMENTS[i % len(DEPARTMENTS)] for i in article_idxs],
        "index_code": ["A" for _ in article_idxs],
        "index_name": ["Ladieswear" for _ in article_idxs],
        "index_group_no": pl.Series(rng.integers(1, 5, N_ARTICLES), dtype=pl.Int8),
        "index_group_name": ["Ladieswear" for _ in article_idxs],
        "section_no": pl.Series(rng.integers(1, 10, N_ARTICLES), dtype=pl.Int16),
        "section_name": [SECTIONS[i % len(SECTIONS)] for i in article_idxs],
        "garment_group_no": pl.Series(rng.integers(1, 10, N_ARTICLES), dtype=pl.Int16),
        "garment_group_name": [GARMENT_GROUPS[i % len(GARMENT_GROUPS)] for i in article_idxs],
    })


def make_customers(rng: np.random.Generator) -> pl.DataFrame:
    customer_idxs = list(range(N_CUSTOMERS))
    # Hex customer_ids for lookup in the API
    rng_py = random.Random(SEED)
    customer_ids = [
        "".join([f"{rng_py.randint(0,255):02x}" for _ in range(32)])
        for _ in customer_idxs
    ]
    ages = rng.integers(18, 70, N_CUSTOMERS).tolist()
    return pl.DataFrame({
        "customer_idx": pl.Series(customer_idxs, dtype=pl.Int32),
        "customer_id": customer_ids,
        "age": pl.Series(ages, dtype=pl.Float32),
        "FN": pl.Series([1.0] * N_CUSTOMERS, dtype=pl.Float32),
        "Active": pl.Series([1.0] * N_CUSTOMERS, dtype=pl.Float32),
        "club_member_status": pl.Series(["ACTIVE"] * N_CUSTOMERS),
        "fashion_news_frequency": pl.Series(["Regularly"] * N_CUSTOMERS),
    })


def make_history(articles_df: pl.DataFrame, customers_df: pl.DataFrame,
                 rng: np.random.Generator) -> pl.DataFrame:
    rows = []
    article_idxs = articles_df["article_idx"].to_list()
    cutoff_date = _week_to_date(AS_OF_WEEK)
    for cidx in customers_df["customer_idx"].to_list():
        n_purchases = int(rng.integers(0, 30))
        for _ in range(n_purchases):
            art_idx = int(rng.choice(article_idxs))
            days_ago = int(rng.integers(1, 730))
            t_dat = cutoff_date - timedelta(days=days_ago)
            week_idx = int((t_dat - ANCHOR_DATE).days // 7)
            if week_idx < 0 or week_idx >= AS_OF_WEEK:
                continue
            rows.append({
                "customer_idx": cidx,
                "article_idx": art_idx,
                "t_dat": t_dat,
                "price": float(rng.uniform(0.01, 0.6)),
                "sales_channel_id": int(rng.choice([1, 2])),
                "week_idx": week_idx,
            })
    return pl.DataFrame(
        rows,
        schema={
            "customer_idx": pl.Int32,
            "article_idx": pl.Int32,
            "t_dat": pl.Date,
            "price": pl.Float32,
            "sales_channel_id": pl.Int8,
            "week_idx": pl.Int16,
        },
    )


def make_gt_week104(articles_df: pl.DataFrame, customers_df: pl.DataFrame,
                    rng: np.random.Generator) -> pl.DataFrame:
    rows = []
    article_idxs = articles_df["article_idx"].to_list()
    for cidx in customers_df["customer_idx"].to_list():
        if rng.random() < 0.5:
            n_buys = int(rng.integers(1, 5))
            for art_idx in rng.choice(article_idxs, n_buys, replace=False):
                rows.append({"customer_idx": int(cidx), "article_idx": int(art_idx)})
    return pl.DataFrame(
        rows if rows else [{"customer_idx": 0, "article_idx": 0}],
        schema={"customer_idx": pl.Int32, "article_idx": pl.Int32},
    )


def make_popularity(articles_df: pl.DataFrame, rng: np.random.Generator):
    n = len(articles_df)
    article_idxs = articles_df["article_idx"].to_list()
    scores_lw = rng.exponential(1.0, n).astype(np.float32)
    pop_lw = (
        pl.DataFrame({
            "article_idx": pl.Series(article_idxs, dtype=pl.Int32),
            "score": pl.Series(scores_lw, dtype=pl.Float32),
        })
        .sort("score", descending=True)
        .with_columns(pl.Series("source_rank", list(range(1, n + 1)), dtype=pl.Int16))
    )

    scores_dec = rng.exponential(1.0, n).astype(np.float32)
    pop_dec = (
        pl.DataFrame({
            "article_idx": pl.Series(article_idxs, dtype=pl.Int32),
            "score": pl.Series(scores_dec, dtype=pl.Float32),
        })
        .sort("score", descending=True)
        .with_columns(pl.Series("source_rank", list(range(1, n + 1)), dtype=pl.Int16))
    )

    prev_sales = pl.DataFrame({
        "article_idx": pl.Series(article_idxs, dtype=pl.Int32),
        "sales_count": pl.Series(rng.integers(0, 100, n), dtype=pl.Int32),
    })
    return pop_lw, pop_dec, prev_sales


def make_seg_pop(articles_df: pl.DataFrame, rng: np.random.Generator
                 ) -> dict[str, pl.DataFrame]:
    article_idxs = articles_df["article_idx"].to_list()
    n = len(article_idxs)
    result = {}
    for bucket in AGE_BUCKETS:
        scores = rng.exponential(1.0, n).astype(np.float32)
        result[bucket] = (
            pl.DataFrame({
                "article_idx": pl.Series(article_idxs, dtype=pl.Int32),
                "score": pl.Series(scores, dtype=pl.Float32),
            })
            .sort("score", descending=True)
            .with_columns(pl.Series("source_rank", list(range(1, n + 1)), dtype=pl.Int16))
        )
    return result


def make_copurchase(articles_df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    article_idxs = articles_df["article_idx"].to_list()
    n_pairs = min(500, N_ARTICLES * (N_ARTICLES - 1) // 2)
    pairs_a, pairs_b, scores = [], [], []
    used = set()
    while len(pairs_a) < n_pairs:
        i, j = rng.choice(N_ARTICLES, 2, replace=False)
        key = (min(i, j), max(i, j))
        if key in used:
            continue
        used.add(key)
        pairs_a.append(article_idxs[i])
        pairs_b.append(article_idxs[j])
        scores.append(float(rng.uniform(0, 1)))
    return pl.DataFrame({
        "seed_article": pl.Series(pairs_a, dtype=pl.Int32),
        "candidate_article": pl.Series(pairs_b, dtype=pl.Int32),
        "cosine_score": pl.Series(scores, dtype=pl.Float64),
    })


def make_age_buckets(customers_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    def bucket(age):
        if age < 25:
            return "<25"
        if age < 35:
            return "25-34"
        if age < 45:
            return "35-44"
        if age < 55:
            return "45-54"
        return "55+"
    ages = customers_df["age"].to_list()
    buckets = [bucket(a) for a in ages]
    ab = pl.DataFrame({
        "customer_idx": customers_df["customer_idx"],
        "age_bucket": pl.Series(buckets),
    })
    # bucket_totals
    from collections import Counter
    counts = Counter(buckets)
    all_buckets = AGE_BUCKETS
    bt = pl.DataFrame({
        "age_bucket": all_buckets,
        "_bucket_total": pl.Series([float(counts.get(b, 1)) for b in all_buckets], dtype=pl.Float32),
    })
    return ab, bt


def make_article_product_codes(articles_df: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({
        "article_idx": articles_df["article_idx"],
        "product_code": articles_df["product_code"],
    })


def make_article_meta(articles_df: pl.DataFrame) -> pl.DataFrame:
    return articles_df.select([
        "article_idx", "article_id", "prod_name",
        "product_type_name", "colour_group_name", "department_name",
    ])


def make_article_feats(articles_df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Synthetic article feature table matching compute_article_features() output schema.

    The bundle stores the output of compute_article_features (22 cols),
    NOT the 68 model input features. The model features are assembled on-the-fly
    by joining candidates + customer_feats + article_feats + interaction_feats.
    """
    n = len(articles_df)
    article_idxs = articles_df["article_idx"].to_list()
    return pl.DataFrame({
        "article_idx": pl.Series(article_idxs, dtype=pl.Int32),
        "a_sales_1w": pl.Series(rng.integers(0, 50, n), dtype=pl.Int32),
        "a_sales_2w": pl.Series(rng.integers(0, 100, n), dtype=pl.Int32),
        "a_sales_4w": pl.Series(rng.integers(0, 200, n), dtype=pl.Int32),
        "a_unique_buyers_1w": pl.Series(rng.integers(0, 30, n), dtype=pl.Int32),
        "a_trend_ratio": pl.Series(rng.uniform(0, 3, n).astype(np.float32), dtype=pl.Float32),
        "a_days_since_first_sale": pl.Series(rng.uniform(0, 700, n).astype(np.float32), dtype=pl.Float32),
        "a_censored_flag": pl.Series(rng.integers(0, 2, n), dtype=pl.Int8),
        "a_days_since_last_sale": pl.Series(rng.uniform(0, 100, n).astype(np.float32), dtype=pl.Float32),
        "a_mean_price_4w": pl.Series(rng.uniform(0.01, 0.6, n).astype(np.float32), dtype=pl.Float32),
        "a_price_vs_own_history": pl.Series(rng.uniform(0.5, 2.0, n).astype(np.float32), dtype=pl.Float32),
        "a_repurchase_rate": pl.Series(rng.uniform(0, 0.3, n).astype(np.float32), dtype=pl.Float32),
        "a_mean_buyer_age_12w": pl.Series(rng.uniform(20, 60, n).astype(np.float32), dtype=pl.Float32),
        "a_std_buyer_age_12w": pl.Series(rng.uniform(5, 20, n).astype(np.float32), dtype=pl.Float32),
        "a_product_type_no": pl.Series(rng.integers(1, 50, n), dtype=pl.Int16),
        "a_product_group_name": pl.Series(rng.integers(0, 4, n), dtype=pl.Int8),
        "a_index_group_name": pl.Series(rng.integers(0, 4, n), dtype=pl.Int8),
        "a_garment_group_name": pl.Series(rng.integers(0, 5, n), dtype=pl.Int8),
        "a_colour_group_name": pl.Series(rng.integers(0, 7, n), dtype=pl.Int8),
        "a_section_name": pl.Series(rng.integers(0, 4, n), dtype=pl.Int8),
        "a_department_no": pl.Series(rng.integers(1, 20, n), dtype=pl.Int16),
        "_article_online_share_4w": pl.Series(rng.uniform(0, 1, n).astype(np.float32), dtype=pl.Float32),
    })


def train_model(article_feats: pl.DataFrame, history_df: pl.DataFrame,
                rng: np.random.Generator) -> lgb.Booster:
    """Train a tiny LightGBM LambdaRank model on synthetic data."""
    from src.model.data import FEATURE_NAMES

    n_rows = min(5_000, N_CUSTOMERS * 50)
    labels = rng.integers(0, 2, n_rows).astype(np.float32)

    X = rng.random((n_rows, len(FEATURE_NAMES))).astype(np.float32)
    groups = []
    i = 0
    while i < n_rows:
        chunk = min(int(rng.integers(5, 30)), n_rows - i)
        groups.append(chunk)
        i += chunk

    ds = lgb.Dataset(X, label=labels, group=groups, free_raw_data=False)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [12],
        "num_leaves": 4,
        "learning_rate": 0.1,
        "n_estimators": 10,
        "verbose": -1,
        "seed": SEED,
        "deterministic": True,
        "force_row_wise": True,
    }
    booster = lgb.train(params, ds, num_boost_round=10, valid_sets=[ds],
                        callbacks=[lgb.log_evaluation(-1)])
    return booster


def _make_ab_arrays(customers_df: pl.DataFrame, gt_df: pl.DataFrame,
                    rng: np.random.Generator) -> pl.DataFrame:
    """Synthetic per-customer hit@12/AP@12 arrays (ranker + heuristic)."""
    customer_idxs = customers_df["customer_idx"].to_list()
    gt_set = set(zip(gt_df["customer_idx"].to_list(), gt_df["article_idx"].to_list()))
    n = len(customer_idxs)
    # Synthetic: ranker outperforms heuristic
    ranker_ap12 = rng.beta(2, 10, n).astype(np.float32)
    heuristic_ap12 = (ranker_ap12 * rng.uniform(0.3, 0.8, n)).astype(np.float32)
    ranker_hit12 = (ranker_ap12 > 0.01).astype(np.int8)
    heuristic_hit12 = (heuristic_ap12 > 0.01).astype(np.int8)
    return pl.DataFrame({
        "customer_idx": pl.Series(customer_idxs, dtype=pl.Int32),
        "ranker_ap12": pl.Series(ranker_ap12, dtype=pl.Float32),
        "ranker_hit12": pl.Series(ranker_hit12, dtype=pl.Int8),
        "heuristic_ap12": pl.Series(heuristic_ap12, dtype=pl.Float32),
        "heuristic_hit12": pl.Series(heuristic_hit12, dtype=pl.Int8),
    })


def _write_synthetic_reports(out_dir: Path) -> None:
    """Write minimal synthetic report JSONs so Docker insights endpoints return data."""
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    placeholder = {"synthetic": True, "note": "Synthetic bundle — values are illustrative only."}

    synthetic_reports = {
        "holdout_results": {
            **placeholder,
            "holdout_week": 104,
            "n_eval_customers": N_CUSTOMERS,
            "primary_lift_vs_heuristic_pct": 45.0,
            "primary_lift_vs_baseline_b_pct": 48.0,
            "models": {
                "primary": {"map@12": 0.036},
                "heuristic": {"map@12": 0.025},
                "baseline_b": {"map@12": 0.024},
            },
        },
        "ab_test": {**placeholder, "aa_test": {"fpr_estimate": 0.05},
                    "peeking": {"peeking_false_positive_rate": 0.22},
                    "simulation": {"hit_rate_z_test": {"z_statistic": 24.0}}},
        "metric_suite": {**placeholder, "ranker": {"hit_rate@12": 0.147}, "heuristic": {"hit_rate@12": 0.086}},
        "segment_analysis": {**placeholder, "segments": []},
        "rolling_origin": {**placeholder, "folds": []},
        "eval_fold103": {**placeholder, "ranker": {"map@12": 0.03583}, "heuristic": {"map@12": 0.02468}},
        "ablations": {**placeholder, "ablations": [], "gain_importance_top25": []},
        "shap_summary": {**placeholder, "top10_gain_importance": [], "top10_shap_all_candidates": [],
                         "top10_contrast_shap_all_candidates": [], "group_shap_all_candidates": {},
                         "group_contrast_shap_all_candidates": {}},
    }
    for name, data in synthetic_reports.items():
        (reports_dir / f"{name}.json").write_text(json.dumps(data, indent=2))


def build_bundle(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir = out_dir / "state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "seg_pop").mkdir(exist_ok=True)

    rng = np.random.default_rng(SEED)

    print("[synth] Building synthetic data ...", flush=True)
    articles_df = make_articles(rng)
    customers_df = make_customers(rng)
    history_df = make_history(articles_df, customers_df, rng)
    gt_df = make_gt_week104(articles_df, customers_df, rng)
    pop_lw, pop_dec, prev_sales = make_popularity(articles_df, rng)
    seg_pop = make_seg_pop(articles_df, rng)
    copurchase = make_copurchase(articles_df, rng)
    age_buckets, bucket_totals = make_age_buckets(customers_df)
    article_product_codes = make_article_product_codes(articles_df)
    article_meta = make_article_meta(articles_df)
    article_feats = make_article_feats(articles_df, rng)

    print("[synth] Training tiny LightGBM model ...", flush=True)
    booster = train_model(article_feats, history_df, rng)

    print("[synth] Writing parquet files ...", flush=True)

    # Per-customer A/B arrays (synthetic)
    ab_arrays = _make_ab_arrays(customers_df, gt_df, rng)

    # Model
    model_path = out_dir / "model.txt"
    booster.save_model(str(model_path))

    # State parquets
    articles_df.write_parquet(state_dir / "articles_raw.parquet", compression="zstd")
    customers_df.write_parquet(state_dir / "customers.parquet", compression="zstd")
    history_df.write_parquet(state_dir / "customer_history.parquet", compression="zstd")
    gt_df.write_parquet(state_dir / "gt_week104.parquet", compression="zstd")
    pop_lw.write_parquet(state_dir / "pop_last_week.parquet", compression="zstd")
    pop_dec.write_parquet(state_dir / "pop_decayed.parquet", compression="zstd")
    prev_sales.write_parquet(state_dir / "prev_week_sales.parquet", compression="zstd")
    copurchase.write_parquet(state_dir / "copurchase.parquet", compression="zstd")
    age_buckets.write_parquet(state_dir / "age_buckets.parquet", compression="zstd")
    bucket_totals.write_parquet(state_dir / "bucket_totals.parquet", compression="zstd")
    article_product_codes.write_parquet(state_dir / "article_product_codes.parquet", compression="zstd")
    article_meta.write_parquet(state_dir / "article_meta.parquet", compression="zstd")
    article_feats.write_parquet(state_dir / "article_feats.parquet", compression="zstd")
    ab_arrays.write_parquet(state_dir / "ab_arrays.parquet", compression="zstd")

    # Snapshot synthetic report JSONs for Docker self-containedness
    _write_synthetic_reports(out_dir)

    # Seg pop files (safe bucket names)
    bucket_safe = {
        "<25": "lt25", "25-34": "25-34", "35-44": "35-44",
        "45-54": "45-54", "55+": "55plus", "missing": "missing",
    }
    for bucket, df in seg_pop.items():
        safe = bucket_safe.get(bucket, bucket)
        df.write_parquet(state_dir / "seg_pop" / f"{safe}.parquet", compression="zstd")

    # Config
    from src.model.data import FEATURE_NAMES
    config = {
        "as_of_week": AS_OF_WEEK,
        "feature_names": FEATURE_NAMES,
        "feature_dtypes": {f: "Float32" for f in FEATURE_NAMES},
        "feature_groups": {"candidate": [], "customer": [], "article": [], "interaction": []},
        "reason_templates": FEATURE_NAMES,
        "total_tx_last_week": float(len(history_df.filter(
            pl.col("week_idx") == AS_OF_WEEK - 1
        ))),
        "total_decayed_tx": float(rng.uniform(10_000, 50_000)),
        "synthetic": True,
        "n_articles": N_ARTICLES,
        "n_customers": N_CUSTOMERS,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Manifest with SHA-256 of every file
    print("[synth] Computing SHA-256 hashes ...", flush=True)
    file_hashes: dict[str, str] = {}
    for f in sorted(out_dir.rglob("*")):
        if f.is_file() and f.name != "manifest.json":
            rel = str(f.relative_to(out_dir))
            file_hashes[rel] = _sha256_file(f)

    manifest = {
        "as_of_week": AS_OF_WEEK,
        "git_commit": "synthetic",
        "synthetic": True,
        "model_metrics": {
            "map12_fold103": 0.0,
            "map12_holdout": 0.0,
        },
        "file_hashes": file_hashes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total_mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
    print(f"[synth] Bundle written to {out_dir} ({total_mb:.1f} MB, {N_ARTICLES} articles, {N_CUSTOMERS} customers)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build synthetic bundle for CI testing")
    parser.add_argument("--out-dir", default="artifacts/bundle_synthetic",
                        help="Output directory (default: artifacts/bundle_synthetic)")
    args = parser.parse_args()
    build_bundle(Path(args.out_dir))
