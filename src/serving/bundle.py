"""Bundle loader: verifies SHA-256 of every artifact against manifest and serves from disk.

load_bundle(path) raises ValueError on any SHA-256 mismatch; never silently uses corrupt data.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import polars as pl


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Bundle:
    """Loaded serving bundle with verified artifacts.

    Attributes
    ----------
    path : Path
        Path to the bundle directory.
    manifest : dict
        Parsed manifest.json.
    booster : lgb.Booster
        Primary week-104 model.
    article_feats : pl.DataFrame
        Article feature table as of week 104.
    pop_last_week : pl.DataFrame
        Top-k articles by last-week popularity.
    pop_decayed : pl.DataFrame
        Top-k articles by decayed popularity.
    seg_pop_by_bucket : dict[str, pl.DataFrame]
        Per-age-bucket top-k articles.
    copurchase : pl.DataFrame
        Symmetric co-purchase cosine matrix.
    prev_week_sales : pl.DataFrame
        Article sales in week 103.
    article_product_codes : pl.DataFrame
        (article_idx, product_code) mapping.
    age_buckets : pl.DataFrame
        (customer_idx, age_bucket) for all customers.
    bucket_totals : pl.DataFrame
        Segment normalization denominators.
    customer_history : pl.DataFrame
        Transaction history up to week 104 (indexed by customer_idx for fast lookup).
    article_meta : pl.DataFrame
        Display metadata (article_id, prod_name, product_type, colour, department).
    feature_names : list[str]
        Canonical feature list.
    feature_dtypes : dict[str, str]
        Polars dtype names per feature.
    reason_templates : dict
        Plain-language template registry.
    gt_week104 : pl.DataFrame
        Week-104 ground-truth purchases (for display only, clearly labelled).
    customers : pl.DataFrame
        Full customers table (age, FN, Active, club_member_status, fashion_news_frequency).
    articles_raw : pl.DataFrame
        Raw articles table (needed by compute_interaction_features: product_code, etc.).
    total_tx_last_week : float
        Popularity share denominator.
    total_decayed_tx : float
        Decayed popularity share denominator.
    ab_arrays : pl.DataFrame or None
        Per-customer hit@12/AP@12 arrays for ranker and heuristic (week 104).
        Columns: customer_idx, ranker_ap12, ranker_hit12, heuristic_ap12, heuristic_hit12.
    reports : dict[str, dict]
        Snapshotted evaluation JSONs keyed by filename stem.
    """

    def __init__(self) -> None:
        self.path: Path | None = None
        self.manifest: dict = {}
        self.booster: lgb.Booster | None = None
        self.article_feats: pl.DataFrame | None = None
        self.pop_last_week: pl.DataFrame | None = None
        self.pop_decayed: pl.DataFrame | None = None
        self.seg_pop_by_bucket: dict[str, pl.DataFrame] = {}
        self.copurchase: pl.DataFrame | None = None
        self.prev_week_sales: pl.DataFrame | None = None
        self.article_product_codes: pl.DataFrame | None = None
        self.age_buckets: pl.DataFrame | None = None
        self.bucket_totals: pl.DataFrame | None = None
        self.customer_history: pl.DataFrame | None = None
        self.article_meta: pl.DataFrame | None = None
        self.customers: pl.DataFrame | None = None
        self.articles_raw: pl.DataFrame | None = None
        self.feature_names: list[str] = []
        self.feature_dtypes: dict[str, str] = {}
        self.reason_templates: dict = {}
        self.gt_week104: pl.DataFrame | None = None
        self.total_tx_last_week: float = 1.0
        self.total_decayed_tx: float = 1.0
        self.ab_arrays: pl.DataFrame | None = None
        self.reports: dict[str, dict] = {}


def load_bundle(path: str | Path) -> Bundle:
    """Load and verify all bundle artifacts from disk.

    Raises ValueError if any file's SHA-256 does not match the manifest.

    Parameters
    ----------
    path : str or Path
        Path to the bundle directory (containing manifest.json).

    Returns
    -------
    Bundle with all fields populated.
    """
    t0 = time.perf_counter()
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found in {path}")

    manifest = json.loads(manifest_path.read_text())
    file_hashes: dict[str, str] = manifest["file_hashes"]

    # Verify every file listed in the manifest
    for rel, expected_sha in file_hashes.items():
        fpath = path / rel
        if not fpath.exists():
            raise ValueError(f"Bundle file missing: {fpath}")
        actual = _sha256_file(fpath)
        if actual != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {rel}:\n"
                f"  expected: {expected_sha}\n"
                f"  actual:   {actual}"
            )

    b = Bundle()
    b.path = path
    b.manifest = manifest

    # Load model
    b.booster = lgb.Booster(model_file=str(path / "model.txt"))

    # Load parquet state files
    b.article_feats = pl.read_parquet(path / "state" / "article_feats.parquet")
    b.pop_last_week = pl.read_parquet(path / "state" / "pop_last_week.parquet")
    b.pop_decayed = pl.read_parquet(path / "state" / "pop_decayed.parquet")
    b.copurchase = pl.read_parquet(path / "state" / "copurchase.parquet")
    b.prev_week_sales = pl.read_parquet(path / "state" / "prev_week_sales.parquet")
    b.article_product_codes = pl.read_parquet(
        path / "state" / "article_product_codes.parquet"
    )
    b.age_buckets = pl.read_parquet(path / "state" / "age_buckets.parquet")
    b.bucket_totals = pl.read_parquet(path / "state" / "bucket_totals.parquet")
    b.customer_history = pl.read_parquet(path / "state" / "customer_history.parquet")
    b.article_meta = pl.read_parquet(path / "state" / "article_meta.parquet")
    b.gt_week104 = pl.read_parquet(path / "state" / "gt_week104.parquet")
    # Full customers table for feature computation (age, FN, Active, etc.)
    customers_path = path / "state" / "customers.parquet"
    if customers_path.exists():
        b.customers = pl.read_parquet(customers_path)
    else:
        b.customers = None
    # Raw articles table (needed by compute_interaction_features for product_code, etc.)
    articles_raw_path = path / "state" / "articles_raw.parquet"
    if articles_raw_path.exists():
        b.articles_raw = pl.read_parquet(articles_raw_path)
    else:
        b.articles_raw = None

    # Load per-bucket segment popularity
    # Reverse the safe-name encoding applied during export_bundle:
    #   lt<X>  →  <<X>   (e.g. lt25 → <25)
    #   <X>plus →  <X>+  (e.g. 55plus → 55+)
    seg_dir = path / "state" / "seg_pop"
    if seg_dir.exists():
        for bucket_file in sorted(seg_dir.glob("*.parquet")):
            bucket_name = bucket_file.stem
            if bucket_name.startswith("lt"):
                bucket_name = "<" + bucket_name[2:]
            if bucket_name.endswith("plus"):
                bucket_name = bucket_name[:-4] + "+"
            b.seg_pop_by_bucket[bucket_name] = pl.read_parquet(bucket_file)

    # Load config files
    config = json.loads((path / "config.json").read_text())
    b.feature_names = config["feature_names"]
    b.feature_dtypes = config["feature_dtypes"]
    b.reason_templates = config.get("reason_templates", {})
    b.total_tx_last_week = float(config.get("total_tx_last_week", 1.0))
    b.total_decayed_tx = float(config.get("total_decayed_tx", 1.0))

    # Per-customer A/B arrays (optional — absent in old bundles)
    ab_arrays_path = path / "state" / "ab_arrays.parquet"
    if ab_arrays_path.exists():
        b.ab_arrays = pl.read_parquet(ab_arrays_path)

    # Snapshotted evaluation report JSONs (optional — absent in old bundles)
    reports_dir = path / "reports"
    if reports_dir.exists():
        for json_file in sorted(reports_dir.glob("*.json")):
            try:
                b.reports[json_file.stem] = json.loads(json_file.read_text())
            except Exception:
                pass

    load_time = time.perf_counter() - t0
    print(f"[bundle] Loaded in {load_time:.2f}s from {path}", flush=True)
    return b
