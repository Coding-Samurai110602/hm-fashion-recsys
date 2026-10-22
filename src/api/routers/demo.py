"""Demo shoppers endpoint — pre-computed diverse customer showcase."""

from __future__ import annotations

import logging
import math
import threading

import polars as pl
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.deps import get_bundle

logger = logging.getLogger("api")
router = APIRouter(tags=["demo"])

# Module-level cache: set once, never recomputed
_demo_shoppers_cache: list[DemoShopper] | None = None
_demo_cache_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class DemoShopper(BaseModel):
    customer_id: str
    customer_idx: int
    age_range: str
    channel: str
    purchase_count: int
    top_product_groups: list[str]
    top_colour: str | None
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _age_to_range(age: float | None) -> str:
    if age is None or (isinstance(age, float) and math.isnan(age)):
        return "25-34"
    if age < 25:
        return "<25"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55+"


def _dominant_channel(hist: pl.DataFrame) -> str:
    """Return 'store' or 'online' based on the dominant sales_channel_id."""
    if hist is None or len(hist) == 0 or "sales_channel_id" not in hist.columns:
        return "online"
    ch_counts = (
        hist.group_by("sales_channel_id")
        .agg(pl.len().alias("cnt"))
        .sort("cnt", descending=True)
    )
    dominant = int(ch_counts["sales_channel_id"][0])
    return "store" if dominant == 1 else "online"


def _compute_demo_shoppers(bundle) -> list[DemoShopper]:
    """Compute diverse demo shoppers from bundle data. Called once at first request."""
    if bundle.customers is None or bundle.customer_history is None:
        logger.warning("Demo shoppers: bundle.customers or customer_history is None")
        return []

    customers = bundle.customers
    history = bundle.customer_history
    article_meta = bundle.article_meta

    # Count purchases per customer
    purchase_counts = history.group_by("customer_idx").agg(
        pl.len().alias("purchase_count")
    )

    # Join with customer info
    is_synthetic = bundle.manifest.get("synthetic", False)
    n_target = 15 if is_synthetic else 40

    # Filter: 5–150 purchases
    eligible = purchase_counts.filter(
        (pl.col("purchase_count") >= 5) & (pl.col("purchase_count") <= 150)
    )

    if len(eligible) == 0:
        # Fallback: take any customers
        eligible = purchase_counts.sort("purchase_count", descending=True).head(
            n_target
        )

    # Join with customers table for age + customer_id
    merged = eligible.join(customers, on="customer_idx", how="inner")

    # Try to get diverse age buckets
    def bucket_fn(age):
        return _age_to_range(age)

    if "age" in merged.columns:
        merged = merged.with_columns(
            pl.col("age")
            .map_elements(bucket_fn, return_dtype=pl.Utf8)
            .alias("age_range_col")
        )

        # Pick diverse selection: sample from each age bucket
        buckets = ["<25", "25-34", "35-44", "45-54", "55+"]
        selected_idxs: list[int] = []
        per_bucket = max(1, n_target // len(buckets))

        for bucket in buckets:
            bucket_rows = merged.filter(pl.col("age_range_col") == bucket)
            take = min(per_bucket, len(bucket_rows))
            if take > 0:
                # Sort by purchase_count for variety, then take
                sub = bucket_rows.sort("purchase_count").head(take)
                selected_idxs.extend(sub["customer_idx"].to_list())

        # If we still need more, take from remaining
        if len(selected_idxs) < n_target:
            remaining_needed = n_target - len(selected_idxs)
            taken_set = set(selected_idxs)
            rest = merged.filter(~pl.col("customer_idx").is_in(list(taken_set)))
            extra = rest.sort("purchase_count", descending=True).head(remaining_needed)
            selected_idxs.extend(extra["customer_idx"].to_list())

        selected_idxs = selected_idxs[:n_target]
        selected_merged = merged.filter(pl.col("customer_idx").is_in(selected_idxs))
    else:
        selected_merged = merged.head(n_target)

    shoppers = []
    for row in selected_merged.iter_rows(named=True):
        cidx = int(row["customer_idx"])
        cid = str(row.get("customer_id", str(cidx)))
        age = row.get("age")
        age_range = _age_to_range(age)
        purchase_count = int(row.get("purchase_count", 0))

        # Get history for this customer
        cust_hist = history.filter(pl.col("customer_idx") == cidx)
        channel = _dominant_channel(cust_hist)

        # Top product groups and colour from purchase history
        top_groups: list[str] = []
        top_colour: str | None = None

        if article_meta is not None and len(cust_hist) > 0:
            art_idxs = cust_hist["article_idx"].to_list()
            subset = article_meta.filter(pl.col("article_idx").is_in(art_idxs))

            if len(subset) > 0 and "product_type_name" in subset.columns:
                group_counts = (
                    subset.group_by("product_type_name")
                    .agg(pl.len().alias("cnt"))
                    .sort("cnt", descending=True)
                )
                top_groups = group_counts["product_type_name"].head(2).to_list()

            if len(subset) > 0 and "colour_group_name" in subset.columns:
                colour_counts = (
                    subset.group_by("colour_group_name")
                    .agg(pl.len().alias("cnt"))
                    .sort("cnt", descending=True)
                )
                top_colour_list = colour_counts["colour_group_name"].head(1).to_list()
                top_colour = top_colour_list[0] if top_colour_list else None

        # Build a human-readable summary
        group_str = " and ".join(top_groups) if top_groups else "various items"
        colour_str = f", favouring {top_colour}" if top_colour else ""
        summary = (
            f"{age_range} {channel} shopper with {purchase_count} purchases, "
            f"mostly {group_str}{colour_str}."
        )

        shoppers.append(
            DemoShopper(
                customer_id=cid,
                customer_idx=cidx,
                age_range=age_range,
                channel=channel,
                purchase_count=purchase_count,
                top_product_groups=top_groups,
                top_colour=top_colour,
                summary=summary,
            )
        )

    return shoppers


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/shoppers",
    response_model=list[DemoShopper],
    summary="Diverse demo customer profiles for showcasing recommendations",
)
async def demo_shoppers(bundle=Depends(get_bundle)):
    global _demo_shoppers_cache

    if _demo_shoppers_cache is not None:
        return _demo_shoppers_cache

    with _demo_cache_lock:
        # Double-checked locking
        if _demo_shoppers_cache is None:
            _demo_shoppers_cache = _compute_demo_shoppers(bundle)

    return _demo_shoppers_cache
