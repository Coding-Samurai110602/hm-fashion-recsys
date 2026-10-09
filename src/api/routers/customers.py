"""Customer and recommendation endpoints."""

from __future__ import annotations

import logging
import random

import polars as pl
from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.config import get_settings
from src.api.deps import get_bundle, get_executor, get_recommender
from src.api.schemas import (
    ArticleMeta,
    CustomerProfile,
    CustomRecommendRequest,
    ExplainResult,
    FunnelItem,
    RecommendedItemNoReasons,
    SampleCustomer,
)
from src.api.services import recommendation as rec_svc

logger = logging.getLogger("api")
router = APIRouter(tags=["customers"])

# Segment definitions (prior purchases thresholds)
_SEGMENTS = {
    "cold-start": (0, 0),
    "light": (1, 4),
    "medium": (5, 19),
    "heavy": (20, 999_999),
}


def _resolve_customer_idx(customer_id: str, bundle) -> int:
    """Map Kaggle customer_id hex string → integer customer_idx via bundle customers table."""
    if bundle.customers is None or "customer_id" not in bundle.customers.columns:
        raise HTTPException(
            status_code=503,
            detail="Customer lookup unavailable: bundle missing customer_id mapping. Re-export the bundle.",
        )
    rows = bundle.customers.filter(pl.col("customer_id") == customer_id)
    if len(rows) == 0:
        raise HTTPException(
            status_code=404, detail=f"Customer {customer_id!r} not found"
        )
    return int(rows["customer_idx"][0])


def _resolve_article_idx(article_id: str, bundle) -> int:
    """Map article_id string → integer article_idx via bundle article_meta."""
    if bundle.article_meta is None:
        raise HTTPException(status_code=404, detail=f"Article {article_id!r} not found")
    rows = bundle.article_meta.filter(pl.col("article_id") == article_id)
    if len(rows) == 0:
        raise HTTPException(status_code=404, detail=f"Article {article_id!r} not found")
    return int(rows["article_idx"][0])


@router.get(
    "/customers/sample",
    response_model=list[SampleCustomer],
    summary="Sample demo customers by segment",
)
async def sample_customers(
    segment: str | None = Query(
        None, description="Segment: cold-start, light, medium, heavy"
    ),
    n: int = Query(5, ge=1, le=20),
    bundle=Depends(get_bundle),
):
    if bundle.customers is None or bundle.customer_history is None:
        raise HTTPException(status_code=503, detail="Customer data not available")

    # Count purchases per customer from history
    purchase_counts = bundle.customer_history.group_by("customer_idx").agg(
        pl.col("article_idx").count().alias("n_purchases")
    )

    lo, hi = 0, 999_999
    if segment and segment in _SEGMENTS:
        lo, hi = _SEGMENTS[segment]

    filtered = purchase_counts.filter(
        (pl.col("n_purchases") >= lo) & (pl.col("n_purchases") <= hi)
    )

    if len(filtered) == 0:
        return []

    # Deterministic sample (seed per segment+n)
    seed = hash(f"{segment}{n}") % (2**31)
    rng = random.Random(seed)
    idx_list = filtered["customer_idx"].to_list()
    rng.shuffle(idx_list)
    selected = idx_list[:n]

    # Build a lookup dict from customer_idx → customer_id (hex) for selected customers
    cid_map: dict[int, str] = {}
    if bundle.customers is not None and "customer_id" in bundle.customers.columns:
        sub = bundle.customers.filter(pl.col("customer_idx").is_in(selected))
        for row in sub.iter_rows(named=True):
            cid_map[int(row["customer_idx"])] = str(row["customer_id"])

    results = []
    for cidx in selected:
        cid = cid_map.get(cidx, str(cidx))
        age_row = (
            bundle.age_buckets.filter(pl.col("customer_idx") == cidx)
            if bundle.age_buckets is not None
            else None
        )
        bucket = (
            str(age_row["age_bucket"][0])
            if (age_row is not None and len(age_row) > 0)
            else "unknown"
        )
        seg_label = segment or "unknown"
        results.append(
            SampleCustomer(
                customer_idx=cidx,
                customer_id=cid,
                segment=seg_label,
                age_bucket=bucket,
            )
        )
    return results


@router.get(
    "/customers/{customer_id}",
    response_model=CustomerProfile,
    summary="Customer profile summary",
)
async def customer_profile(customer_id: str, bundle=Depends(get_bundle)):
    customer_idx = _resolve_customer_idx(customer_id, bundle)

    # Purchase counts
    hist = bundle.customer_history.filter(pl.col("customer_idx") == customer_idx)
    n_all = len(hist)
    if "week_idx" in hist.columns:
        n_4w = len(hist.filter(pl.col("week_idx") >= 100))
    else:
        n_4w = 0

    # Age bucket
    age_row = (
        bundle.age_buckets.filter(pl.col("customer_idx") == customer_idx)
        if bundle.age_buckets is not None
        else None
    )
    bucket = (
        str(age_row["age_bucket"][0])
        if (age_row is not None and len(age_row) > 0)
        else "unknown"
    )

    # Recent 10 purchases
    recent_hist = (
        hist.sort("t_dat", descending=True).head(10) if len(hist) > 0 else hist
    )
    recent_items = []
    for row in recent_hist.iter_rows(named=True):
        art_idx = int(row["article_idx"])
        if bundle.article_meta is not None:
            meta_rows = bundle.article_meta.filter(pl.col("article_idx") == art_idx)
            if len(meta_rows) > 0:
                r = meta_rows.row(0, named=True)
                recent_items.append(
                    ArticleMeta(
                        article_idx=art_idx,
                        article_id=str(r.get("article_id", art_idx)),
                        prod_name=str(r.get("prod_name", "")),
                        product_type=str(r.get("product_type_name", "")),
                        colour=str(r.get("colour_group_name", "")),
                        department=str(r.get("department_name", "")),
                    )
                )

    return CustomerProfile(
        customer_idx=customer_idx,
        customer_id=customer_id,
        age_bucket=bucket,
        n_purchases_all_time=n_all,
        n_purchases_last_4w=n_4w,
        recent_history=recent_items,
    )


@router.get(
    "/customers/{customer_id}/recommendations",
    summary="Ranked recommendations for a customer",
)
async def recommendations(
    customer_id: str,
    k: int = Query(12, ge=1, le=50),
    include_reasons: bool = Query(True),
    bundle=Depends(get_bundle),
    recommender=Depends(get_recommender),
    executor=Depends(get_executor),
):
    customer_idx = _resolve_customer_idx(customer_id, bundle)
    results = await rec_svc.get_recommendations(
        recommender, customer_idx, k, include_reasons, executor
    )
    return results


@router.get(
    "/customers/{customer_id}/baseline",
    response_model=list[RecommendedItemNoReasons],
    summary="Heuristic baseline top-k (for side-by-side comparison)",
)
async def baseline(
    customer_id: str,
    k: int = Query(12, ge=1, le=50),
    bundle=Depends(get_bundle),
    recommender=Depends(get_recommender),
    executor=Depends(get_executor),
):
    customer_idx = _resolve_customer_idx(customer_id, bundle)
    return await rec_svc.get_baseline(recommender, customer_idx, k, executor)


@router.get(
    "/customers/{customer_id}/actual-purchases",
    response_model=list[ArticleMeta],
    summary="Week-104 actual purchases (display only, labelled)",
)
async def actual_purchases(
    customer_id: str,
    bundle=Depends(get_bundle),
    recommender=Depends(get_recommender),
    executor=Depends(get_executor),
):
    customer_idx = _resolve_customer_idx(customer_id, bundle)
    return await rec_svc.get_actual_purchases(recommender, customer_idx, executor)


@router.get(
    "/customers/{customer_id}/funnel",
    response_model=list[FunnelItem],
    summary="Full candidate funnel: source, score, final rank",
)
async def funnel(
    customer_id: str,
    bundle=Depends(get_bundle),
    recommender=Depends(get_recommender),
    executor=Depends(get_executor),
):
    customer_idx = _resolve_customer_idx(customer_id, bundle)
    return await rec_svc.get_funnel(recommender, customer_idx, executor)


@router.get(
    "/customers/{customer_id}/explain/{article_id}",
    response_model=ExplainResult,
    summary="Full contrast-SHAP breakdown for one (customer, article) pair",
)
async def explain(
    customer_id: str,
    article_id: str,
    bundle=Depends(get_bundle),
    recommender=Depends(get_recommender),
    executor=Depends(get_executor),
):
    customer_idx = _resolve_customer_idx(customer_id, bundle)
    article_idx = _resolve_article_idx(article_id, bundle)
    result = await rec_svc.get_explain(recommender, customer_idx, article_idx, executor)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post(
    "/recommendations/custom",
    summary="Recommendations for a custom history (build-your-own customer)",
)
async def custom_recommendations(
    body: CustomRecommendRequest,
    k: int = Query(12, ge=1, le=50),
    bundle=Depends(get_bundle),
    recommender=Depends(get_recommender),
    executor=Depends(get_executor),
):
    settings = get_settings()
    if len(body.history) > settings.MAX_CUSTOM_HISTORY:
        raise HTTPException(
            status_code=422,
            detail=f"History too long: {len(body.history)} > MAX_CUSTOM_HISTORY={settings.MAX_CUSTOM_HISTORY}",
        )
    history_rows = []
    for item in body.history:
        art_idx = (
            _resolve_article_idx(item.article_id, bundle)
            if bundle.article_meta is not None
            else 0
        )
        history_rows.append(
            {
                "article_idx": art_idx,
                "t_dat": str(item.t_dat),
                "price": item.price or 0.05,
                "sales_channel_id": item.sales_channel_id or 2,
            }
        )
    return await rec_svc.get_custom_recommendations(
        recommender, history_rows, k, executor
    )
