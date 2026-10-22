"""Profile CRUD, recommendations, and comparison endpoints."""

from __future__ import annotations

import logging
import math
import time
from datetime import date as date_type

import polars as pl
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from src.api import db
from src.api.deps import get_bundle, get_recommender

logger = logging.getLogger("api")
router = APIRouter(tags=["profiles"])

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

_VALID_AGE_RANGES = {"<25", "25-34", "35-44", "45-54", "55+"}
_VALID_CHANNELS = {"online", "store"}

# Age range → age_bucket (direct passthrough for our buckets)
_AGE_RANGE_TO_BUCKET: dict[str, str] = {
    "<25": "<25",
    "25-34": "25-34",
    "35-44": "35-44",
    "45-54": "45-54",
    "55+": "55+",
}

# Channel → sales_channel_id
_CHANNEL_TO_ID: dict[str, int] = {
    "online": 2,
    "store": 1,
}

_HISTORY_DATE = date_type(2020, 9, 15)


class CreateProfileRequest(BaseModel):
    nickname: str = Field(..., min_length=1, max_length=30)
    age_range: str
    channel: str

    @field_validator("age_range")
    @classmethod
    def validate_age_range(cls, v: str) -> str:
        if v not in _VALID_AGE_RANGES:
            raise ValueError(f"age_range must be one of {sorted(_VALID_AGE_RANGES)}")
        return v

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        if v not in _VALID_CHANNELS:
            raise ValueError(f"channel must be one of {sorted(_VALID_CHANNELS)}")
        return v


class Profile(BaseModel):
    id: str
    nickname: str
    age_range: str
    channel: str
    created_at: str
    purchase_count: int


class ProfileList(BaseModel):
    profiles: list[Profile]
    total: int


class CheckoutRequest(BaseModel):
    article_ids: list[str] = Field(..., min_length=1)


class CheckoutResponse(BaseModel):
    purchase_count: int
    new_count: int


class ProfileRecItem(BaseModel):
    article_idx: int
    article_id: str
    prod_name: str
    product_group: str
    colour: str
    image_url: str
    score: float | None
    reasons: list[str]
    source: str
    latency_ms: float
    cold_start: bool


class ProfileRecsResponse(BaseModel):
    profile_id: str
    cold_start: bool
    items: list[ProfileRecItem]
    latency_ms: float


class ProfileSummary(BaseModel):
    profile_id: str
    nickname: str
    age_range: str
    channel: str
    purchase_count: int
    top_product_groups: list[str]
    top_colour: str | None


class CompareResult(BaseModel):
    a_summary: ProfileSummary
    b_summary: ProfileSummary
    a_recommendations: list[ProfileRecItem]
    b_recommendations: list[ProfileRecItem]
    overlap: list[str]
    overlap_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_profile_or_404(profile_id: str) -> dict:
    profile = db.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id!r} not found")
    return profile


def _resolve_article_idx_from_id(article_id: str, bundle) -> int | None:
    """Map article_id → article_idx; returns None if not found."""
    if bundle.article_meta is None:
        return None
    rows = bundle.article_meta.filter(pl.col("article_id") == article_id)
    if len(rows) == 0:
        return None
    return int(rows["article_idx"][0])


def _image_url(article_id: str) -> str:
    return f"/api/v1/articles/{article_id}/image"


def _get_article_meta_by_idx(article_idx: int, bundle) -> dict:
    """Return prod_name, product_group, colour from bundle.article_meta."""
    if bundle.article_meta is None:
        return {
            "prod_name": "",
            "product_group": "",
            "colour": "",
            "article_id": str(article_idx),
        }
    rows = bundle.article_meta.filter(pl.col("article_idx") == article_idx)
    if len(rows) == 0:
        return {
            "prod_name": "",
            "product_group": "",
            "colour": "",
            "article_id": str(article_idx),
        }
    r = rows.row(0, named=True)
    return {
        "prod_name": str(r.get("prod_name", "")),
        "product_group": str(r.get("product_type_name", "")),
        "colour": str(r.get("colour_group_name", "")),
        "article_id": str(r.get("article_id", article_idx)),
    }


def _build_rec_items(
    raw_results: list[dict],
    bundle,
    cold_start: bool,
    latency_ms: float,
) -> list[ProfileRecItem]:
    items = []
    for r in raw_results:
        art_idx = int(r.get("article_idx", 0))
        art_id = str(r.get("article_id", ""))
        prod_name = str(r.get("prod_name", ""))
        product_group = str(r.get("product_type", r.get("product_group", "")))
        colour = str(r.get("colour", ""))

        # Fill from article_meta if missing
        if not art_id or not prod_name:
            meta = _get_article_meta_by_idx(art_idx, bundle)
            art_id = art_id or meta["article_id"]
            prod_name = prod_name or meta["prod_name"]
            product_group = product_group or meta["product_group"]
            colour = colour or meta["colour"]

        score = r.get("score")
        reasons = r.get("top_3_reasons", r.get("reasons", []))
        source = r.get("source", "trending" if cold_start else "model")

        items.append(
            ProfileRecItem(
                article_idx=art_idx,
                article_id=art_id,
                prod_name=prod_name,
                product_group=product_group,
                colour=colour,
                image_url=_image_url(art_id),
                score=float(score) if score is not None else None,
                reasons=reasons if isinstance(reasons, list) else [],
                source=source,
                latency_ms=latency_ms,
                cold_start=cold_start,
            )
        )
    return items


def _build_summary(profile: dict, purchases: list[dict], bundle) -> ProfileSummary:
    """Aggregate purchased articles into top product groups and top colour."""
    top_groups: list[str] = []
    top_colour: str | None = None

    if purchases and bundle.article_meta is not None:
        art_idxs = [p["article_idx"] for p in purchases]
        subset = bundle.article_meta.filter(pl.col("article_idx").is_in(art_idxs))

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

    return ProfileSummary(
        profile_id=profile["id"],
        nickname=profile["nickname"],
        age_range=profile["age_range"],
        channel=profile["channel"],
        purchase_count=profile["purchase_count"],
        top_product_groups=top_groups,
        top_colour=top_colour,
    )


def _get_recs_for_profile(
    profile: dict, purchases: list[dict], bundle, recommender
) -> tuple[list[ProfileRecItem], bool, float]:
    """Return (items, cold_start, latency_ms)."""
    t0 = time.perf_counter()
    cold_start = len(purchases) == 0

    if cold_start:
        raw = recommender._popularity_fallback_with_meta(k=12)
        # Ensure we get items from pop_last_week with correct field names
        latency_ms = (time.perf_counter() - t0) * 1000
        items = []
        for r in raw[:12]:
            art_idx = int(r.get("article_idx", 0))
            art_id = str(r.get("article_id", ""))
            meta = _get_article_meta_by_idx(art_idx, bundle)
            items.append(
                ProfileRecItem(
                    article_idx=art_idx,
                    article_id=art_id or meta["article_id"],
                    prod_name=r.get("prod_name") or meta["prod_name"],
                    product_group=r.get("product_group") or meta["product_group"],
                    colour=r.get("colour") or meta["colour"],
                    image_url=_image_url(art_id or meta["article_id"]),
                    score=r.get("score"),
                    reasons=[],
                    source="trending",
                    latency_ms=latency_ms,
                    cold_start=True,
                )
            )
        return items, True, latency_ms
    else:
        # Build history_rows for recommend_custom
        age_bucket = _AGE_RANGE_TO_BUCKET.get(profile["age_range"], "missing")
        channel_id = _CHANNEL_TO_ID.get(profile["channel"], 2)

        history_rows = [
            {
                "article_idx": p["article_idx"],
                "t_dat": str(_HISTORY_DATE),
                "price": 0.05,
                "sales_channel_id": channel_id,
            }
            for p in purchases
        ]

        raw = recommender.recommend_custom(history_rows, k=12, age_bucket=age_bucket)
        latency_ms = (time.perf_counter() - t0) * 1000
        items = _build_rec_items(raw, bundle, cold_start=False, latency_ms=latency_ms)
        return items, False, latency_ms


def _try_resolve_as_bundle_customer(entity_id: str, bundle) -> ProfileSummary | None:
    """If entity_id looks like a customer_id hex, resolve from bundle. Return None if not found."""
    if bundle.customers is None or "customer_id" not in bundle.customers.columns:
        return None
    rows = bundle.customers.filter(pl.col("customer_id") == entity_id)
    if len(rows) == 0:
        return None
    row = rows.row(0, named=True)
    cidx = int(row["customer_idx"])
    age = row.get("age")

    # Derive age_range from age
    if age is None or (isinstance(age, float) and math.isnan(age)) or age < 25:
        age_range = "<25"
    elif age < 35:
        age_range = "25-34"
    elif age < 45:
        age_range = "35-44"
    elif age < 55:
        age_range = "45-54"
    else:
        age_range = "55+"

    # Determine dominant channel from history
    channel = "online"
    if bundle.customer_history is not None:
        hist = bundle.customer_history.filter(pl.col("customer_idx") == cidx)
        if len(hist) > 0 and "sales_channel_id" in hist.columns:
            ch_counts = (
                hist.group_by("sales_channel_id")
                .agg(pl.len().alias("cnt"))
                .sort("cnt", descending=True)
            )
            dominant_ch = ch_counts["sales_channel_id"][0]
            channel = "store" if int(dominant_ch) == 1 else "online"

    # Purchase history for summary
    purchases_list: list[dict] = []
    if bundle.customer_history is not None:
        hist = bundle.customer_history.filter(pl.col("customer_idx") == cidx)
        purchases_list = [
            {"article_idx": int(r["article_idx"])} for r in hist.iter_rows(named=True)
        ]

    synthetic_profile = {
        "id": entity_id,
        "nickname": entity_id[:8],
        "age_range": age_range,
        "channel": channel,
        "purchase_count": len(purchases_list),
    }
    return _build_summary(synthetic_profile, purchases_list, bundle)


def _resolve_entity_summary(entity_id: str, bundle) -> ProfileSummary:
    """Resolve entity_id as profile UUID first, then as bundle customer_id."""
    profile = db.get_profile(entity_id)
    if profile is not None:
        purchases = db.get_purchases(entity_id)
        return _build_summary(profile, purchases, bundle)

    # Try as bundle customer_id
    summary = _try_resolve_as_bundle_customer(entity_id, bundle)
    if summary is not None:
        return summary

    raise HTTPException(
        status_code=404, detail=f"Entity {entity_id!r} not found as profile or customer"
    )


def _resolve_entity_recs(
    entity_id: str, bundle, recommender
) -> tuple[list[ProfileRecItem], bool, float]:
    """Resolve entity_id as profile UUID or bundle customer_id and return recs."""
    profile = db.get_profile(entity_id)
    if profile is not None:
        purchases = db.get_purchases(entity_id)
        return _get_recs_for_profile(profile, purchases, bundle, recommender)

    # Try as bundle customer_id
    if bundle.customers is not None and "customer_id" in bundle.customers.columns:
        rows = bundle.customers.filter(pl.col("customer_id") == entity_id)
        if len(rows) > 0:
            row = rows.row(0, named=True)
            cidx = int(row["customer_idx"])
            t0 = time.perf_counter()
            raw = recommender.recommend(cidx, k=12, compute_reasons=False)
            latency_ms = (time.perf_counter() - t0) * 1000
            items = _build_rec_items(
                raw, bundle, cold_start=False, latency_ms=latency_ms
            )
            return items, False, latency_ms

    raise HTTPException(status_code=404, detail=f"Entity {entity_id!r} not found")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/profiles",
    response_model=Profile,
    status_code=201,
    summary="Create a new shopping profile",
)
async def create_profile(body: CreateProfileRequest):
    try:
        result = db.create_profile(body.nickname, body.age_range, body.channel)
    except OverflowError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return Profile(**result)


@router.get(
    "/profiles",
    response_model=ProfileList,
    summary="List profiles with pagination",
)
async def list_profiles(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    profiles_raw, total = db.list_profiles(limit=limit, offset=offset)
    return ProfileList(
        profiles=[Profile(**p) for p in profiles_raw],
        total=total,
    )


@router.get(
    "/profiles/{profile_id}",
    response_model=Profile,
    summary="Get a profile by ID",
)
async def get_profile(profile_id: str):
    return Profile(**_get_profile_or_404(profile_id))


@router.delete(
    "/profiles/{profile_id}",
    status_code=204,
    summary="Delete a profile",
)
async def delete_profile(profile_id: str):
    deleted = db.delete_profile(profile_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id!r} not found")


@router.post(
    "/profiles/{profile_id}/checkout",
    response_model=CheckoutResponse,
    summary="Record article purchases on a profile",
)
async def checkout(
    profile_id: str,
    body: CheckoutRequest,
    bundle=Depends(get_bundle),
):
    _get_profile_or_404(profile_id)

    # Resolve article_ids → article_idx
    article_idxs = []
    valid_article_ids = []
    for aid in body.article_ids:
        aidx = _resolve_article_idx_from_id(aid, bundle)
        if aidx is not None:
            article_idxs.append(aidx)
            valid_article_ids.append(aid)
        else:
            logger.warning("article_id %r not found in bundle; skipping", aid)

    if not article_idxs:
        raise HTTPException(
            status_code=422,
            detail="None of the provided article_ids were found in the bundle",
        )

    try:
        new_count = db.add_purchases(profile_id, article_idxs, valid_article_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return CheckoutResponse(
        purchase_count=new_count,  # return the new total (not the old count)
        new_count=new_count,
    )


@router.get(
    "/profiles/{profile_id}/recommendations",
    response_model=ProfileRecsResponse,
    summary="Personalized recommendations for a profile",
)
async def profile_recommendations(
    profile_id: str,
    include_reasons: bool = Query(True),
    bundle=Depends(get_bundle),
    recommender=Depends(get_recommender),
):
    profile = _get_profile_or_404(profile_id)
    purchases = db.get_purchases(profile_id)

    items, cold_start, latency_ms = _get_recs_for_profile(
        profile, purchases, bundle, recommender
    )

    return ProfileRecsResponse(
        profile_id=profile_id,
        cold_start=cold_start,
        items=items[:12],
        latency_ms=latency_ms,
    )


@router.get(
    "/profiles/{profile_id}/summary",
    response_model=ProfileSummary,
    summary="Preference summary for a profile",
)
async def profile_summary(
    profile_id: str,
    bundle=Depends(get_bundle),
):
    profile = _get_profile_or_404(profile_id)
    purchases = db.get_purchases(profile_id)
    return _build_summary(profile, purchases, bundle)


@router.get(
    "/compare",
    response_model=CompareResult,
    summary="Compare two profiles or customers side-by-side",
)
async def compare(
    a: str = Query(..., description="Profile UUID or customer_id hex"),
    b: str = Query(..., description="Profile UUID or customer_id hex"),
    bundle=Depends(get_bundle),
    recommender=Depends(get_recommender),
):
    a_summary = _resolve_entity_summary(a, bundle)
    b_summary = _resolve_entity_summary(b, bundle)
    a_items, _, _ = _resolve_entity_recs(a, bundle, recommender)
    b_items, _, _ = _resolve_entity_recs(b, bundle, recommender)

    a_ids = {item.article_id for item in a_items}
    b_ids = {item.article_id for item in b_items}
    overlap_ids = sorted(a_ids & b_ids)

    return CompareResult(
        a_summary=a_summary,
        b_summary=b_summary,
        a_recommendations=a_items,
        b_recommendations=b_items,
        overlap=overlap_ids,
        overlap_count=len(overlap_ids),
    )
