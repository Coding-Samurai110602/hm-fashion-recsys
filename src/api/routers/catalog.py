"""Catalog endpoints: trending, browse, filters."""

from __future__ import annotations

import logging

import polars as pl
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from src.api.deps import get_bundle

logger = logging.getLogger("api")
router = APIRouter(tags=["catalog"])

_MAX_PAGE_SIZE = 100
_DEFAULT_PAGE_SIZE = 24


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TrendingItem(BaseModel):
    article_idx: int
    article_id: str
    prod_name: str
    product_group: str
    colour: str
    image_url: str
    popularity_rank: int


class CatalogBrowseItem(BaseModel):
    article_idx: int
    article_id: str
    prod_name: str
    product_group: str
    colour: str
    image_url: str
    popularity_rank: int | None = None


class CatalogBrowseResponse(BaseModel):
    items: list[CatalogBrowseItem]
    total: int
    page: int
    page_size: int


class CatalogFilters(BaseModel):
    product_groups: list[str]
    colours: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_url(article_id: str) -> str:
    return f"/api/v1/articles/{article_id}/image"


def _build_trending_df(bundle) -> pl.DataFrame | None:
    """Join pop_last_week with article_meta, sorted by source_rank."""
    if bundle.pop_last_week is None or bundle.article_meta is None:
        return None
    pop = bundle.pop_last_week
    meta = bundle.article_meta
    joined = pop.join(meta, on="article_idx", how="inner")
    if "source_rank" in joined.columns:
        joined = joined.sort("source_rank")
    return joined


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/trending",
    response_model=list[TrendingItem],
    summary="Top trending articles by last-week popularity",
)
async def trending(
    k: int = Query(20, ge=1, le=200),
    bundle=Depends(get_bundle),
):
    df = _build_trending_df(bundle)
    if df is None or len(df) == 0:
        return []

    results = []
    for i, row in enumerate(df.head(k).iter_rows(named=True), start=1):
        art_id = str(row.get("article_id", ""))
        results.append(
            TrendingItem(
                article_idx=int(row.get("article_idx", 0)),
                article_id=art_id,
                prod_name=str(row.get("prod_name", "")),
                product_group=str(row.get("product_type_name", "")),
                colour=str(row.get("colour_group_name", "")),
                image_url=_image_url(art_id),
                popularity_rank=int(row.get("source_rank", i)),
            )
        )
    return results


@router.get(
    "/browse",
    response_model=CatalogBrowseResponse,
    summary="Browse and filter the article catalog",
)
async def browse(
    product_group: str | None = Query(None, description="Filter by product_type_name"),
    colour: str | None = Query(None, description="Filter by colour_group_name"),
    q: str | None = Query(
        None, description="Search prod_name (case-insensitive contains)"
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(_DEFAULT_PAGE_SIZE, ge=1, le=_MAX_PAGE_SIZE),
    bundle=Depends(get_bundle),
):
    if bundle.article_meta is None:
        return CatalogBrowseResponse(items=[], total=0, page=page, page_size=page_size)

    df = bundle.article_meta

    # Filters
    if product_group and "product_type_name" in df.columns:
        df = df.filter(
            pl.col("product_type_name").str.to_lowercase() == product_group.lower()
        )
    if colour and "colour_group_name" in df.columns:
        df = df.filter(pl.col("colour_group_name").str.to_lowercase() == colour.lower())
    if q:
        q_lower = q.lower()
        if "prod_name" in df.columns:
            df = df.filter(pl.col("prod_name").str.to_lowercase().str.contains(q_lower))

    # Join with pop_last_week for sorting
    if bundle.pop_last_week is not None:
        pop = bundle.pop_last_week.select(["article_idx", "score", "source_rank"])
        df = df.join(pop, on="article_idx", how="left")
        df = df.sort("score", descending=True, nulls_last=True)
    else:
        df = df.with_columns(
            pl.lit(None).cast(pl.Float32).alias("score"),
            pl.lit(None).cast(pl.Int16).alias("source_rank"),
        )

    total = len(df)
    offset = (page - 1) * page_size
    page_df = df.slice(offset, page_size)

    items = []
    for row in page_df.iter_rows(named=True):
        art_id = str(row.get("article_id", ""))
        src_rank = row.get("source_rank")
        items.append(
            CatalogBrowseItem(
                article_idx=int(row.get("article_idx", 0)),
                article_id=art_id,
                prod_name=str(row.get("prod_name", "")),
                product_group=str(row.get("product_type_name", "")),
                colour=str(row.get("colour_group_name", "")),
                image_url=_image_url(art_id),
                popularity_rank=int(src_rank) if src_rank is not None else None,
            )
        )

    return CatalogBrowseResponse(
        items=items, total=total, page=page, page_size=page_size
    )


@router.get(
    "/filters",
    response_model=CatalogFilters,
    summary="Distinct product groups and colours for filter UI",
)
async def filters(bundle=Depends(get_bundle)):
    if bundle.article_meta is None:
        return CatalogFilters(product_groups=[], colours=[])

    df = bundle.article_meta
    product_groups: list[str] = []
    colours: list[str] = []

    if "product_type_name" in df.columns:
        product_groups = sorted(df["product_type_name"].drop_nulls().unique().to_list())
    if "colour_group_name" in df.columns:
        colours = sorted(df["colour_group_name"].drop_nulls().unique().to_list())

    return CatalogFilters(product_groups=product_groups, colours=colours)
