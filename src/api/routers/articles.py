"""Article catalog endpoints."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response

from src.api.deps import get_bundle
from src.api.schemas import ArticleMeta

logger = logging.getLogger("api")
router = APIRouter(tags=["articles"])

# Placeholder image (1×1 white PNG, base64 encoded)
_PLACEHOLDER_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a"
    "4944415478016360000000020001e221bc330000000049454e44ae426082"
)


@router.get(
    "/articles/search",
    response_model=list[ArticleMeta],
    summary="Search articles by name/type",
)
async def search_articles(
    q: str | None = Query(None, description="Name or keyword search"),
    product_type: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    bundle=Depends(get_bundle),
):
    if bundle.article_meta is None:
        return []
    df = bundle.article_meta
    if q:
        q_lower = q.lower()
        df = (
            df.filter(
                pl.col("prod_name").str.to_lowercase().str.contains(q_lower)
                | pl.col("product_type_name").str.to_lowercase().str.contains(q_lower)
            )
            if "prod_name" in df.columns
            else df
        )
    if product_type:
        df = (
            df.filter(pl.col("product_type_name") == product_type)
            if "product_type_name" in df.columns
            else df
        )
    df = df.head(limit)
    results = []
    for row in df.iter_rows(named=True):
        results.append(
            ArticleMeta(
                article_idx=int(row.get("article_idx", 0)),
                article_id=str(row.get("article_id", "")),
                prod_name=str(row.get("prod_name", "")),
                product_type=str(row.get("product_type_name", "")),
                colour=str(row.get("colour_group_name", "")),
                department=str(row.get("department_name", "")),
            )
        )
    return results


@router.get(
    "/articles/{article_id}",
    response_model=ArticleMeta,
    summary="Article metadata by ID",
)
async def get_article(article_id: str, bundle=Depends(get_bundle)):
    if bundle.article_meta is None:
        raise HTTPException(status_code=404, detail=f"Article {article_id!r} not found")
    rows = bundle.article_meta.filter(pl.col("article_id") == article_id)
    if len(rows) == 0:
        raise HTTPException(status_code=404, detail=f"Article {article_id!r} not found")
    r = rows.row(0, named=True)
    return ArticleMeta(
        article_idx=int(r.get("article_idx", 0)),
        article_id=str(r.get("article_id", "")),
        prod_name=str(r.get("prod_name", "")),
        product_type=str(r.get("product_type_name", "")),
        colour=str(r.get("colour_group_name", "")),
        department=str(r.get("department_name", "")),
    )


@router.get(
    "/articles/{article_id}/image",
    summary="Article image (local if available, placeholder otherwise)",
)
async def get_article_image(article_id: str):
    # Validate article_id: only hex digits (10 chars) to prevent path traversal
    if not article_id.isdigit() or len(article_id) != 10:
        raise HTTPException(status_code=422, detail="Invalid article_id format")
    # H&M image path convention: images/{prefix}/{article_id}.jpg
    prefix = article_id[:3]
    img_path = Path("data") / "images" / prefix / f"{article_id}.jpg"
    if img_path.exists():
        return FileResponse(str(img_path), media_type="image/jpeg")
    # Return placeholder PNG
    return Response(content=_PLACEHOLDER_PNG, media_type="image/png")
