"""Thin service layer over BundleRecommender — keeps routers free of polars/numpy.

Thread-safety notes:
- _cache and counters are protected by _cache_lock.
- LightGBM Booster.predict() is thread-safe for concurrent reads on an immutable model.
- Polars DataFrames are immutable; all operations create new frames without shared mutation.
- The ThreadPoolExecutor in deps.py queues scoring tasks; the Lock ensures counters and
  cache dict are updated atomically even when multiple threads call get_recommendations.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

import polars as pl  # noqa: F401 (kept for potential future use by format helpers)

logger = logging.getLogger("api")

# Thread-safe LRU cache keyed by (customer_idx, k, include_reasons).
_cache: dict[tuple, Any] = {}
_cache_hits = 0
_cache_misses = 0
_cache_lock = Lock()


def cache_stats() -> dict:
    with _cache_lock:
        return {"hits": _cache_hits, "misses": _cache_misses, "size": len(_cache)}


def cache_clear() -> None:
    global _cache_hits, _cache_misses
    with _cache_lock:
        _cache.clear()
        _cache_hits = 0
        _cache_misses = 0


def cache_info():
    """Return a simple namespace with hits, misses, currsize (compat with lru_cache API)."""
    with _cache_lock:

        class _Info:
            hits = _cache_hits
            misses = _cache_misses
            currsize = len(_cache)

        return _Info()


def _cache_key(customer_idx: int, k: int, include_reasons: bool) -> tuple:
    return (customer_idx, k, include_reasons)


async def get_recommendations(
    recommender,
    customer_idx: int,
    k: int,
    include_reasons: bool,
    executor: ThreadPoolExecutor,
) -> list[dict[str, Any]]:
    global _cache_hits, _cache_misses
    key = _cache_key(customer_idx, k, include_reasons)
    with _cache_lock:
        if key in _cache:
            _cache_hits += 1
            return _cache[key]
        _cache_misses += 1

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(
        executor,
        lambda: recommender.recommend(customer_idx, k=k),
    )
    results = _format_recommendations(raw, include_reasons)
    with _cache_lock:
        # Another thread may have computed the same key concurrently — that's fine;
        # the result is deterministic so overwriting is safe.
        _cache[key] = results
    return results


def _format_recommendations(raw: list[dict], include_reasons: bool) -> list[dict]:
    out = []
    for rank, item in enumerate(raw, start=1):
        d = {
            "rank": rank,
            "article_idx": item.get("article_idx", 0),
            "article_id": item.get("article_id", ""),
            "prod_name": item.get("prod_name", ""),
            "product_type": item.get("product_type", ""),
            "colour": item.get("colour", ""),
            "department": item.get("department", ""),
            "sources": item.get("sources", []),
        }
        if include_reasons:
            d["reasons"] = item.get("top_3_reasons", [])
        out.append(d)
    return out


async def get_baseline(
    recommender,
    customer_idx: int,
    k: int,
    executor: ThreadPoolExecutor,
) -> list[dict[str, Any]]:
    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(
        executor,
        lambda: recommender.baseline(customer_idx, k=k),
    )
    return [
        {
            "rank": i + 1,
            "article_idx": item.get("article_idx", 0),
            "article_id": item.get("article_id", ""),
            "prod_name": item.get("prod_name", ""),
            "product_type": item.get("product_type", ""),
            "colour": item.get("colour", ""),
            "department": item.get("department", ""),
            "sources": item.get("sources", []),
        }
        for i, item in enumerate(raw)
    ]


async def get_funnel(
    recommender,
    customer_idx: int,
    executor: ThreadPoolExecutor,
) -> list[dict[str, Any]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: recommender.candidate_funnel(customer_idx),
    )


async def get_explain(
    recommender,
    customer_idx: int,
    article_idx: int,
    executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: recommender.explain(customer_idx, article_idx),
    )


async def get_actual_purchases(
    recommender,
    customer_idx: int,
    executor: ThreadPoolExecutor,
) -> list[dict[str, Any]]:
    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(
        executor,
        lambda: recommender.actual_purchases(customer_idx),
    )
    return [
        {
            "article_idx": item.get("article_idx", 0),
            "article_id": item.get("article_id", ""),
            "prod_name": item.get("prod_name", ""),
            "product_type": item.get("product_type", ""),
            "colour": item.get("colour", ""),
            "department": item.get("department", ""),
        }
        for item in raw
    ]


async def get_custom_recommendations(
    recommender,
    history_rows: list[dict],
    k: int,
    executor: ThreadPoolExecutor,
) -> list[dict[str, Any]]:
    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(
        executor,
        lambda: recommender.recommend_custom(history_rows, k=k),
    )
    return _format_recommendations(raw, include_reasons=True)
