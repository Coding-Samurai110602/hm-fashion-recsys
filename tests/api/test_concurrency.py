"""Concurrency test: 20 concurrent requests across 5 customers.

Tests that:
1. All 20 requests complete successfully.
2. Results are identical to sequential calls (LightGBM predict + polars are thread-safe).
3. Cache counters are consistent with the number of distinct (customer_idx, k, reasons) keys.

Thread-safety reasoning:
- LightGBM Booster.predict() is thread-safe when called concurrently on separate inputs
  (no shared mutable state per prediction; reads from the same immutable model tree data).
- Polars DataFrames are immutable; operations produce new frames and are thread-safe.
- The LRU recommendation cache uses a threading.Lock (in services/recommendation.py) to
  protect cache writes; reads are lock-free (CPython GIL ensures dict lookup atomicity).
- The ThreadPoolExecutor in deps.py serializes model-scoring tasks internally.
"""

from __future__ import annotations

import concurrent.futures
import threading

import pytest

from tests.api.conftest import SYNTHETIC_BUNDLE


def _get_customers(n: int = 5) -> list[str]:
    """Return n distinct hex customer_ids from the synthetic bundle."""
    from src.serving.bundle import load_bundle

    b = load_bundle(SYNTHETIC_BUNDLE)
    return [
        str(row["customer_id"]) for row in b.customers.head(n).iter_rows(named=True)
    ]


@pytest.mark.api
class TestConcurrency:
    def test_20_concurrent_requests_all_succeed(self, synth_client):
        """20 concurrent GET /recommendations requests must all return 200."""
        customer_ids = _get_customers(5)
        # 4 requests per customer = 20 total
        urls = [
            f"/api/v1/customers/{cid}/recommendations?k=12&include_reasons=false"
            for cid in customer_ids
            for _ in range(4)
        ]

        results = []
        lock = threading.Lock()

        def _call(url: str) -> dict:
            try:
                r = synth_client.get(url)
                return {
                    "status": r.status_code,
                    "n_items": len(r.json()) if r.status_code == 200 else 0,
                }
            except Exception as exc:
                return {"status": -1, "error": str(exc)}

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(_call, url) for url in urls]
            for f in concurrent.futures.as_completed(futures):
                with lock:
                    results.append(f.result())

        assert len(results) == 20, f"Expected 20 results, got {len(results)}"
        bad = [r for r in results if r["status"] != 200]
        assert not bad, f"{len(bad)} requests failed: {bad[:3]}"

    def test_concurrent_results_match_sequential(self, synth_client):
        """Top-12 article list must be identical whether called sequentially or concurrently."""
        customer_ids = _get_customers(5)
        url_tmpl = "/api/v1/customers/{cid}/recommendations?k=12&include_reasons=false"

        # Sequential baseline
        sequential = {}
        for cid in customer_ids:
            r = synth_client.get(url_tmpl.format(cid=cid))
            assert r.status_code == 200
            sequential[cid] = [item["article_idx"] for item in r.json()]

        # Concurrent results (4 calls per customer)
        def _call(cid: str) -> tuple[str, list[int]]:
            r = synth_client.get(url_tmpl.format(cid=cid))
            assert r.status_code == 200
            return cid, [item["article_idx"] for item in r.json()]

        concurrent_results: list[tuple[str, list[int]]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            futures = [
                pool.submit(_call, cid) for cid in customer_ids for _ in range(4)
            ]
            for f in concurrent.futures.as_completed(futures):
                concurrent_results.append(f.result())

        for cid, article_idxs in concurrent_results:
            assert article_idxs == sequential[cid], (
                f"Concurrent result differs from sequential for customer {cid}:\n"
                f"  sequential:  {sequential[cid]}\n"
                f"  concurrent:  {article_idxs}"
            )

    def test_cache_counters_consistent(self, synth_client):
        """Cache hit + miss count must equal total requests after repeated calls."""
        from src.api.services import recommendation as rec_svc

        rec_svc.cache_clear()

        customer_ids = _get_customers(5)
        url_tmpl = "/api/v1/customers/{cid}/recommendations?k=12&include_reasons=false"
        n_calls_per_customer = 4
        total_calls = len(customer_ids) * n_calls_per_customer

        def _call(cid: str) -> int:
            r = synth_client.get(url_tmpl.format(cid=cid))
            return r.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            statuses = list(
                pool.map(
                    _call,
                    [cid for cid in customer_ids for _ in range(n_calls_per_customer)],
                )
            )

        assert all(s == 200 for s in statuses), f"Some requests failed: {set(statuses)}"

        # Cache info: hits + misses == total calls (each call goes through cache)
        info = rec_svc.cache_info()
        assert info.hits + info.misses == total_calls, (
            f"Cache hits ({info.hits}) + misses ({info.misses}) = {info.hits + info.misses} "
            f"!= total calls {total_calls}"
        )
        # In concurrent access, multiple threads for the same key can simultaneously
        # miss before any stores the result (race between check and store).
        # Misses must be at least len(customer_ids) (one per distinct key)
        # and at most total_calls (fully concurrent, each call sees an empty cache).
        # The important invariant is that ZERO results were wrong — tested above.
        assert len(customer_ids) <= info.misses <= total_calls, (
            f"Cache misses {info.misses} not in expected range "
            f"[{len(customer_ids)}, {total_calls}]"
        )
