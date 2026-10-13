"""Async load test: 10 concurrent users for 60 seconds on a realistic endpoint mix.

Targets the REAL bundle (must be running). Reports throughput, p50/p95/p99, error rate,
and cache hit rate. Draws from 1,000+ distinct real customers (seeded).
Saves results to reports/api/load_test.json.

Usage:
    # Start API first: make api
    PYTHONPATH=. python scripts/load_test.py [--url http://localhost:8000] [--duration 60] [--users 10]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx
import polars as pl

BUNDLE_PATH = Path("artifacts/bundle_week104")
N_DISTINCT_CUSTOMERS = 1000  # draw from at least this many distinct customers

ENDPOINT_MIX = [
    # (weight, path_template)
    (3, "/health"),
    (2, "/ready"),
    (1, "/version"),
    (2, "/api/v1/insights/summary"),
    (1, "/api/v1/insights/ablations"),
    (1, "/api/v1/insights/importance"),
    (4, "/api/v1/customers/{cid}/recommendations?k=12&include_reasons=false"),
    (2, "/api/v1/customers/{cid}/recommendations?k=12&include_reasons=true"),
    (1, "/api/v1/customers/{cid}/baseline?k=12"),
    (1, "/api/v1/customers/{cid}/actual-purchases"),
    (1, "/api/v1/articles/search?limit=20"),
]


def _load_real_customer_ids(seed: int, n: int) -> list[str]:
    """Load n distinct customer hex IDs from the real bundle (seeded)."""
    customers_path = BUNDLE_PATH / "state" / "customers.parquet"
    if not customers_path.exists():
        return []
    df = pl.read_parquet(customers_path)
    if "customer_id" not in df.columns:
        return []
    sample = df.sample(n=min(n, len(df)), seed=seed)["customer_id"].to_list()
    return [str(cid) for cid in sample]


async def single_user(
    base_url: str,
    duration: float,
    sample_customer_ids: list[str],
    results: list[dict],
    user_id: int,
):
    rng = random.Random(user_id * 1337)
    weights = [w for w, _ in ENDPOINT_MIX]
    paths = [p for _, p in ENDPOINT_MIX]

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        deadline = time.perf_counter() + duration
        while time.perf_counter() < deadline:
            path_template = rng.choices(paths, weights=weights)[0]
            cid = rng.choice(sample_customer_ids)
            path = path_template.replace("{cid}", str(cid))
            t0 = time.perf_counter()
            try:
                r = await client.get(path)
                elapsed = (time.perf_counter() - t0) * 1000
                results.append({
                    "path": path,
                    "status": r.status_code,
                    "elapsed_ms": elapsed,
                    "error": r.status_code >= 400,
                })
            except Exception as e:  # noqa: BLE001
                elapsed = (time.perf_counter() - t0) * 1000
                results.append({
                    "path": path,
                    "status": 0,
                    "elapsed_ms": elapsed,
                    "error": True,
                    "exception": str(e),
                })


async def main(base_url: str, n_users: int, duration: float, seed: int):
    print(f"[load_test] {n_users} users, {duration}s, target: {base_url}", flush=True)

    # Load distinct real customers from bundle (seeded for reproducibility)
    sample_cidxs = _load_real_customer_ids(seed=seed, n=N_DISTINCT_CUSTOMERS)
    if not sample_cidxs:
        # Fallback: fetch from API sample endpoint
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            r = await client.get("/api/v1/customers/sample?n=20&segment=medium")
            sample_cidxs = [item["customer_id"] for item in r.json()] if r.status_code == 200 else []

    n_distinct = len(set(sample_cidxs))
    print(f"[load_test] {n_distinct} distinct customers loaded (seed={seed})", flush=True)
    print(f"[load_test] Sample: {sample_cidxs[0][:16]}...", flush=True)

    all_results: list[dict] = []
    tasks = [
        single_user(base_url, duration, sample_cidxs, all_results, i)
        for i in range(n_users)
    ]

    t_start = time.perf_counter()
    await asyncio.gather(*tasks)
    elapsed_total = time.perf_counter() - t_start

    # Fetch cache stats from API
    cache_hit_rate = None
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
        try:
            cr = await client.get("/api/v1/metrics")
            if cr.status_code == 200:
                metrics_text = cr.text
                # Extract cache_hits and cache_misses from Prometheus metrics text
                hits = misses = 0
                for line in metrics_text.splitlines():
                    if line.startswith("recommendation_cache_hits_total "):
                        hits = int(float(line.split()[-1]))
                    elif line.startswith("recommendation_cache_misses_total "):
                        misses = int(float(line.split()[-1]))
                total = hits + misses
                if total > 0:
                    cache_hit_rate = round(100.0 * hits / total, 1)
                    print(f"[load_test] Cache: {hits} hits, {misses} misses ({cache_hit_rate}% hit rate)")
        except Exception:  # noqa: BLE001
            pass

    # Compute stats
    times = [r["elapsed_ms"] for r in all_results]
    errors = [r for r in all_results if r["error"]]
    n_total = len(all_results)
    n_errors = len(errors)
    error_rate = n_errors / n_total if n_total > 0 else 0.0

    times.sort()
    p50 = statistics.median(times) if times else 0
    p95 = times[int(0.95 * len(times))] if times else 0
    p99 = times[int(0.99 * len(times))] if times else 0
    throughput = n_total / elapsed_total

    summary = {
        "n_users": n_users,
        "duration_s": round(elapsed_total, 1),
        "n_requests": n_total,
        "n_errors": n_errors,
        "error_rate_pct": round(error_rate * 100, 2),
        "throughput_rps": round(throughput, 1),
        "n_distinct_customers": n_distinct,
        "cache_hit_rate_pct": cache_hit_rate,
        "note": "Cold-cache numbers: 1000+ distinct customers seeded so most requests miss the warm LRU cache",
        "latency_ms": {
            "p50": round(p50, 1),
            "p95": round(p95, 1),
            "p99": round(p99, 1),
            "mean": round(statistics.mean(times) if times else 0, 1),
        },
    }

    print("\n[load_test] Results:")
    print(f"  Requests: {n_total} total, {n_errors} errors ({error_rate*100:.1f}%)")
    print(f"  Throughput: {throughput:.1f} req/s")
    print(f"  Latency: p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms")
    print(f"  Distinct customers: {n_distinct}  Cache hit rate: {cache_hit_rate}%")

    out_path = Path("reports/api/load_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[load_test] Saved to {out_path}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--users", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.users, args.duration, args.seed))
