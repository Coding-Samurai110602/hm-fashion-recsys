"""Async load test: 10 concurrent users for 60 seconds on a realistic endpoint mix.

Targets the REAL bundle (must be running). Reports throughput, p50/p95/p99, error rate.
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


async def single_user(
    base_url: str,
    duration: float,
    sample_customer_idxs: list[str],
    results: list[dict],
    user_id: int,
):
    rng = random.Random(user_id * 1337)
    weights = [w for w, _ in ENDPOINT_MIX]
    paths = [p for _, p in ENDPOINT_MIX]

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        deadline = time.perf_counter() + duration
        while time.perf_counter() < deadline:
            path_template = rng.choices(paths, weights=weights)[0]
            cid = rng.choice(sample_customer_idxs)
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
            except Exception as e:
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

    # Get sample customers
    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        r = await client.get("/api/v1/customers/sample?n=20&segment=medium")
        if r.status_code != 200:
            print(f"[load_test] Failed to get sample customers: {r.status_code}")
            sample_cidxs = list(range(0, 20))
        else:
            sample_cidxs = [item["customer_id"] for item in r.json()] or []

    print(f"[load_test] Sample customers: {sample_cidxs[:3]} ...", flush=True)

    all_results: list[dict] = []
    tasks = [
        single_user(base_url, duration, sample_cidxs, all_results, i)
        for i in range(n_users)
    ]

    t_start = time.perf_counter()
    await asyncio.gather(*tasks)
    elapsed_total = time.perf_counter() - t_start

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
