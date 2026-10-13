"""Load test against the real bundle API.

Runs three profiles (1, 3, 10 concurrent users) for --duration seconds each.
Reports throughput, p50/p95/p99, error rate, cache hit rate, and capacity math.
Saves all three profiles to reports/api/load_test.json.

Usage:
    # Start API first: make api
    PYTHONPATH=. python scripts/load_test.py [--url http://localhost:8000] [--duration 30]
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
N_DISTINCT_CUSTOMERS = 1000

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
) -> None:
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


def _fetch_cache_stats(base_url: str) -> tuple[int, int]:
    """Fetch cache hit/miss counts from Prometheus /metrics.  Raises on failure."""
    import httpx as _httpx

    with _httpx.Client(base_url=base_url, timeout=5.0) as client:
        cr = client.get("/api/v1/metrics")

    if cr.status_code != 200:
        raise RuntimeError(
            f"[load_test] /metrics returned HTTP {cr.status_code}: {cr.text[:200]}"
        )

    hits = misses = 0
    found_hits = found_misses = False
    for line in cr.text.splitlines():
        if line.startswith("recommendation_cache_hits_total "):
            hits = int(float(line.split()[-1]))
            found_hits = True
        elif line.startswith("recommendation_cache_misses_total "):
            misses = int(float(line.split()[-1]))
            found_misses = True

    if not found_hits or not found_misses:
        raise RuntimeError(
            "[load_test] Could not find recommendation_cache_hits_total / "
            "recommendation_cache_misses_total in /metrics output.\n"
            f"Got:\n{cr.text[:500]}"
        )

    return hits, misses


async def run_profile(
    base_url: str, n_users: int, duration: float, seed: int,
    sample_cidxs: list[str],
) -> dict:
    """Run one load-test profile and return its summary dict."""
    print(f"\n[load_test] ── profile: {n_users} users × {duration}s ──", flush=True)
    n_distinct = len(set(sample_cidxs))

    all_results: list[dict] = []
    tasks = [
        single_user(base_url, duration, sample_cidxs, all_results, i)
        for i in range(n_users)
    ]

    t_start = time.perf_counter()
    await asyncio.gather(*tasks)
    elapsed_total = time.perf_counter() - t_start

    times = sorted(r["elapsed_ms"] for r in all_results)
    errors = [r for r in all_results if r["error"]]
    n_total = len(all_results)
    n_errors = len(errors)
    error_rate = n_errors / n_total if n_total > 0 else 0.0

    p50 = statistics.median(times) if times else 0
    p95 = times[int(0.95 * len(times))] if times else 0
    p99 = times[int(0.99 * len(times))] if times else 0
    throughput = n_total / elapsed_total

    print(
        f"  Requests: {n_total} total, {n_errors} errors ({error_rate*100:.1f}%)",
        flush=True,
    )
    print(f"  Throughput: {throughput:.1f} req/s", flush=True)
    print(
        f"  Latency: p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms",
        flush=True,
    )

    return {
        "n_users": n_users,
        "duration_s": round(elapsed_total, 1),
        "n_requests": n_total,
        "n_errors": n_errors,
        "error_rate_pct": round(error_rate * 100, 2),
        "throughput_rps": round(throughput, 1),
        "n_distinct_customers": n_distinct,
        "latency_ms": {
            "p50": round(p50, 1),
            "p95": round(p95, 1),
            "p99": round(p99, 1),
            "mean": round(statistics.mean(times) if times else 0, 1),
        },
    }


async def main(base_url: str, duration: float, seed: int, profiles: list[int]) -> None:
    print(
        f"[load_test] target={base_url}  duration={duration}s  profiles={profiles}",
        flush=True,
    )

    sample_cidxs = _load_real_customer_ids(seed=seed, n=N_DISTINCT_CUSTOMERS)
    if not sample_cidxs:
        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            r = await client.get("/api/v1/customers/sample?n=20&segment=medium")
            sample_cidxs = (
                [item["customer_id"] for item in r.json()]
                if r.status_code == 200
                else []
            )

    n_distinct = len(set(sample_cidxs))
    print(
        f"[load_test] {n_distinct} distinct customers (seed={seed}): "
        f"{sample_cidxs[0][:16]}...",
        flush=True,
    )

    profile_results = []
    for n_users in profiles:
        summary = await run_profile(base_url, n_users, duration, seed, sample_cidxs)
        profile_results.append(summary)

    # Fetch cache stats once after all profiles complete (cumulative)
    hits, misses = _fetch_cache_stats(base_url)
    total = hits + misses
    cache_hit_rate = round(100.0 * hits / total, 1) if total > 0 else 0.0
    print(
        f"\n[load_test] Cache (cumulative): {hits} hits, {misses} misses"
        f" ({cache_hit_rate}% hit rate)",
        flush=True,
    )

    report = {
        "seed": seed,
        "note": (
            "Three profiles (1, 3, 10 users). 1000+ distinct customers → "
            "most requests are cache misses (cold path). "
            "Cache hit rate is cumulative across all profiles."
        ),
        "executor_config": "ThreadPoolExecutor(max_workers=4) in src/api/deps.py",
        "cache_hit_rate_pct": cache_hit_rate,
        "cache_hits": hits,
        "cache_misses": misses,
        "profiles": profile_results,
    }

    out_path = Path("reports/api/load_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[load_test] Saved to {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--profiles",
        nargs="+",
        type=int,
        default=[1, 3, 10],
        help="Number of concurrent users for each profile (default: 1 3 10)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.url, args.duration, args.seed, args.profiles))
