# H&M Recommendation System — API Reference

## Run locally

```bash
# 1. Build the serving bundle (requires Kaggle data in data/processed/)
make bundle        # → artifacts/bundle_week104/

# 2. Start the API
make api           # → http://localhost:8000

# or with Docker Compose (mounts the real bundle)
docker compose up
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BUNDLE_PATH` | `artifacts/bundle_week104` | Path to the serving bundle directory |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated allowed origins |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `CACHE_SIZE` | `512` | Max cached recommendation responses (LRU) |
| `MAX_CUSTOM_HISTORY` | `200` | Max items in `/recommendations/custom` body |

Copy `.env.example` to `.env` to override defaults.

## Endpoint overview

All endpoints are versioned under `/api/v1`. Operational endpoints are at root.

### Ops

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe (always 200) |
| GET | `/ready` | Readiness probe (503 until bundle loaded) |
| GET | `/version` | Git commit, bundle metadata, model MAP@12 |
| GET | `/metrics` | Prometheus metrics + LRU cache counters |

### Customers & Recommendations

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/customers/sample` | Sample demo customers by segment |
| GET | `/api/v1/customers/{customer_id}` | Profile: age bucket, purchase counts, recent history |
| GET | `/api/v1/customers/{customer_id}/recommendations` | Top-k ranked items with SHAP reason codes |
| GET | `/api/v1/customers/{customer_id}/baseline` | Heuristic top-k (for side-by-side comparison) |
| GET | `/api/v1/customers/{customer_id}/actual-purchases` | Week-104 purchases (display only, clearly labelled) |
| GET | `/api/v1/customers/{customer_id}/funnel` | Full candidate funnel: source, score, final rank |
| GET | `/api/v1/customers/{customer_id}/explain/{article_id}` | Contrast-SHAP breakdown for one item |
| POST | `/api/v1/recommendations/custom` | Recommendations for a hand-built history |

### Catalog

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/articles/search` | Search by name / product type |
| GET | `/api/v1/articles/{article_id}` | Article metadata |
| GET | `/api/v1/articles/{article_id}/image` | Local JPEG or placeholder PNG |

### Insights

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/insights/summary` | Headline metrics incl. week-104 holdout |
| GET | `/api/v1/insights/weekly-lift` | Rolling-origin MAP@12 by training week |
| GET | `/api/v1/insights/segments` | 21-segment analysis with Holm-corrected p-values |
| GET | `/api/v1/insights/ablations` | Feature group ablation deltas |
| GET | `/api/v1/insights/importance` | SHAP vs gain importance comparison |

### Experiment

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/experiment/power` | Sample size & power for an A/B test |
| POST | `/api/v1/experiment/simulate` | Run vectorized A/B or A/A simulation |

## Error format

All errors return a consistent envelope:

```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Customer '12345' not found",
    "request_id": "a3f92b1c"
  }
}
```

HTTP status codes: 404 unknown ID, 422 validation, 503 bundle not loaded.

## Input limits

| Parameter | Limit |
|-----------|-------|
| `k` (recommendations) | 1–50 |
| `n_sims` (simulate) | 1–2000 |
| `history` length (custom) | 1–`MAX_CUSTOM_HISTORY` (200) |
| `t_dat` (custom history) | must be ≤ 2020-09-22 (bundle cutoff) |

## Latency budgets — measured (local, real bundle, HTTP over loopback)

**Primary figures are cold-cache (distinct customers, fresh customer → cache miss); warm-cache (same customer repeated → LRU hit) shown separately. Stage breakdown from direct BundleRecommender timing (profiling_stages.json).**

| Endpoint group | Budget p95 | Cold p50 | Cold p95 | Warm p50 | Warm p95 | Status (cold) |
|----------------|-----------|---------|---------|---------|---------|--------------|
| Ops + insights | < 50 ms | 1 ms | 1 ms | 1 ms | 1 ms | PASS |
| Recommendations — no reasons | < 150 ms | 56 ms | 58 ms | 1 ms | 2 ms | PASS |
| Recommendations — with reasons | < 700 ms | 409 ms | 480 ms | 1 ms | 2 ms | PASS |
| `/explain` (after recommend) | < 700 ms | 2 ms | 3 ms | — | — | PASS |
| `/explain` (cold first call) | < 700 ms | 364 ms | 408 ms | — | — | PASS |
| `/recommendations/custom` | < 700 ms | 427 ms | 458 ms | — | — | PASS |
| `/experiment/power` | < 50 ms | 2 ms | 2 ms | — | — | PASS |
| `/experiment/simulate` (50 sims) | < 500 ms | 223 ms | 225 ms | — | — | PASS |

**Stage breakdown (no-reasons cold, 20 customers, seed=42):**

| Stage | Mean | p95 |
|-------|------|-----|
| candidate_gen | 9 ms | 11 ms |
| feature_assembly | 38 ms | 41 ms |
| lgbm_predict | 1.9 ms | 2.1 ms |
| treeshap | 342 ms | 369 ms |
| reason_format | 0.1 ms | 0.3 ms |
| **no_reasons total** | **49 ms** | **52 ms** |
| **with_reasons total** | **391 ms** | **419 ms** |

TreeSHAP with n_jobs=4 (saved in model params): 299 ms per call (200 candidates × 68 features × 421 trees). Single-threaded: 1118 ms — would exceed the 700 ms budget without the 3.7× parallelism gain.

**Capacity math — executor: ThreadPoolExecutor(max_workers=4), os.cpu_count()=8:**
- Cold no-reasons HTTP p50: 56 ms → 4 workers / 56 ms ≈ **71 cold req/s** (executor-bound)
- Cold with-reasons HTTP p50: 409 ms → 4 workers / 409 ms ≈ **10 cold req/s** (executor-bound)
  - SHAP uses n_jobs=4 per call; at 4 workers: 16 SHAP threads on 8 cores → 2× oversubscription reduces CPU capacity to ~5.6 cold with-reasons req/s during full SHAP saturation
- Warm (LRU hit) service time: ~2 ms → ~**2000 req/s** effective with warm cache
- HTTP overhead vs direct: no-reasons p50 56ms HTTP vs 49ms direct → **~7ms overhead** (loopback + FastAPI + JSON)

**Load test (mixed endpoint traffic, 1000 distinct seeded customers, 30s per profile):**

| Profile | Requests | Throughput | p50 | p95 | p99 | Errors |
|---------|---------|-----------|-----|-----|-----|--------|
| 1 user | 2,841 | 94 req/s | 1 ms | 6 ms | 437 ms | 0 |
| 3 users | 5,125 | 167 req/s | 3 ms | 31 ms | 588 ms | 0 |
| 10 users | 5,147 | 166 req/s | 10 ms | 184 ms | 1375 ms | 0 |

Cache hit rate (cumulative, all profiles): **81.9%** (6435 hits / 7855 total). High hit rate because the load test runs three profiles back-to-back against the same 1,000-customer pool: the first profile populates the LRU cache with the most-recently-requested customers (up to the cache size of 512), and subsequent profiles re-request many of the same customers, generating hits. 1,000 > 512, so not all customers fit simultaneously; the high rate comes from repeat visits across the three profiles rather than all customers fitting at once. Tail latency (p99=437ms for 1 user, p99=1375ms for 10 users) comes from cold with-reasons calls (409ms service time) queueing behind other cold calls.

**Notes:**
- Cold no-reasons: candidate_gen (9ms) + feature_assembly (38ms) + predict (2ms) = 49ms total, well under 150ms.
- **Explain after recommend**: 3ms (SHAP cache hit). Cold first call: 430ms (SHAP only; score cache reused, no candidate gen).
- **Custom recs**: 398ms p50, 448ms p95 — includes full candidate gen + feature assembly + TreeSHAP.

## Single-worker note

The API runs a single uvicorn worker. The bundle RSS is ~5.7 GB; running multiple workers would require that memory for each copy. Use a load balancer + single worker per pod in production.

## Interactive docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json (committed snapshot: `docs/openapi.json`)
