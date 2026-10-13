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

## Latency budgets — measured (local, real bundle, 30 calls per endpoint)

**Primary figures are cold-cache (distinct customers); warm-cache shown separately.**

| Endpoint group | Budget p95 | Cold p50 | Cold p95 | Warm p50 | Warm p95 | Status (cold) |
|----------------|-----------|---------|---------|---------|---------|--------------|
| Ops + insights | < 50 ms | 2 ms | 4 ms | 2 ms | 4 ms | PASS |
| Recommendations — no reasons | < 150 ms | 397 ms | 422 ms | 2 ms | 2 ms | **MISS** (cold) |
| Recommendations — with reasons | < 700 ms | 396 ms | 431 ms | 2 ms | 2 ms | PASS |
| `/explain` (after recommend) | < 700 ms | 3 ms | 3 ms | 3 ms | 3 ms | PASS |
| `/explain` (cold first call) | < 700 ms | ~430 ms | ~450 ms | — | — | PASS |
| `/recommendations/custom` | < 700 ms | ~430 ms | ~480 ms | — | — | PASS |
| `/experiment/power` | < 50 ms | 2 ms | 3 ms | — | — | PASS |
| `/experiment/simulate` | < 500 ms | 4 ms | 5 ms | — | — | PASS |

**Notes:**
- **No-reasons cold MISS**: candidate generation + feature assembly + LightGBM scoring takes ~400ms per new customer. The 150ms budget assumed a warm LRU cache. Warm (LRU hit) is 2ms — well within budget.
- **Explain improvement**: Previously 1330ms/1839ms (cold TreeSHAP, no cache). After this session: 3ms after a prior `recommend-with-reasons` call (SHAP cache hit); ~430ms if no prior recommend call (score cache reused, only SHAP runs).
- **Custom improvement**: Previously 551ms/947ms. Now ~430ms cold because the SHAP cache avoids redundant computation.
- **Load test (10 users, 60s, 1000 distinct customers)**: 561 requests, 0 errors (0.0%), throughput 9.1 req/s, p50=10ms, p95=4298ms, p99=4549ms. The high p95 reflects cold TreeSHAP calls dominating the tail; most requests (ops/insights endpoints) return in <10ms.

## Single-worker note

The API runs a single uvicorn worker. The bundle RSS is ~5.7 GB; running multiple workers would require that memory for each copy. Use a load balancer + single worker per pod in production.

## Interactive docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json (committed snapshot: `docs/openapi.json`)
