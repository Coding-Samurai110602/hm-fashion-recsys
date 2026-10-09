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

## Latency budgets — measured (local, real bundle, warm cache, 100 calls, 5 warmups)

| Endpoint group | Budget p50 | Budget p95 | Measured p50 | Measured p95 | Status |
|----------------|-----------|-----------|-------------|-------------|--------|
| Ops + insights | < 5 ms | < 50 ms | 1.9 ms | 3.9 ms | OK |
| Recommendations without reasons | < 150 ms | < 150 ms | 12 ms | 39 ms | OK |
| Recommendations with reasons | < 500 ms | < 700 ms | 17 ms | 47 ms | OK (LRU cache hit) |
| `/experiment/power` | < 5 ms | < 50 ms | 2.5 ms | 2.9 ms | OK |
| `/experiment/simulate` | < 500 ms | < 500 ms | 4.1 ms | 5.0 ms | OK |
| `/explain` | < 500 ms | < 700 ms | 1330 ms | 1839 ms | **MISS** |
| `/recommendations/custom` | < 500 ms | < 700 ms | 551 ms | 947 ms | **MISS** (p95) |

**Notes on misses:**
- `/explain`: per-call TreeSHAP on all ~200 candidates in the pool takes ~1.3 s on this hardware (MacBook Air, loaded). Session 6 measured 449ms on a lightly-loaded machine; 700ms budget is not achievable under concurrent load. The computation is O(200 × 68 features) and cannot be reduced without approximating SHAP.
- `/recommendations/custom`: builds the full candidate + feature pipeline for a new customer on each call (no LRU cache benefit); p95 exceeds 700ms under load. Cold-path inference is bounded by LightGBM predict on 200 candidates (~550ms).
- Recommendations **with reasons** appears fast (p50=17ms) because the LRU cache returns cached results for 99/100 warm calls. Cold-cache first call ≈ 450–700ms (TreeSHAP-dominated).

## Single-worker note

The API runs a single uvicorn worker. The bundle RSS is ~5.7 GB; running multiple workers would require that memory for each copy. Use a load balancer + single worker per pod in production.

## Interactive docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json (committed snapshot: `docs/openapi.json`)
