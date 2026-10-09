"""Export the OpenAPI schema to docs/openapi.json.

Run this whenever the API contract changes intentionally, then commit the result.
The contract test (tests/api/test_contract.py) fails if the live schema diverges.

Usage:
    python scripts/export_openapi.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("BUNDLE_PATH", "artifacts/bundle_synthetic")

from src.api.config import get_settings
get_settings.cache_clear()
from src.api.main import create_app

# Build schema without loading the bundle (just routes)
from fastapi import FastAPI
from src.api.routers import customers, articles, insights, experiment, ops
from src.api.config import get_settings as gs

settings = gs()
prefix = f"/api/{settings.API_VERSION}"
app = FastAPI(
    title="H&M Fashion Recommendation API",
    description="Personalized fashion recommendations via LightGBM LambdaRank.",
    version=settings.API_VERSION,
)
app.include_router(ops.router)
app.include_router(customers.router, prefix=prefix)
app.include_router(articles.router, prefix=prefix)
app.include_router(insights.router, prefix=prefix)
app.include_router(experiment.router, prefix=prefix)

schema = app.openapi()
out = Path("docs/openapi.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(schema, indent=2))
n_paths = len(schema.get("paths", {}))
print(f"[export_openapi] Written {n_paths} paths to {out}")
