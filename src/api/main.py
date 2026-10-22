"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import get_settings
from src.api.deps import load_bundle_at_startup
from src.api.errors import http_exception_handler, validation_exception_handler
from src.api.middleware import RequestMiddleware
from src.api.routers import (
    articles,
    catalog,
    customers,
    demo,
    experiment,
    insights,
    ops,
    profiles,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.api.db import get_db_path

    _logger = logging.getLogger("api")
    settings = get_settings()
    db_path = get_db_path()
    _logger.info("Profiles DB path: %s", db_path)
    load_bundle_at_startup(settings)
    yield
    _logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="H&M Fashion Recommendation API",
        description=(
            "Production-grade API for personalized fashion recommendations. "
            "Serves LightGBM LambdaRank scores with SHAP-derived reason codes, "
            "NDCG/MAP@12 evaluation results, and A/B experiment utilities.\n\n"
            "**Single-worker deployment**: the bundle RSS is ~5.7 GB; multiple workers "
            "would each load a full copy. Use a load balancer + single uvicorn worker."
        ),
        version=settings.API_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestMiddleware)

    # Exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Routers
    prefix = f"/api/{settings.API_VERSION}"
    app.include_router(ops.router)  # no prefix (health/ready at root)
    app.include_router(customers.router, prefix=prefix)
    app.include_router(articles.router, prefix=prefix)
    app.include_router(insights.router, prefix=prefix)
    app.include_router(experiment.router, prefix=prefix)
    app.include_router(
        profiles.router, prefix=prefix
    )  # /api/v1/profiles + /api/v1/compare
    app.include_router(
        catalog.router, prefix=f"{prefix}/catalog"
    )  # /api/v1/catalog/...
    app.include_router(demo.router, prefix=f"{prefix}/demo")  # /api/v1/demo/...

    return app


app = create_app()
