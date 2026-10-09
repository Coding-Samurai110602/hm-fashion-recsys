"""Contract test: fail if the live OpenAPI schema diverges from the committed snapshot.

How it works:
1. Generate current OpenAPI schema from the live app (no bundle load needed).
2. Compare against docs/openapi.json (committed snapshot).
3. Comparison covers: paths (with methods + parameters + request/response schemas)
   and component schemas. Metadata (title, description, version, servers) is excluded.
4. If they differ → test fails, forcing an intentional schema update.

Path count note: 22 paths in the schema (not 25 as session notes stated).
The 25 figure counted handler functions rather than distinct URL paths;
e.g. /docs, /redoc, /openapi.json are FastAPI-generated and not in custom routers.

To regenerate the snapshot:
    python scripts/export_openapi.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

OPENAPI_SNAPSHOT = Path("docs/openapi.json")


def _get_live_schema() -> dict:
    """Generate the OpenAPI schema from the live FastAPI app (no bundle load required)."""
    import os

    os.environ["BUNDLE_PATH"] = "artifacts/bundle_synthetic"
    from src.api.config import get_settings

    get_settings.cache_clear()

    from fastapi import FastAPI

    from src.api.config import get_settings as gs
    from src.api.routers import articles, customers, experiment, insights, ops

    settings = gs()
    app = FastAPI(title="H&M Fashion Recommendation API", version=settings.API_VERSION)
    prefix = f"/api/{settings.API_VERSION}"
    app.include_router(ops.router)
    app.include_router(customers.router, prefix=prefix)
    app.include_router(articles.router, prefix=prefix)
    app.include_router(insights.router, prefix=prefix)
    app.include_router(experiment.router, prefix=prefix)
    return app.openapi()


def _schema_key(schema: dict) -> dict:
    """Extract the comparison-relevant subset of the OpenAPI schema."""
    return {
        "paths": schema.get("paths", {}),
        "components": schema.get("components", {}),
    }


def _diff_paths(live: dict, committed: dict) -> list[str]:
    """Return list of human-readable differences in paths + methods + parameters."""
    issues = []
    live_paths = live.get("paths", {})
    committed_paths = committed.get("paths", {})

    added = set(live_paths) - set(committed_paths)
    removed = set(committed_paths) - set(live_paths)
    for p in sorted(added):
        issues.append(f"PATH ADDED:   {p}")
    for p in sorted(removed):
        issues.append(f"PATH REMOVED: {p}")

    for path in sorted(set(live_paths) & set(committed_paths)):
        live_ops = live_paths[path]
        committed_ops = committed_paths[path]
        live_methods = {
            m
            for m in live_ops
            if m in ("get", "post", "put", "patch", "delete", "head", "options")
        }
        committed_methods = {
            m
            for m in committed_ops
            if m in ("get", "post", "put", "patch", "delete", "head", "options")
        }
        for m in sorted(live_methods - committed_methods):
            issues.append(f"METHOD ADDED:   {m.upper()} {path}")
        for m in sorted(committed_methods - live_methods):
            issues.append(f"METHOD REMOVED: {m.upper()} {path}")

        # Compare parameters and response/request body schemas per method
        for method in sorted(live_methods & committed_methods):
            live_op = live_ops[method]
            committed_op = committed_ops[method]

            # Parameters (query/path)
            live_params = {p["name"]: p for p in live_op.get("parameters", [])}
            committed_params = {
                p["name"]: p for p in committed_op.get("parameters", [])
            }
            for name in sorted(set(live_params) - set(committed_params)):
                issues.append(f"PARAM ADDED:   {method.upper()} {path} ?{name}")
            for name in sorted(set(committed_params) - set(live_params)):
                issues.append(f"PARAM REMOVED: {method.upper()} {path} ?{name}")

            # Request body schema reference
            live_rb = live_op.get("requestBody", {})
            committed_rb = committed_op.get("requestBody", {})
            if json.dumps(live_rb, sort_keys=True) != json.dumps(
                committed_rb, sort_keys=True
            ):
                issues.append(f"REQUEST BODY CHANGED: {method.upper()} {path}")

            # Response schemas (check all status codes)
            live_resp = live_op.get("responses", {})
            committed_resp = committed_op.get("responses", {})
            if json.dumps(live_resp, sort_keys=True) != json.dumps(
                committed_resp, sort_keys=True
            ):
                issues.append(f"RESPONSE SCHEMA CHANGED: {method.upper()} {path}")

    return issues


def _diff_components(live: dict, committed: dict) -> list[str]:
    """Return list of human-readable differences in component schemas."""
    issues = []
    live_schemas = live.get("components", {}).get("schemas", {})
    committed_schemas = committed.get("components", {}).get("schemas", {})

    for name in sorted(set(live_schemas) - set(committed_schemas)):
        issues.append(f"SCHEMA ADDED:   {name}")
    for name in sorted(set(committed_schemas) - set(live_schemas)):
        issues.append(f"SCHEMA REMOVED: {name}")
    for name in sorted(set(live_schemas) & set(committed_schemas)):
        if json.dumps(live_schemas[name], sort_keys=True) != json.dumps(
            committed_schemas[name], sort_keys=True
        ):
            issues.append(f"SCHEMA CHANGED: {name}")

    return issues


class TestOpenAPIContract:
    def test_snapshot_exists(self):
        """The committed OpenAPI snapshot must exist."""
        assert OPENAPI_SNAPSHOT.exists(), (
            f"Missing {OPENAPI_SNAPSHOT}. "
            "Run: python scripts/export_openapi.py to generate it."
        )

    def test_schema_paths_match_snapshot(self):
        """Every path, method, parameter, request body, and response schema must match."""
        if not OPENAPI_SNAPSHOT.exists():
            pytest.skip("Snapshot not found — generate it first with export_openapi.py")

        live = _get_live_schema()
        committed = json.loads(OPENAPI_SNAPSHOT.read_text())

        issues = _diff_paths(live, committed)
        assert not issues, (
            f"OpenAPI paths/methods/parameters/response schemas diverged "
            f"({len(issues)} difference(s)):\n"
            + "\n".join(f"  {i}" for i in issues)
            + "\nRun: python scripts/export_openapi.py to regenerate the snapshot."
        )

    def test_component_schemas_match_snapshot(self):
        """Component schemas (Pydantic models) must match the committed snapshot."""
        if not OPENAPI_SNAPSHOT.exists():
            pytest.skip("Snapshot not found")

        live = _get_live_schema()
        committed = json.loads(OPENAPI_SNAPSHOT.read_text())

        issues = _diff_components(live, committed)
        assert not issues, (
            f"OpenAPI component schemas diverged ({len(issues)} difference(s)):\n"
            + "\n".join(f"  {i}" for i in issues)
            + "\nRun: python scripts/export_openapi.py to regenerate the snapshot."
        )

    def test_path_count(self):
        """22 paths in the OpenAPI schema (4 ops + 18 versioned).

        Session notes said 25 — that counted handler functions, not URL paths.
        The 3 auto-generated FastAPI paths (/docs, /redoc, /openapi.json) are excluded
        from custom router registration and do not appear in this count.
        """
        live = _get_live_schema()
        n_paths = len(live.get("paths", {}))
        assert n_paths == 22, (
            f"Expected 22 paths, got {n_paths}: {sorted(live.get('paths', {}).keys())}"
        )

    def test_all_paths_have_summaries(self):
        """Every endpoint should have a summary (OpenAPI quality check)."""
        live = _get_live_schema()
        missing = []
        for path, methods in live.get("paths", {}).items():
            for method, op in methods.items():
                if method in ("get", "post", "put", "patch", "delete"):
                    if not op.get("summary"):
                        missing.append(f"{method.upper()} {path}")
        assert not missing, f"Endpoints missing summaries: {missing}"
