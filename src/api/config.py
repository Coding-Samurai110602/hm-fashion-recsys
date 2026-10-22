"""API configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


def _parse_cors(v: object) -> list[str]:
    """Accept comma-separated string or JSON array; pydantic_settings 2.6+ compat."""
    import json as _json
    from collections.abc import Iterable

    if isinstance(v, str):
        v = v.strip()
        if v.startswith("["):
            # JSON array: '["http://a","http://b"]'
            return _json.loads(v)
        # Comma-separated: "http://a,http://b"
        return [x.strip() for x in v.split(",") if x.strip()]
    if isinstance(v, Iterable):
        return [str(x) for x in v]
    return []


class Settings(BaseSettings):
    BUNDLE_PATH: str = "artifacts/bundle_week104"
    # Declare as str; split/validate via validator so pydantic_settings 2.6+
    # does not attempt json.loads() before the validator fires.
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    LOG_LEVEL: str = "INFO"
    CACHE_SIZE: int = 512
    MAX_CUSTOM_HISTORY: int = 200
    API_VERSION: str = "v1"
    GIT_COMMIT: str = "unknown"

    def get_cors_origins(self) -> list[str]:
        return _parse_cors(self.CORS_ORIGINS)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
