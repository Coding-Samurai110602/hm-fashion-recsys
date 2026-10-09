# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# libgomp1 is required by LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# libgomp1 needed at runtime by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ src/
COPY scripts/make_synthetic_bundle.py scripts/

# Synthetic bundle is generated at startup if BUNDLE_PATH points to a missing dir
# (CI smoke test path). In production the real bundle is mounted read-only via volume.
COPY artifacts/bundle_synthetic/ artifacts/bundle_synthetic/

# Ownership
RUN chown -R appuser:appuser /app

USER appuser

# Bundle path is overridden by the env var or docker-compose volume mount
ENV BUNDLE_PATH=/bundle \
    LOG_LEVEL=INFO \
    PYTHONPATH=/app

# Healthcheck: poll /ready every 30s; give 120s for bundle load at startup
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')" || exit 1

EXPOSE 8000

# Single worker: the bundle RSS is ~5.7 GB; multiple workers = OOM.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
