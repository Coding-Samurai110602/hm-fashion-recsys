#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Wait for the API container to report /ready (up to 120 s)
# ---------------------------------------------------------------------------
echo "Waiting for /ready..."
for i in $(seq 1 60); do
    STATUS=$(curl -sf http://localhost:8000/ready 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('ready','false'))" 2>/dev/null \
        || echo "false")
    if [ "$STATUS" = "True" ]; then
        echo "/ready returned true after ${i}s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "Timed out waiting for /ready"
        docker logs hm-api
        exit 1
    fi
    sleep 2
done

# ---------------------------------------------------------------------------
# Smoke-test key endpoints
# ---------------------------------------------------------------------------
echo "=== /health ==="
curl -sf http://localhost:8000/health | python3 -m json.tool

echo "=== /version ==="
curl -sf http://localhost:8000/version | python3 -m json.tool

echo "=== /api/v1/insights/summary ==="
curl -sf http://localhost:8000/api/v1/insights/summary | python3 -m json.tool

# ---------------------------------------------------------------------------
# Report image size (docker format uses {{}} which must live in a .sh, not
# inline YAML, because ': ' inside the awk format string trips the YAML parser)
# ---------------------------------------------------------------------------
docker image inspect hm-recsys-api:ci --format '{{.Size}}' \
    | awk '{printf "Image size: %.1f MB\n", $1/1048576}'
