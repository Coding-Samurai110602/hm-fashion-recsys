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
# Customer + recommendation + explain smoke test
# ---------------------------------------------------------------------------
echo "=== /api/v1/customers/sample?n=1 ==="
SAMPLE_JSON=$(curl -sf "http://localhost:8000/api/v1/customers/sample?n=1")
echo "$SAMPLE_JSON" | python3 -m json.tool
CUSTOMER_ID=$(echo "$SAMPLE_JSON" | python3 -c \
    "import sys, json; print(json.load(sys.stdin)[0]['customer_id'])")
echo "Selected customer_id: $CUSTOMER_ID"

echo "=== recommendations (include_reasons=false) ==="
RECS_NO=$(curl -sf \
    "http://localhost:8000/api/v1/customers/${CUSTOMER_ID}/recommendations?include_reasons=false&k=12")
echo "$RECS_NO" | python3 -m json.tool
N_UNIQUE_NO=$(echo "$RECS_NO" | python3 -c \
    "import sys, json; d=json.load(sys.stdin); print(len({x['article_id'] for x in d}))")
if [ "$N_UNIQUE_NO" -ne 12 ]; then
    echo "FAIL: recommendations (no reasons): expected 12 unique article_ids, got $N_UNIQUE_NO"
    exit 1
fi
echo "PASS: $N_UNIQUE_NO unique article_ids (include_reasons=false)"

echo "=== recommendations (include_reasons=true) ==="
RECS_YES=$(curl -sf \
    "http://localhost:8000/api/v1/customers/${CUSTOMER_ID}/recommendations?include_reasons=true&k=12")
echo "$RECS_YES" | python3 -m json.tool
N_UNIQUE_YES=$(echo "$RECS_YES" | python3 -c \
    "import sys, json; d=json.load(sys.stdin); print(len({x['article_id'] for x in d}))")
if [ "$N_UNIQUE_YES" -ne 12 ]; then
    echo "FAIL: recommendations (with reasons): expected 12 unique article_ids, got $N_UNIQUE_YES"
    exit 1
fi
echo "PASS: $N_UNIQUE_YES unique article_ids (include_reasons=true)"

ARTICLE_ID=$(echo "$RECS_NO" | python3 -c \
    "import sys, json; print(json.load(sys.stdin)[0]['article_id'])")
echo "=== explain for article $ARTICLE_ID ==="
curl -sf "http://localhost:8000/api/v1/customers/${CUSTOMER_ID}/explain/${ARTICLE_ID}" \
    | python3 -m json.tool
echo "PASS: explain HTTP 200"

# ---------------------------------------------------------------------------
# Report image size (docker format uses {{}} which must live in a .sh, not
# inline YAML, because ': ' inside the awk format string trips the YAML parser)
# ---------------------------------------------------------------------------
docker image inspect hm-recsys-api:ci --format '{{.Size}}' \
    | awk '{printf "Image size: %.1f MB\n", $1/1048576}'
