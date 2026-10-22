.PHONY: install lint format test test-all synthetic-bundle bundle api \
        docker-build docker-web-build docker-run compose-up ci-lint ci-test ci-docker-test \
        frontend-install frontend-dev frontend-build frontend-lint \
        frontend-test frontend-check-types frontend-all \
        e2e e2e-compose e2e-real

PYTHON = .venv/bin/python
PIP    = .venv/bin/pip

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-api.txt

lint:
	$(PYTHON) -m ruff check src/api/ tests/api/

format:
	$(PYTHON) -m ruff format src/api/ tests/api/

# Run fast tests only (no real bundle, no slow tests)
test:
	BUNDLE_PATH=artifacts/bundle_synthetic PYTHONPATH=. $(PYTHON) -m pytest \
		-q -m "not integration_real and not slow" --tb=short tests/

# Run all tests including real-bundle integration tests
test-all:
	PYTHONPATH=. $(PYTHON) -m pytest -q --tb=short tests/

# Build the synthetic CI bundle (0.2 MB, no real H&M data)
synthetic-bundle:
	PYTHONPATH=. $(PYTHON) scripts/make_synthetic_bundle.py

# Export the real serving bundle from the full processed data
bundle:
	caffeinate -i PYTHONPATH=. $(PYTHON) -u scripts/export_bundle.py

# Start the API locally with uvicorn (dev mode, real bundle)
# --reload-dir src prevents node_modules changes from restarting the API process
api:
	BUNDLE_PATH=artifacts/bundle_week104 PYTHONPATH=. \
		$(PYTHON) -m uvicorn src.api.main:app --reload --reload-dir src --host 0.0.0.0 --port 8000

docker-build:
	docker build -t hm-recsys-api:latest .

# Build the frontend web image
docker-web-build:
	docker build -t hm-recsys-web:latest frontend/

# Run container with the real bundle mounted read-only
docker-run:
	docker run --rm -p 8000:8000 \
		-v $$(pwd)/artifacts/bundle_week104:/bundle:ro \
		-e BUNDLE_PATH=/bundle \
		hm-recsys-api:latest

# Bring up the full stack (api + web) — http://localhost:8080
compose-up:
	docker compose up

ci-lint:
	python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('ci.yml YAML OK')"
	python3 -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml')); print('release.yml YAML OK')"
	actionlint .github/workflows/ci.yml .github/workflows/release.yml

# Run the exact CI pytest command locally with an empty data dir so
# requires_data tests are skipped — mirrors what GitHub Actions does.
ci-test:
	$(eval TMP := $(shell mktemp -d))
	HM_DATA_DIR=$(TMP) BUNDLE_PATH=artifacts/bundle_synthetic PYTHONPATH=. \
		$(PYTHON) -m pytest -q \
		-m "not integration_real and not slow and not requires_data" \
		--tb=short tests/
	rm -rf $(TMP)

# Run the CI test command inside a clean python:3.12-slim container with
# only the repo mounted (no data/, models/, or real bundle).
ci-docker-test:
	docker run --rm \
		-v $$(pwd):/repo:ro \
		-w /repo \
		-e BUNDLE_PATH=/repo/artifacts/bundle_synthetic \
		-e HM_DATA_DIR=/tmp/no_data \
		python:3.12-slim \
		sh -c "apt-get update -qq && apt-get install -y -q libgomp1 && \
			pip install -q -r requirements.txt -r requirements-api.txt 'pytest>=8.0.0' 'statsmodels==0.14.4' && \
			PYTHONPATH=. python -m pytest -q \
			-m 'not integration_real and not slow and not requires_data' \
			--tb=short tests/"

frontend-install:
	cd frontend && npm ci

frontend-dev:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

frontend-lint:
	cd frontend && npm run lint && npm run typecheck

frontend-test:
	cd frontend && npm run test -- --run

frontend-check-types:
	cd frontend && npm run check-types-fresh

frontend-all: frontend-lint frontend-test frontend-build

# Run e2e tests against the Vite dev server (requires both servers running)
# Starts the API with a temp DB to keep profiles.db clean.
e2e:
	SQLITE_DB_PATH=/tmp/profiles_e2e.db BUNDLE_PATH=artifacts/bundle_synthetic PYTHONPATH=. \
		$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
	sleep 3 && cd frontend && npm run e2e; \
	kill %1 2>/dev/null; true

# Run e2e tests against the containerised stack (synthetic bundle, port 8080)
# Use BUNDLE_DIR=./artifacts/bundle_week104 for the real bundle.
e2e-compose:
	BUNDLE_DIR=./artifacts/bundle_synthetic SQLITE_DB_PATH=/tmp/profiles_e2e.db \
		docker compose up -d
	@echo "Waiting for http://localhost:8080/ ..."; \
	for i in $$(seq 1 40); do \
		curl -sf http://localhost:8080/ > /dev/null 2>&1 && break; \
		echo "  $$i/40"; sleep 5; \
	done
	cd frontend && BASE_URL=http://localhost:8080 npm run e2e; STATUS=$$?; \
	docker compose down -v; exit $$STATUS

# Run e2e tests against the real bundle stack (requires artifacts/bundle_week104)
e2e-real:
	docker compose up -d
	@echo "Waiting for http://localhost:8080/ ..."; \
	for i in $$(seq 1 40); do \
		curl -sf http://localhost:8080/ > /dev/null 2>&1 && break; \
		echo "  $$i/40"; sleep 5; \
	done
	cd frontend && BASE_URL=http://localhost:8080 npm run e2e; STATUS=$$?; \
	docker compose down; exit $$STATUS
