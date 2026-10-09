.PHONY: install lint format test test-all synthetic-bundle bundle api \
        docker-build docker-run compose-up ci-lint

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
api:
	BUNDLE_PATH=artifacts/bundle_week104 PYTHONPATH=. \
		$(PYTHON) -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t hm-recsys-api:latest .

# Run container with the real bundle mounted read-only
docker-run:
	docker run --rm -p 8000:8000 \
		-v $$(pwd)/artifacts/bundle_week104:/bundle:ro \
		-e BUNDLE_PATH=/bundle \
		hm-recsys-api:latest

compose-up:
	docker compose up

ci-lint:
	python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('YAML OK')"
	actionlint .github/workflows/ci.yml
