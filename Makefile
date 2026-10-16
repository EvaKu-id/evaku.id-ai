.PHONY: install install-all lint task-ids format typecheck security audit secrets-scan test coverage check \
	test-integration api openapi up dev down tippecanoe

TIPPECANOE_TAG := 2.79.0
TIPPECANOE_COMMIT := 68ab8dcc229f95b8b25877697d5e8d66783af503
TIPPECANOE_DIR := data/tools/tippecanoe-$(TIPPECANOE_TAG)
TIPPECANOE_SRC := data/tools/src/tippecanoe-$(TIPPECANOE_TAG)

install:
	uv sync --extra api --extra geo --group dev

install-all:
	uv sync --all-extras --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run lint-imports
	@$(MAKE) --no-print-directory task-ids

# Task IDs belong in md/TASK.MD only.
task-ids:
	@if grep -rnIE --exclude-dir=__pycache__ '\b(SET|KON|INF|DAT|ENG|PIP|API|PRK|RAD|QA|OPS|DEM|BFF|DB|TG|UI)-[0-9]+' \
		src tests scripts config .github Dockerfile compose.yaml compose.dev.yaml pyproject.toml; then \
		echo "task IDs found in code"; exit 1; fi

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

# Security lint (ruff S rules) runs inside `lint`; this adds dependency and secret scanning.
security: audit secrets-scan

audit:
	@req=$$(mktemp); \
	uv export --frozen --all-extras --no-emit-project --format requirements-txt -o $$req >/dev/null && \
	uv run pip-audit --disable-pip -r $$req; status=$$?; rm -f $$req; exit $$status

secrets-scan:
	@if command -v gitleaks >/dev/null 2>&1; then \
		gitleaks dir . --config .gitleaks.toml --redact; \
	else \
		echo "gitleaks not installed: secret scan skipped (runs in pre-commit and CI)"; \
	fi

test:
	uv run pytest

coverage:
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck security coverage

test-integration:
	docker compose --profile db up -d db
	uv run pytest -m integration

api:
	uv run uvicorn evaku_ai.api.app:app --reload --port 8000

openapi:
	uv run python scripts/export_openapi.py

up:
	docker compose up --build api

dev:
	docker compose -f compose.yaml -f compose.dev.yaml up --build --watch api

down:
	docker compose --profile db --profile pipeline down

# Built from a pinned commit; data/ is git-ignored.
tippecanoe:
	@if [ -x $(TIPPECANOE_DIR)/tippecanoe ]; then echo "tippecanoe $(TIPPECANOE_TAG) already built"; exit 0; fi; \
	set -e; \
	if [ ! -d $(TIPPECANOE_SRC) ]; then \
		git clone --quiet --depth 1 --branch $(TIPPECANOE_TAG) https://github.com/felt/tippecanoe.git $(TIPPECANOE_SRC); \
	fi; \
	head=$$(git -C $(TIPPECANOE_SRC) rev-parse HEAD); \
	if [ "$$head" != "$(TIPPECANOE_COMMIT)" ]; then \
		echo "tippecanoe $(TIPPECANOE_TAG) is at $$head, expected $(TIPPECANOE_COMMIT)"; exit 1; \
	fi; \
	$(MAKE) -C $(TIPPECANOE_SRC) tippecanoe tippecanoe-decode; \
	mkdir -p $(TIPPECANOE_DIR); \
	cp $(TIPPECANOE_SRC)/tippecanoe $(TIPPECANOE_SRC)/tippecanoe-decode $(TIPPECANOE_DIR)/; \
	rm -rf $(TIPPECANOE_SRC)
