# Makefile
# Makefile — Common development commands

.PHONY: help setup run test db-init db-migrate db-reset seed shadows analytics lint seed-knowledge

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*
$$
' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n",
$$
1, $\$2}'

# ── Setup ──

setup: ## Initial project setup
	pip install -r requirements.txt
	cp -n .env.example .env || true
	mkdir -p data/vector_store data/screenshots data/exports data/market_data_cache
	python scripts/init_db.py
	@echo "✅ Setup complete! Edit .env with your settings."

# ── Run ──

run: ## Run the backend server (PROD — ptis DB, port 8000)
	PTIS_ENV=prod uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

run-test: ## Run the backend server (TEST — ptis_test DB, port 8001)
	PTIS_ENV=test uvicorn api.main:app --reload --host 127.0.0.1 --port 8001

run-frontend: ## Run the frontend dev server (PROD — port 5173)
	cd frontend && npm run dev -- --port 5173

run-test-frontend: ## Run the frontend dev server (TEST — port 5174)
	cd frontend && npm run dev -- --mode test --port 5174

run-all: ## Run everything with Docker
	docker-compose up --build

# ── Database ──

db-init: ## Initialize database tables (PROD)
	PTIS_ENV=prod python scripts/init_db.py

db-migrate: ## Run database migrations (PROD)
	PTIS_ENV=prod alembic upgrade head

db-reset: ## Reset database (DANGER: deletes all prod data)
	@echo "⚠️  This will DELETE ALL PROD DATA. Press Ctrl+C to cancel."
	@sleep 3
	PTIS_ENV=prod alembic downgrade base
	PTIS_ENV=prod alembic upgrade head
	PTIS_ENV=prod python scripts/init_db.py

test-setup: ## First-time setup of test environment (run once)
	PTIS_ENV=test python scripts/setup_test_env.py

test-seed: ## Seed / re-seed test data (wipes previous test data first)
	PTIS_ENV=test python scripts/seed_test_data.py

test-reset: ## Wipe and re-seed test database from scratch
	PTIS_ENV=test alembic downgrade base
	PTIS_ENV=test alembic upgrade head
	PTIS_ENV=test python scripts/setup_test_env.py
	PTIS_ENV=test python scripts/seed_test_data.py

# ── Data ──

seed: ## Seed strategies from YAML files
	python scripts/seed_strategies.py

shadows: ## Backfill shadow trades for historical data
	python scripts/backfill_shadows.py --days 90

analytics: ## Recompute all analytics
	python scripts/recompute_analytics.py --days 180

daily-summary: ## Generate today's daily summary
	python scripts/generate_daily_summary.py

import-csv: ## Import trades from CSV (usage: make import-csv FILE=path.csv BROKER=generic)
	python scripts/import_historical.py $(FILE) --broker $(BROKER)

export: ## Export knowledge base and DNA
	python scripts/export_knowledge.py

seed-knowledge: ## Index strategies + docs into RAG vector store
	python scripts/seed_knowledge.py

# ── Testing ──

test: ## Run all tests
	pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v

test-api: ## Run API tests
	pytest tests/api/ -v

# ── Quality ──

lint: ## Run linter
	flake8 --max-line-length 120 --exclude=alembic,frontend .

format: ## Format code
	black --line-length 120 --exclude="alembic|frontend" .

typecheck: ## Run type checker
	mypy --ignore-missing-imports .