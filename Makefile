.PHONY: help setup preprocess train pipeline evaluate start stop restart logs dev-api dev-frontend test clean

# Default target
help:
	@echo ""
	@echo "  CineMatch - Movie Recommendation System"
	@echo ""
	@echo "  Setup"
	@echo "    make setup          Install Python dependencies"
	@echo ""
	@echo "  ML Pipeline (run in order after downloading the dataset)"
	@echo "    make preprocess     Step 1+2: clean raw data + create train/test split"
	@echo "    make train          Step 3:   train SVD + TF-IDF, export artifacts"
	@echo "    make pipeline       Run preprocess + train in sequence"
	@echo "    make evaluate       Step 4:   run full offline evaluation suite"
	@echo ""
	@echo "  Development"
	@echo "    make dev-api        Run FastAPI with hot-reload (port 8000)"
	@echo "    make dev-frontend   Run Vite dev server (port 5173)"
	@echo "    make test           Run test suite"
	@echo ""
	@echo "  Docker"
	@echo "    make start          Build + start all services (API :8000, App :3000)"
	@echo "    make stop           Stop all services"
	@echo "    make restart        Rebuild + restart"
	@echo "    make logs           Tail live logs from all containers"
	@echo ""
	@echo "  Maintenance"
	@echo "    make clean          Remove __pycache__, .pyc files, and build artifacts"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────────

setup:
	pip install -r requirements.txt

# ── ML pipeline ───────────────────────────────────────────────────────────────

preprocess:
	@echo "Cleaning raw data and creating train/test split ..."
	python scripts/run_preprocessing.py

train:
	@echo "Training models and exporting artifacts to models/artifacts/ ..."
	python scripts/train_and_export.py

pipeline: preprocess train
	@echo "Full pipeline complete. Start the API with: make dev-api or make start"

evaluate:
	@echo "Running evaluation suite ..."
	python scripts/evaluate.py

# ── Local development ─────────────────────────────────────────────────────────

dev-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm install && npm run dev

# ── Docker ────────────────────────────────────────────────────────────────────

start:
	docker compose up --build -d
	@echo ""
	@echo "  Services started:"
	@echo "    API      http://localhost:8000"
	@echo "    API docs http://localhost:8000/docs"
	@echo "    App      http://localhost:3000"

stop:
	docker compose down

restart:
	docker compose down
	docker compose up --build -d

logs:
	docker compose logs -f

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

# ── Maintenance ───────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned build artifacts"
