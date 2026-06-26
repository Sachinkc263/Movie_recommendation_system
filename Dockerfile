# FastAPI backend — multi-stage build
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Application source
COPY api/     ./api/
COPY src/     ./src/
COPY scripts/ ./scripts/

# Static runtime data that ships with the repo
COPY models/popularity_scores.csv        ./models/popularity_scores.csv
COPY data/processed/movies_integrated.csv ./data/processed/movies_integrated.csv

# Pre-create directories for artifacts downloaded at startup and runtime state
RUN mkdir -p models/artifacts data/cache logs

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# scripts/start.sh downloads ML artifacts then starts uvicorn
CMD ["/bin/bash", "/app/scripts/start.sh"]
