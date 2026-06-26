#!/usr/bin/env bash
# Container startup: download ML artifacts then hand off to uvicorn.
# Used as the Dockerfile CMD for Render deployment.
set -euo pipefail

# Download SVD/TF-IDF artifacts from GitHub Releases (if not already present).
bash /app/scripts/download_artifacts.sh

# Hand off to uvicorn — exec replaces this shell so uvicorn becomes PID 1
# and receives Docker stop signals correctly.
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers 1
