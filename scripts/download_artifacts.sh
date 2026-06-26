#!/usr/bin/env bash
# Downloads ML artifacts from GitHub Releases at Render startup.
# Runs before uvicorn so the model service can load them on first request.
set -euo pipefail

RELEASE_URL="https://github.com/Sachinkc263/Movie_recommendation_system/releases/download/v1.0-artifacts"
ARTIFACTS_DIR="models/artifacts"
DATA_DIR="data/processed"

mkdir -p "$ARTIFACTS_DIR" "$DATA_DIR"

dl() {
    local file="$1"
    local dest="$2/$file"
    if [ -f "$dest" ]; then
        echo "[skip] $file already present"
        return
    fi
    echo "[download] $file ..."
    curl -fsSL -o "$dest" "$RELEASE_URL/$file"
    echo "[done] $file"
}

# SVD artifacts
dl "svd_U.npy"        "$ARTIFACTS_DIR"
dl "svd_Vt.npy"       "$ARTIFACTS_DIR"
dl "svd_meta.pkl"     "$ARTIFACTS_DIR"

# TF-IDF artifacts
dl "tfidf_matrix.npz" "$ARTIFACTS_DIR"
dl "tfidf_meta.pkl"   "$ARTIFACTS_DIR"

# Movie metadata (not in git — too large)
dl "movies_integrated.csv" "$DATA_DIR"

echo "All artifacts ready."
