"""Application configuration via environment variables."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    APP_TITLE: str = "Movie Recommendation System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Paths
    DATA_DIR: Path = BASE_DIR / "data" / "processed"
    MODELS_DIR: Path = BASE_DIR / "models"
    ARTIFACTS_DIR: Path = BASE_DIR / "models" / "artifacts"
    DB_PATH: str = str(BASE_DIR / "data" / "db.sqlite3")

    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://frontend:80",
    ]

    # Recommendation defaults
    DEFAULT_N_RECS: int = 20
    COLD_START_THRESHOLD: int = 5  # interactions before switching to SVD

    # TMDB (optional — only used if frontend calls TMDB directly)
    TMDB_IMAGE_BASE: str = "https://image.tmdb.org/t/p/w500"

settings = Settings()
