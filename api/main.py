"""FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.database import create_tables
from api.models.schemas import HealthResponse
from api.routes import movies, recommendations, users
from api.services.model_service import model_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="Hybrid Movie Recommendation System — FastAPI backend",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    logger.info("Creating database tables…")
    create_tables()

    logger.info("Loading ML model artifacts…")
    model_service.load()

    # Pre-warm TMDB poster cache for the top 300 most popular movies so that
    # the first page users see always has real posters (not text fallbacks).
    # The prewarm runs in a daemon thread and does not block startup.
    try:
        from api.services.tmdb_service import tmdb_service

        top_pairs = model_service.get_popular(n=300)
        tmdb_ids: list[int] = []
        for movie_id, _ in top_pairs:
            meta = model_service.get_movie_meta(movie_id)
            if meta and meta.get("tmdbId"):
                try:
                    tmdb_ids.append(int(float(meta["tmdbId"])))
                except (ValueError, TypeError):
                    pass

        if tmdb_ids:
            tmdb_service.prewarm(tmdb_ids, rate=12.0)
    except Exception as exc:
        logger.warning("Poster prewarm setup failed (non-fatal): %s", exc)

    logger.info("API ready.")


app.include_router(movies.router)
app.include_router(recommendations.router)
app.include_router(users.router)


@app.get("/", tags=["health"])
def root():
    return {"message": "Movie Recommendation API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return HealthResponse(
        status="ok",
        models_loaded=model_service.is_svd_ready or model_service.is_tfidf_ready,
        movie_count=model_service.movie_count,
        version=settings.APP_VERSION,
    )
