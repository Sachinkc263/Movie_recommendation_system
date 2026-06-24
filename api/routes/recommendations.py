"""Recommendation endpoints — personalized hybrid recs."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.config import settings
from api.database import get_db
from api.models.schemas import RecommendationResponse
from api.services import movie_service, user_service
from api.services.model_service import model_service

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/user/{user_id}", response_model=RecommendationResponse)
def recommend_for_user(
    user_id: int,
    n: int = Query(20, ge=1, le=50),
    db: Session = Depends(get_db),
):
    user = user_service.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    exclude_ids = user_service.get_viewed_movie_ids(db, user_id)
    rated = user_service.get_rated_movies(db, user_id)
    liked = user_service.get_liked_movies(db, user_id)

    strategy = "popularity"
    id_score_pairs = []

    # ── Strategy 1: SVD fold-in (≥ cold_start_threshold rated interactions) ──
    if len(rated) >= settings.COLD_START_THRESHOLD and model_service.is_svd_ready:
        movie_ids, ratings = zip(*rated)
        result = model_service.recommend_fold_in(
            rated_movie_ids=list(movie_ids),
            ratings=list(ratings),
            n=n,
            exclude_ids=exclude_ids,
        )
        if result:
            id_score_pairs = result
            strategy = "hybrid"

    # ── Strategy 2: TF-IDF profile (≥2 liked/viewed movies) ─────────────────
    if not id_score_pairs and len(liked) >= 2 and model_service.is_tfidf_ready:
        lids, lrats = zip(*liked)
        result = model_service.recommend_by_profile(
            liked_movie_ids=list(lids),
            liked_ratings=list(lrats),
            n=n,
            exclude_ids=exclude_ids,
        )
        if result:
            id_score_pairs = result
            strategy = "content"

    # ── Strategy 3: Genre-based from onboarding ────────────────────────────────
    if not id_score_pairs and user.preference is not None:
        genres = user.preference.genres_list
        if genres:
            id_score_pairs = model_service.recommend_by_genres(
                preferred_genres=genres, n=n, exclude_ids=exclude_ids
            )
            strategy = "genre"

    # ── Strategy 4: Popularity fallback ──────────────────────────────────────
    if not id_score_pairs:
        id_score_pairs = model_service.get_popular(n=n, exclude_ids=exclude_ids)
        strategy = "popularity"

    movies = movie_service.enrich_ids(id_score_pairs[:n])
    label = {
        "hybrid": "Recommended for you",
        "content": "Based on movies you liked",
        "genre": "Based on your interests",
        "popularity": "Trending now",
    }.get(strategy, "Recommended")
    for m in movies:
        m.reason = label

    return RecommendationResponse(
        user_id=user_id,
        strategy=strategy,
        recommendations=movies,
        total=len(movies),
    )


@router.get("/popular", response_model=RecommendationResponse)
def popular_recommendations(
    n: int = Query(20, ge=1, le=50),
    user_id: int = Query(0),
):
    id_score_pairs = model_service.get_popular(n=n)
    movies = movie_service.enrich_ids(id_score_pairs)
    for m in movies:
        m.reason = "Popular"
    return RecommendationResponse(
        user_id=user_id,
        strategy="popularity",
        recommendations=movies,
        total=len(movies),
    )


@router.get("/because-you-watched/{movie_id}", response_model=RecommendationResponse)
def because_you_watched(
    movie_id: int,
    n: int = Query(10, ge=1, le=30),
    user_id: int = Query(0),
):
    movie = movie_service.get_movie_detail(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    id_score_pairs = model_service.get_similar_movies(movie_id=movie_id, n=n)
    movies = movie_service.enrich_ids(id_score_pairs)
    for m in movies:
        m.reason = f"Because you watched {movie.title}"
    return RecommendationResponse(
        user_id=user_id,
        strategy="content",
        recommendations=movies,
        total=len(movies),
        context=movie.title,   # BUG FIX: expose title so frontend can use it in section header
    )
