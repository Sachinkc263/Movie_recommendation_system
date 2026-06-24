"""Movie endpoints — popular, search, suggestions, detail, similar, poster."""

import ast
import math

from fastapi import APIRouter, HTTPException, Query

from api.models.schemas import Movie, MovieSearchResult, MovieSuggestion, MovieWithScore
from api.services import movie_service
from api.services.model_service import model_service
from api.services.tmdb_service import tmdb_service

router = APIRouter(prefix="/movies", tags=["movies"])


@router.get("/popular", response_model=list[MovieWithScore])
def popular_movies(n: int = Query(20, ge=1, le=100)):
    return movie_service.get_popular_movies(n=n)


@router.get("/search", response_model=MovieSearchResult)
def search_movies(
    q: str = Query(..., min_length=1, description="Search query"),
    n: int = Query(20, ge=1, le=100),
):
    results = movie_service.search_movies(query=q, n=n)
    return MovieSearchResult(movies=results, total=len(results), query=q)


@router.get("/suggestions", response_model=list[MovieSuggestion])
def movie_suggestions(
    q: str = Query(..., min_length=1, description="Autocomplete query"),
    n: int = Query(6, ge=1, le=10),
):
    """Fast autocomplete — lightweight movie stubs for the Navbar dropdown."""
    pairs = model_service.get_search_suggestions(query=q, n=n)
    results = []
    for movie_id, _ in pairs:
        meta = model_service.get_movie_meta(movie_id)
        if meta is None:
            continue

        def _safe_list(v):
            if isinstance(v, list):
                return v
            if isinstance(v, str) and v.startswith("["):
                try:
                    return ast.literal_eval(v)
                except Exception:
                    pass
            return []

        genres = _safe_list(meta.get("genre_names"))
        year_raw = meta.get("year")
        try:
            year = int(float(year_raw)) if year_raw is not None else None
            if year and math.isnan(float(year_raw)):
                year = None
        except (ValueError, TypeError):
            year = None

        # Resolve poster from cache (fast path — no API call during autocomplete)
        tmdb_id_raw = meta.get("tmdbId")
        tmdb_id = int(float(tmdb_id_raw)) if tmdb_id_raw else None
        poster_path = meta.get("poster_path") or None
        if tmdb_id:
            fresh = tmdb_service.get_poster_path(tmdb_id)
            if fresh:
                poster_path = fresh

        results.append(MovieSuggestion(
            movie_id=movie_id,
            title=str(meta.get("title") or f"Movie {movie_id}"),
            year=year,
            poster_path=poster_path,
            genres=genres[:2],
        ))
    return results


@router.get("/{movie_id}/poster-url")
def get_fresh_poster_url(movie_id: int):
    """
    Return the current TMDB CDN poster URL for *movie_id*.

    Frontend calls this as an `onError` fallback when the static CSV-derived URL
    returns 404.  Result is cached server-side so repeated calls are instant.
    """
    meta = model_service.get_movie_meta(movie_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Movie not found")

    tmdb_id_raw = meta.get("tmdbId")
    if not tmdb_id_raw:
        return {"poster_url": None}

    try:
        tmdb_id = int(float(tmdb_id_raw))
    except (ValueError, TypeError):
        return {"poster_url": None}

    poster_url = tmdb_service.get_poster_url(tmdb_id, size="w500")
    return {"poster_url": poster_url}


@router.get("/{movie_id}", response_model=Movie)
def movie_detail(movie_id: int):
    movie = movie_service.get_movie_detail(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie


@router.get("/{movie_id}/similar", response_model=list[MovieWithScore])
def similar_movies(movie_id: int, n: int = Query(10, ge=1, le=30)):
    movie = movie_service.get_movie_detail(movie_id)
    if movie is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return movie_service.get_similar_movies(movie_id=movie_id, n=n)
