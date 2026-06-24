"""Movie metadata service — converts raw ML data to API response schemas."""

import ast
from typing import List, Optional

from api.config import settings
from api.models.schemas import Movie, MovieBase, MovieWithScore
from api.services.model_service import model_service


def _safe_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("["):
        try:
            return ast.literal_eval(value)
        except Exception:
            pass
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _safe_str(value) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None


def _safe_float(value) -> Optional[float]:
    try:
        v = float(value)
        import math
        return v if not math.isnan(v) else None
    except Exception:
        return None


def _safe_int(value) -> Optional[int]:
    try:
        v = float(value)
        import math
        return int(v) if not math.isnan(v) else None
    except Exception:
        return None


def _row_to_movie_base(movie_id: int, meta: dict) -> MovieBase:
    poster = _safe_str(meta.get("poster_path"))
    return MovieBase(
        movie_id=movie_id,
        title=_safe_str(meta.get("title")) or f"Movie {movie_id}",
        year=_safe_int(meta.get("year")),
        genres=_safe_list(meta.get("genre_names")),
        overview=_safe_str(meta.get("overview")),
        poster_path=poster,
        tmdb_id=_safe_int(meta.get("tmdbId")),
        vote_average=_safe_float(meta.get("vote_average")),
        popularity=_safe_float(meta.get("popularity")),
    )


def _row_to_movie(movie_id: int, meta: dict) -> Movie:
    base = _row_to_movie_base(movie_id, meta)
    director_raw = meta.get("director")
    if isinstance(director_raw, list):
        director = director_raw[0] if director_raw else None
    else:
        director = _safe_str(director_raw)
    return Movie(
        **base.model_dump(),
        director=director,
        cast=_safe_list(meta.get("cast_names"))[:5],
        keywords=_safe_list(meta.get("keyword_names"))[:10],
        runtime=_safe_float(meta.get("runtime")),
        original_language=_safe_str(meta.get("original_language")),
        tagline=_safe_str(meta.get("tagline")),
    )


# ── Public functions ──────────────────────────────────────────────────────────

def get_popular_movies(n: int = 20, exclude_ids: Optional[List[int]] = None) -> List[MovieWithScore]:
    results = model_service.get_popular(n=n, exclude_ids=exclude_ids)
    movies = []
    for movie_id, score in results:
        meta = model_service.get_movie_meta(movie_id)
        if meta is None:
            continue
        base = _row_to_movie_base(movie_id, meta)
        movies.append(MovieWithScore(**base.model_dump(), score=score, reason="Popular"))
    return movies


def get_movie_detail(movie_id: int) -> Optional[Movie]:
    meta = model_service.get_movie_meta(movie_id)
    if meta is None:
        return None
    return _row_to_movie(movie_id, meta)


def get_similar_movies(movie_id: int, n: int = 10) -> List[MovieWithScore]:
    results = model_service.get_similar_movies(movie_id=movie_id, n=n)
    movies = []
    for mid, score in results:
        meta = model_service.get_movie_meta(mid)
        if meta is None:
            continue
        base = _row_to_movie_base(mid, meta)
        movies.append(MovieWithScore(**base.model_dump(), score=score, reason="Similar content"))
    return movies


def search_movies(query: str, n: int = 20) -> List[MovieWithScore]:
    results = model_service.search_movies(query=query, n=n)
    movies = []
    for movie_id, score in results:
        meta = model_service.get_movie_meta(movie_id)
        if meta is None:
            continue
        base = _row_to_movie_base(movie_id, meta)
        movies.append(MovieWithScore(**base.model_dump(), score=score))
    return movies


def enrich_ids(id_score_pairs: List[tuple]) -> List[MovieWithScore]:
    """Convert (movie_id, score) pairs to MovieWithScore objects."""
    movies = []
    for movie_id, score in id_score_pairs:
        meta = model_service.get_movie_meta(movie_id)
        if meta is None:
            continue
        base = _row_to_movie_base(movie_id, meta)
        movies.append(MovieWithScore(**base.model_dump(), score=score))
    return movies
