"""Pydantic request / response schemas."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Movie schemas ────────────────────────────────────────────────────────────

class MovieBase(BaseModel):
    movie_id: int
    title: str
    year: Optional[int] = None
    genres: List[str] = []
    overview: Optional[str] = None
    poster_path: Optional[str] = None
    tmdb_id: Optional[int] = None
    vote_average: Optional[float] = None
    popularity: Optional[float] = None

class Movie(MovieBase):
    director: Optional[str] = None
    cast: List[str] = []
    keywords: List[str] = []
    runtime: Optional[float] = None
    original_language: Optional[str] = None
    tagline: Optional[str] = None

class MovieWithScore(MovieBase):
    score: float = 0.0
    reason: Optional[str] = None

class MovieSearchResult(BaseModel):
    movies: List[MovieWithScore]
    total: int
    query: str

class MovieSuggestion(BaseModel):
    movie_id: int
    title: str
    year: Optional[int] = None
    poster_path: Optional[str] = None
    genres: List[str] = []


# ── User schemas ─────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    session_id: str

class UserOut(BaseModel):
    id: int
    session_id: str
    created_at: datetime
    has_onboarding: bool = False
    has_preferences: bool = False   # True only if onboarding was completed with genres
    interaction_count: int = 0

    class Config:
        from_attributes = True

class OnboardingRequest(BaseModel):
    preferred_genres: List[str] = Field(default=[], description="Favorite genres")
    preferred_languages: List[str] = Field(default=[], description="Preferred languages")
    favorite_actors: List[str] = Field(default=[], description="Favorite actors (optional)")

class OnboardingResponse(BaseModel):
    success: bool
    message: str


# ── Interaction schemas ───────────────────────────────────────────────────────

VALID_TYPES = {"click", "view", "like", "dislike", "rate", "search", "watch"}

class InteractionCreate(BaseModel):
    movie_id: Optional[int] = None
    interaction_type: str
    rating: Optional[float] = Field(None, ge=0.5, le=5.0)
    search_query: Optional[str] = None

class InteractionOut(BaseModel):
    id: int
    movie_id: Optional[int]
    interaction_type: str
    rating: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


# ── Recommendation schemas ────────────────────────────────────────────────────

class RecommendationResponse(BaseModel):
    user_id: int
    strategy: str  # "hybrid", "content", "popularity", "genre"
    recommendations: List[MovieWithScore]
    total: int
    context: Optional[str] = None   # e.g. movie title for "Because You Watched"


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    movie_count: int
    version: str
