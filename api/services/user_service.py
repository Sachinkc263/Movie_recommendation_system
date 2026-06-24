"""User management service — CRUD for users, preferences, interactions."""

import json
import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from api.models.db_models import Interaction, User, UserPreference
from api.models.schemas import InteractionCreate, OnboardingRequest

logger = logging.getLogger(__name__)


def get_or_create_user(db: Session, session_id: str) -> User:
    user = db.query(User).filter(User.session_id == session_id).first()
    if user is None:
        user = User(session_id=session_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def save_onboarding(db: Session, user_id: int, data: OnboardingRequest) -> UserPreference:
    pref = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
    if pref is None:
        pref = UserPreference(user_id=user_id)
        db.add(pref)

    pref.preferred_genres = json.dumps(data.preferred_genres)
    pref.preferred_languages = json.dumps(data.preferred_languages)
    pref.favorite_actors = json.dumps(data.favorite_actors)
    db.commit()
    db.refresh(pref)
    return pref


def log_interaction(db: Session, user_id: int, data: InteractionCreate) -> Interaction:
    inter = Interaction(
        user_id=user_id,
        movie_id=data.movie_id,
        interaction_type=data.interaction_type,
        rating=data.rating,
        search_query=data.search_query,
    )
    db.add(inter)
    db.commit()
    db.refresh(inter)
    return inter


def get_interactions(db: Session, user_id: int, limit: int = 100) -> List[Interaction]:
    return (
        db.query(Interaction)
        .filter(Interaction.user_id == user_id)
        .order_by(Interaction.created_at.desc())
        .limit(limit)
        .all()
    )


def get_rated_movies(db: Session, user_id: int):
    """Return (movie_id, rating) for movies the user has explicitly rated."""
    rows = (
        db.query(Interaction.movie_id, Interaction.rating)
        .filter(
            Interaction.user_id == user_id,
            Interaction.rating.isnot(None),
            Interaction.movie_id.isnot(None),
        )
        .all()
    )
    return [(r.movie_id, r.rating) for r in rows]


def get_viewed_movie_ids(db: Session, user_id: int) -> List[int]:
    """Return unique movie_ids the user has interacted with (used to exclude from recs)."""
    rows = (
        db.query(Interaction.movie_id)
        .filter(
            Interaction.user_id == user_id,
            Interaction.movie_id.isnot(None),
        )
        .distinct()
        .all()
    )
    return [r.movie_id for r in rows]


def get_liked_movies(db: Session, user_id: int):
    """
    Return (movie_id, implicit_rating) for positive-signal interactions.

    Hierarchy:
      explicit rating ≥ 3.5  → use rating as-is
      like                    → 5.0
      view (opened detail)    → 3.5  (positive implicit signal)
      click (card click)      → 3.0  (weak implicit signal)
      watch                   → 4.0
    """
    IMPLICIT = {
        "like": 5.0,
        "watch": 4.0,
        "view": 3.5,
        "click": 3.0,
    }

    rows = (
        db.query(Interaction.movie_id, Interaction.rating, Interaction.interaction_type)
        .filter(
            Interaction.user_id == user_id,
            Interaction.movie_id.isnot(None),
            Interaction.interaction_type.in_(list(IMPLICIT.keys()) + ["rate"]),
        )
        .all()
    )

    seen: dict = {}  # movie_id → best rating (de-duplicate, keep max)
    for r in rows:
        mid = r.movie_id
        if r.rating is not None and r.rating >= 3.5:
            weight = float(r.rating)
        elif r.interaction_type in IMPLICIT:
            weight = IMPLICIT[r.interaction_type]
        else:
            continue

        if mid not in seen or seen[mid] < weight:
            seen[mid] = weight

    return list(seen.items())
