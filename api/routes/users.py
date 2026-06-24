"""User endpoints — create, onboarding, interactions, history."""

import json
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import get_db
from api.models.schemas import (
    InteractionCreate,
    InteractionOut,
    OnboardingRequest,
    OnboardingResponse,
    UserCreate,
    UserOut,
)
from api.services import user_service

router = APIRouter(prefix="/users", tags=["users"])


def _build_user_out(user, db: Session) -> UserOut:
    from api.models.db_models import Interaction
    from sqlalchemy import func

    interaction_count = (
        db.query(func.count(Interaction.id))
        .filter(Interaction.user_id == user.id)
        .scalar() or 0
    )

    has_pref = user.preference is not None
    if has_pref:
        genres = json.loads(user.preference.preferred_genres or "[]")
        has_preferences = bool(genres)
    else:
        has_preferences = False

    return UserOut(
        id=user.id,
        session_id=user.session_id,
        created_at=user.created_at,
        has_onboarding=has_pref,
        has_preferences=has_preferences,
        interaction_count=interaction_count,
    )


@router.post("/", response_model=UserOut, status_code=201)
def create_or_get_user(body: UserCreate, db: Session = Depends(get_db)):
    user = user_service.get_or_create_user(db, session_id=body.session_id)
    return _build_user_out(user, db)


@router.get("/{user_id}", response_model=UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = user_service.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _build_user_out(user, db)


@router.post("/{user_id}/onboarding", response_model=OnboardingResponse)
def save_onboarding(
    user_id: int,
    body: OnboardingRequest,
    db: Session = Depends(get_db),
):
    user = user_service.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user_service.save_onboarding(db, user_id, body)
    return OnboardingResponse(success=True, message="Preferences saved")


@router.post("/{user_id}/interactions", response_model=InteractionOut, status_code=201)
def log_interaction(
    user_id: int,
    body: InteractionCreate,
    db: Session = Depends(get_db),
):
    user = user_service.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    inter = user_service.log_interaction(db, user_id, body)
    return inter


@router.get("/{user_id}/history", response_model=List[InteractionOut])
def interaction_history(
    user_id: int,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    user = user_service.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user_service.get_interactions(db, user_id, limit=limit)
