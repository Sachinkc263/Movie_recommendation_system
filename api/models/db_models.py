"""SQLAlchemy ORM models."""

import json
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from api.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    preference = relationship("UserPreference", back_populates="user", uselist=False)
    interactions = relationship("Interaction", back_populates="user")


class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    preferred_genres = Column(Text, default="[]")   # JSON list
    preferred_languages = Column(Text, default="[]")
    favorite_actors = Column(Text, default="[]")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="preference")

    @property
    def genres_list(self):
        return json.loads(self.preferred_genres or "[]")

    @property
    def languages_list(self):
        return json.loads(self.preferred_languages or "[]")

    @property
    def actors_list(self):
        return json.loads(self.favorite_actors or "[]")


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    movie_id = Column(Integer, nullable=True, index=True)
    interaction_type = Column(String, nullable=False)  # click, view, like, dislike, rate, search
    rating = Column(Float, nullable=True)
    search_query = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="interactions")
