"""Models module for movie recommendation system."""

from .popularity import PopularityRecommender
from .collaborative_filtering import UserBasedCF, ItemBasedCF, MatrixFactorizationCF
from .content_based import GenreBasedRecommender, TFIDFContentRecommender
from .hybrid import HybridRecommender

__all__ = [
    'PopularityRecommender',
    'UserBasedCF',
    'ItemBasedCF',
    'MatrixFactorizationCF',
    'GenreBasedRecommender',
    'TFIDFContentRecommender',
    'HybridRecommender',
]
