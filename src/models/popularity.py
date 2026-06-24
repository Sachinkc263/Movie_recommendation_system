"""Popularity-based recommendation model (The Movies Dataset - Kaggle)."""

from pathlib import Path
from typing import Optional, Union, List

import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from src.utils.helpers import ensure_movie_id, parse_list_columns


class PopularityRecommender:
    """Popularity-based recommendation system using Bayesian average."""

    def __init__(self, train_ratings: pd.DataFrame, movies_df: pd.DataFrame, top_n: int = 10):
        self.train_ratings = train_ratings
        self.movies_df = ensure_movie_id(parse_list_columns(movies_df.copy(), ['genre_names']))
        self.top_n = top_n
        self.popularity_scores = None
        self._user_seen_movies = train_ratings.groupby('userId')['movieId'].apply(set).to_dict()
        self._calculate_popularity()

    def _calculate_popularity(self):
        movie_stats = self.train_ratings.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std'],
            'userId': 'nunique',
        }).reset_index()
        movie_stats.columns = [
            'movieId', 'n_ratings', 'avg_rating', 'std_rating', 'n_unique_users',
        ]

        global_mean = movie_stats['avg_rating'].mean()
        smoothing_count = movie_stats['n_ratings'].quantile(0.90)

        movie_stats['weighted_score'] = (
            (movie_stats['n_ratings'] / (movie_stats['n_ratings'] + smoothing_count))
            * movie_stats['avg_rating']
            + (smoothing_count / (movie_stats['n_ratings'] + smoothing_count))
            * global_mean
        )

        self.popularity_scores = movie_stats.sort_values(
            'weighted_score', ascending=False
        )

    @staticmethod
    def _normalize_genre_filter(genre: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        if genre is None:
            return None
        if isinstance(genre, str):
            return [genre]
        return list(genre)

    def recommend(
        self,
        user_id: Optional[int] = None,
        n: Optional[int] = None,
        genre: Optional[Union[str, List[str]]] = None,
    ) -> pd.DataFrame:
        if n is None:
            n = self.top_n

        recommendations = self.popularity_scores.copy()

        genre_filters = self._normalize_genre_filter(genre)
        if genre_filters:
            genre_movies = self.movies_df[
                self.movies_df['genre_names'].apply(
                    lambda names: isinstance(names, list)
                    and any(g in names for g in genre_filters)
                )
            ]
            recommendations = recommendations[
                recommendations['movieId'].isin(genre_movies['movieId'])
            ]

        if user_id is not None:
            seen = self._user_seen_movies.get(user_id, set())
            if seen:
                recommendations = recommendations[~recommendations['movieId'].isin(seen)]

        recommendations = recommendations.head(n)

        result = recommendations.merge(
            self.movies_df[['movieId', 'title', 'genre_names']],
            on='movieId',
            how='left',
        )
        return result[
            ['movieId', 'title', 'genre_names', 'n_ratings', 'avg_rating', 'weighted_score']
        ]

    def save_model(self, save_dir: Path):
        import pickle

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / 'popularity_recommender.pkl', 'wb') as f:
            pickle.dump(self, f)

        self.popularity_scores.to_csv(save_dir / 'popularity_scores.csv', index=False)
