"""Feature engineering module for movie recommendation system."""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

from src.utils.helpers import parse_list_columns


class FeatureEngineer:
    """Engineer features for movie recommendation models (The Movies Dataset - Kaggle)."""
    
    def __init__(self, processed_dir: Path = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            processed_dir: Path to processed data directory
        """
        if processed_dir is None:
            self.processed_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
        else:
            self.processed_dir = Path(processed_dir)

    @staticmethod
    def _normalize_rating_movie_args(first: pd.DataFrame,
                                     second: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Accept either (ratings, movies) or the older notebook order (movies, ratings).
        """
        first_is_ratings = {'userId', 'movieId', 'rating'}.issubset(first.columns)
        second_is_ratings = {'userId', 'movieId', 'rating'}.issubset(second.columns)
        if first_is_ratings and not second_is_ratings:
            return first, second
        if second_is_ratings and not first_is_ratings:
            return second, first
        return first, second
    
    def create_user_activity_features(self, train_ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Create user activity features.
        
        Args:
            train_ratings: Training ratings DataFrame
            
        Returns:
            User activity features DataFrame
        """
        user_activity = train_ratings.groupby('userId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'movieId': 'nunique'
        }).reset_index()
        user_activity.columns = ['userId', 'n_ratings', 'avg_rating', 'std_rating',
                                  'min_rating', 'max_rating', 'n_unique_movies']
        
        # Calculate rating diversity
        user_activity['rating_diversity'] = user_activity['max_rating'] - user_activity['min_rating']
        
        return user_activity
    
    def create_user_genre_preferences(self, train_ratings: pd.DataFrame,
                                     movies_integrated: pd.DataFrame) -> pd.DataFrame:
        """
        Create user genre preferences from genre_names list.
        
        Args:
            train_ratings: Training ratings DataFrame
            movies_integrated: Integrated movies DataFrame
            
        Returns:
            User genre preferences DataFrame
        """
        # Use movieId from ratings to merge with movies_integrated
        # For unified dataset, we need to map movieId to tmdbId first
        movies_integrated = parse_list_columns(movies_integrated.copy(), ['genre_names'])
        if 'movieId' in movies_integrated.columns:
            movies_with_genres = movies_integrated[['movieId', 'genre_names']].copy()
        else:
            # If no movieId column, we need to use links to map
            movies_with_genres = movies_integrated[['id', 'genre_names']].copy()
            movies_with_genres = movies_with_genres.rename(columns={'id': 'tmdbId'})
        
        movies_with_genres = movies_with_genres[movies_with_genres['genre_names'].notna()]
        
        user_genre_ratings = train_ratings.merge(movies_with_genres, on='movieId', how='inner')
        
        def get_top_genres(genre_names_series):
            all_genres = []
            for genre_names in genre_names_series:
                if isinstance(genre_names, list):
                    all_genres.extend(genre_names)
            counter = Counter(all_genres)
            return [g for g, c in counter.most_common(3)]
        
        user_top_genres = user_genre_ratings.groupby('userId')['genre_names'].apply(
            get_top_genres
        ).reset_index()
        user_top_genres.columns = ['userId', 'top_genres']
        
        return user_top_genres
    
    def create_movie_popularity_features(self, train_ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Create movie popularity features.
        
        Args:
            train_ratings: Training ratings DataFrame
            
        Returns:
            Movie popularity features DataFrame
        """
        movie_popularity = train_ratings.groupby('movieId').agg({
            'rating': ['count', 'mean', 'std', 'min', 'max'],
            'userId': 'nunique'
        }).reset_index()
        movie_popularity.columns = ['movieId', 'n_ratings', 'avg_rating', 'std_rating',
                                     'min_rating', 'max_rating', 'n_unique_users']
        
        # Calculate rating consistency
        movie_popularity['rating_consistency'] = 1 / (1 + movie_popularity['std_rating'].fillna(0))
        
        # Calculate popularity rank
        movie_popularity['popularity_rank'] = movie_popularity['n_ratings'].rank(ascending=False)
        
        # Calculate average rating rank
        movie_popularity['avg_rating_rank'] = movie_popularity['avg_rating'].rank(ascending=False)
        
        return movie_popularity
    
    def create_temporal_features(self, movies_integrated: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from movie release dates.
        
        Args:
            movies_integrated: Integrated movies DataFrame
            
        Returns:
            Temporal features DataFrame
        """
        # Use appropriate ID column
        id_col = 'movieId' if 'movieId' in movies_integrated.columns else 'id'
        
        movies_temporal = movies_integrated[[id_col, 'year', 'month']].copy()
        movies_temporal = movies_temporal[movies_temporal['year'].notna()]
        
        # Calculate movie age
        latest_year = movies_temporal['year'].max()
        movies_temporal['movie_age_years'] = latest_year - movies_temporal['year']
        
        # Calculate decade
        movies_temporal['decade'] = (movies_temporal['year'] // 10) * 10
        
        return movies_temporal
    
    def create_collaborative_features(self, train_ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create collaborative filtering bias features.
        
        Args:
            train_ratings: Training ratings DataFrame
            
        Returns:
            Tuple of (user_bias_df, movie_bias_df)
        """
        # Calculate user average rating deviation from global mean
        global_mean_rating = train_ratings['rating'].mean()
        user_avg_rating = train_ratings.groupby('userId')['rating'].mean()
        user_rating_bias = user_avg_rating - global_mean_rating
        
        # Calculate movie average rating deviation from global mean
        movie_avg_rating = train_ratings.groupby('movieId')['rating'].mean()
        movie_rating_bias = movie_avg_rating - global_mean_rating
        
        # Create bias features
        user_bias_df = user_rating_bias.reset_index()
        user_bias_df.columns = ['userId', 'user_rating_bias']
        
        movie_bias_df = movie_rating_bias.reset_index()
        movie_bias_df.columns = ['movieId', 'movie_rating_bias']
        
        return user_bias_df, movie_bias_df
    
    def engineer_user_features(self, train_ratings: pd.DataFrame,
                              movies_integrated: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer complete user features.
        
        Args:
            train_ratings: Training ratings DataFrame
            movies_integrated: Integrated movies DataFrame
            
        Returns:
            Engineered user features DataFrame
        """
        train_ratings, movies_integrated = self._normalize_rating_movie_args(train_ratings, movies_integrated)
        user_activity = self.create_user_activity_features(train_ratings)
        user_top_genres = self.create_user_genre_preferences(train_ratings, movies_integrated)
        user_bias_df, _ = self.create_collaborative_features(train_ratings)
        
        # Merge all user features
        user_engineered = user_activity.merge(user_top_genres, on='userId', how='left')
        user_engineered = user_engineered.merge(user_bias_df, on='userId', how='left')
        
        return user_engineered
    
    def engineer_movie_features(self, train_ratings: pd.DataFrame,
                               movies_integrated: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer complete movie features.
        
        Args:
            train_ratings: Training ratings DataFrame
            movies_integrated: Integrated movies DataFrame
            
        Returns:
            Engineered movie features DataFrame
        """
        train_ratings, movies_integrated = self._normalize_rating_movie_args(train_ratings, movies_integrated)
        movie_popularity = self.create_movie_popularity_features(train_ratings)
        movies_temporal = self.create_temporal_features(movies_integrated)
        _, movie_bias_df = self.create_collaborative_features(train_ratings)
        
        # Use appropriate ID column for merging
        id_col = 'movieId' if 'movieId' in movies_integrated.columns else 'id'
        
        # Rename temporal features ID column to match
        movies_temporal = movies_temporal.rename(columns={id_col: 'movieId'})
        
        # Merge all movie features
        movie_engineered = movie_popularity.merge(movies_temporal, on='movieId', how='left')
        movie_engineered = movie_engineered.merge(movie_bias_df, on='movieId', how='left')
        
        return movie_engineered
    
    def save_engineered_features(self, user_engineered: pd.DataFrame,
                                 movie_engineered: pd.DataFrame):
        """
        Save engineered features.
        
        Args:
            user_engineered: Engineered user features DataFrame
            movie_engineered: Engineered movie features DataFrame
        """
        user_engineered.to_csv(self.processed_dir / 'user_engineered.csv', index=False)
        movie_engineered.to_csv(self.processed_dir / 'movie_engineered.csv', index=False)
        
        print("Engineered features saved successfully!")

    def save_features(self, user_features: pd.DataFrame, movie_features: pd.DataFrame):
        """
        Backward-compatible alias used by older notebooks.
        """
        self.save_engineered_features(user_features, movie_features)
