"""Feature selection module for movie recommendation system."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """Select and filter features for modeling (The Movies Dataset - Kaggle)."""
    
    def __init__(self, processed_dir: Path = None):
        """
        Initialize FeatureSelector.
        
        Args:
            processed_dir: Path to processed data directory
        """
        if processed_dir is None:
            self.processed_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
        else:
            self.processed_dir = Path(processed_dir)
    
    def remove_low_variance(self, df: pd.DataFrame, threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with low variance.
        
        Args:
            df: DataFrame to filter
            threshold: Variance threshold
            
        Returns:
            Tuple of (filtered DataFrame, list of removed columns)
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        variances = df[numeric_cols].var()
        low_variance_cols = variances[variances < threshold].index.tolist()
        
        print(f"Low variance columns (threshold={threshold}): {low_variance_cols}")
        
        df_filtered = df.drop(columns=low_variance_cols)
        return df_filtered, low_variance_cols
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df: DataFrame with missing values
            
        Returns:
            DataFrame with missing values filled
        """
        df_filled = df.copy()
        for col in df_filled.columns:
            if df_filled[col].isnull().sum() > 0:
                if df_filled[col].dtype in ['float64', 'int64']:
                    df_filled[col].fillna(df_filled[col].median(), inplace=True)
                else:
                    df_filled[col].fillna('Unknown', inplace=True)
        
        return df_filled
    
    def select_user_features(self, user_engineered: pd.DataFrame) -> pd.DataFrame:
        """
        Select final user feature set.
        
        Args:
            user_engineered: Engineered user features DataFrame
            
        Returns:
            Selected user features DataFrame
        """
        candidate_cols = ['userId', 'n_ratings', 'avg_rating', 'std_rating',
                          'n_unique_movies', 'user_rating_bias']
        user_feature_cols = [col for col in candidate_cols if col in user_engineered.columns]
        
        optional_user_cols = ['rating_span_days', 'rating_diversity', 'top_genres']
        for col in optional_user_cols:
            if col in user_engineered.columns:
                user_feature_cols.append(col)
        
        return user_engineered[user_feature_cols]
    
    def select_movie_features(self, movie_engineered: pd.DataFrame) -> pd.DataFrame:
        """
        Select final movie feature set.
        
        Args:
            movie_engineered: Engineered movie features DataFrame
            
        Returns:
            Selected movie features DataFrame
        """
        candidate_cols = ['movieId', 'n_ratings', 'avg_rating', 'std_rating',
                          'n_unique_users', 'movie_rating_bias']
        movie_feature_cols = [col for col in candidate_cols if col in movie_engineered.columns]
        
        optional_movie_cols = ['rating_consistency', 'popularity_rank', 'avg_rating_rank',
                               'movie_age_years', 'decade']
        for col in optional_movie_cols:
            if col in movie_engineered.columns:
                movie_feature_cols.append(col)
        
        return movie_engineered[movie_feature_cols]
    
    def process_features(self, user_engineered: pd.DataFrame,
                        movie_engineered: pd.DataFrame,
                        variance_threshold: float = 0.01) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete feature selection pipeline.
        
        Args:
            user_engineered: Engineered user features DataFrame
            movie_engineered: Engineered movie features DataFrame
            variance_threshold: Threshold for variance filtering
            
        Returns:
            Tuple of (selected user features, selected movie features)
        """
        # Remove low variance features
        user_filtered, user_low_var = self.remove_low_variance(user_engineered, variance_threshold)
        movie_filtered, movie_low_var = self.remove_low_variance(movie_engineered, variance_threshold)
        
        # Handle missing values
        user_final = self.handle_missing_values(user_filtered)
        movie_final = self.handle_missing_values(movie_filtered)
        
        # Select final feature sets
        user_final_features = self.select_user_features(user_final)
        movie_final_features = self.select_movie_features(movie_final)
        
        return user_final_features, movie_final_features
    
    def save_selected_features(self, user_final: pd.DataFrame = None,
                               movie_final: pd.DataFrame = None,
                               user_features: pd.DataFrame = None,
                               movie_features: pd.DataFrame = None):
        """
        Save selected features.
        
        Args:
            user_final: Final user features DataFrame
            movie_final: Final movie features DataFrame
        """
        if user_final is None:
            user_final = user_features
        if movie_final is None:
            movie_final = movie_features

        user_final.to_csv(self.processed_dir / 'user_features_selected.csv', index=False)
        movie_final.to_csv(self.processed_dir / 'movie_features_selected.csv', index=False)
        
        print("Selected features saved successfully!")
