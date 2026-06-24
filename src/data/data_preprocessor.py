"""Data preprocessing module for movie recommendation system."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from collections import Counter
from typing import Tuple, Dict
import pickle
import warnings
warnings.filterwarnings('ignore')

from src.utils.helpers import parse_list_columns


class DataPreprocessor:
    """Preprocess cleaned data for feature engineering and modeling (The Movies Dataset - Kaggle)."""
    
    def __init__(self, processed_dir: Path = None):
        """
        Initialize DataPreprocessor.
        
        Args:
            processed_dir: Path to processed data directory
        """
        if processed_dir is None:
            self.processed_dir = Path(__file__).parent.parent.parent / 'data' / 'processed'
        else:
            self.processed_dir = Path(processed_dir)
    
    def merge_datasets(self, movies_clean: pd.DataFrame, credits_clean: pd.DataFrame,
                      keywords_clean: pd.DataFrame, links_clean: pd.DataFrame = None) -> pd.DataFrame:
        """
        Merge unified dataset components.
        
        Args:
            movies_clean: Cleaned movies metadata
            credits_clean: Cleaned credits
            keywords_clean: Cleaned keywords
            links_clean: Optional links for MovieLens ID mapping
            
        Returns:
            Integrated movies DataFrame
        """
        movies_integrated = movies_clean.copy()
        movies_integrated = parse_list_columns(movies_integrated, ['genres_parsed', 'genre_names'])
        credits_clean = parse_list_columns(credits_clean.copy(), ['cast_parsed', 'crew_parsed', 'cast_names', 'director'])
        keywords_clean = parse_list_columns(keywords_clean.copy(), ['keywords_parsed', 'keyword_names'])
        
        # Merge with credits
        movies_integrated = movies_integrated.merge(
            credits_clean[['id', 'cast_names', 'director']],
            on='id',
            how='left'
        )
        
        # Merge with keywords
        movies_integrated = movies_integrated.merge(
            keywords_clean[['id', 'keyword_names']],
            on='id',
            how='left'
        )
        
        # Merge with links if provided (for MovieLens compatibility)
        if links_clean is not None:
            links_clean = links_clean.copy()
            links_clean['tmdbId'] = pd.to_numeric(links_clean['tmdbId'], errors='coerce')
            links_clean['movieId'] = pd.to_numeric(links_clean['movieId'], errors='coerce')
            links_clean = links_clean.dropna(subset=['tmdbId', 'movieId'])
            links_clean[['tmdbId', 'movieId']] = links_clean[['tmdbId', 'movieId']].astype(int)
            movies_integrated = movies_integrated.merge(
                links_clean[['tmdbId', 'movieId']],
                left_on='id',
                right_on='tmdbId',
                how='left'
            )
        
        return movies_integrated
    
    def process_ratings(self, ratings_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Process ratings data with temporal features.
        
        Args:
            ratings_clean: Cleaned ratings DataFrame
            
        Returns:
            Processed ratings DataFrame
        """
        ratings = ratings_clean.copy()
        
        # Convert timestamp to temporal features (memory-efficient)
        ratings['year'] = (ratings['timestamp'] / (365.25 * 24 * 3600) + 1970).astype(int)
        ratings['month'] = ((ratings['timestamp'] / (30.44 * 24 * 3600)) % 12 + 1).astype(int)
        ratings['day_of_week'] = ((ratings['timestamp'] / (24 * 3600)) % 7).astype(int)
        
        # Normalize ratings to 0-1 scale
        ratings['rating_normalized'] = (ratings['rating'] - 0.5) / 4.5
        
        return ratings
    
    
    def create_user_item_matrix(self, ratings: pd.DataFrame) -> Tuple[csr_matrix, Dict, Dict]:
        """
        Create sparse user-item interaction matrix.
        
        Args:
            ratings: Ratings DataFrame
            
        Returns:
            Tuple of (sparse matrix, user_to_index mapping, movie_to_index mapping)
        """
        n_users = ratings['userId'].nunique()
        n_movies = ratings['movieId'].nunique()
        
        # Create mappings
        user_ids = ratings['userId'].unique()
        movie_ids = ratings['movieId'].unique()
        
        user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        movie_to_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        
        # Map ratings to indices
        ratings['user_idx'] = ratings['userId'].map(user_to_index)
        ratings['movie_idx'] = ratings['movieId'].map(movie_to_index)
        
        # Create sparse matrix
        user_item_matrix = csr_matrix(
            (ratings['rating'], (ratings['user_idx'], ratings['movie_idx'])),
            shape=(n_users, n_movies)
        )
        
        return user_item_matrix, user_to_index, movie_to_index
    
    def normalize_features(self, movies_integrated: pd.DataFrame, 
                         numerical_features: list) -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Args:
            movies_integrated: Integrated movies DataFrame
            numerical_features: List of numerical feature names to normalize
            
        Returns:
            DataFrame with normalized features
        """
        movies = movies_integrated.copy()
        movies = parse_list_columns(movies, ['genre_names'])
        scaler = StandardScaler()
        
        for feature in numerical_features:
            if feature in movies.columns:
                mask = movies[feature].notna()
                if mask.sum() > 0:
                    movies[f'{feature}_normalized'] = np.nan
                    movies.loc[mask, f'{feature}_normalized'] = scaler.fit_transform(
                        movies.loc[mask, [feature]]
                    ).ravel()
        
        return movies
    
    def create_genre_dummies(self, movies_integrated: pd.DataFrame, 
                            top_n: int = 20) -> Tuple[pd.DataFrame, list]:
        """
        Create genre dummy variables from genre_names list.
        
        Args:
            movies_integrated: Integrated movies DataFrame
            top_n: Number of top genres to create dummies for
            
        Returns:
            Tuple of (DataFrame with genre dummy variables, list of top genres)
        """
        movies = movies_integrated.copy()
        
        # Extract all genres from genre_names list
        all_genres = []
        for genre_names in movies['genre_names'].dropna():
            if isinstance(genre_names, list):
                all_genres.extend(genre_names)
        
        genre_counts = Counter(all_genres)
        top_genres = [g for g, c in genre_counts.most_common(top_n)]
        
        # Create genre columns
        for genre in top_genres:
            movies[f'genre_{genre}'] = movies['genre_names'].apply(
                lambda x: 1 if isinstance(x, list) and genre in x else 0
            )
        
        return movies, top_genres
    
    def create_train_test_split(self, ratings: pd.DataFrame,
                                test_size: float = 0.2,
                                method: str = 'user_based',
                                min_ratings_per_user: int = 5,
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split.

        Args:
            ratings: Ratings DataFrame
            test_size: Proportion of data for test set
            method: Split method ('user_based' or 'temporal').
                    Defaults to 'user_based' — guarantees every test user
                    has training history, eliminating the coverage gap.
            min_ratings_per_user: Minimum ratings a user must have to be
                    included in the user-based split.
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (train_ratings, test_ratings) where every userId in
            test_ratings is also present in train_ratings.
        """
        if method == 'user_based':
            user_counts = ratings['userId'].value_counts()
            active_users = user_counts[user_counts >= min_ratings_per_user].index

            ratings_filtered = ratings[ratings['userId'].isin(active_users)].copy()

            train_list = []
            test_list = []

            rng = np.random.default_rng(random_state)
            for user_id, user_df in ratings_filtered.groupby('userId'):
                n_test = max(1, int(len(user_df) * test_size))
                shuffled = user_df.sample(frac=1, random_state=int(rng.integers(0, 2**31)))
                test_list.append(shuffled.iloc[:n_test])
                train_list.append(shuffled.iloc[n_test:])

            train_ratings = pd.concat(train_list, ignore_index=True)
            test_ratings = pd.concat(test_list, ignore_index=True)

            # Enforce overlap — every test user must have train history
            overlap_users = set(train_ratings['userId']) & set(test_ratings['userId'])
            train_ratings = train_ratings[train_ratings['userId'].isin(overlap_users)].copy()
            test_ratings = test_ratings[test_ratings['userId'].isin(overlap_users)].copy()

            print(f"User-based split: {len(overlap_users):,} overlap users, "
                  f"{len(train_ratings):,} train rows, {len(test_ratings):,} test rows")

        else:
            # Temporal split — kept for backward compatibility only
            cutoff_timestamp = ratings['timestamp'].quantile(1 - test_size)
            train_ratings = ratings.loc[ratings['timestamp'] <= cutoff_timestamp].copy()
            test_ratings = ratings.loc[ratings['timestamp'] > cutoff_timestamp].copy()

        return train_ratings, test_ratings
    
    def create_user_features(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-level features.
        
        Args:
            ratings: Ratings DataFrame
            
        Returns:
            User features DataFrame
        """
        user_features = ratings.groupby('userId').agg({
            'rating': ['mean', 'std', 'count'],
            'timestamp': ['min', 'max']
        }).reset_index()
        user_features.columns = ['userId', 'avg_rating', 'std_rating', 'n_ratings', 
                                'first_rating', 'last_rating']
        user_features['rating_span_days'] = (
            (user_features['last_rating'] - user_features['first_rating']) / (24 * 3600)
        ).astype(int)
        
        return user_features
    
    def create_movie_features(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Create movie-level features.
        
        Args:
            ratings: Ratings DataFrame
            
        Returns:
            Movie features DataFrame
        """
        movie_features = ratings.groupby('movieId').agg({
            'rating': ['mean', 'std', 'count'],
            'userId': 'nunique'
        }).reset_index()
        movie_features.columns = ['movieId', 'avg_rating', 'std_rating', 'n_ratings', 
                                'n_unique_users']
        
        return movie_features
    
    def save_preprocessed_data(self, movies_integrated: pd.DataFrame,
                              train_ratings: pd.DataFrame, test_ratings: pd.DataFrame,
                              user_features: pd.DataFrame, movie_features: pd.DataFrame,
                              user_item_matrix: csr_matrix = None,
                              user_to_index: Dict = None, movie_to_index: Dict = None):
        """
        Save all preprocessed data.
        
        Args:
            movies_integrated: Integrated movies DataFrame
            train_ratings: Training ratings DataFrame
            test_ratings: Test ratings DataFrame
            user_features: User features DataFrame
            movie_features: Movie features DataFrame
            user_item_matrix: Optional sparse user-item matrix
            user_to_index: Optional user to index mapping
            movie_to_index: Optional movie to index mapping
        """
        movies_integrated.to_csv(self.processed_dir / 'movies_integrated.csv', index=False)
        train_ratings.to_csv(self.processed_dir / 'train_ratings.csv', index=False)
        test_ratings.to_csv(self.processed_dir / 'test_ratings.csv', index=False)
        user_features.to_csv(self.processed_dir / 'user_features.csv', index=False)
        movie_features.to_csv(self.processed_dir / 'movie_features.csv', index=False)
        
        # Save sparse matrix and mappings if provided
        if user_item_matrix is not None:
            from scipy.sparse import save_npz
            save_npz(self.processed_dir / 'user_item_matrix.npz', user_item_matrix)
        
        if user_to_index is not None:
            with open(self.processed_dir / 'user_to_index.pkl', 'wb') as f:
                pickle.dump(user_to_index, f)
        
        if movie_to_index is not None:
            with open(self.processed_dir / 'movie_to_index.pkl', 'wb') as f:
                pickle.dump(movie_to_index, f)
        
        print("Preprocessed data saved successfully!")
