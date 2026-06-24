"""Data cleaning module for movie recommendation system."""

import pandas as pd
import numpy as np
import json
import ast
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """Clean and preprocess raw movie recommendation datasets (The Movies Dataset - Kaggle)."""
    
    def __init__(self, data_dir: Path = None, processed_dir: Path = None):
        """
        Initialize DataCleaner.
        
        Args:
            data_dir: Path to raw data directory
            processed_dir: Path to save cleaned data
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        else:
            self.data_dir = Path(data_dir)
        
        if processed_dir is None:
            self.processed_dir = self.data_dir.parent / 'processed'
        else:
            self.processed_dir = Path(processed_dir)
    
    @staticmethod
    def parse_json_field(json_str):
        """
        Parse JSON field from string.
        
        Args:
            json_str: JSON string to parse
            
        Returns:
            Parsed JSON object or empty list if parsing fails
        """
        try:
            if pd.isna(json_str):
                return []
            json_str = str(json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(json_str)
                return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    
    def clean_movies_metadata(self, movies_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Clean movies metadata dataset.
        
        Args:
            movies_metadata: Raw movies metadata DataFrame
            
        Returns:
            Cleaned movies metadata DataFrame
        """
        movies_clean = movies_metadata.copy()
        
        # Fix ID column - handle non-numeric values
        movies_clean['id_numeric'] = pd.to_numeric(movies_clean['id'], errors='coerce')
        movies_clean = movies_clean.dropna(subset=['id_numeric']).copy()
        movies_clean['id'] = movies_clean['id_numeric'].astype(int)
        movies_clean = movies_clean.drop(columns=['id_numeric'])
        
        # Clean numeric fields
        numeric_fields = ['budget', 'revenue', 'runtime', 'popularity', 'vote_average', 'vote_count']
        for field in numeric_fields:
            if field in movies_clean.columns:
                movies_clean[field] = pd.to_numeric(movies_clean[field], errors='coerce')
        
        # Replace 0 budget and revenue with NaN
        if 'budget' in movies_clean.columns:
            movies_clean['budget'] = movies_clean['budget'].replace(0, np.nan)
        if 'revenue' in movies_clean.columns:
            movies_clean['revenue'] = movies_clean['revenue'].replace(0, np.nan)
        
        # Clean date fields
        if 'release_date' in movies_clean.columns:
            movies_clean['release_date'] = pd.to_datetime(
                movies_clean['release_date'], errors='coerce'
            )
            movies_clean['year'] = movies_clean['release_date'].dt.year
            movies_clean['month'] = movies_clean['release_date'].dt.month
        
        # Parse genres
        if 'genres' in movies_clean.columns:
            movies_clean['genres_parsed'] = movies_clean['genres'].apply(self.parse_json_field)
            movies_clean['genre_names'] = movies_clean['genres_parsed'].apply(
                lambda x: [g['name'] for g in x] if x else []
            )
        
        # Remove outliers
        current_year = 2026
        if 'runtime' in movies_clean.columns:
            movies_clean = movies_clean[
                (movies_clean['runtime'] >= 30) & 
                (movies_clean['runtime'] <= 300)
            ]
        if 'year' in movies_clean.columns:
            movies_clean = movies_clean[
                (movies_clean['year'] >= 1900) & 
                (movies_clean['year'] <= current_year)
            ]
        
        return movies_clean
    
    def clean_credits(self, credits: pd.DataFrame) -> pd.DataFrame:
        """
        Clean credits dataset.
        
        Args:
            credits: Raw credits DataFrame
            
        Returns:
            Cleaned credits DataFrame
        """
        credits_clean = credits.copy()
        
        # Fix id column
        credits_clean['id_numeric'] = pd.to_numeric(credits_clean['id'], errors='coerce')
        credits_clean = credits_clean.dropna(subset=['id_numeric']).copy()
        credits_clean['id'] = credits_clean['id_numeric'].astype(int)
        credits_clean = credits_clean.drop(columns=['id_numeric'])
        
        # Parse cast and crew
        credits_clean['cast_parsed'] = credits_clean['cast'].apply(self.parse_json_field)
        credits_clean['crew_parsed'] = credits_clean['crew'].apply(self.parse_json_field)
        
        # Extract top 5 cast names
        credits_clean['cast_names'] = credits_clean['cast_parsed'].apply(
            lambda x: [c['name'] for c in x[:5]] if x else []
        )
        
        # Extract director
        credits_clean['director'] = credits_clean['crew_parsed'].apply(
            lambda x: [c['name'] for c in x if c.get('job') == 'Director'] if x else []
        )
        
        return credits_clean
    
    def clean_keywords(self, keywords: pd.DataFrame) -> pd.DataFrame:
        """
        Clean keywords dataset.
        
        Args:
            keywords: Raw keywords DataFrame
            
        Returns:
            Cleaned keywords DataFrame
        """
        keywords_clean = keywords.copy()
        
        # Fix id column
        keywords_clean['id_numeric'] = pd.to_numeric(keywords_clean['id'], errors='coerce')
        keywords_clean = keywords_clean.dropna(subset=['id_numeric']).copy()
        keywords_clean['id'] = keywords_clean['id_numeric'].astype(int)
        keywords_clean = keywords_clean.drop(columns=['id_numeric'])
        
        # Parse keywords
        keywords_clean['keywords_parsed'] = keywords_clean['keywords'].apply(self.parse_json_field)
        keywords_clean['keyword_names'] = keywords_clean['keywords_parsed'].apply(
            lambda x: [k['name'] for k in x] if x else []
        )
        
        return keywords_clean
    
    def clean_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Clean ratings dataset.
        
        Args:
            ratings: Raw ratings DataFrame
            
        Returns:
            Cleaned ratings DataFrame
        """
        ratings_clean = ratings.copy()
        
        # Remove duplicates
        ratings_clean = ratings_clean.drop_duplicates()
        
        # Validate ratings range
        ratings_clean = ratings_clean[
            (ratings_clean['rating'] >= 0.5) & (ratings_clean['rating'] <= 5.0)
        ]
        
        return ratings_clean
    
    def clean_links(self, links: pd.DataFrame) -> pd.DataFrame:
        """
        Clean links dataset (MovieLens to TMDB/IMDB mapping).
        
        Args:
            links: Raw links DataFrame
            
        Returns:
            Cleaned links DataFrame
        """
        links_clean = links.copy()
        
        # Convert IDs to numeric
        if 'movieId' in links_clean.columns:
            links_clean['movieId'] = pd.to_numeric(links_clean['movieId'], errors='coerce')
        if 'tmdbId' in links_clean.columns:
            links_clean['tmdbId'] = pd.to_numeric(links_clean['tmdbId'], errors='coerce')
        if 'imdbId' in links_clean.columns:
            links_clean['imdbId'] = pd.to_numeric(links_clean['imdbId'], errors='coerce')
        
        # Keep rows with a valid MovieLens ID and de-duplicate mappings.
        links_clean = links_clean.dropna(subset=['movieId'])
        links_clean['movieId'] = links_clean['movieId'].astype(int)
        if 'tmdbId' in links_clean.columns:
            links_clean = links_clean.dropna(subset=['tmdbId'])
            links_clean['tmdbId'] = links_clean['tmdbId'].astype(int)
        links_clean = links_clean.drop_duplicates(subset=['movieId'])
        
        return links_clean
    
    def save_cleaned_data(self, movies_clean: pd.DataFrame, credits_clean: pd.DataFrame,
                         keywords_clean: pd.DataFrame, ratings_clean: pd.DataFrame,
                         links_clean: pd.DataFrame = None):
        """
        Save all cleaned datasets to processed directory.
        
        Args:
            movies_clean: Cleaned movies metadata DataFrame
            credits_clean: Cleaned credits DataFrame
            keywords_clean: Cleaned keywords DataFrame
            ratings_clean: Cleaned ratings DataFrame
            links_clean: Optional cleaned links DataFrame
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        movies_clean.to_csv(self.processed_dir / 'movies_clean.csv', index=False)
        credits_clean.to_csv(self.processed_dir / 'credits_clean.csv', index=False)
        keywords_clean.to_csv(self.processed_dir / 'keywords_clean.csv', index=False)
        ratings_clean.to_csv(self.processed_dir / 'ratings_clean.csv', index=False)
        
        if links_clean is not None:
            links_clean.to_csv(self.processed_dir / 'links_clean.csv', index=False)
        
        print("Cleaned data saved successfully!")
