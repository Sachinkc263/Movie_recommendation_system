"""Data loading module for movie recommendation system."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.utils.helpers import parse_list_columns


class DataLoader:
    """Load and manage movie recommendation datasets (The Movies Dataset - Kaggle)."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Path to data directory. Defaults to ../data/raw
        """
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent.parent / 'data' / 'raw'
        else:
            self.data_dir = Path(data_dir)
        
        self.processed_dir = self.data_dir.parent / 'processed'
    
    def load_movies_metadata(self, nrows: Optional[int] = None, clean: bool = False) -> pd.DataFrame:
        """
        Load movies metadata dataset.
        
        Args:
            nrows: If specified, limit number of rows
            clean: If True, load cleaned data from processed directory
            
        Returns:
            Movies metadata DataFrame
        """
        if clean:
            df = pd.read_csv(self.processed_dir / 'movies_clean.csv', nrows=nrows)
            return parse_list_columns(df, ['genres_parsed', 'genre_names'])
        return pd.read_csv(self.data_dir / 'movies_metadata.csv', nrows=nrows)
    
    def load_credits(self, nrows: Optional[int] = None, clean: bool = False) -> pd.DataFrame:
        """
        Load credits dataset (cast and crew).
        
        Args:
            nrows: If specified, limit number of rows
            clean: If True, load cleaned data from processed directory
            
        Returns:
            Credits DataFrame
        """
        if clean:
            df = pd.read_csv(self.processed_dir / 'credits_clean.csv', nrows=nrows)
            return parse_list_columns(df, ['cast_parsed', 'crew_parsed', 'cast_names', 'director'])
        return pd.read_csv(self.data_dir / 'credits.csv', nrows=nrows)
    
    def load_keywords(self, nrows: Optional[int] = None, clean: bool = False) -> pd.DataFrame:
        """
        Load keywords dataset.
        
        Args:
            nrows: If specified, limit number of rows
            clean: If True, load cleaned data from processed directory
            
        Returns:
            Keywords DataFrame
        """
        if clean:
            df = pd.read_csv(self.processed_dir / 'keywords_clean.csv', nrows=nrows)
            return parse_list_columns(df, ['keywords_parsed', 'keyword_names'])
        return pd.read_csv(self.data_dir / 'keywords.csv', nrows=nrows)
    
    def load_links(self, use_small: bool = False, clean: bool = False) -> pd.DataFrame:
        """
        Load links dataset.
        
        Args:
            use_small: If True, load links_small.csv instead of links.csv
            clean: If True, load cleaned data from processed directory
            
        Returns:
            Links DataFrame
        """
        if clean:
            return pd.read_csv(self.processed_dir / 'links_clean.csv')
        filename = 'links_small.csv' if use_small else 'links.csv'
        return pd.read_csv(self.data_dir / filename)
    
    def load_ratings(self, use_small: bool = False, chunksize: Optional[int] = None,
                     nrows: Optional[int] = None, clean: bool = False) -> pd.DataFrame:
        """
        Load ratings dataset.
        
        Args:
            use_small: If True, load ratings_small.csv instead of ratings.csv
            chunksize: If specified, load in chunks
            nrows: If specified, limit number of rows
            clean: If True, load cleaned data from processed directory
            
        Returns:
            Ratings DataFrame
        """
        if clean:
            return pd.read_csv(self.processed_dir / 'ratings_clean.csv', nrows=nrows)
        filename = 'ratings_small.csv' if use_small else 'ratings.csv'
        
        if chunksize:
            chunks = []
            for chunk in pd.read_csv(self.data_dir / filename, chunksize=chunksize):
                chunks.append(chunk)
                if nrows and len(chunks) * chunksize >= nrows:
                    break
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.read_csv(self.data_dir / filename, nrows=nrows)
    
    def load_cleaned_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all cleaned datasets.
        
        Returns:
            Tuple of (movies_clean, credits_clean, keywords_clean, ratings_clean)
        """
        movies_clean = self.load_movies_metadata(clean=True)
        credits_clean = self.load_credits(clean=True)
        keywords_clean = self.load_keywords(clean=True)
        ratings_clean = pd.read_csv(self.processed_dir / 'ratings_clean.csv')
        
        return movies_clean, credits_clean, keywords_clean, ratings_clean
    
    def load_preprocessed_data(self, filename: Optional[str] = None,
                               nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed data for modeling.

        Args:
            filename: Optional processed filename to load. If omitted, load the
                standard modeling tuple.
            nrows: If filename is provided, limit rows loaded.
        
        Returns:
            Tuple of (movies_integrated, train_ratings, test_ratings), or a
            single DataFrame when filename is provided.
        """
        if filename is not None:
            aliases = {
                'merged_data.csv': 'movies_integrated.csv',
            }
            filename = aliases.get(filename, filename)
            df = pd.read_csv(self.processed_dir / filename, nrows=nrows)
            if filename in {'movies_integrated.csv', 'movies_clean.csv', 'credits_clean.csv', 'keywords_clean.csv'}:
                df = parse_list_columns(
                    df,
                    ['genres_parsed', 'genre_names', 'cast_names', 'director', 'keyword_names']
                )
            return df

        movies_integrated = pd.read_csv(self.processed_dir / 'movies_integrated.csv')
        movies_integrated = parse_list_columns(
            movies_integrated,
            ['genres_parsed', 'genre_names', 'cast_names', 'director', 'keyword_names']
        )
        train_ratings = pd.read_csv(self.processed_dir / 'train_ratings.csv')
        test_ratings = pd.read_csv(self.processed_dir / 'test_ratings.csv')
        
        return movies_integrated, train_ratings, test_ratings
    
    def load_sampled_data(self, train_n: int = 100000, test_n: int = 20000,
                         use_small: bool = False, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load sampled data for memory-efficient modeling.
        
        Args:
            train_n: Number of training samples
            test_n: Number of test samples
            use_small: If True, use ratings_small.csv
            random_state: Random seed
            
        Returns:
            Tuple of (movies_integrated, train_ratings, test_ratings)
        """
        movies_integrated = pd.read_csv(self.processed_dir / 'movies_integrated.csv')
        movies_integrated = parse_list_columns(
            movies_integrated,
            ['genres_parsed', 'genre_names', 'cast_names', 'director', 'keyword_names']
        )
        
        # Load sampled training data
        chunksize = 100000
        train_chunks = []
        for chunk in pd.read_csv(self.processed_dir / 'train_ratings.csv', 
                                chunksize=chunksize):
            train_chunks.append(chunk)
            if len(train_chunks) >= 1:
                break
        train_ratings = pd.concat(train_chunks, ignore_index=True)
        train_ratings = train_ratings.sample(n=min(train_n, len(train_ratings)), random_state=random_state)
        
        # Load sampled test data
        test_ratings = pd.read_csv(self.processed_dir / 'test_ratings.csv', 
                                   nrows=test_n * 2)
        test_ratings = test_ratings.sample(n=min(test_n, len(test_ratings)), random_state=random_state)
        
        return movies_integrated, train_ratings, test_ratings
