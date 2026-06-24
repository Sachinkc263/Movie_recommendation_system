"""Visualization module for movie recommendation system (The Movies Dataset - Kaggle)."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from src.utils.helpers import parse_list_columns


class Plotter:
    """Create simple, clean visualizations for recommendation system analysis (The Movies Dataset - Kaggle)."""
    
    def __init__(self, figures_dir: Optional[Path] = None):
        """
        Initialize Plotter.
        
        Args:
            figures_dir: Directory to save figures. Defaults to data/figures
        """
        if figures_dir is None:
            self.figures_dir = Path(__file__).parent.parent.parent / 'data' / 'figures'
        else:
            self.figures_dir = Path(figures_dir)
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set simple, clean style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_rating_distribution(self, ratings: pd.DataFrame, save: bool = True):
        """
        Plot rating distribution (simple, clean bar chart).
        
        Args:
            ratings: Ratings DataFrame
            save: Whether to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        rating_counts = ratings['rating'].value_counts().sort_index()
        ax.bar(rating_counts.index, rating_counts.values, color='steelblue', alpha=0.7, width=0.4)
        ax.set_xlabel('Rating')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Ratings')
        ax.set_xticks(rating_counts.index)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.figures_dir / 'rating_distribution.png', dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_user_activity(self, ratings: pd.DataFrame, save: bool = True):
        """
        Plot user activity distribution.
        
        Args:
            ratings: Ratings DataFrame
            save: Whether to save the plot
        """
        user_activity = ratings.groupby("userId").size()
        
        fig, axes = plt.subplots(1, 1, figsize=(12, 5))
        
        top_users = user_activity.sort_values(ascending=False).head(10)
        axes.bar(
            top_users.index.astype(str),
            top_users.values,
            alpha=0.8
        )
        axes.set_title("Top 10 Most Active Users")
        axes.set_xlabel("User ID")
        axes.set_ylabel("Number of Ratings")
        axes.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.figures_dir / 'user_activity.png', dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_movie_popularity(self, ratings: pd.DataFrame, movies: pd.DataFrame, save: bool = True):
        """
        Plot movie popularity.
        
        Args:
            ratings: Ratings DataFrame
            movies: Movies DataFrame
            save: Whether to save the plot
        """
        movie_popularity = ratings.groupby('movieId').size()
        
        fig, axes = plt.subplots(1, 1, figsize=(14, 10))
        
        top_movies = movie_popularity.nlargest(20)
        top_movie_titles = (
            movies[['movieId', 'title']]
            .drop_duplicates('movieId')
            .set_index('movieId')
            .reindex(top_movies.index)['title']
            .fillna('Unknown')
        )
        axes.barh(range(len(top_movies)), top_movies.values, color='green', alpha=0.7)
        axes.set_yticks(range(len(top_movies)))
        axes.set_yticklabels(top_movie_titles, fontsize=8)
        axes.set_xlabel('Number of Ratings')
        axes.set_title('Top 20 Movies by Rating Count')
        axes.invert_yaxis()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.figures_dir / 'movie_popularity.png', dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_genre_distribution(self, movies: pd.DataFrame, save: bool = True):
        """
        Plot genre distribution (simple, clean bar chart).
        
        Args:
            movies: Movies DataFrame with genre_names (list format)
            save: Whether to save the plot
        """
        from collections import Counter
        
        movies = parse_list_columns(movies.copy(), ['genre_names'])
        all_genres = []
        for genre_names in movies['genre_names'].dropna():
            if isinstance(genre_names, list):
                all_genres.extend(genre_names)
        
        genre_counts = Counter(all_genres)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        genres_df = pd.DataFrame.from_dict(genre_counts, orient='index', columns=['count'])
        genres_df = genres_df.sort_values('count', ascending=True)
        ax.barh(genres_df.index, genres_df['count'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Count')
        ax.set_title('Genre Distribution')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.figures_dir / 'genre_distribution.png', dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_temporal_analysis(self, ratings: pd.DataFrame, movies: pd.DataFrame, save: bool = True):
        """
        Plot temporal analysis of ratings and movie releases (simple, clean line chart).
        
        Args:
            ratings: Ratings DataFrame
            movies: Movies DataFrame with year column
            save: Whether to save the plot
        """
        ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
        ratings['year'] = ratings['datetime'].dt.year
        
        # Use the year column from movies metadata instead of extracting from title
        movie_years = movies['year'].dropna() if 'year' in movies.columns else pd.Series()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        yearly_ratings = ratings.groupby('year').size()
        axes[0].plot(yearly_ratings.index, yearly_ratings.values, marker='o', color='steelblue', linewidth=2)
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of Ratings')
        axes[0].set_title('Ratings Over Time (Yearly)')
        axes[0].tick_params(axis='x', rotation=45)
        
        if len(movie_years) > 0:
            axes[1].hist(movie_years, bins=30, color='green', alpha=0.7)
            axes[1].set_xlabel('Release Year')
            axes[1].set_ylabel('Number of Movies')
            axes[1].set_title('Movie Release Year Distribution')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.figures_dir / 'temporal_analysis.png', dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, metrics_dict: dict, save: bool = True):
        """
        Plot model comparison based on metrics (simple, clean bar chart).
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics
            save: Whether to save the plot
        """
        comparison_df = pd.DataFrame(metrics_dict).T
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Plot F1 scores
        comparison_df['f1@k'].plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
        ax.set_ylabel('F1 Score')
        ax.set_title('Model Comparison - F1 Score')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.figures_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
        
        plt.show()
