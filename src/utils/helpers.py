"""Utility helper functions for movie recommendation system."""

import ast
import json
import pickle
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in {'.yaml', '.yml'}:
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: Path):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in {'.yaml', '.yml'}:
            import yaml
            yaml.safe_dump(config, f, sort_keys=False)
        else:
            json.dump(config, f, indent=2)


def parse_list_cell(value: Any) -> list:
    """
    Parse list-like values loaded from CSV back into Python lists.
    """
    if isinstance(value, list):
        return value
    if value is None:
        return []
    try:
        if hasattr(value, "__float__") and not isinstance(value, str) and np.isnan(value):
            return []
    except Exception:
        pass

    if not isinstance(value, str):
        return []

    value = value.strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return []

    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def parse_list_columns(df, columns):
    """
    Convert selected DataFrame columns from CSV strings to Python lists.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(parse_list_cell)
    return df


def ensure_movie_id(df, processed_dir: Path = None):
    """
    Add MovieLens movieId using links_clean.csv when a movie DataFrame only has TMDB id.
    """
    if 'movieId' in df.columns or 'id' not in df.columns:
        return df

    if processed_dir is None:
        processed_dir = get_processed_dir()
    else:
        processed_dir = Path(processed_dir)

    links_path = processed_dir / 'links_clean.csv'
    if not links_path.exists():
        return df

    links = pd.read_csv(links_path, usecols=['movieId', 'tmdbId'])
    links = links.dropna(subset=['movieId', 'tmdbId']).copy()
    links[['movieId', 'tmdbId']] = links[['movieId', 'tmdbId']].astype(int)

    movies = df.copy()
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.merge(links, left_on='id', right_on='tmdbId', how='left')
    return movies


def ensure_directory(directory: Path):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    directory.mkdir(parents=True, exist_ok=True)


def save_model(model: Any, save_path: Path):
    """
    Save model to pickle file.
    
    Args:
        model: Model object to save
        save_path: Path to save model
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(load_path: Path) -> Any:
    """
    Load model from pickle file.
    
    Args:
        load_path: Path to load model from
        
    Returns:
        Loaded model object
    """
    with open(load_path, 'rb') as f:
        return pickle.load(f)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """
    Get the data directory.
    
    Returns:
        Path to data directory
    """
    return get_project_root() / 'data'


def get_processed_dir() -> Path:
    """
    Get the processed data directory.
    
    Returns:
        Path to processed data directory
    """
    return get_data_dir() / 'processed'


def get_models_dir() -> Path:
    """
    Get the models directory.
    
    Returns:
        Path to models directory
    """
    return get_project_root() / 'models'


def get_reports_dir() -> Path:
    """
    Get the reports directory.
    
    Returns:
        Path to reports directory
    """
    return get_project_root() / 'reports'
