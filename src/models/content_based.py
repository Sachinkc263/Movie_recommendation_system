"""Content-based recommendation models (The Movies Dataset - Kaggle)."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

from src.utils.helpers import ensure_movie_id, parse_list_columns


def _list_to_text(values) -> str:
    if isinstance(values, list):
        return ' '.join(str(v) for v in values if v)
    if values is None or (isinstance(values, float) and np.isnan(values)):
        return ''
    return str(values)


class GenreBasedRecommender:
    """Genre-based content recommendation with Bayesian movie scoring and enhanced features."""

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, use_keywords: bool = True):
        self.movies_df = ensure_movie_id(parse_list_columns(movies_df.copy(), ['genre_names', 'keyword_names']))
        self.ratings_df = ratings_df
        self.genre_stats = None
        self.keyword_stats = None
        self.user_genre_preferences: Dict[int, Dict[str, float]] = {}
        self.user_keyword_preferences: Dict[int, Dict[str, float]] = {}
        self.movie_scores: Dict[int, float] = {}
        self.use_keywords = use_keywords
        self._build_model()

    def _build_model(self):
        movie_genres = self.movies_df[['movieId', 'genre_names']].dropna(subset=['movieId'])
        genre_rows = []
        for _, row in movie_genres.iterrows():
            genres = row['genre_names']
            if isinstance(genres, list):
                for genre in genres:
                    genre_rows.append({'movieId': row['movieId'], 'genre': genre})

        self._genre_df = pd.DataFrame(genre_rows)
        genre_df = self._genre_df

        # Compute per-movie rating stats once (43k rows) — avoids the 56M-row full join.
        movie_stats = self.ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()

        # Genre stats: genre_df (86k rows) × movie_stats (43k rows) = 86k rows
        if genre_df.empty:
            self.genre_stats = pd.DataFrame(
                columns=['genre', 'avg_rating', 'n_ratings', 'std_rating', 'n_movies']
            )
        else:
            gm = genre_df.merge(movie_stats, on='movieId', how='inner')
            self.genre_stats = gm.groupby('genre').agg(
                avg_rating=('mean', 'mean'),
                n_ratings=('count', 'sum'),
                std_rating=('mean', 'std'),
                n_movies=('movieId', 'count'),
            ).reset_index()

        # Keyword stats: keyword_df (~430k rows) × movie_stats (43k rows) = ~430k rows
        self._keyword_df = pd.DataFrame()
        if self.use_keywords and 'keyword_names' in self.movies_df.columns:
            movie_keywords = self.movies_df[['movieId', 'keyword_names']].dropna(
                subset=['movieId', 'keyword_names']
            )
            keyword_rows = []
            for _, row in movie_keywords.iterrows():
                keywords = row['keyword_names']
                if isinstance(keywords, list):
                    for keyword in keywords:
                        keyword_rows.append({'movieId': row['movieId'], 'keyword': keyword})
            self._keyword_df = pd.DataFrame(keyword_rows)
            if not self._keyword_df.empty:
                km = self._keyword_df.merge(movie_stats, on='movieId', how='inner')
                self.keyword_stats = km.groupby('keyword').agg(
                    avg_rating=('mean', 'mean'),
                    n_ratings=('count', 'sum'),
                    n_movies=('movieId', 'count'),
                ).reset_index()

        global_mean = movie_stats['mean'].mean()
        smoothing = movie_stats['count'].quantile(0.90)
        movie_stats['score'] = (
            (movie_stats['count'] / (movie_stats['count'] + smoothing)) * movie_stats['mean']
            + (smoothing / (movie_stats['count'] + smoothing)) * global_mean
        )
        self.movie_scores = dict(zip(movie_stats['movieId'], movie_stats['score']))

        # User genre and keyword preferences are computed on-demand in recommend_for_user()
        # (only ~100 eval users ever call it).  Pre-computing for all 256k users would
        # require 800MB defaultdicts + a 500MB ratings_mini slice which OOMs in the
        # hybrid notebook after UserBasedCF has claimed 1.8GB for its similarity matrix.

        # Pre-build vectorized movie score arrays for fast recommend()
        self._movie_genre_sets: Dict[int, set] = {
            int(row['movieId']): set(row['genre_names']) if isinstance(row['genre_names'], list) else set()
            for _, row in self.movies_df.iterrows()
        }
        self._movie_keyword_sets: Dict[int, set] = {}
        if self.use_keywords and 'keyword_names' in self.movies_df.columns:
            self._movie_keyword_sets = {
                int(row['movieId']): set(row['keyword_names']) if isinstance(row['keyword_names'], list) else set()
                for _, row in self.movies_df.iterrows()
            }

        print(
            f"Genre-based model: {len(self.genre_stats)} genres, "
            f"{len(self.keyword_stats) if self.keyword_stats is not None else 0} keywords, "
            f"user prefs computed on-demand"
        )

    def recommend_for_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        uid = int(user_id)
        # One 20M-row scan per user; reuse for both user_rated and genre prefs.
        user_rows = self.ratings_df[self.ratings_df['userId'] == uid]
        user_rated = set(user_rows['movieId'].values)

        # Compute genre preferences on-demand: user_rows (~82 rows) × genre_df (86k → filtered
        # to user's movies) = tiny merge, negligible memory.
        if uid not in self.user_genre_preferences and user_rated and not self._genre_df.empty:
            user_movie_genres = self._genre_df[self._genre_df['movieId'].isin(user_rated)]
            if not user_movie_genres.empty:
                merged_small = user_rows[['movieId', 'rating']].merge(
                    user_movie_genres, on='movieId', how='inner'
                )
                if not merged_small.empty:
                    self.user_genre_preferences[uid] = (
                        merged_small.groupby('genre')['rating'].mean().to_dict()
                    )

        # Get preferred genres
        if uid in self.user_genre_preferences:
            preferred_genres = sorted(
                self.user_genre_preferences[uid].items(),
                key=lambda item: item[1], reverse=True,
            )[:5]
            genres = set(g for g, _ in preferred_genres)
        else:
            genres = set(self.genre_stats.sort_values('n_ratings', ascending=False)['genre'].head(5))

        # Get preferred keywords (lazy — only computed for queried users, ~100 total)
        keywords: set = set()
        if self.use_keywords and self.keyword_stats is not None:
            if uid not in self.user_keyword_preferences and user_rated and not self._keyword_df.empty:
                user_movie_kw = self._keyword_df[
                    self._keyword_df['movieId'].isin(user_rated)
                ].copy()
                if not user_movie_kw.empty:
                    user_ratings_row = self.ratings_df[
                        self.ratings_df['userId'] == user_id
                    ][['movieId', 'rating']]
                    user_movie_kw['rating'] = user_movie_kw['movieId'].map(
                        dict(zip(user_ratings_row['movieId'], user_ratings_row['rating']))
                    )
                    self.user_keyword_preferences[uid] = (
                        user_movie_kw.groupby('keyword')['rating'].mean().to_dict()
                    )
            if uid in self.user_keyword_preferences:
                keywords = set(
                    kw for kw, _ in sorted(
                        self.user_keyword_preferences[uid].items(),
                        key=lambda x: x[1], reverse=True,
                    )[:5]
                )
            else:
                keywords = set(
                    self.keyword_stats.sort_values('n_ratings', ascending=False)['keyword'].head(5)
                )

        # Fully vectorized scoring — avoids iterrows over 43k movies
        movies = self.movies_df[['movieId', 'title', 'genre_names']].copy()
        movie_ids = movies['movieId'].values.astype(int)

        if user_rated:
            unseen_mask = ~np.isin(movie_ids, list(user_rated))
            movies = movies[unseen_mask]
            movie_ids = movie_ids[unseen_mask]

        if len(movie_ids) == 0:
            return pd.DataFrame()

        genre_scores = np.array([
            float(len(genres & self._movie_genre_sets.get(mid, set()))) for mid in movie_ids
        ])
        keyword_scores = 0.5 * np.array([
            float(len(keywords & self._movie_keyword_sets.get(mid, set()))) for mid in movie_ids
        ])
        base_scores = np.array([self.movie_scores.get(mid, 0.0) for mid in movie_ids])
        total_scores = base_scores + genre_scores + keyword_scores

        top_count = min(n, len(total_scores))
        top_local = np.argpartition(total_scores, -top_count)[-top_count:]
        top_local = top_local[np.argsort(total_scores[top_local])[::-1]]

        result = movies.iloc[top_local].copy()
        result['score'] = total_scores[top_local]
        return result.reset_index(drop=True)

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        return self.recommend_for_user(user_id=user_id, n=n)


class TFIDFContentRecommender:
    """TF-IDF content recommender using combined 'soup' of all features (reference approach)."""

    def __init__(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        max_features: int = 10000,
        min_df: int = 2,
        max_df: float = 0.8,
        min_user_rating: float = 3.5,
    ):
        self.movies_df = ensure_movie_id(parse_list_columns(movies_df.copy(), [
            'genre_names', 'keyword_names', 'cast_names', 'director',
        ]))
        self.ratings_df = ratings_df
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.min_user_rating = min_user_rating
        self.vectorizer = None
        self.tfidf_matrix = None
        self.movie_id_to_idx: Dict[int, int] = {}
        self.idx_to_movie_id: Dict[int, int] = {}
        self._build_model()

    def _build_text_corpus(self, movies: pd.DataFrame) -> List[str]:
        texts = []
        for _, row in movies.iterrows():
            # genres repeated 3x — cheap weighting without a separate TF-IDF
            genres_raw = row.get('genre_names')
            genres_once = _list_to_text(genres_raw)
            genres_3x = (genres_once + ' ') * 3 if genres_once else ''

            # director: unwrap list → single string
            director_raw = row.get('director')
            if isinstance(director_raw, list):
                director_str = ' '.join(str(d) for d in director_raw if d)
            else:
                director_str = str(director_raw or '')

            # top-3 cast members only
            cast_raw = row.get('cast_names')
            if isinstance(cast_raw, list):
                cast_str = ' '.join(str(c) for c in cast_raw[:3] if c)
            else:
                cast_str = _list_to_text(cast_raw)

            parts = [
                genres_3x.strip(),
                str(row.get('overview', '') or ''),
                director_str,
                cast_str,
                _list_to_text(row.get('keyword_names')),
                str(row.get('title', '') or ''),
            ]
            texts.append(' '.join(part for part in parts if part))
        return texts

    def _build_model(self):
        movies = self.movies_df.dropna(subset=['movieId']).drop_duplicates('movieId')
        corpus = self._build_text_corpus(movies)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=self.min_df,
            max_df=self.max_df,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.movie_id_to_idx = {
            int(movie_id): idx for idx, movie_id in enumerate(movies['movieId'].values)
        }
        self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_idx.items()}
        self.movies_lookup = movies.set_index('movieId')
        print(f"TF-IDF content model: {len(self.movie_id_to_idx)} movies built")

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        # On-demand scan — only ~100 eval users call this, one 20M-row scan per user is fine
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        if len(user_ratings) == 0:
            return pd.DataFrame()

        liked = user_ratings[user_ratings['rating'] >= self.min_user_rating]
        if len(liked) == 0:
            liked = user_ratings.sort_values('rating', ascending=False).head(5)

        profile = None
        total_weight = 0.0
        for _, row in liked.iterrows():
            movie_id = int(row['movieId'])
            if movie_id not in self.movie_id_to_idx:
                continue
            idx = self.movie_id_to_idx[movie_id]
            weight = float(row['rating'])
            vector = self.tfidf_matrix[idx]
            profile = vector if profile is None else profile + (vector * weight)
            total_weight += weight

        if profile is None or total_weight == 0:
            return pd.DataFrame()

        profile = profile / total_weight
        similarities = cosine_similarity(profile, self.tfidf_matrix).ravel()

        seen = set(int(m) for m in user_ratings['movieId'])

        # Vectorized top-n selection (avoids Python loop over all 43k movies)
        all_ids = np.array([self.idx_to_movie_id[i] for i in range(len(similarities))])
        unseen_mask = ~np.isin(all_ids, list(seen))
        unseen_indices = np.where(unseen_mask)[0]
        if len(unseen_indices) == 0:
            return pd.DataFrame()

        top_n_local = np.argsort(similarities[unseen_indices])[::-1][:n]
        top_indices = unseen_indices[top_n_local]

        rows = []
        for idx in top_indices:
            movie_id = self.idx_to_movie_id[idx]
            if movie_id not in self.movies_lookup.index:
                continue
            movie = self.movies_lookup.loc[movie_id]
            rows.append({
                'movieId': movie_id,
                'title': movie.get('title', ''),
                'genre_names': movie.get('genre_names', []),
                'score': float(similarities[idx]),
            })
        return pd.DataFrame(rows)
