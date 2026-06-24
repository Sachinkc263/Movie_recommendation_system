"""Hybrid recommendation system combining multiple approaches (The Movies Dataset - Kaggle)."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from src.utils.helpers import ensure_movie_id, parse_list_columns


def _extract_score(row: pd.Series) -> float:
    for col in ('hybrid_score', 'weighted_score', 'score', 'pred'):
        if col in row and pd.notna(row[col]):
            return float(row[col])
    return 0.0


def _score_dict(recommendations: pd.DataFrame) -> Dict[int, float]:
    if len(recommendations) == 0:
        return {}
    return {int(row['movieId']): _extract_score(row) for _, row in recommendations.iterrows()}


def _minmax_normalize(scores: Dict[int, float]) -> Dict[int, float]:
    """Min-max normalize scores over all candidates for this user."""
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRecommender:
    """Two-tier hybrid recommender with min-max normalization and cold-start fallback.

    Tier 1 (warm users, ≥3 train ratings):
        collab_score = mean(norm(user_cf), norm(item_cf), norm(svd_mf))  # available signals only
        final_score  = 0.55 * norm(content) + 0.45 * collab_score

    Tier 2 cold-start (< 3 train ratings):
        final_score  = 0.70 * final_score + 0.30 * norm(popularity)

    All normalization is min-max per component over each user's candidate pool.
    """

    def __init__(
        self,
        pop_recommender,
        genre_recommender,
        movies_df: Optional[pd.DataFrame] = None,
        train_ratings: Optional[pd.DataFrame] = None,
        user_cf_recommender=None,
        item_cf_recommender=None,
        mf_recommender=None,
        content_recommender=None,
        candidate_pool_size: int = 500,
        content_weight: float = 0.55,
        collab_weight: float = 0.45,
        cold_start_threshold: int = 3,
        cold_start_pop_weight: float = 0.30,
        # Legacy RRF params kept for backward-compat but ignored
        pop_weight: float = 0.15,
        genre_weight: float = 0.25,
        user_cf_weight: float = 0.15,
        item_cf_weight: float = 0.15,
        mf_weight: float = 0.20,
        rrf_k: int = 50,
        use_simple_hybrid: bool = True,
        content_weight_simple: float = 0.55,
        collab_weight_simple: float = 0.45,
    ):
        self.pop_recommender = pop_recommender
        self.genre_recommender = genre_recommender
        self.user_cf_recommender = user_cf_recommender
        self.item_cf_recommender = item_cf_recommender
        self.mf_recommender = mf_recommender
        self.content_recommender = content_recommender
        self.candidate_pool_size = candidate_pool_size
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        self.cold_start_threshold = cold_start_threshold
        self.cold_start_pop_weight = cold_start_pop_weight
        self.movies_df = (
            ensure_movie_id(parse_list_columns(movies_df.copy(), ['genre_names']))
            if movies_df is not None
            else None
        )
        # Pre-build user → train count lookup for cold-start detection
        if train_ratings is not None:
            self._user_train_counts: Dict[int, int] = (
                train_ratings.groupby('userId').size().to_dict()
            )
        else:
            self._user_train_counts = {}

    def _get_scores(self, recommender, user_id: int) -> Dict[int, float]:
        if recommender is None:
            return {}
        try:
            recs = recommender.recommend(user_id, n=self.candidate_pool_size)
        except Exception:
            return {}
        return _score_dict(recs)

    def _get_pop_scores(self, user_id: int) -> Dict[int, float]:
        try:
            recs = self.pop_recommender.recommend(user_id=user_id, n=self.candidate_pool_size)
        except Exception:
            return {}
        return _score_dict(recs)

    def _collect_candidates(self, user_id: int) -> Dict[int, float]:
        """Two-tier hybrid scoring."""
        content_scores = self._get_scores(self.content_recommender, user_id)
        user_cf_scores = self._get_scores(self.user_cf_recommender, user_id)
        item_cf_scores = self._get_scores(self.item_cf_recommender, user_id)
        mf_scores = self._get_scores(self.mf_recommender, user_id)
        pop_scores = self._get_pop_scores(user_id)

        # Normalize per component over this user's candidates
        content_norm = _minmax_normalize(content_scores)
        user_cf_norm = _minmax_normalize(user_cf_scores)
        item_cf_norm = _minmax_normalize(item_cf_scores)
        mf_norm = _minmax_normalize(mf_scores)
        pop_norm = _minmax_normalize(pop_scores)

        all_movies = (
            set(content_norm)
            | set(user_cf_norm)
            | set(item_cf_norm)
            | set(mf_norm)
            | set(pop_norm)
        )

        if not all_movies:
            return {}

        final_scores: Dict[int, float] = {}
        for movie_id in all_movies:
            # Collaborative score = mean of available CF signals for this movie
            cf_parts: List[float] = []
            if movie_id in user_cf_norm:
                cf_parts.append(user_cf_norm[movie_id])
            if movie_id in item_cf_norm:
                cf_parts.append(item_cf_norm[movie_id])
            if movie_id in mf_norm:
                cf_parts.append(mf_norm[movie_id])

            collab_score = float(np.mean(cf_parts)) if cf_parts else 0.0
            content_score = content_norm.get(movie_id, 0.0)

            # Tier 1 score
            score = self.content_weight * content_score + self.collab_weight * collab_score
            final_scores[movie_id] = score

        # Tier 2: cold-start fallback for users with <threshold train ratings
        user_train_count = self._user_train_counts.get(user_id, 0)
        if user_train_count < self.cold_start_threshold:
            main_w = 1.0 - self.cold_start_pop_weight
            for movie_id in final_scores:
                pop_score = pop_norm.get(movie_id, 0.0)
                final_scores[movie_id] = (
                    main_w * final_scores[movie_id]
                    + self.cold_start_pop_weight * pop_score
                )

        return final_scores

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        movies_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        final_scores = self._collect_candidates(user_id)

        if not final_scores:
            return pd.DataFrame()

        top_recs = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)[:n]
        scores_df = pd.DataFrame(top_recs, columns=['movieId', 'hybrid_score'])

        mdf = movies_df if movies_df is not None else self.movies_df
        if mdf is not None:
            mdf = ensure_movie_id(parse_list_columns(mdf.copy(), ['genre_names']))
            return scores_df.merge(
                mdf[['movieId', 'title', 'genre_names']],
                on='movieId',
                how='left',
            )
        return scores_df

    def save_model(self, save_dir):
        import pickle
        from pathlib import Path

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / 'hybrid_recommender.pkl', 'wb') as f:
            pickle.dump(self, f)
