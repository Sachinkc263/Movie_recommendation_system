"""Evaluation metrics for recommendation systems (The Movies Dataset - Kaggle)."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Set

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class MetricsCalculator:
    """Calculate evaluation metrics for recommendation systems."""

    @staticmethod
    def _get_relevant_test_movies(
        user_test: pd.DataFrame,
        min_rating: Optional[float] = None,
    ) -> Set[int]:
        """Return test movie IDs considered relevant for a user."""
        if min_rating is not None:
            user_test = user_test[user_test['rating'] >= min_rating]
        return set(user_test['movieId'].values)

    @staticmethod
    def _select_evaluation_users(
        test_ratings: pd.DataFrame,
        train_ratings: Optional[pd.DataFrame],
        n_users: int,
        random_state: int,
        min_train_ratings: int = 5,
        min_test_items: int = 1,
    ) -> np.ndarray:
        """Select users with sufficient train history and test items."""
        test_counts = test_ratings.groupby('userId').size()
        eligible = test_counts[test_counts >= min_test_items].index

        if train_ratings is not None:
            train_users = set(train_ratings['userId'].unique())
            train_counts = train_ratings.groupby('userId').size()
            active_train_users = set(
                train_counts[train_counts >= min_train_ratings].index
            )
            eligible = [u for u in eligible if u in train_users and u in active_train_users]

        eligible = np.array(sorted(eligible))
        if len(eligible) == 0:
            return eligible

        rng = np.random.default_rng(random_state)
        if len(eligible) > n_users:
            eligible = rng.choice(eligible, size=n_users, replace=False)
        return eligible

    @staticmethod
    def _filter_seen_movies(
        recommendations: pd.DataFrame,
        user_id: int,
        train_ratings: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Remove movies the user already rated in training."""
        if train_ratings is None or len(recommendations) == 0:
            return recommendations

        seen = set(
            train_ratings.loc[train_ratings['userId'] == user_id, 'movieId'].values
        )
        if not seen:
            return recommendations
        return recommendations[~recommendations['movieId'].isin(seen)]

    @staticmethod
    def calculate_metrics(
        recommendations: pd.DataFrame,
        test_ratings: pd.DataFrame,
        k: int = 10,
        min_rating: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculate metrics for a single global recommendation list (non-personalized baselines).
        """
        recommended_movies = set(recommendations['movieId'].values[:k])
        test_users = test_ratings['userId'].unique()

        precision_scores = []
        recall_scores = []
        hit_count = 0

        for user_id in test_users:
            user_test = test_ratings[test_ratings['userId'] == user_id]
            user_test_movies = MetricsCalculator._get_relevant_test_movies(
                user_test, min_rating=min_rating
            )
            if not user_test_movies:
                continue

            hits = len(recommended_movies & user_test_movies)
            hit_count += hits
            precision_scores.append(hits / k)
            recall_scores.append(hits / len(user_test_movies))

        avg_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
        avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
        f1_score = (
            2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0.0
        )
        hit_rate = hit_count / (len(precision_scores) * k) if precision_scores else 0.0

        return {
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'f1@k': f1_score,
            'hit_rate': hit_rate,
            'evaluated_users': int(len(precision_scores)),
            'coverage': 1.0 if precision_scores else 0.0,
            'catalog_coverage': len(recommended_movies),
        }

    @staticmethod
    def evaluate_model_with_recommendations(
        model,
        test_ratings: pd.DataFrame,
        k: int = 10,
        n_users: int = 100,
        train_ratings: Optional[pd.DataFrame] = None,
        random_state: int = 42,
        min_rating: Optional[float] = 4.0,
        min_train_ratings: int = 5,
        min_test_items: int = 1,
        candidate_pool_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a model with a recommend() method using consistent holdout users.
        """
        test_users = MetricsCalculator._select_evaluation_users(
            test_ratings=test_ratings,
            train_ratings=train_ratings,
            n_users=n_users,
            random_state=random_state,
            min_train_ratings=min_train_ratings,
            min_test_items=min_test_items,
        )

        # Pre-build O(1) lookup dicts only for evaluation users (not full 256k set)
        eval_user_set = set(int(u) for u in test_users)

        _seen_by_user: Dict[int, Set[int]] = {}
        if train_ratings is not None:
            eval_train = train_ratings[train_ratings['userId'].isin(eval_user_set)]
            for uid, grp in eval_train.groupby('userId')['movieId']:
                _seen_by_user[int(uid)] = set(grp.values)

        eval_test = test_ratings[test_ratings['userId'].isin(eval_user_set)]
        _test_by_user: Dict[int, pd.DataFrame] = {
            int(uid): grp for uid, grp in eval_test.groupby('userId')
        }

        precision_scores = []
        recall_scores = []
        hit_count = 0
        evaluated_users = 0
        recommended_items: Set[int] = set()

        pool_size = candidate_pool_size or max(k * 5, k)

        for user_id in test_users:
            recommendations = model.recommend(user_id, n=pool_size)

            # Fast seen-movie filter using pre-built dict
            seen = _seen_by_user.get(int(user_id), set())
            if seen and len(recommendations) > 0:
                recommendations = recommendations[
                    ~recommendations['movieId'].isin(seen)
                ]

            if len(recommendations) == 0:
                continue

            recommendations = recommendations.head(k)
            evaluated_users += 1
            recommended_movies = set(recommendations['movieId'].values)
            recommended_items.update(recommended_movies)

            user_test = _test_by_user.get(int(user_id))
            if user_test is None:
                continue
            user_test_movies = MetricsCalculator._get_relevant_test_movies(
                user_test, min_rating=min_rating
            )
            if not user_test_movies:
                continue

            hits = len(recommended_movies & user_test_movies)
            hit_count += hits
            precision_scores.append(hits / k)
            recall_scores.append(hits / len(user_test_movies))

        avg_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
        avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
        f1_score = (
            2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            if (avg_precision + avg_recall) > 0
            else 0.0
        )
        hit_rate = hit_count / (evaluated_users * k) if evaluated_users > 0 else 0.0
        coverage = evaluated_users / len(test_users) if len(test_users) > 0 else 0.0
        all_test_movies = test_ratings['movieId'].nunique()
        catalog_coverage = (
            len(recommended_items) / all_test_movies if all_test_movies > 0 else 0.0
        )

        return {
            'precision@k': avg_precision,
            'recall@k': avg_recall,
            'f1@k': f1_score,
            'hit_rate': hit_rate,
            'evaluated_users': int(evaluated_users),
            'coverage': coverage,
            'catalog_coverage': catalog_coverage,
        }

    @staticmethod
    def compare_models(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Compare multiple models based on their metrics."""
        comparison_df = pd.DataFrame(metrics_dict).T
        sort_col = 'f1@k' if 'f1@k' in comparison_df.columns else comparison_df.columns[0]
        return comparison_df.sort_values(sort_col, ascending=False)

    @staticmethod
    def save_metrics(metrics: Dict[str, Any], save_path: str):
        """Save metrics to JSON file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

    @staticmethod
    def load_metrics(load_path: str) -> Dict[str, Any]:
        """Load metrics from JSON file."""
        with open(load_path, 'r', encoding='utf-8') as f:
            return json.load(f)
