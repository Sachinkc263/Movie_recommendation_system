"""Unified evaluation pipeline for all recommendation models.

Evaluation contract (must match across ALL runs):
  - K = 10
  - Relevance threshold = 4.0  (rating >= 4.0)
  - random_state = 42
  - Same test user set (derived deterministically from fixed seed)

Key fixes in this version:
  - Eval users are pre-selected BEFORE CF training so every CF model
    sees all 100 evaluation users in its training matrix (fixes Users=1 bug).
  - ImplicitALS replaces explicit SVD as the 'matrix_factorization' model
    because ALS optimises ranking (P@k) rather than RMSE.
  - SVD and ALS both implement fold-in so they can recommend for any user
    even if they were not present in the CF training subset.
  - Hybrid weights rebalanced: collab 0.70 / content 0.30.
"""

import io
import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.evaluation.metrics import MetricsCalculator
from src.models.collaborative_filtering import (
    ImplicitALSRecommender,
    ItemBasedCF,
    MatrixFactorizationCF,
    UserBasedCF,
)
from src.models.content_based import GenreBasedRecommender, TFIDFContentRecommender
from src.models.hybrid import HybridRecommender
from src.models.popularity import PopularityRecommender
from src.utils.helpers import get_reports_dir

# ── Evaluation constants ──────────────────────────────────────────────────────
K = 10
MIN_RATING = 4.0
RANDOM_STATE = 42
N_EVAL_USERS = 100
CF_MIN_RATINGS = 3
CF_TARGET_USERS = 5_000   # total CF training users (includes all eval users)
CONTENT_MAX_FEATURES = 10_000

# Hybrid weights: favour ALS (strong ranking signal) over content
HYBRID_COLLAB_WEIGHT = 0.70
HYBRID_CONTENT_WEIGHT = 0.30

# ALS hyper-parameters (Hu, Koren & Volinsky 2008)
ALS_FACTORS = 100
ALS_ITERATIONS = 20
ALS_ALPHA = 40.0
ALS_REG = 0.01
# ─────────────────────────────────────────────────────────────────────────────


def save_checkpoint(metrics: dict, tag: str, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = results_dir / f'checkpoint_{ts}_{tag}.json'
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Checkpoint saved -> {path.name}")
    return path


def load_evaluation_data(
    sample_size: int = 0,
    random_state: int = RANDOM_STATE,
    min_user_ratings: int = 5,
):
    """Load the user-based train/test split and enforce overlap."""
    loader = DataLoader()
    movies_df, train_ratings, test_ratings = loader.load_preprocessed_data()

    overlap_users = set(train_ratings['userId']) & set(test_ratings['userId'])
    train_ratings = train_ratings[train_ratings['userId'].isin(overlap_users)].copy()
    test_ratings = test_ratings[test_ratings['userId'].isin(overlap_users)].copy()

    print(f"Overlap users: {len(overlap_users):,}")
    print(f"Train ratings: {len(train_ratings):,} | Test ratings: {len(test_ratings):,}")

    if sample_size > 0 and len(train_ratings) > sample_size:
        rng = np.random.default_rng(random_state)
        user_counts = train_ratings.groupby('userId').size()
        eligible = user_counts[user_counts >= min_user_ratings].index.tolist()
        n_sample = min(len(eligible), sample_size // 100)
        sampled_users = set(rng.choice(eligible, size=n_sample, replace=False).tolist())
        sampled_users |= set(test_ratings['userId'].unique())
        train_ratings = train_ratings[train_ratings['userId'].isin(sampled_users)].copy()
        print(f"Sampled train: {len(train_ratings):,} ratings from {train_ratings['userId'].nunique():,} users")

    return (
        movies_df,
        train_ratings.reset_index(drop=True),
        test_ratings.reset_index(drop=True),
    )


def prepare_cf_train(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    eval_users: Optional[np.ndarray] = None,
    target_users: int = CF_TARGET_USERS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Build CF training data that guarantees ALL eval_users are present.

    Strategy:
      1. Force-include all evaluation users (so CF models see them in training).
      2. Fill remaining slots with the most-active non-eval users for matrix density.

    This eliminates the "Users=1" bug where eval users were never in the CF
    training matrix and CF models returned empty recommendations for 99/100 users.
    """
    rng = np.random.default_rng(random_state)

    # Step 1 -- force all evaluation users into CF training
    forced_ids: set = set(int(u) for u in eval_users) if eval_users is not None and len(eval_users) > 0 else set()
    forced_train = train_ratings[train_ratings['userId'].isin(forced_ids)]

    # Step 2 -- fill remaining capacity with active non-eval users
    remaining = max(0, target_users - len(forced_ids))
    other_train = train_ratings[~train_ratings['userId'].isin(forced_ids)]

    if remaining > 0:
        other_counts = other_train.groupby('userId').size()
        other_eligible = other_counts[other_counts >= CF_MIN_RATINGS].index
        n_extra = min(remaining, len(other_eligible))
        if n_extra > 0:
            extra_users = set(rng.choice(other_eligible, size=n_extra, replace=False).tolist())
            extra_train = other_train[other_train['userId'].isin(extra_users)]
            cf_train = pd.concat([forced_train, extra_train], ignore_index=True)
        else:
            cf_train = forced_train.copy()
    else:
        cf_train = forced_train.copy()

    n_ratings = len(cf_train)
    n_users = cf_train['userId'].nunique()
    n_movies = cf_train['movieId'].nunique()
    n_forced = len(forced_ids & set(cf_train['userId'].unique()))
    print(f"CF train: {n_ratings:,} ratings | {n_users:,} users | {n_movies:,} movies")
    print(f"  (forced {n_forced}/{len(forced_ids)} eval users + {n_users - n_forced:,} filler users)")
    return cf_train


def evaluate_all_models(
    movies_df: pd.DataFrame,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    cf_train_ratings: pd.DataFrame,
    k: int = K,
    n_users: int = N_EVAL_USERS,
    random_state: int = RANDOM_STATE,
    min_rating: float = MIN_RATING,
) -> dict:
    eval_kwargs = dict(
        k=k,
        n_users=n_users,
        train_ratings=train_ratings,
        random_state=random_state,
        min_rating=min_rating,
    )

    print("\n-- Training models -------------------------------------------------")

    print("Training popularity baseline...")
    pop_model = PopularityRecommender(train_ratings, movies_df)

    print("Training genre content model...")
    genre_model = GenreBasedRecommender(movies_df, train_ratings)

    print("Training TF-IDF content model (max_features=10000, genres 3x)...")
    tfidf_model = TFIDFContentRecommender(
        movies_df, train_ratings, max_features=CONTENT_MAX_FEATURES
    )

    print("Training user-based CF (cosine similarity, min_ratings=3)...")
    user_cf = UserBasedCF(
        cf_train_ratings,
        n_neighbors=50,
        min_ratings=CF_MIN_RATINGS,
    )

    print("Training item-based CF (adjusted cosine, min_ratings=3)...")
    item_cf = ItemBasedCF(
        cf_train_ratings,
        n_neighbors=50,
        min_ratings=CF_MIN_RATINGS,
    )

    # SVD: kept for comparison; fold-in handles any user not in CF training
    print("Training centered SVD (n=50, no grid search)...")
    svd_model = MatrixFactorizationCF(
        cf_train_ratings,
        n_components=50,
        min_ratings=CF_MIN_RATINGS,
        random_state=random_state,
        validate=False,
    )

    # ImplicitALS: optimises ranking (P@k) rather than RMSE -- main CF model
    print(f"Training ImplicitALS (k={ALS_FACTORS}, iter={ALS_ITERATIONS}, alpha={ALS_ALPHA})...")
    als_model = ImplicitALSRecommender(
        cf_train_ratings,
        n_factors=ALS_FACTORS,
        n_iterations=ALS_ITERATIONS,
        regularization=ALS_REG,
        alpha=ALS_ALPHA,
        min_ratings=CF_MIN_RATINGS,
        random_state=random_state,
    )

    # Hybrid: ALS (70%) + TF-IDF (30%)
    print(f"Training hybrid (ALS {HYBRID_COLLAB_WEIGHT:.0%} / TF-IDF {HYBRID_CONTENT_WEIGHT:.0%})...")
    hybrid_model = HybridRecommender(
        pop_recommender=pop_model,
        genre_recommender=genre_model,
        movies_df=movies_df,
        train_ratings=train_ratings,
        user_cf_recommender=None,        # excluded: too sparse on 5K subset
        item_cf_recommender=None,        # excluded: same reason
        mf_recommender=als_model,        # ALS is the primary collab signal
        content_recommender=tfidf_model,
        content_weight=HYBRID_CONTENT_WEIGHT,
        collab_weight=HYBRID_COLLAB_WEIGHT,
        candidate_pool_size=500,
        cold_start_threshold=3,
        cold_start_pop_weight=0.30,
    )

    models = {
        'popularity':          pop_model,
        'user_cf':             user_cf,
        'item_cf':             item_cf,
        'matrix_factorization': als_model,   # ALS -- best ranking model
        'genre_content':       genre_model,
        'tfidf_content':       tfidf_model,
        'hybrid':              hybrid_model,
    }

    print("\n-- Evaluating models (K=10, threshold=4.0) -------------------------")
    metrics = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        metrics[name] = MetricsCalculator.evaluate_model_with_recommendations(
            model=model,
            test_ratings=test_ratings,
            **eval_kwargs,
        )
        m = metrics[name]
        print(
            f"  P@10={m['precision@k']:.4f}  R@10={m['recall@k']:.4f}  "
            f"F1={m['f1@k']:.4f}  HR={m['hit_rate']:.4f}  "
            f"Cov={m['coverage']:.3f}  Users={m['evaluated_users']}"
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate all recommendation models.')
    parser.add_argument('--k', type=int, default=K)
    parser.add_argument('--n-users', type=int, default=N_EVAL_USERS)
    parser.add_argument('--sample-size', type=int, default=0)
    parser.add_argument('--cf-target-users', type=int, default=CF_TARGET_USERS)
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE)
    parser.add_argument('--min-rating', type=float, default=MIN_RATING)
    args = parser.parse_args()

    reports_dir = get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)

    movies_df, train_ratings, test_ratings = load_evaluation_data(
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    # Pre-select the evaluation users BEFORE building the CF training set so
    # prepare_cf_train can guarantee every eval user appears in cf_train.
    # Without this, CF models see ~0 of the 100 eval users and return empty.
    print("\nPre-selecting evaluation users...")
    eval_users = MetricsCalculator._select_evaluation_users(
        test_ratings=test_ratings,
        train_ratings=train_ratings,
        n_users=args.n_users,
        random_state=args.random_state,
    )
    print(f"Selected {len(eval_users)} evaluation users")

    cf_train = prepare_cf_train(
        train_ratings,
        test_ratings,
        eval_users=eval_users,
        target_users=args.cf_target_users,
        random_state=args.random_state,
    )

    new_metrics = evaluate_all_models(
        movies_df=movies_df,
        train_ratings=train_ratings,
        test_ratings=test_ratings,
        cf_train_ratings=cf_train,
        k=args.k,
        n_users=args.n_users,
        random_state=args.random_state,
        min_rating=args.min_rating,
    )

    # Save metrics
    MetricsCalculator.save_metrics(new_metrics, reports_dir / 'all_models_metrics.json')
    MetricsCalculator.save_metrics(
        new_metrics['popularity'], reports_dir / 'popularity_baseline_metrics.json'
    )
    MetricsCalculator.save_metrics(
        {
            'user_based': new_metrics['user_cf'],
            'item_based': new_metrics['item_cf'],
            'matrix_factorization': new_metrics['matrix_factorization'],
        },
        reports_dir / 'collaborative_filtering_metrics.json',
    )
    MetricsCalculator.save_metrics(
        {'genre_based': new_metrics['genre_content'], 'tfidf_based': new_metrics['tfidf_content']},
        reports_dir / 'content_based_metrics.json',
    )
    MetricsCalculator.save_metrics(new_metrics['hybrid'], reports_dir / 'hybrid_metrics.json')

    save_checkpoint(new_metrics, 'all_fixes', reports_dir)

    print('\n=== Final Metrics ===')
    for model, m in new_metrics.items():
        print(
            f"{model:25s}  P@10={m['precision@k']:.4f}  R@10={m['recall@k']:.4f}  "
            f"F1={m['f1@k']:.4f}  Users={m['evaluated_users']}"
        )


if __name__ == '__main__':
    main()
