"""Unified evaluation pipeline for all recommendation models.

Evaluation contract (must match across ALL runs):
  - K = 10
  - Relevance threshold = 4.0  (rating >= 4.0)
  - random_state = 42
  - Same test user set (derived deterministically from fixed seed)
"""

import io
import sys
# Ensure Unicode box-drawing chars work on Windows (cp1252 console)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.evaluation.metrics import MetricsCalculator
from src.models.collaborative_filtering import ItemBasedCF, MatrixFactorizationCF, UserBasedCF
from src.models.content_based import GenreBasedRecommender, TFIDFContentRecommender
from src.models.hybrid import HybridRecommender
from src.models.popularity import PopularityRecommender
from src.utils.helpers import get_reports_dir, load_config

# ── Evaluation constants ──────────────────────────────────────────────────────
K = 10
MIN_RATING = 4.0
RANDOM_STATE = 42
N_EVAL_USERS = 100
CF_MIN_RATINGS = 3          # Fix #2: was 5
CF_MAX_USERS = None         # UserBasedCF/ItemBasedCF internal cap (None = use all rows in cf_train)
CF_MAX_MOVIES = None        # same for movies
CONTENT_MAX_FEATURES = 10000
HYBRID_CONTENT_WEIGHT = 0.55
HYBRID_COLLAB_WEIGHT = 0.45
# ─────────────────────────────────────────────────────────────────────────────


def save_checkpoint(metrics: dict, tag: str, results_dir: Path) -> Path:
    """Save metrics to a timestamped checkpoint in results/."""
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = results_dir / f'checkpoint_{ts}_{tag}.json'
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Checkpoint saved → {path.name}")
    return path


def load_evaluation_data(
    sample_size: int = 0,
    random_state: int = RANDOM_STATE,
    min_user_ratings: int = 5,
):
    """Load the user-based train/test split, enforce overlap, return consistent data.

    If the existing processed files were created with a temporal split they will
    still work but overlap enforcement will drop cold-start test users.  Run
    data_preprocessing.ipynb first to regenerate with user_based method.
    """
    loader = DataLoader()
    movies_df, train_ratings, test_ratings = loader.load_preprocessed_data()

    # Enforce overlap — every evaluated user must have training history
    overlap_users = set(train_ratings['userId']) & set(test_ratings['userId'])
    train_ratings = train_ratings[train_ratings['userId'].isin(overlap_users)].copy()
    test_ratings = test_ratings[test_ratings['userId'].isin(overlap_users)].copy()

    print(f"Overlap users: {len(overlap_users):,}")
    print(f"Train ratings: {len(train_ratings):,} | Test ratings: {len(test_ratings):,}")

    if sample_size > 0 and len(train_ratings) > sample_size:
        # Sample-users approach: take all ratings for a random user subset
        rng = np.random.default_rng(random_state)
        user_counts = train_ratings.groupby('userId').size()
        eligible = user_counts[user_counts >= min_user_ratings].index.tolist()
        n_sample = min(len(eligible), sample_size // 100)  # ~100 ratings/user
        sampled_users = set(rng.choice(eligible, size=n_sample, replace=False).tolist())
        # Always include all test users so eval coverage is 100%
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
    target_users: int = 5_000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Prepare CF training data that guarantees sampled test users are included.

    Strategy:
      1. Sample up to target_users//2 test users (so evaluation covers them all).
      2. Fill the remainder with active non-test users for density.

    Capped at target_users to keep cosine-similarity matrices in memory.
    """
    rng = np.random.default_rng(random_state)
    all_test_users = test_ratings['userId'].unique()

    # Sample a manageable subset of test users
    n_test_sample = min(len(all_test_users), target_users // 2)
    sampled_test_users = set(
        rng.choice(all_test_users, size=n_test_sample, replace=False).tolist()
    )
    test_user_train = train_ratings[train_ratings['userId'].isin(sampled_test_users)]

    # Fill up to target_users with other active users
    other_train = train_ratings[~train_ratings['userId'].isin(sampled_test_users)]
    other_user_counts = other_train.groupby('userId').size()
    other_eligible = other_user_counts[other_user_counts >= CF_MIN_RATINGS].index

    n_extra = max(0, target_users - len(sampled_test_users))
    n_extra = min(n_extra, len(other_eligible))
    if n_extra > 0:
        extra_users = set(rng.choice(other_eligible, size=n_extra, replace=False).tolist())
        extra_train = other_train[other_train['userId'].isin(extra_users)]
        cf_train = pd.concat([test_user_train, extra_train], ignore_index=True)
    else:
        cf_train = test_user_train.copy()

    n_ratings = len(cf_train)
    n_users = cf_train['userId'].nunique()
    n_movies = cf_train['movieId'].nunique()
    print(f"CF train: {n_ratings:,} ratings | {n_users:,} users | {n_movies:,} movies")
    print(f"  (sampled {len(sampled_test_users):,} test users + {n_extra:,} other users)")
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

    print("\n── Training models ─────────────────────────────────────────────")
    print("Training popularity baseline...")
    pop_model = PopularityRecommender(train_ratings, movies_df)

    print("Training genre content model...")
    genre_model = GenreBasedRecommender(movies_df, train_ratings)

    print("Training TF-IDF content model (max_features=10000, genres 3×)...")
    tfidf_model = TFIDFContentRecommender(
        movies_df, train_ratings, max_features=CONTENT_MAX_FEATURES
    )

    print("Training user-based CF (min_ratings=3, no caps)...")
    user_cf = UserBasedCF(
        cf_train_ratings,
        n_neighbors=50,
        min_ratings=CF_MIN_RATINGS,
        max_users=CF_MAX_USERS,
        max_movies=CF_MAX_MOVIES,
    )

    print("Training item-based CF (min_ratings=3, no caps)...")
    item_cf = ItemBasedCF(
        cf_train_ratings,
        n_neighbors=50,
        min_ratings=CF_MIN_RATINGS,
        max_users=CF_MAX_USERS,
        max_movies=CF_MAX_MOVIES,
    )

    print("Training centered SVD matrix factorization...")
    mf_model = MatrixFactorizationCF(
        cf_train_ratings,
        n_components=50,
        min_ratings=CF_MIN_RATINGS,
        max_users=CF_MAX_USERS,
        max_movies=CF_MAX_MOVIES,
        random_state=random_state,
        validate=True,
    )

    print("Training two-tier hybrid recommender...")
    hybrid_model = HybridRecommender(
        pop_recommender=pop_model,
        genre_recommender=genre_model,
        movies_df=movies_df,
        train_ratings=train_ratings,
        user_cf_recommender=user_cf,
        item_cf_recommender=item_cf,
        mf_recommender=mf_model,
        content_recommender=tfidf_model,
        content_weight=HYBRID_CONTENT_WEIGHT,
        collab_weight=HYBRID_COLLAB_WEIGHT,
        candidate_pool_size=500,
        cold_start_threshold=3,
        cold_start_pop_weight=0.30,
    )

    models = {
        'popularity': pop_model,
        'user_cf': user_cf,
        'item_cf': item_cf,
        'matrix_factorization': mf_model,
        'genre_content': genre_model,
        'tfidf_content': tfidf_model,
        'hybrid': hybrid_model,
    }

    print("\n── Evaluating models (K=10, threshold=4.0) ─────────────────────")
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
    parser.add_argument('--sample-size', type=int, default=0,
                        help='0 = use all train data (recommended after user-based split)')
    parser.add_argument('--cf-target-users', type=int, default=5_000)
    parser.add_argument('--random-state', type=int, default=RANDOM_STATE)
    parser.add_argument('--min-rating', type=float, default=MIN_RATING)
    args = parser.parse_args()

    reports_dir = get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir = reports_dir  # checkpoints go alongside final reports

    movies_df, train_ratings, test_ratings = load_evaluation_data(
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    cf_train = prepare_cf_train(
        train_ratings, test_ratings,
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
    MetricsCalculator.save_metrics(new_metrics['popularity'],
                                   reports_dir / 'popularity_baseline_metrics.json')
    MetricsCalculator.save_metrics(
        {'user_based': new_metrics['user_cf'],
         'item_based': new_metrics['item_cf'],
         'matrix_factorization': new_metrics['matrix_factorization']},
        reports_dir / 'collaborative_filtering_metrics.json',
    )
    MetricsCalculator.save_metrics(
        {'genre_based': new_metrics['genre_content'],
         'tfidf_based': new_metrics['tfidf_content']},
        reports_dir / 'content_based_metrics.json',
    )
    MetricsCalculator.save_metrics(new_metrics['hybrid'], reports_dir / 'hybrid_metrics.json')

    # Checkpoint
    save_checkpoint(new_metrics, 'all_fixes', results_dir)

    print('\n=== Final Metrics ===')
    for model, m in new_metrics.items():
        print(f"{model:25s}  P@10={m['precision@k']:.4f}  R@10={m['recall@k']:.4f}  "
              f"F1={m['f1@k']:.4f}  Users={m['evaluated_users']}")


if __name__ == '__main__':
    main()
