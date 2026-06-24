"""
Run the full data preprocessing pipeline.

Step 1: Clean raw datasets  → data/processed/*_clean.csv
Step 2: Merge + split       → movies_integrated.csv, train/test_ratings.csv
Step 3: Create features     → user/movie_features.csv, user_item_matrix.npz

Run before train_and_export.py:
    python scripts/run_preprocessing.py
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.data.data_preprocessor import DataPreprocessor


def main():
    loader = DataLoader()
    cleaner = DataCleaner()
    preprocessor = DataPreprocessor()

    # ── Step 1: Clean raw data ────────────────────────────────────────────────
    log.info("Step 1/3 — Cleaning raw datasets")

    log.info("  Loading movies_metadata.csv ...")
    movies_raw = loader.load_movies_metadata()
    movies_clean = cleaner.clean_movies_metadata(movies_raw)
    log.info("  movies: %d -> %d rows", len(movies_raw), len(movies_clean))
    del movies_raw

    log.info("  Loading credits.csv ...")
    credits_raw = loader.load_credits()
    credits_clean = cleaner.clean_credits(credits_raw)
    log.info("  credits: %d -> %d rows", len(credits_raw), len(credits_clean))
    del credits_raw

    log.info("  Loading keywords.csv ...")
    keywords_raw = loader.load_keywords()
    keywords_clean = cleaner.clean_keywords(keywords_raw)
    log.info("  keywords: %d -> %d rows", len(keywords_raw), len(keywords_clean))
    del keywords_raw

    log.info("  Loading links.csv ...")
    links_raw = loader.load_links()
    links_clean = cleaner.clean_links(links_raw)
    log.info("  links: %d -> %d rows", len(links_raw), len(links_clean))
    del links_raw

    log.info("  Loading ratings.csv (677 MB — this takes a few minutes) ...")
    ratings_raw = loader.load_ratings()
    ratings_clean = cleaner.clean_ratings(ratings_raw)
    log.info("  ratings: %d -> %d rows", len(ratings_raw), len(ratings_clean))
    del ratings_raw

    cleaner.save_cleaned_data(
        movies_clean=movies_clean,
        credits_clean=credits_clean,
        keywords_clean=keywords_clean,
        ratings_clean=ratings_clean,
        links_clean=links_clean,
    )
    log.info("Step 1 complete — cleaned files saved to data/processed/")

    # ── Step 2: Merge + split ─────────────────────────────────────────────────
    log.info("Step 2/3 — Merging datasets and creating train/test split")

    movies_integrated = preprocessor.merge_datasets(
        movies_clean=movies_clean,
        credits_clean=credits_clean,
        keywords_clean=keywords_clean,
        links_clean=links_clean,
    )
    log.info("  movies_integrated: %s", movies_integrated.shape)

    movies_integrated = preprocessor.normalize_features(
        movies_integrated,
        ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"],
    )
    movies_integrated, _ = preprocessor.create_genre_dummies(movies_integrated)

    ratings_processed = preprocessor.process_ratings(ratings_clean)
    del ratings_clean

    user_item_matrix, user_mapping, movie_mapping = preprocessor.create_user_item_matrix(ratings_processed)
    log.info("  user-item matrix: %s", user_item_matrix.shape)

    log.info("  Creating user-based train/test split (80/20) ...")
    train_ratings, test_ratings = preprocessor.create_train_test_split(
        ratings_processed,
        test_size=0.2,
        method="user_based",
        min_ratings_per_user=5,
        random_state=42,
    )
    log.info("  train: %d rows  test: %d rows", len(train_ratings), len(test_ratings))

    user_features = preprocessor.create_user_features(train_ratings)
    movie_features = preprocessor.create_movie_features(train_ratings)

    preprocessor.save_preprocessed_data(
        movies_integrated=movies_integrated,
        train_ratings=train_ratings,
        test_ratings=test_ratings,
        user_features=user_features,
        movie_features=movie_features,
        user_item_matrix=user_item_matrix,
        user_to_index=user_mapping,
        movie_to_index=movie_mapping,
    )
    log.info("Step 2 complete — preprocessed files saved to data/processed/")

    log.info("Step 3/3 — Done. Run next: python scripts/train_and_export.py")


if __name__ == "__main__":
    main()
