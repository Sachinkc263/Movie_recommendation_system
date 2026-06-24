"""
Train ML models and export lightweight artifacts for fast API startup.

Run once before starting the API:
    python scripts/train_and_export.py

Artifacts saved to models/artifacts/:
  svd_U.npy          — user factors  (n_users × n_components)
  svd_Vt.npy         — item factors  (n_components × n_items)
  svd_meta.pkl       — dicts: user_idx, movie_idx, idx_movie, user_mean, global_mean
  tfidf_matrix.npz   — sparse TF-IDF matrix (n_movies × n_features)
  tfidf_meta.pkl     — dicts: movie_id_to_idx, idx_to_movie_id

Startup time after export: < 30 s (vs 30+ min of retraining).
"""

import gc
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Allow importing from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = ROOT / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = ROOT / "data" / "processed"


def load_data():
    logger.info("Loading processed data…")
    train = pd.read_csv(DATA_DIR / "train_ratings.csv")
    movies = pd.read_csv(DATA_DIR / "movies_integrated.csv", low_memory=False)
    logger.info("Train ratings: %s | Movies: %s", len(train), len(movies))
    return train, movies


def export_svd(train_ratings: pd.DataFrame):
    logger.info("Training centered TruncatedSVD (n_components=50)…")
    from scipy.sparse import csr_matrix as _csr
    from sklearn.decomposition import TruncatedSVD

    # Filter low-count users / movies
    user_counts = train_ratings["userId"].value_counts()
    movie_counts = train_ratings["movieId"].value_counts()
    eligible_users = user_counts[user_counts >= 3].index
    eligible_movies = movie_counts[movie_counts >= 3].index
    filtered = train_ratings[
        train_ratings["userId"].isin(eligible_users) &
        train_ratings["movieId"].isin(eligible_movies)
    ].copy()
    logger.info("Filtered ratings: %d  users: %d  movies: %d",
                len(filtered), filtered["userId"].nunique(), filtered["movieId"].nunique())

    unique_users = sorted(filtered["userId"].unique())
    unique_movies = sorted(filtered["movieId"].unique())
    user_idx = {u: i for i, u in enumerate(unique_users)}
    movie_idx = {m: i for i, m in enumerate(unique_movies)}
    idx_movie = {i: m for m, i in movie_idx.items()}

    # User-mean centering
    user_mean = filtered.groupby("userId")["rating"].mean().to_dict()
    global_mean = float(filtered["rating"].mean())

    centered_df = filtered.copy()
    centered_df["rating"] = centered_df["rating"] - centered_df["userId"].map(user_mean)

    rows = centered_df["userId"].map(user_idx).values
    cols = centered_df["movieId"].map(movie_idx).values
    vals = centered_df["rating"].values.astype(np.float32)
    centered_matrix = _csr((vals, (rows, cols)), shape=(len(unique_users), len(unique_movies)))
    del centered_df, filtered
    gc.collect()

    logger.info("Fitting SVD…")
    svd = TruncatedSVD(n_components=50, random_state=42)
    U = svd.fit_transform(centered_matrix)  # (n_users, 50)
    Vt = svd.components_                    # (50, n_items)

    logger.info("Saving SVD artifacts…")
    np.save(ARTIFACTS_DIR / "svd_U.npy", U.astype(np.float32))
    np.save(ARTIFACTS_DIR / "svd_Vt.npy", Vt.astype(np.float32))
    meta = {
        "user_idx": user_idx,
        "movie_idx": movie_idx,
        "idx_movie": idx_movie,
        "user_mean": user_mean,
        "global_mean": global_mean,
    }
    with open(ARTIFACTS_DIR / "svd_meta.pkl", "wb") as f:
        pickle.dump(meta, f, protocol=4)
    logger.info("SVD artifacts saved: U=%s  Vt=%s", U.shape, Vt.shape)
    del U, Vt, centered_matrix
    gc.collect()


def export_tfidf(movies_df: pd.DataFrame):
    logger.info("Building TF-IDF corpus…")
    import ast

    from sklearn.feature_extraction.text import TfidfVectorizer

    def _safe_list(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str) and v.startswith("["):
            try:
                return ast.literal_eval(v)
            except Exception:
                pass
        return []

    movies = movies_df.dropna(subset=["movieId"]).drop_duplicates("movieId").copy()
    movies["movieId"] = movies["movieId"].astype(int)

    corpus = []
    for _, row in movies.iterrows():
        genres = _safe_list(row.get("genre_names"))
        genres_3x = (" ".join(genres) + " ") * 3 if genres else ""

        director_raw = row.get("director")
        if isinstance(director_raw, list):
            director = " ".join(str(d) for d in director_raw if d)
        else:
            director = str(director_raw or "")

        cast_raw = row.get("cast_names")
        cast = " ".join(str(c) for c in (_safe_list(cast_raw)[:3]) if c)

        parts = [
            genres_3x.strip(),
            str(row.get("overview") or ""),
            director,
            cast,
            " ".join(_safe_list(row.get("keyword_names"))),
            str(row.get("title") or ""),
        ]
        corpus.append(" ".join(p for p in parts if p))

    vectorizer = TfidfVectorizer(
        max_features=10_000, stop_words="english",
        ngram_range=(1, 2), min_df=2, max_df=0.8,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    movie_ids = movies["movieId"].values
    movie_id_to_idx = {int(mid): i for i, mid in enumerate(movie_ids)}
    idx_to_movie_id = {i: int(mid) for mid, i in movie_id_to_idx.items()}

    logger.info("TF-IDF matrix: %s", tfidf_matrix.shape)
    sp.save_npz(str(ARTIFACTS_DIR / "tfidf_matrix.npz"), tfidf_matrix)
    with open(ARTIFACTS_DIR / "tfidf_meta.pkl", "wb") as f:
        pickle.dump({
            "movie_id_to_idx": movie_id_to_idx,
            "idx_to_movie_id": idx_to_movie_id,
        }, f, protocol=4)
    logger.info("TF-IDF artifacts saved.")


def export_popularity(train_ratings: pd.DataFrame, movies_df: pd.DataFrame):
    """Bayesian popularity score: weighted rating + log(count)."""
    logger.info("Computing popularity scores…")
    stats = train_ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    smoothing = float(stats["count"].quantile(0.90))
    global_mean = float(stats["mean"].mean())
    stats["score"] = (
        (stats["count"] / (stats["count"] + smoothing)) * stats["mean"]
        + (smoothing / (stats["count"] + smoothing)) * global_mean
    )
    out = ROOT / "models" / "popularity_scores.csv"
    stats[["movieId", "score"]].to_csv(out, index=False)
    logger.info("Popularity scores saved → %s", out)


if __name__ == "__main__":
    train, movies = load_data()
    export_popularity(train, movies)
    export_svd(train)
    export_tfidf(movies)
    logger.info("All artifacts exported to %s", ARTIFACTS_DIR)
    logger.info("You can now start the API: uvicorn api.main:app --host 0.0.0.0 --port 8000")
