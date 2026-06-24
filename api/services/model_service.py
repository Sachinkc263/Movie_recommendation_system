"""ML recommendation service — loads pre-saved artifacts and serves recs."""

import gc
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from api.config import settings

logger = logging.getLogger(__name__)


class ModelService:
    """
    Loads SVD-MF + TF-IDF artifacts saved by scripts/train_and_export.py.
    Falls back to popularity when artifacts are absent.

    Folding-in: new API users with ≥ cold_start_threshold rated interactions
    get a virtual SVD user-factor via the normal equations, giving genuine
    personalisation without retraining.
    """

    def __init__(self):
        self._svd_ready = False
        self._tfidf_ready = False

        # SVD artifacts
        self._U: Optional[np.ndarray] = None          # (n_users, n_components)
        self._Vt: Optional[np.ndarray] = None         # (n_components, n_items)
        self._user_mean: Dict[int, float] = {}
        self._global_mean: float = 3.5
        self._user_idx: Dict[int, int] = {}
        self._movie_idx: Dict[int, int] = {}
        self._idx_movie: Dict[int, int] = {}

        # TF-IDF artifacts
        self._tfidf_matrix = None                     # scipy sparse (n_movies, n_features)
        self._movie_id_to_idx: Dict[int, int] = {}
        self._idx_to_movie_id: Dict[int, int] = {}
        self._movie_ids_array: Optional[np.ndarray] = None

        # Popularity
        self._pop_scores: Dict[int, float] = {}

        # Movie metadata lookup (movieId → row dict)
        self._movies_meta: Optional[pd.DataFrame] = None
        self._movie_count: int = 0

        # Regularisation for fold-in (empirically: 0.01 * n_components)
        self._lam: float = 0.5

    # ─── Loading ──────────────────────────────────────────────────────────────

    def load(self):
        """Load all artifacts. Called once at API startup."""
        self._load_movies_metadata()
        self._load_popularity()
        self._load_svd()
        self._load_tfidf()
        gc.collect()
        logger.info(
            "ModelService ready — SVD:%s  TF-IDF:%s  movies:%d",
            self._svd_ready, self._tfidf_ready, self._movie_count,
        )

    def _load_movies_metadata(self):
        path = settings.DATA_DIR / "movies_integrated.csv"
        if not path.exists():
            logger.warning("movies_integrated.csv not found at %s", path)
            return
        cols = [
            "movieId", "title", "year", "genre_names", "overview",
            "poster_path", "tmdbId", "vote_average", "vote_count",
            "popularity", "director", "cast_names", "keyword_names",
            "runtime", "original_language", "tagline",
        ]
        avail = pd.read_csv(path, nrows=0).columns.tolist()
        load_cols = [c for c in cols if c in avail]
        df = pd.read_csv(path, usecols=load_cols, low_memory=False)
        df = df.dropna(subset=["movieId"]).drop_duplicates("movieId")
        df["movieId"] = df["movieId"].astype(int)
        df = df.set_index("movieId")
        self._movies_meta = df
        self._movie_count = len(df)
        logger.info("Loaded %d movies from metadata", self._movie_count)

    def _load_popularity(self):
        path = settings.MODELS_DIR / "popularity_scores.csv"
        if not path.exists():
            logger.warning("popularity_scores.csv not found")
            return
        df = pd.read_csv(path)
        id_col = "movieId" if "movieId" in df.columns else df.columns[0]
        score_col = "score" if "score" in df.columns else df.columns[1]
        self._pop_scores = dict(zip(df[id_col].astype(int), df[score_col].astype(float)))
        logger.info("Loaded %d popularity scores", len(self._pop_scores))

    def _load_svd(self):
        art = settings.ARTIFACTS_DIR
        required = ["svd_U.npy", "svd_Vt.npy", "svd_meta.pkl"]
        if not all((art / f).exists() for f in required):
            logger.warning("SVD artifacts missing — run scripts/train_and_export.py first")
            return
        try:
            self._U = np.load(art / "svd_U.npy", mmap_mode="r")
            self._Vt = np.load(art / "svd_Vt.npy")
            with open(art / "svd_meta.pkl", "rb") as f:
                meta = pickle.load(f)
            self._user_mean = meta["user_mean"]
            self._global_mean = float(meta["global_mean"])
            self._user_idx = meta["user_idx"]
            self._movie_idx = meta["movie_idx"]
            self._idx_movie = meta["idx_movie"]
            self._lam = 0.01 * self._Vt.shape[0]
            self._svd_ready = True
            logger.info(
                "SVD loaded — %d users, %d items, %d components",
                self._U.shape[0], self._Vt.shape[1], self._Vt.shape[0],
            )
        except Exception as exc:
            logger.error("Failed loading SVD artifacts: %s", exc)

    def _load_tfidf(self):
        art = settings.ARTIFACTS_DIR
        required = ["tfidf_matrix.npz", "tfidf_meta.pkl"]
        if not all((art / f).exists() for f in required):
            logger.warning("TF-IDF artifacts missing — run scripts/train_and_export.py first")
            return
        try:
            import scipy.sparse as sp
            self._tfidf_matrix = sp.load_npz(str(art / "tfidf_matrix.npz"))
            with open(art / "tfidf_meta.pkl", "rb") as f:
                meta = pickle.load(f)
            self._movie_id_to_idx = meta["movie_id_to_idx"]
            self._idx_to_movie_id = meta["idx_to_movie_id"]
            self._movie_ids_array = np.array(
                [self._idx_to_movie_id[i] for i in range(self._tfidf_matrix.shape[0])]
            )
            self._tfidf_ready = True
            logger.info(
                "TF-IDF loaded — %d movies, %d features",
                self._tfidf_matrix.shape[0], self._tfidf_matrix.shape[1],
            )
        except Exception as exc:
            logger.error("Failed loading TF-IDF artifacts: %s", exc)

    # ─── Public API ───────────────────────────────────────────────────────────

    @property
    def is_svd_ready(self) -> bool:
        return self._svd_ready

    @property
    def is_tfidf_ready(self) -> bool:
        return self._tfidf_ready

    @property
    def movie_count(self) -> int:
        return self._movie_count

    # ── Popular ───────────────────────────────────────────────────────────────

    def get_popular(
        self,
        n: int = 20,
        exclude_ids: Optional[List[int]] = None,
        genres: Optional[List[str]] = None,
    ) -> List[Tuple[int, float]]:
        """Return (movie_id, score) sorted by popularity score."""
        scores = self._pop_scores
        if not scores and self._movies_meta is not None:
            # Fallback: sort by vote_average × log(vote_count)
            df = self._movies_meta.copy()
            if "vote_average" in df.columns and "vote_count" in df.columns:
                df["_s"] = df["vote_average"] * np.log1p(df["vote_count"].fillna(0))
                scores = df["_s"].to_dict()

        exclude = set(exclude_ids or [])
        results = []
        for mid, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if mid in exclude:
                continue
            if genres and self._movies_meta is not None:
                gnr = self._parse_genres(mid)
                if not any(g in gnr for g in genres):
                    continue
            results.append((int(mid), float(sc)))
            if len(results) >= n:
                break
        return results

    # ── SVD-based recs ────────────────────────────────────────────────────────

    def recommend_svd(
        self,
        user_id: int,
        n: int = 20,
        exclude_ids: Optional[List[int]] = None,
    ) -> Optional[List[Tuple[int, float]]]:
        """Return top-n SVD recommendations for a known MovieLens user."""
        if not self._svd_ready or user_id not in self._user_idx:
            return None
        u = self._user_idx[user_id]
        mean = self._user_mean.get(user_id, self._global_mean)
        scores = mean + self._U[u] @ self._Vt
        return self._top_n_from_scores(scores, n, exclude_ids)

    def recommend_fold_in(
        self,
        rated_movie_ids: List[int],
        ratings: List[float],
        n: int = 20,
        exclude_ids: Optional[List[int]] = None,
    ) -> Optional[List[Tuple[int, float]]]:
        """
        Compute a virtual user factor via SVD folding-in and return top-n recs.
        Works for API users who haven't trained the original SVD.
        """
        if not self._svd_ready or len(rated_movie_ids) == 0:
            return None

        # Collect item indices and centered ratings
        item_idxs, r_centered = [], []
        for mid, r in zip(rated_movie_ids, ratings):
            idx = self._movie_idx.get(mid)
            if idx is not None:
                item_idxs.append(idx)
                r_centered.append(r - self._global_mean)

        if len(item_idxs) < 2:
            return None

        Vt_sub = self._Vt[:, item_idxs]            # (n_comp, k)
        r_vec = np.array(r_centered, dtype=np.float64)

        # Normal equations: (Vt_sub @ Vt_sub.T + λI) u = Vt_sub @ r
        A = Vt_sub @ Vt_sub.T + self._lam * np.eye(self._Vt.shape[0])
        b = Vt_sub @ r_vec
        u_factor = np.linalg.solve(A, b)

        scores = self._global_mean + u_factor @ self._Vt
        return self._top_n_from_scores(scores, n, exclude_ids)

    # ── TF-IDF similarity ─────────────────────────────────────────────────────

    def get_similar_movies(
        self,
        movie_id: int,
        n: int = 10,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Return movies similar to movie_id by TF-IDF cosine similarity."""
        if not self._tfidf_ready:
            return self.get_popular(n, exclude_ids)

        idx = self._movie_id_to_idx.get(movie_id)
        if idx is None:
            return self.get_popular(n, exclude_ids)

        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self._tfidf_matrix[idx]
        sims = cosine_similarity(query_vec, self._tfidf_matrix).ravel()
        sims[idx] = -1.0  # exclude self

        exclude = set(exclude_ids or [])
        exclude.add(movie_id)

        top_local = np.argpartition(sims, -min(n + len(exclude), len(sims)))[-min(n + len(exclude), len(sims)):]
        top_local = top_local[np.argsort(sims[top_local])[::-1]]

        results = []
        for i in top_local:
            mid = int(self._movie_ids_array[i])
            if mid in exclude:
                continue
            results.append((mid, float(sims[i])))
            if len(results) >= n:
                break
        return results

    def recommend_by_profile(
        self,
        liked_movie_ids: List[int],
        liked_ratings: Optional[List[float]] = None,
        n: int = 20,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Content-based profile: weighted avg of liked movie TF-IDF vectors."""
        if not self._tfidf_ready or not liked_movie_ids:
            return []

        weights = liked_ratings if liked_ratings else [1.0] * len(liked_movie_ids)
        profile = None
        total_w = 0.0
        for mid, w in zip(liked_movie_ids, weights):
            idx = self._movie_id_to_idx.get(mid)
            if idx is None:
                continue
            vec = self._tfidf_matrix[idx]
            profile = vec if profile is None else profile + vec * w
            total_w += w

        if profile is None or total_w == 0:
            return []

        profile = profile / total_w
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(profile, self._tfidf_matrix).ravel()
        return self._top_n_from_scores(sims, n, list(set(exclude_ids or []) | set(liked_movie_ids)))

    # ── Genre-based cold start ────────────────────────────────────────────────

    def recommend_by_genres(
        self,
        preferred_genres: List[str],
        n: int = 20,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """Return popular movies filtered by preferred genres."""
        return self.get_popular(n, exclude_ids, genres=preferred_genres)

    # ── Search ────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase + strip all non-alphanumeric chars for fuzzy matching."""
        import re
        return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()

    def search_movies(self, query: str, n: int = 20) -> List[Tuple[int, float]]:
        """
        Fuzzy title + genre search with normalization.

        Normalisation removes hyphens and punctuation so that, e.g.:
          'spiderman'  matches  'Spider-Man'
          'star wars'  matches  'Star Wars: Episode IV'
        Ranking: exact-norm > starts-with-norm > word-overlap > genre-match, then × popularity.
        """
        if self._movies_meta is None or not query.strip():
            return []

        q_raw = query.strip().lower()
        q_norm = self._normalize(q_raw)
        q_words = [w for w in q_norm.split() if len(w) > 1]

        df = self._movies_meta.copy()
        titles_raw = df["title"].fillna("").str.lower()

        # Pre-compute normalised title series once
        import re
        titles_norm = titles_raw.apply(lambda t: re.sub(r'[^a-z0-9\s]', '', t).strip())

        # ── Scoring tiers ──────────────────────────────────────────────────────
        # Tier 4: exact normalised substring match (highest signal)
        exact_norm = titles_norm.str.contains(q_norm, regex=False) if q_norm else pd.Series(False, index=df.index)
        # Tier 3: raw query substring match (catches accented chars etc.)
        exact_raw = titles_raw.str.contains(q_raw, regex=False)
        # Tier 2: starts-with normalised
        starts_norm = titles_norm.str.startswith(q_norm) if q_norm else pd.Series(False, index=df.index)
        # Tier 1: every query word appears somewhere in the normalised title
        if q_words:
            word_match = pd.Series(True, index=df.index)
            for w in q_words:
                word_match &= titles_norm.str.contains(w, regex=False)
        else:
            word_match = pd.Series(False, index=df.index)

        df["_match"] = (
            exact_norm.astype(int) * 4
            + starts_norm.astype(int) * 3
            + exact_raw.astype(int) * 2
            + word_match.astype(int) * 1
        ).astype(float)

        # Genre match bonus
        if "genre_names" in df.columns:
            genre_match = df["genre_names"].fillna("").str.lower().str.contains(q_raw, regex=False)
            df["_match"] += genre_match.astype(float) * 0.5

        candidates = df[df["_match"] > 0].copy()

        # ── Last-resort fallback: any single word match ─────────────────────
        if candidates.empty and q_words:
            mask = pd.Series(False, index=df.index)
            for w in q_words:
                mask |= titles_norm.str.contains(w, regex=False)
            candidates = df[mask].copy()
            candidates["_match"] = 0.5

        if candidates.empty:
            return []

        # ── Multiply by popularity ──────────────────────────────────────────
        pop = pd.Series(self._pop_scores)
        if not pop.empty:
            candidates["_pop"] = candidates.index.map(pop).fillna(0)
            candidates["_score"] = candidates["_match"] * 2 + candidates["_pop"]
        else:
            candidates["_score"] = candidates["_match"]

        candidates = candidates.sort_values("_score", ascending=False).head(n)
        return [(int(mid), float(row["_score"])) for mid, row in candidates.iterrows()]

    def get_search_suggestions(self, query: str, n: int = 6) -> List[Tuple[int, float]]:
        """Fast suggestion lookup — returns top-n titles for autocomplete dropdown."""
        return self.search_movies(query=query, n=n)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _top_n_from_scores(
        self,
        scores: np.ndarray,
        n: int,
        exclude_ids: Optional[List[int]],
    ) -> List[Tuple[int, float]]:
        exclude = set(exclude_ids or [])
        n_items = len(scores)
        k = min(n + len(exclude) + 50, n_items)
        top_local = np.argpartition(scores, -k)[-k:]
        top_local = top_local[np.argsort(scores[top_local])[::-1]]

        results = []
        for idx in top_local:
            mid = self._idx_movie.get(int(idx))
            if mid is None or mid in exclude:
                continue
            results.append((int(mid), float(scores[idx])))
            if len(results) >= n:
                break
        return results

    def _parse_genres(self, movie_id: int) -> List[str]:
        if self._movies_meta is None or movie_id not in self._movies_meta.index:
            return []
        raw = self._movies_meta.loc[movie_id, "genre_names"]
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            import ast
            try:
                return ast.literal_eval(raw)
            except Exception:
                return [g.strip() for g in raw.split(",") if g.strip()]
        return []

    def get_movie_meta(self, movie_id: int) -> Optional[dict]:
        if self._movies_meta is None or movie_id not in self._movies_meta.index:
            return None
        row = self._movies_meta.loc[movie_id]
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}

    def list_all_movie_ids(self) -> List[int]:
        if self._movies_meta is not None:
            return self._movies_meta.index.tolist()
        return list(self._pop_scores.keys())


# Singleton — imported by routes
model_service = ModelService()
