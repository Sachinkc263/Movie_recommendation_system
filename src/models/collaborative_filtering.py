"""Collaborative filtering recommendation models (The Movies Dataset - Kaggle)."""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import warnings

warnings.filterwarnings('ignore')


def _build_sparse_user_item_matrix(
    ratings: pd.DataFrame,
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, Dict[int, int], Dict[int, int]]:
    """Build a sparse user-item matrix without treating missing ratings as zero."""
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    rows = ratings['userId'].map(user_to_idx).values
    cols = ratings['movieId'].map(movie_to_idx).values
    data = ratings['rating'].values.astype(float)

    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(user_ids), len(movie_ids)),
    )
    return matrix, user_ids, movie_ids, user_to_idx, movie_to_idx


def _center_sparse_by_user(matrix: csr_matrix) -> Tuple[csr_matrix, np.ndarray]:
    """Mean-center only observed ratings per user."""
    user_means = np.array(matrix.sum(axis=1)).flatten() / np.maximum(
        matrix.getnnz(axis=1), 1
    )
    coo = matrix.tocoo()
    centered_data = coo.data - user_means[coo.row]
    centered = csr_matrix(
        (centered_data, (coo.row, coo.col)),
        shape=matrix.shape,
    )
    return centered, user_means


def _center_sparse_by_item_rows(item_user_matrix: csr_matrix) -> Tuple[csr_matrix, np.ndarray]:
    """Mean-center only observed ratings per item (items as rows)."""
    item_means = np.array(item_user_matrix.sum(axis=1)).flatten() / np.maximum(
        item_user_matrix.getnnz(axis=1), 1
    )
    coo = item_user_matrix.tocoo()
    centered_data = coo.data - item_means[coo.row]
    centered = csr_matrix(
        (centered_data, (coo.row, coo.col)),
        shape=item_user_matrix.shape,
    )
    return centered, item_means


def _filter_training_ratings(
    train_ratings: pd.DataFrame,
    min_ratings: int = 3,
    min_movie_ratings: int = 3,
    max_users: Optional[int] = None,
    max_movies: Optional[int] = None,
) -> pd.DataFrame:
    """Filter users and movies by minimum interaction counts.

    max_users/max_movies default to None (no hard cap) to avoid the original
    98.4% data-loss bug.  min_movie_ratings drops long-tail movies that have
    too few co-ratings to produce reliable similarity estimates — this also
    keeps the ItemBasedCF similarity matrix to a manageable size.
    """
    user_counts = train_ratings['userId'].value_counts()
    active_users = user_counts[user_counts >= min_ratings]
    if max_users is not None:
        active_users = active_users.head(max_users)
    active_user_ids = set(active_users.index)

    movie_counts = train_ratings['movieId'].value_counts()
    if max_movies is not None:
        active_movie_ids = set(movie_counts.head(max_movies).index)
    elif min_movie_ratings > 1:
        active_movie_ids = set(movie_counts[movie_counts >= min_movie_ratings].index)
    else:
        active_movie_ids = set(train_ratings['movieId'].unique())

    filtered = train_ratings[
        train_ratings['userId'].isin(active_user_ids)
        & train_ratings['movieId'].isin(active_movie_ids)
    ].copy()
    return filtered


class UserBasedCF:
    """User-based collaborative filtering with proper sparse mean-centering."""

    def __init__(
        self,
        train_ratings: pd.DataFrame,
        n_neighbors: int = 50,
        min_ratings: int = 3,
        min_movie_ratings: int = 3,
        max_users: Optional[int] = None,
        max_movies: Optional[int] = None,
    ):
        self.train_ratings = train_ratings
        self.n_neighbors = n_neighbors
        self.min_ratings = min_ratings
        self.min_movie_ratings = min_movie_ratings
        self.max_users = max_users
        self.max_movies = max_movies
        self.user_similarity = None
        self.user_item_matrix = None
        self.user_means = None
        self.user_ids = None
        self.movie_ids = None
        self.user_to_idx = None
        self._build_model()

    def _build_model(self):
        filtered_ratings = _filter_training_ratings(
            self.train_ratings,
            min_ratings=self.min_ratings,
            min_movie_ratings=self.min_movie_ratings,
            max_users=self.max_users,
            max_movies=self.max_movies,
        )

        matrix, self.user_ids, self.movie_ids, self.user_to_idx, _ = (
            _build_sparse_user_item_matrix(filtered_ratings)
        )
        centered, self.user_means = _center_sparse_by_user(matrix)
        self.user_similarity = cosine_similarity(centered)

        self.user_item_matrix = matrix
        print(
            f"User-based CF: {matrix.shape[0]} users, "
            f"{matrix.shape[1]} movies, {matrix.nnz} ratings"
        )

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if user_id not in self.user_to_idx:
            return pd.DataFrame()

        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_item_matrix.getrow(user_idx).toarray().ravel()
        rated_indices = np.where(user_vector > 0)[0]
        if len(rated_indices) == 0:
            return pd.DataFrame()

        similarities = self.user_similarity[user_idx]
        neighbor_indices = np.argsort(similarities)[::-1]
        neighbor_indices = [
            idx for idx in neighbor_indices
            if idx != user_idx and similarities[idx] > 0
        ][: self.n_neighbors]

        candidate_indices = set()
        for neighbor_idx in neighbor_indices:
            neighbor_row = self.user_item_matrix.getrow(neighbor_idx).toarray().ravel()
            candidate_indices.update(np.where(neighbor_row > 0)[0])
        candidate_indices -= set(rated_indices)

        predictions: Dict[int, float] = {}
        for movie_idx in candidate_indices:
            numerator = 0.0
            denominator = 0.0
            for neighbor_idx in neighbor_indices:
                neighbor_rating = self.user_item_matrix[neighbor_idx, movie_idx]
                if neighbor_rating == 0:
                    continue
                sim = similarities[neighbor_idx]
                numerator += sim * (neighbor_rating - self.user_means[neighbor_idx])
                denominator += abs(sim)

            if denominator > 0:
                movie_id = self.movie_ids[movie_idx]
                predictions[movie_id] = self.user_means[user_idx] + numerator / denominator

        if not predictions:
            return pd.DataFrame()

        top = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
        return pd.DataFrame(top, columns=['movieId', 'score'])


class ItemBasedCF:
    """Item-based collaborative filtering with adjusted cosine similarity."""

    def __init__(
        self,
        train_ratings: pd.DataFrame,
        n_neighbors: int = 50,
        min_ratings: int = 3,
        min_movie_ratings: int = 5,
        max_users: Optional[int] = None,
        max_movies: Optional[int] = None,
    ):
        self.train_ratings = train_ratings
        self.n_neighbors = n_neighbors
        self.min_ratings = min_ratings
        self.min_movie_ratings = min_movie_ratings
        self.max_users = max_users
        self.max_movies = max_movies
        self.item_similarity = None
        self.user_item_matrix = None
        self.item_means = None
        self.user_ids = None
        self.movie_ids = None
        self.user_to_idx = None
        self.movie_to_idx = None
        self._build_model()

    def _build_model(self):
        filtered_ratings = _filter_training_ratings(
            self.train_ratings,
            min_ratings=self.min_ratings,
            min_movie_ratings=self.min_movie_ratings,
            max_users=self.max_users,
            max_movies=self.max_movies,
        )

        matrix, self.user_ids, self.movie_ids, self.user_to_idx, self.movie_to_idx = (
            _build_sparse_user_item_matrix(filtered_ratings)
        )
        item_user = matrix.T.tocsr()
        centered, self.item_means = _center_sparse_by_item_rows(item_user)
        self.item_similarity = cosine_similarity(centered)
        self.user_item_matrix = matrix

        print(
            f"Item-based CF: {matrix.shape[0]} users, "
            f"{matrix.shape[1]} movies, {matrix.nnz} ratings"
        )

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if user_id not in self.user_to_idx:
            return pd.DataFrame()

        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_item_matrix.getrow(user_idx).toarray().ravel()
        rated_indices = np.where(user_vector > 0)[0]
        if len(rated_indices) == 0:
            return pd.DataFrame()

        candidate_indices = set()
        for rated_idx in rated_indices:
            similarities = self.item_similarity[rated_idx]
            neighbor_indices = np.argsort(similarities)[::-1]
            neighbor_indices = [
                idx for idx in neighbor_indices
                if idx != rated_idx and similarities[idx] > 0
            ][: self.n_neighbors]
            candidate_indices.update(neighbor_indices)
        candidate_indices -= set(rated_indices)

        predictions: Dict[int, float] = {}
        for candidate_idx in candidate_indices:
            numerator = 0.0
            denominator = 0.0
            for rated_idx in rated_indices:
                sim = self.item_similarity[candidate_idx, rated_idx]
                if sim <= 0:
                    continue
                user_rating = user_vector[rated_idx]
                numerator += sim * (user_rating - self.item_means[rated_idx])
                denominator += abs(sim)

            if denominator > 0:
                movie_id = self.movie_ids[candidate_idx]
                predictions[movie_id] = self.item_means[candidate_idx] + numerator / denominator

        if not predictions:
            return pd.DataFrame()

        top = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
        return pd.DataFrame(top, columns=['movieId', 'score'])


class MatrixFactorizationCF:
    """Centered TruncatedSVD matrix factorization.

    Treats only observed entries (sparse), centers by user mean, then
    factorizes with TruncatedSVD.  Prediction for (user, movie) is:
        user_mean[user] + U[user] @ Vt[:, movie]

    The constructor auto-selects n_components via a validation-RMSE grid
    search over [20, 50, 100] if validate=True.
    """

    def __init__(
        self,
        train_ratings: pd.DataFrame,
        n_components: int = 50,
        min_ratings: int = 3,
        min_movie_ratings: int = 3,
        max_users: Optional[int] = None,
        max_movies: Optional[int] = None,
        random_state: int = 42,
        validate: bool = True,
        # kept for API compat but ignored
        n_epochs: int = 20,
    ):
        self.train_ratings = train_ratings
        self.n_components = n_components
        self.min_ratings = min_ratings
        self.min_movie_ratings = min_movie_ratings
        self.max_users = max_users
        self.max_movies = max_movies
        self.random_state = random_state
        self.validate = validate
        self.val_rmse: Optional[float] = None
        self.val_mae: Optional[float] = None
        self.global_mean: float = float(train_ratings['rating'].mean())

        self.U: Optional[np.ndarray] = None
        self.Vt: Optional[np.ndarray] = None
        self.user_mean: Optional[Dict[int, float]] = None
        self.user_idx: Optional[Dict[int, int]] = None
        self.movie_idx: Optional[Dict[int, int]] = None
        self.idx_user: Optional[np.ndarray] = None
        self.idx_movie: Optional[np.ndarray] = None
        self.user_item_matrix = None

        self._build_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fit_svd(
        self, centered_matrix, n_comp: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_comp = min(n_comp, min(centered_matrix.shape) - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        U = svd.fit_transform(centered_matrix)
        Vt = svd.components_
        return U, Vt

    def _validation_rmse(
        self,
        U: np.ndarray,
        Vt: np.ndarray,
        val_df: pd.DataFrame,
    ) -> Tuple[float, float]:
        """Vectorized RMSE/MAE on a held-out validation set."""
        uids = val_df['userId'].values.astype(int)
        mids = val_df['movieId'].values.astype(int)
        actuals = val_df['rating'].values.astype(float)

        u_idxs = np.array([self.user_idx.get(uid, -1) for uid in uids])
        m_idxs = np.array([self.movie_idx.get(mid, -1) for mid in mids])
        means  = np.array([self.user_mean.get(uid, self.global_mean) for uid in uids])

        preds = means.copy()
        known = (u_idxs >= 0) & (m_idxs >= 0)
        if known.any():
            # batched dot: (n_known, k) * (k, n_known)^T → (n_known,)
            preds[known] += (U[u_idxs[known]] * Vt[:, m_idxs[known]].T).sum(axis=1)

        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
        mae  = float(np.mean(np.abs(preds - actuals)))
        return rmse, mae

    def _predict_one(
        self,
        user_id: int,
        movie_id: int,
        U: np.ndarray,
        Vt: np.ndarray,
    ) -> float:
        u = self.user_idx.get(user_id)
        m = self.movie_idx.get(movie_id)
        mean = self.user_mean.get(user_id, self.global_mean)
        if u is None or m is None:
            return mean
        return float(mean + U[u] @ Vt[:, m])

    def _build_model(self):
        filtered = _filter_training_ratings(
            self.train_ratings,
            min_ratings=self.min_ratings,
            min_movie_ratings=self.min_movie_ratings,
            max_users=self.max_users,
            max_movies=self.max_movies,
        )
        n_filtered = len(filtered)

        # Step 1 — compute user means from observed ratings only
        self.user_mean = filtered.groupby('userId')['rating'].mean().to_dict()

        # Step 2 — center ratings (vectorized) and build index maps
        unique_users = filtered['userId'].unique()
        unique_movies = filtered['movieId'].unique()
        self.user_idx = {u: i for i, u in enumerate(unique_users)}
        self.movie_idx = {m: i for i, m in enumerate(unique_movies)}
        self.idx_user = unique_users
        self.idx_movie = unique_movies

        # Step 3 — build sparse centered matrix; free centered_df immediately after
        from scipy.sparse import csr_matrix as _csr
        centered_df = filtered.copy()
        centered_df['rating'] = (centered_df['rating']
                                 - centered_df['userId'].map(self.user_mean))
        rows = centered_df['userId'].map(self.user_idx).values
        cols = centered_df['movieId'].map(self.movie_idx).values
        vals = centered_df['rating'].values.astype(float)
        centered_matrix = _csr((vals, (rows, cols)),
                                shape=(len(unique_users), len(unique_movies)))
        del centered_df  # free 1× copy of filtered (potentially 1.3 GB) immediately

        # Step 4 — validation grid search if requested
        best_n = self.n_components
        best_rmse = float('inf')

        if self.validate and n_filtered >= 200:
            rng = np.random.default_rng(self.random_state)
            val_mask = rng.random(n_filtered) < 0.20
            train_inner = filtered[~val_mask]
            val_df = filtered[val_mask]

            # rebuild centered matrix on train_inner
            inner_mean = train_inner.groupby('userId')['rating'].mean().to_dict()
            train_inner_c = train_inner.copy()
            train_inner_c['rating'] = (train_inner_c['rating']
                                       - train_inner_c['userId'].map(inner_mean)
                                       .fillna(self.global_mean))
            r2 = train_inner_c['userId'].map(self.user_idx)
            c2 = train_inner_c['movieId'].map(self.movie_idx)
            v2 = train_inner_c['rating'].values.astype(float)
            valid_mask = r2.notna() & c2.notna()
            r2 = r2[valid_mask].astype(int).values
            c2 = c2[valid_mask].astype(int).values
            v2 = v2[valid_mask]
            inner_matrix = _csr((v2, (r2, c2)),
                                  shape=(len(unique_users), len(unique_movies)))
            del train_inner, train_inner_c  # free 2× copies (potentially 2.1 GB) before SVD loop

            # Temporarily swap user_mean for inner mean during search
            saved_mean = self.user_mean
            self.user_mean = inner_mean

            n_candidates = [20, 50, 100]
            print("SVD validation grid search:")
            for n_comp in n_candidates:
                if n_comp >= min(inner_matrix.shape):
                    continue
                U_v, Vt_v = self._fit_svd(inner_matrix, n_comp)
                rmse, mae = self._validation_rmse(U_v, Vt_v, val_df)
                print(f"  n_components={n_comp}: val_RMSE={rmse:.4f}, val_MAE={mae:.4f}")
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_n = n_comp

            self.user_mean = saved_mean
            print(f"Selected n_components={best_n} (val_RMSE={best_rmse:.4f})")
        else:
            # validate=False: filtered is not referenced again — free it now so the
            # SVD decomposition (which peaks at ~2× the matrix size) has more headroom.
            import gc
            del filtered
            gc.collect()

        # Step 5 — fit final model on full filtered data
        self.U, self.Vt = self._fit_svd(centered_matrix, best_n)
        self.n_components = best_n

        # Final validation RMSE on a small held-out set (informational)
        if self.validate and n_filtered >= 200:
            rng2 = np.random.default_rng(self.random_state + 1)
            val_mask2 = rng2.random(n_filtered) < 0.10
            val_df2 = filtered[val_mask2]
            self.val_rmse, self.val_mae = self._validation_rmse(
                self.U, self.Vt, val_df2
            )
            print(
                f"Matrix Factorization (centered SVD n={self.n_components}): "
                f"{len(unique_users)} users, {len(unique_movies)} movies, "
                f"{n_filtered} ratings | val_RMSE={self.val_rmse:.4f}"
            )
        else:
            print(
                f"Matrix Factorization (centered SVD n={self.n_components}): "
                f"{len(unique_users)} users, {len(unique_movies)} movies, "
                f"{n_filtered} ratings"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, user_id: int, movie_id: int) -> float:
        return self._predict_one(user_id, movie_id, self.U, self.Vt)

    def _fold_in_recommend(self, user_id: int, n: int) -> pd.DataFrame:
        """SVD fold-in: estimate a virtual user vector via normal equations.

        For a user not in the training matrix, solve:
            x_u = argmin ||Vt[:,seen].T @ x_u - (r_seen - mean_u)||^2
        using least-squares.  Then score all movies with mean_u + x_u @ Vt.
        """
        user_rows = self.train_ratings[self.train_ratings['userId'] == user_id]
        if len(user_rows) == 0:
            return pd.DataFrame()
        user_mean = float(user_rows['rating'].mean())

        seen_local, deltas = [], []
        for _, row in user_rows.iterrows():
            local = self.movie_idx.get(int(row['movieId']))
            if local is not None:
                seen_local.append(local)
                deltas.append(float(row['rating']) - user_mean)
        if not seen_local:
            return pd.DataFrame()

        Vt_sub = self.Vt[:, seen_local].T  # (n_seen, k)
        x_u, _, _, _ = np.linalg.lstsq(Vt_sub, np.array(deltas), rcond=None)

        scores = user_mean + x_u @ self.Vt
        for idx in seen_local:
            scores[idx] = -np.inf

        n_top = min(n, len(scores))
        top_local = np.argpartition(scores, -n_top)[-n_top:]
        top_local = top_local[np.argsort(scores[top_local])[::-1]]
        return pd.DataFrame({
            'movieId': self.idx_movie[top_local],
            'score': scores[top_local].astype(float),
        })

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if user_id not in self.user_idx:
            return self._fold_in_recommend(user_id, n)

        u = self.user_idx[user_id]
        mean = self.user_mean.get(user_id, self.global_mean)

        # Vectorized predicted scores for all movies in SVD space
        scores_vec = (mean + self.U[u] @ self.Vt).copy()

        # On-demand rated-movie lookup
        user_rated = set(
            self.train_ratings.loc[
                self.train_ratings['userId'] == user_id, 'movieId'
            ].values
        )
        for mid in user_rated:
            idx = self.movie_idx.get(mid)
            if idx is not None:
                scores_vec[idx] = -np.inf

        n_movies = len(scores_vec)
        k = min(n, n_movies)
        top_local = np.argpartition(scores_vec, -k)[-k:]
        top_local = top_local[np.argsort(scores_vec[top_local])[::-1]]

        return pd.DataFrame({
            'movieId': self.idx_movie[top_local],
            'score': scores_vec[top_local].astype(float),
        })


class ImplicitALSRecommender:
    """Alternating Least Squares for implicit collaborative filtering.

    Treats every rating as positive evidence with confidence c_ui = 1 + alpha * r_ui,
    then alternates closed-form user and item factor updates.  Substantially outperforms
    explicit-rating SVD on ranking metrics (P@k, NDCG@k) because it optimises pairwise
    preference rather than RMSE.

    Reference: Hu, Koren & Volinsky (2008) — Collaborative Filtering for Implicit
    Feedback Datasets, ICDM 2008.
    """

    def __init__(
        self,
        train_ratings: pd.DataFrame,
        n_factors: int = 100,
        n_iterations: int = 20,
        regularization: float = 0.01,
        alpha: float = 40.0,
        random_state: int = 42,
        min_ratings: int = 3,
    ):
        self.train_ratings = train_ratings
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.alpha = alpha
        self.random_state = random_state
        self.min_ratings = min_ratings

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_idx: Dict[int, int] = {}
        self.item_idx: Dict[int, int] = {}
        self.idx_user: Optional[np.ndarray] = None
        self.idx_item: Optional[np.ndarray] = None
        self._user_rated_local: Dict[int, np.ndarray] = {}  # local user idx → local item indices

        self._build_model()

    def _build_model(self):
        from scipy.sparse import csr_matrix as _csr

        user_counts = self.train_ratings.groupby('userId').size()
        valid_users = user_counts[user_counts >= self.min_ratings].index
        filtered = self.train_ratings[self.train_ratings['userId'].isin(valid_users)]

        unique_users = filtered['userId'].unique()
        unique_items = filtered['movieId'].unique()
        self.user_idx = {int(u): i for i, u in enumerate(unique_users)}
        self.item_idx = {int(m): i for i, m in enumerate(unique_items)}
        self.idx_user = unique_users
        self.idx_item = unique_items

        n_users, n_items, k = len(unique_users), len(unique_items), self.n_factors

        rows = filtered['userId'].map(self.user_idx).values
        cols = filtered['movieId'].map(self.item_idx).values
        conf_vals = (1.0 + self.alpha * filtered['rating'].values).astype(np.float64)
        del filtered

        # CSR by users for user-step; transposed CSR for item-step
        C = _csr((conf_vals, (rows, cols)), shape=(n_users, n_items))
        Ct = C.T.tocsr()

        # Cache each user's rated local item indices for fast filtering in recommend()
        for u in range(n_users):
            s, e = int(C.indptr[u]), int(C.indptr[u + 1])
            if s < e:
                self._user_rated_local[u] = C.indices[s:e].copy()

        rng = np.random.default_rng(self.random_state)
        X = (rng.standard_normal((n_users, k)) * 0.01).astype(np.float64)
        Y = (rng.standard_normal((n_items, k)) * 0.01).astype(np.float64)
        lam_I = self.regularization * np.eye(k, dtype=np.float64)

        for it in range(self.n_iterations):
            # ── User step: solve (YtY + Yt diag(c_u-1) Y + λI) x_u = Yt c_u ──
            YtY = Y.T @ Y
            for u in range(n_users):
                s, e = int(C.indptr[u]), int(C.indptr[u + 1])
                if s == e:
                    continue
                ii, cu = C.indices[s:e], C.data[s:e]
                Yu = Y[ii]
                A = YtY + (Yu * (cu - 1.0)[:, None]).T @ Yu + lam_I
                X[u] = np.linalg.solve(A, Yu.T @ cu)

            # ── Item step: solve (XtX + Xt diag(c_i-1) X + λI) y_i = Xt c_i ──
            XtX = X.T @ X
            for i in range(n_items):
                s, e = int(Ct.indptr[i]), int(Ct.indptr[i + 1])
                if s == e:
                    continue
                uu, ci = Ct.indices[s:e], Ct.data[s:e]
                Xi = X[uu]
                A = XtX + (Xi * (ci - 1.0)[:, None]).T @ Xi + lam_I
                Y[i] = np.linalg.solve(A, Xi.T @ ci)

            print(f"  ALS iteration {it + 1}/{self.n_iterations}", flush=True)

        self.user_factors = X
        self.item_factors = Y
        print(
            f"Implicit ALS: {n_users} users, {n_items} items, "
            f"k={k}, alpha={self.alpha}, reg={self.regularization}"
        )

    def _fold_in_recommend(self, user_id: int, n: int) -> pd.DataFrame:
        """ALS fold-in: solve normal equations for a user not in training.

        For user u with observed ratings r_u, compute:
            x_u = (Y[seen].T diag(c-1) Y[seen] + YtY + lambda*I)^-1 Y[seen].T c_u
        where c_ui = 1 + alpha * r_ui.
        """
        uid = int(user_id)
        user_rows = self.train_ratings[self.train_ratings['userId'] == uid]
        if len(user_rows) == 0:
            return pd.DataFrame()

        seen_local, confs = [], []
        for _, row in user_rows.iterrows():
            local = self.item_idx.get(int(row['movieId']))
            if local is not None:
                seen_local.append(local)
                confs.append(1.0 + self.alpha * float(row['rating']))
        if not seen_local:
            return pd.DataFrame()

        seen_arr = np.array(seen_local, dtype=int)
        conf_arr = np.array(confs, dtype=np.float64)
        Y_sub = self.item_factors[seen_arr]  # (n_seen, k)
        nf = self.n_factors
        lam_I = self.regularization * np.eye(nf, dtype=np.float64)
        A = Y_sub.T @ (Y_sub * conf_arr[:, None]) + lam_I
        b = Y_sub.T @ conf_arr
        x_u = np.linalg.solve(A, b)

        scores = (x_u @ self.item_factors.T).copy()
        scores[seen_arr] = -np.inf

        n_top = min(n, len(scores))
        top_local = np.argpartition(scores, -n_top)[-n_top:]
        top_local = top_local[np.argsort(scores[top_local])[::-1]]
        return pd.DataFrame({
            'movieId': self.idx_item[top_local],
            'score': scores[top_local].astype(float),
        })

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        uid = int(user_id)
        if uid not in self.user_idx:
            return self._fold_in_recommend(uid, n)

        u = self.user_idx[uid]
        scores = (self.user_factors[u] @ self.item_factors.T).copy()

        # Zero out already-rated items using cached local indices
        rated_local = self._user_rated_local.get(u)
        if rated_local is not None and len(rated_local):
            scores[rated_local] = -np.inf

        n_top = min(n, len(scores))
        top_local = np.argpartition(scores, -n_top)[-n_top:]
        top_local = top_local[np.argsort(scores[top_local])[::-1]]

        return pd.DataFrame({
            'movieId': self.idx_item[top_local],
            'score': scores[top_local].astype(float),
        })
