# CineMatch - System Overview

This document gives a broad, readable overview of the CineMatch recommendation system.
It is written for teachers, recruiters, GitHub visitors, and future contributors.
For deep technical details see docs/ARCHITECTURE.md.

---

## What is CineMatch?

CineMatch is a full-stack movie recommendation system that uses machine learning
to suggest personalised films. It combines two established recommendation techniques
-- collaborative filtering and content-based filtering -- into a hybrid model that
outperforms either approach alone.

The project demonstrates a complete ML workflow, from raw data through to a live
web interface, including:

- Data cleaning and preprocessing at scale (26 million ratings)
- Model training with hyperparameter tuning
- Offline evaluation with standard recommender metrics
- A REST API serving real-time inference
- A React single-page application consuming that API
- Docker packaging for reproducible deployment

---

## Dataset

**Source**: The Movies Dataset (Kaggle, rounakbanik)
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

This dataset bundles MovieLens ratings with TMDB movie metadata.

| File | Description | Size |
|---|---|---|
| movies_metadata.csv | 45,466 movies with title, genres, overview, budget, revenue | 33 MB |
| ratings.csv | 26,024,289 ratings from 270,896 users | 677 MB |
| credits.csv | Cast and crew for each movie (JSON columns) | 182 MB |
| keywords.csv | Movie keywords/tags (JSON columns) | 6 MB |
| links.csv | TMDB and IMDb ID cross-reference | 1 MB |

After cleaning and merging: **43,549 movies** with full TMDB metadata.

---

## Recommendation Approaches

### 1. Collaborative Filtering (SVD)

Collaborative filtering finds patterns in user behaviour.
The key insight is: users who agreed on ratings in the past will likely agree again.

We use **Centered Truncated SVD** -- a matrix factorisation technique that decomposes
the user-item rating matrix into compact latent factor vectors.

    User-Item Matrix (270,896 users x 45,115 movies)
         |
         v
    TruncatedSVD (n_components = 50)
         |
         v
    svd_U  (n_users  x 50) -- user latent factors
    svd_Vt (50 x n_movies) -- movie latent factors

    Recommendation score for user u and movie m:
    score(u, m) = u_vector @ movie_m_vector

**Cold-start solution**: new users not in the training matrix get a virtual
latent vector computed by SVD fold-in (solving the normal equations from their
rated movies). This requires no retraining.

### 2. Content-Based Filtering (TF-IDF)

Content-based filtering recommends movies similar to ones the user already liked,
based purely on movie attributes -- not other users.

We build a **feature soup** for each movie combining:
- Genres (repeated 3x for higher weight)
- Overview / plot summary
- Director name
- Top-3 cast names
- TMDB keywords/tags

This soup is vectorised with TF-IDF (Term Frequency - Inverse Document Frequency)
and recommendations are found by computing cosine similarity against liked movies.

### 3. Hybrid Model

The hybrid combines both approaches:

    final_score = 0.45 * svd_score + 0.55 * tfidf_profile_score

Content-based gets a slightly higher weight because it generalises better to
the long-tail of the catalogue (movies with few ratings).

### 4. Popularity Baseline

A Bayesian-smoothed popularity score is used as a fallback:

    score = (count / (count + C)) * mean_rating
          + (C    / (count + C)) * global_mean

Where C is the 90th-percentile vote count. This prevents movies with
one 5-star rating from dominating the popularity charts.

---

## Recommendation Waterfall

When a user requests recommendations, the system picks a strategy based on
how much interaction history they have:

    1. SVD fold-in            (>= 3 interactions with any movie)
    2. TF-IDF content profile (has liked/viewed at least one movie)
    3. Genre-based popularity (selected genres during onboarding)
    4. Global popularity      (brand-new user, no history at all)

---

## Model Performance

Evaluation contract (fixed across all runs):
- K = 10 (top-10 recommendations evaluated)
- Relevance threshold: rating >= 4.0 (movies the user would like)
- Split: user-based 80/20 (random seed 42)

| Model | P@10 | R@10 | F1@10 | Hit Rate |
|---|---|---|---|---|
| Popularity baseline | 0.0429 | 0.0347 | 0.0383 | 0.0360 |
| User-Based CF | 0.0048 | 0.0036 | 0.0041 | 0.0040 |
| Item-Based CF | 0.0083 | 0.0135 | 0.0103 | 0.0070 |
| Genre content-based | 0.0107 | 0.0066 | 0.0082 | 0.0090 |
| TF-IDF content-based | 0.0095 | 0.0091 | 0.0093 | 0.0080 |
| ImplicitALS (k=100) | 0.0500 | 0.0829 | 0.0624 | 0.0420 |
| **Hybrid (ALS + TF-IDF)** | **0.0536** | **0.0853** | **0.0658** | **0.0450** |

Evaluation contract: K=10, threshold >= 4.0, 100 users, seed=42 (fixed across all models).

The hybrid model achieves the best precision and recall by combining
ImplicitALS (70%) for personalised ranking with TF-IDF (30%) for content diversity.

ImplicitALS is selected over explicit SVD because it optimises ranking (P@k) directly
rather than RMSE. SVD validation RMSE from grid search: **0.8601** (n=50, full training set).

---

## User Experience

### Onboarding

When a new user opens the app, they are prompted to select favourite genres.
This is optional -- they can skip and go directly to the homepage.
If genres are saved, the app uses them for cold-start recommendations
before enough interaction history has been collected.

### Homepage Sections (in order)

1. **Recommended For You** -- personalised SVD/hybrid picks (shown once user has interactions)
2. **Similar To Your Interests** -- content-based from most recently liked movie
3. **Popular Movies** -- always shown, useful for new visitors

### Search

The search bar normalises queries before matching, so `spiderman` finds
`Spider-Man`, `godfather` finds `The Godfather`, etc. As you type, a debounced
autocomplete dropdown shows title suggestions.

Results are paginated at 12 per page with a page picker.

### Movie Detail Page

Each movie page shows: title, year, rating, genres, overview, director, cast,
and a "Like" button. Liking a movie immediately improves recommendations
(the interaction is logged and SVD fold-in re-runs on the next request).

---

## Poster Images

The dataset was compiled in 2020. TMDB periodically rotates their poster file
paths, so approximately 83% of the CSV poster_path values now return HTTP 404.

**Solution**: The frontend tries the CSV path first. If the browser fires
an `img.onError` event, it calls the backend, which looks up the current
poster from the TMDB API and caches the result locally in
`data/cache/poster_cache.json`. The image then reloads with the fresh URL.

At startup, the backend pre-warms the cache for the top 300 popular movies
in a background thread (at 12 requests/second to stay within TMDB rate limits).

---

## Technology Stack

| Component | Technology | Why |
|---|---|---|
| Backend | Python 3.10, FastAPI | Fast async API, automatic OpenAPI docs |
| ORM | SQLAlchemy | Type-safe DB access, easy migration |
| Database | SQLite | Zero-config, sufficient for prototype |
| ML | scikit-learn, NumPy, SciPy | Industry-standard, well-tested |
| Frontend | React 18, Vite, Tailwind CSS | Fast build, great DX, modern UI |
| Serving | uvicorn, nginx | Production-grade ASGI/HTTP servers |
| Packaging | Docker, Docker Compose | Reproducible, one-command startup |

---

## File Structure (Top Level)

    api/         FastAPI backend (routes, services, models)
    frontend/    React + Vite SPA (components, hooks, pages)
    src/         ML library: data loading, cleaning, models, evaluation
    scripts/     Runnable pipeline scripts (preprocessing, training, evaluation)
    notebooks/   Jupyter exploration (01 EDA through 08 Hybrid)
    data/        Raw and processed datasets, EDA figures, poster cache
    models/      Trained artifacts and popularity scores
    docs/        This document, ARCHITECTURE.md, PROJECT_IDEA.md
    reports/     Offline evaluation metrics JSON files
    config/      Training hyperparameters (config.yaml)

---

## Running From Scratch

Full setup instructions are in README.md. The short version:

    # 1. Download dataset to data/raw/ (Kaggle link above)
    # 2. Set TMDB_API_KEY in .env
    # 3. Clean and preprocess data
    python scripts/run_preprocessing.py

    # 4. Train models and export artifacts
    python scripts/train_and_export.py

    # 5. Start the app
    make start   (Docker)
    # or
    make dev-api + make dev-frontend   (local dev)

---

## Contributing

1. Follow the existing code style (type hints, no unnecessary comments)
2. Add tests for new backend routes in tests/
3. Do not change K=10 or relevance_threshold=4.0 without updating all reports
4. Keep notebooks self-contained and runnable in order
5. Never commit secrets; use .env for API keys

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) -- deep-dive component and data flow diagrams
- [PROJECT_IDEA.md](PROJECT_IDEA.md) -- original project brief and goals
