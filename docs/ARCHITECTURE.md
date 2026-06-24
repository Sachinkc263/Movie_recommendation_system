# CineMatch - Architecture Documentation

## System Overview

CineMatch is a hybrid movie recommendation system with three main layers:

1. **React Frontend** (nginx, port 3000) - user interface and interaction logging
2. **FastAPI Backend** (uvicorn, port 8000) - inference engine and REST API
3. **ML Artifacts** (models/artifacts/) - pre-trained files loaded at startup

A SQLite database stores user sessions and interaction history. Models are trained offline
and exported as lightweight numpy/scipy files; no retraining happens during operation.

---

## Component Architecture

    api/
      main.py                 App factory, startup hooks, /health endpoint
      models/
        db_models.py          SQLAlchemy ORM: User, Interaction tables
        schemas.py            Pydantic request/response schemas
      routes/
        movies.py             /movies/* (search, details, similar, poster-url)
        recommendations.py    /recommendations/* (For You, Because You Watched)
        users.py              /users/* and /interactions/*
      services/
        model_service.py      ModelService singleton: SVD + TF-IDF inference
        user_service.py       Interaction aggregation, implicit feedback weights
        tmdb_service.py       TMDB poster cache (lazy fetch + background prewarm)

    frontend/
      src/
        App.jsx               Router, onboarding guard, user context
        api/client.js         Axios wrapper for all backend API calls
        hooks/
          useUser.js          Session init, cold-start flags, markInteracted()
          usePoster.js        CSV path -> TMDB API fallback chain
        pages/
          Home.jsx            3-section layout: For You / Interests / Popular
          Search.jsx          Search + autocomplete + paginated results
          MovieDetail.jsx     Movie page, like button, similar movies
          Onboarding.jsx      Genre preference form (skippable)
        components/
          Navbar.jsx          Search bar with debounced autocomplete dropdown
          MovieCard.jsx       Movie card with automatic poster fallback
          MovieGrid.jsx       Horizontal scrolling carousel
          RecommendationSection.jsx

    src/                      ML library used by notebooks and scripts
      data/
        data_loader.py        Load raw or cleaned datasets
        data_cleaner.py       Parse JSON columns, fix types, dedup
        data_preprocessor.py  Merge datasets, train/test split, user-item matrix
      models/
        collaborative.py      CenteredSVD wrapper
        content_based.py      TF-IDF cosine similarity
        hybrid.py             Hybrid combiner
      evaluation/
        metrics.py            MetricsCalculator: Precision@K, RMSE, coverage
      features/
        feature_engineering.py
        feature_selection.py

    scripts/
      run_preprocessing.py    Clean raw data and create train/test split
      train_and_export.py     Train models and export numpy artifacts
      evaluate.py             Full offline evaluation suite

    notebooks/                Run in order 01 through 08
      01_eda.ipynb
      02_data_cleaning.ipynb
      03_data_preprocessing.ipynb
      04_feature_engineering.ipynb
      05_popularity_baseline.ipynb
      06_collaborative_filtering.ipynb
      07_content_based_filtering.ipynb
      08_hybrid_recommendation.ipynb

    data/
      raw/                    Original CSVs from Kaggle (not committed)
      processed/              Generated outputs (not committed)
      figures/                EDA plots (not committed)
      cache/
        poster_cache.json     TMDB poster path cache (committed, ~400 entries)

    models/
      artifacts/              Trained numpy/scipy files (not committed)
      popularity_scores.csv   Bayesian popularity scores

---

## ML Pipeline

### Setup Pipeline (one-time, run before starting the API)

    data/raw/  (Kaggle: rounakbanik/the-movies-dataset)
        |
        v
    python scripts/run_preprocessing.py
        Step 1 - DataCleaner:
          Fix non-numeric IDs in movies_metadata
          Parse JSON cast/crew/genre/keyword columns
          Remove outlier ratings (outside 0.5-5.0 range)
          Save: movies_clean.csv, credits_clean.csv,
                keywords_clean.csv, ratings_clean.csv
        Step 2 - DataPreprocessor:
          Merge movies + credits + keywords + links
          Normalise numeric features (budget, revenue, runtime, etc.)
          User-based 80/20 train/test split (fixed seed 42)
          Save: movies_integrated.csv, train_ratings.csv, test_ratings.csv
        |
        v
    python scripts/train_and_export.py
        export_svd():
          Centered TruncatedSVD (n_components=50) on user-item matrix
          Save: svd_U.npy (n_users x 50), svd_Vt.npy (50 x n_movies)
          Save: svd_meta.pkl (index dicts + global mean)
        export_tfidf():
          Feature soup: genres(3x) + overview + director + cast + keywords
          TF-IDF (max_features=10000, ngram_range=(1,2))
          Save: tfidf_matrix.npz, tfidf_meta.pkl
        export_popularity():
          Bayesian score: weighted_mean + log(count)
          Save: models/popularity_scores.csv

### Inference Waterfall (per API request)

    User interaction count >= cold_start_threshold (default 3)?
      YES --> SVD fold-in
              final_score = 0.45 * svd_score + 0.55 * tfidf_profile_score

    Has liked or viewed any movie?
      YES --> TF-IDF content profile
              aggregate cosine similarity from liked movie vectors

    Has genre preferences from onboarding?
      YES --> Popularity filtered by preferred genres

    Otherwise:
      --> Global Bayesian popularity baseline

### SVD Fold-In (cold-start without retraining)

New users are not in the training matrix. A virtual latent vector u is
computed from the user's rated movies using the normal equations:

    Inputs:
      r       = user rating vector (sparse, with implicit weights)
      Vt_sub  = item factor rows for rated movies (from svd_Vt)

    Solve:
      A = Vt_sub @ Vt_sub.T + lambda * I
      u = inv(A) @ (Vt_sub @ r_centered)

    Score all movies:
      scores = u @ svd_Vt

Implicit feedback weights: like=5.0, watch=4.0, view=3.5, click=3.0

---

## Data Flow

### Recommendation Request

    Browser -- GET /recommendations/{user_id}
                |
                v
    FastAPI (recommendations.py)
                |
                |-- user_service.get_liked_movies(user_id)
                |     reads SQLite interaction history
                |
                |-- model_service.get_recommendations(user_id, rated_movies)
                |     determine strategy (SVD / TF-IDF / genre / popularity)
                |     compute scores for all 43,549 movies
                |     return top-N movie_ids
                |
                |-- enrich with metadata from movies_integrated dataframe
                v
    RecommendationResponse { recommendations: [...], context: "..." }

### Poster Fallback Chain

    MovieCard mounts
      |-- render img src = TMDB_CDN + movie.poster_path
      |   (poster_path from 2020 dataset -- 83% return HTTP 404)
      v
    img.onError fires
      |-- GET /movies/{id}/poster-url
      v
    tmdb_service.get_poster_url(tmdb_id)
      |-- cache hit  --> return immediately (O(1))
      |-- cache miss --> call api.themoviedb.org
                     --> store in data/cache/poster_cache.json
                     --> return fresh URL
      v
    img re-renders with working CDN URL

---

## API Reference

| Method | Route | Description |
|---|---|---|
| GET | /health | Liveness probe |
| POST | /users/ | Create anonymous session user |
| GET | /users/{id} | User profile (has_onboarding, has_preferences, interaction_count) |
| PUT | /users/{id}/preferences | Save genre preferences |
| POST | /interactions/ | Log any interaction (like/view/click/search/watch) |
| GET | /interactions/{id} | Get user interaction history |
| GET | /movies/popular?n= | Bayesian popularity ranking |
| GET | /movies/search?q=&n= | Normalised full-text search |
| GET | /movies/suggestions?q=&n= | Autocomplete title stubs (6 results) |
| GET | /movies/{id} | Full movie detail with cast, genres, overview |
| GET | /movies/{id}/similar?n= | TF-IDF content-similar movies |
| GET | /movies/{id}/poster-url | Fresh TMDB poster URL via cache |
| GET | /recommendations/{user_id}?n= | Personalised hybrid recommendations |
| GET | /recommendations/because-you-watched/{movie_id} | Item-based recommendations |

Interactive docs: http://localhost:8000/docs

---

## Docker Architecture

    docker-compose.yml
      |
      +-- api service (movierec-api)
      |     Build: Dockerfile (python:3.10-slim, multi-stage)
      |     Port: 8000:8000
      |     Volumes:
      |       ./data   -> /app/data   (DB, poster cache, processed CSVs)
      |       ./models -> /app/models (artifacts, popularity_scores.csv)
      |       ./logs   -> /app/logs
      |     Environment: TMDB_API_KEY, COLD_START_THRESHOLD, DEBUG
      |     Healthcheck: curl /health (90s start_period, 5 retries)
      |
      +-- frontend service (movierec-frontend)
            Build: frontend/Dockerfile (node:20-alpine -> nginx:alpine)
            Port: 3000:80
            Build arg: VITE_API_URL=/api
            DependsOn: api (condition: service_healthy)

    nginx.conf:
      location /        React SPA with try_files index.html fallback
      location /api/    proxy_pass http://api:8000/ (strips /api prefix)
      static assets     Cache-Control: public, immutable, max-age=1y
      gzip              enabled for JS/CSS/JSON/SVG/XML
      Security headers  X-Frame-Options SAMEORIGIN
                        X-Content-Type-Options nosniff

---

## Security

### TMDB API Key

  Stored in:   environment variable TMDB_API_KEY only
  Backend:     os.getenv("TMDB_API_KEY", "")
  If missing:  poster fetching disabled with debug log -- no crash
  Not present in: source code, notebooks, config files, or documentation
  .env:        listed in .gitignore
  .env.example: placeholder text only

### Input Validation

  Pydantic schemas validate all request bodies and query parameters.
  SQLAlchemy ORM uses parameterised queries (no raw SQL injection risk).
  Search queries normalised: re.sub(r'[^a-z0-9\s]', '', text.lower())

### Notes for Production

  For production deployment:
  - Add HTTPS via TLS-terminating reverse proxy or load balancer
  - Add authentication (JWT or session tokens)
  - Rate-limit recommendation and search endpoints
  - Migrate from SQLite to PostgreSQL for concurrent writes
  - Set DEBUG=false and configure structured JSON logging
