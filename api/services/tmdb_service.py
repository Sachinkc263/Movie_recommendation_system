"""TMDB poster cache — fetches current poster paths and persists them locally."""

import json
import logging
import os
import threading
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()  # ensure .env is loaded before reading env vars at module level

logger = logging.getLogger(__name__)

_MOVIE_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}"
_IMAGE_BASE = "https://image.tmdb.org/t/p"
_CACHE_FILE = "data/cache/poster_cache.json"

# Sentinel stored in cache when TMDB confirms a movie has no poster
_NO_POSTER = "__none__"


class TmdbService:
    """
    Lightweight TMDB poster cache.

    - Loads from `data/poster_cache.json` on first use.
    - Fetches from TMDB API on cache miss (synchronous, ~200 ms).
    - Persists to disk every 100 new entries and on explicit flush.
    - `prewarm()` runs a background thread for top-N movies at startup.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Optional[str]] = {}
        self._lock = threading.Lock()
        self._dirty = 0
        self._load_cache()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        if os.path.exists(_CACHE_FILE):
            try:
                with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info("Poster cache loaded: %d entries", len(self._cache))
            except Exception as exc:
                logger.warning("Could not load poster cache: %s", exc)

    def _save_cache(self) -> None:
        try:
            os.makedirs("data/cache", exist_ok=True)
            with self._lock:
                snapshot = dict(self._cache)
            with open(_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(snapshot, f)
            self._dirty = 0
        except Exception as exc:
            logger.warning("Could not save poster cache: %s", exc)

    # ── Core fetch ─────────────────────────────────────────────────────────────

    def get_poster_path(self, tmdb_id: int) -> Optional[str]:
        """
        Return the current TMDB poster_path for *tmdb_id*.

        Result is cached so repeated calls are O(1).
        Returns None when TMDB has no poster for the movie.
        """
        key = str(int(tmdb_id))

        with self._lock:
            if key in self._cache:
                val = self._cache[key]
                return None if val == _NO_POSTER else val

        # Cache miss — call TMDB API (read key at call time, not at import time)
        api_key = os.getenv("TMDB_API_KEY", "")
        if not api_key:
            logger.debug("TMDB_API_KEY not set; skipping poster fetch for tmdb_id=%s", tmdb_id)
            return None

        try:
            url = _MOVIE_URL.format(tmdb_id=tmdb_id, api_key=api_key)
            resp = requests.get(url, timeout=6, headers={"Accept": "application/json"})
            resp.raise_for_status()
            path: Optional[str] = resp.json().get("poster_path") or None
        except Exception as exc:
            logger.debug("TMDB fetch failed for tmdb_id=%s: %s", tmdb_id, exc)
            return None

        stored = path if path else _NO_POSTER
        with self._lock:
            self._cache[key] = stored
            self._dirty += 1

        if self._dirty >= 100:
            self._save_cache()

        return path

    def get_poster_url(self, tmdb_id: int, size: str = "w500") -> Optional[str]:
        """Return a full TMDB CDN URL for *tmdb_id*, or None."""
        path = self.get_poster_path(tmdb_id)
        return f"{_IMAGE_BASE}/{size}{path}" if path else None

    def flush(self) -> None:
        """Force-write cache to disk."""
        self._save_cache()

    # ── Pre-warming ────────────────────────────────────────────────────────────

    def prewarm(self, tmdb_ids: list[int], rate: float = 15.0) -> None:
        """
        Background-thread pre-fetch for *tmdb_ids*.

        *rate* is requests-per-second (TMDB free tier allows ~40 req/s).
        """
        to_fetch = [t for t in tmdb_ids if str(t) not in self._cache]
        if not to_fetch:
            logger.info("Poster prewarm: all %d movies already cached", len(tmdb_ids))
            return

        delay = 1.0 / rate

        def _run() -> None:
            logger.info("Poster prewarm started: %d movies to fetch", len(to_fetch))
            for i, tid in enumerate(to_fetch):
                self.get_poster_path(tid)
                if (i + 1) % 50 == 0:
                    logger.info("Poster prewarm: %d/%d done", i + 1, len(to_fetch))
                time.sleep(delay)
            self._save_cache()
            logger.info(
                "Poster prewarm complete. Cache now has %d entries", len(self._cache)
            )

        threading.Thread(target=_run, daemon=True, name="tmdb-prewarm").start()


# Module-level singleton
tmdb_service = TmdbService()
