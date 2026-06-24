import { useState, useCallback } from 'react'
import { getFreshPosterUrl } from '../api/client'

const TMDB_BASE = 'https://image.tmdb.org/t/p/w500'

/**
 * Manages a movie poster URL with automatic TMDB fallback.
 *
 * First tries the static URL derived from `movie.poster_path`.
 * When the image fires `onError` (≈83% of CSV paths are stale),
 * calls `GET /movies/{id}/poster-url` to get the current TMDB URL,
 * then retries the image.
 *
 * Usage:
 *   const { src, onError } = usePoster(movie)
 *   <img src={src} onError={onError} />
 *   // If src is null, render a text fallback
 */
export function usePoster(movie) {
  const initialSrc = movie?.poster_path ? `${TMDB_BASE}${movie.poster_path}` : null
  const [src, setSrc] = useState(initialSrc)
  const [retried, setRetried] = useState(false)

  const onError = useCallback(async () => {
    if (retried) {
      // Already tried a fresh URL — give up, let parent render text fallback
      setSrc(null)
      return
    }
    setRetried(true)
    setSrc(null) // clear broken image while we fetch

    const movieId = movie?.movie_id
    if (!movieId) return

    try {
      const result = await getFreshPosterUrl(movieId)
      if (result?.poster_url) {
        setSrc(result.poster_url)
      }
    } catch {
      // Network error or movie not found — text fallback will render
    }
  }, [movie?.movie_id, retried])

  return { src, onError }
}
