import { useState, useCallback } from 'react'
import { getFreshPosterUrl } from '../api/client'

const TMDB_BASE = 'https://image.tmdb.org/t/p/w500'

export function usePoster(movie) {
  const initialSrc = movie?.poster_path ? `${TMDB_BASE}${movie.poster_path}` : null
  const [src, setSrc] = useState(initialSrc)
  const [loading, setLoading] = useState(false)
  const [retried, setRetried] = useState(false)

  const onError = useCallback(async () => {
    if (retried || loading) return
    setRetried(true)
    setLoading(true)
    // Keep src as-is (broken) while fetching — avoids flash of text fallback

    const movieId = movie?.movie_id
    if (!movieId) {
      setLoading(false)
      setSrc(null)
      return
    }

    try {
      const result = await getFreshPosterUrl(movieId)
      if (result?.poster_url) {
        setSrc(result.poster_url)
      } else {
        setSrc(null)
      }
    } catch {
      setSrc(null)
    } finally {
      setLoading(false)
    }
  }, [movie?.movie_id, retried, loading])

  return { src, loading, onError }
}
