import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { getMovie, getSimilarMovies, logInteraction, getUserHistory } from '../api/client'
import { usePoster } from '../hooks/usePoster'
import RecommendationSection from '../components/RecommendationSection'

function SkeletonDetail() {
  return (
    <div className="animate-pulse max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-12">
      <div className="flex gap-8 flex-col sm:flex-row">
        <div className="flex-shrink-0 w-full sm:w-56 aspect-[2/3] bg-surface rounded-xl" />
        <div className="flex-1 space-y-4 pt-2">
          <div className="h-8 bg-surface rounded w-3/4" />
          <div className="h-4 bg-surface rounded w-1/3" />
          <div className="h-4 bg-surface rounded w-full" />
          <div className="h-4 bg-surface rounded w-5/6" />
          <div className="h-4 bg-surface rounded w-4/5" />
        </div>
      </div>
    </div>
  )
}

function PosterBox({ movie }) {
  const { src, onError } = usePoster(movie)
  return (
    <div className="relative rounded-xl overflow-hidden aspect-[2/3] bg-card border border-border">
      {src ? (
        <img
          src={src}
          alt={movie.title}
          className="w-full h-full object-cover"
          onError={onError}
        />
      ) : (
        <div className="w-full h-full flex items-center justify-center p-4 bg-surface">
          <span className="text-text-secondary text-sm text-center leading-tight font-medium">
            {movie.title}
          </span>
        </div>
      )}
    </div>
  )
}

export default function MovieDetail({ userId, onInteraction }) {
  const { id } = useParams()
  const navigate = useNavigate()
  const [movie, setMovie] = useState(null)
  const [similar, setSimilar] = useState([])
  const [similarLoading, setSimilarLoading] = useState(true)
  const [loading, setLoading] = useState(true)
  const [liked, setLiked] = useState(false)
  const [liking, setLiking] = useState(false)
  const [error, setError] = useState(null)

  // Load movie + check like state from history
  useEffect(() => {
    if (!id) return
    setLoading(true)
    setError(null)
    setLiked(false)

    const movieId = parseInt(id, 10)

    Promise.all([
      getMovie(id),
      userId ? getUserHistory(userId, 100) : Promise.resolve([]),
    ])
      .then(([m, history]) => {
        setMovie(m)
        // Restore persisted like state
        if (Array.isArray(history)) {
          setLiked(
            history.some(
              (h) => h.movie_id === movieId && h.interaction_type === 'like'
            )
          )
        }
        setSimilarLoading(true)
        return getSimilarMovies(id, 10)
      })
      .then((s) => setSimilar(Array.isArray(s) ? s : []))
      .catch((err) => {
        console.error(err)
        setError('Could not load movie details.')
      })
      .finally(() => {
        setLoading(false)
        setSimilarLoading(false)
      })
  }, [id, userId])

  // Log "view" interaction + update cold-start state when movie loads
  useEffect(() => {
    if (!userId || !id || loading) return
    const movieId = parseInt(id, 10)
    if (!movieId) return
    logInteraction(userId, {
      movie_id: movieId,
      interaction_type: 'view',
    })
      .then(() => onInteraction?.())  // mark user as having interacted
      .catch(() => {})
  }, [userId, id, loading]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleLike = useCallback(async () => {
    if (!userId || !movie || liking || liked) return
    setLiking(true)
    try {
      await logInteraction(userId, {
        movie_id: movie.movie_id,
        interaction_type: 'like',
        rating: 5,
      })
      setLiked(true)
      onInteraction?.()
    } catch (err) {
      console.error(err)
    } finally {
      setLiking(false)
    }
  }, [userId, movie, liking, liked, onInteraction])

  if (loading) return <SkeletonDetail />

  if (error || !movie) {
    return (
      <div className="max-w-7xl mx-auto px-4 pt-24 pb-12 text-center">
        <p className="text-text-secondary text-lg mb-4">{error || 'Movie not found.'}</p>
        <button onClick={() => navigate(-1)} className="text-accent hover:underline text-sm">
          Go back
        </button>
      </div>
    )
  }

  const rating = movie.vote_average ? Number(movie.vote_average).toFixed(1) : null
  const genres = Array.isArray(movie.genres) ? movie.genres : []
  const cast = Array.isArray(movie.cast) ? movie.cast.slice(0, 6) : []

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-12">
      <button
        onClick={() => navigate(-1)}
        className="flex items-center gap-1.5 text-text-secondary hover:text-text-primary text-sm mb-6 transition-colors"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        Back
      </button>

      <div className="flex flex-col sm:flex-row gap-8 mb-12">
        <div className="flex-shrink-0 w-full sm:w-56">
          {/* usePoster handles stale path → TMDB fallback automatically */}
          <PosterBox movie={movie} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-4 mb-3">
            <h1 className="text-text-primary text-2xl sm:text-3xl font-bold leading-tight">
              {movie.title}
              {movie.year && (
                <span className="text-text-secondary font-normal text-xl ml-2">
                  ({movie.year})
                </span>
              )}
            </h1>

            <button
              onClick={handleLike}
              disabled={liking || liked}
              title={liked ? 'Liked!' : 'Like this movie'}
              className={`flex-shrink-0 flex items-center gap-1.5 px-3 py-2 rounded-lg border text-sm font-medium transition-colors ${
                liked
                  ? 'bg-red-900/30 border-red-700/50 text-red-400 cursor-default'
                  : 'border-border text-text-secondary hover:border-accent hover:text-accent'
              }`}
            >
              <svg
                className={`w-4 h-4 ${liked ? 'fill-red-500' : 'fill-none stroke-current'}`}
                viewBox="0 0 24 24"
                strokeWidth={liked ? 0 : 2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"
                />
              </svg>
              {liked ? 'Liked' : 'Like'}
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-2 mb-4">
            {rating && (
              <span className="flex items-center gap-1 bg-yellow-500/10 border border-yellow-500/30 text-yellow-400 text-sm px-2.5 py-1 rounded-full font-medium">
                <svg className="w-3.5 h-3.5 fill-yellow-400" viewBox="0 0 20 20">
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                </svg>
                {rating}
              </span>
            )}
            {genres.map((g) => (
              <span key={g} className="bg-surface border border-border text-text-secondary text-xs px-2.5 py-1 rounded-full">
                {g}
              </span>
            ))}
          </div>

          {movie.overview && (
            <p className="text-text-secondary text-sm sm:text-base leading-relaxed mb-5">
              {movie.overview}
            </p>
          )}

          <div className="space-y-3 text-sm">
            {movie.director && (
              <div className="flex gap-2">
                <span className="text-muted w-20 flex-shrink-0">Director</span>
                <span className="text-text-primary font-medium">{movie.director}</span>
              </div>
            )}
            {cast.length > 0 && (
              <div className="flex gap-2">
                <span className="text-muted w-20 flex-shrink-0">Cast</span>
                <span className="text-text-primary">{cast.join(', ')}</span>
              </div>
            )}
            {movie.tmdb_id && (
              <div className="flex gap-2">
                <span className="text-muted w-20 flex-shrink-0">TMDB</span>
                <a
                  href={`https://www.themoviedb.org/movie/${movie.tmdb_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-accent hover:underline"
                >
                  View on TMDB
                </a>
              </div>
            )}
          </div>
        </div>
      </div>

      <RecommendationSection
        title="Similar Movies"
        movies={similar}
        loading={similarLoading}
        userId={userId}
        onInteraction={onInteraction}
        skeletonCount={5}
      />
    </main>
  )
}
