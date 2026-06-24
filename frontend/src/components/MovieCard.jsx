import { useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { logInteraction } from '../api/client'
import { usePoster } from '../hooks/usePoster'

function StarIcon() {
  return (
    <svg className="w-3 h-3 fill-yellow-400" viewBox="0 0 20 20">
      <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
    </svg>
  )
}

// className: parent overrides card width. Default = horizontal scroll card.
// In grid layouts (Search), pass className="w-full".
// onInteraction: optional callback fired after a successful interaction log
// (used to update cold-start state in parent without prop-drilling a refresh fn).
export default function MovieCard({
  movie,
  userId,
  className = 'flex-shrink-0 w-40 sm:w-44',
  onInteraction,
}) {
  const navigate = useNavigate()
  const { src: posterSrc, onError: handlePosterError } = usePoster(movie)

  const handleClick = useCallback(async () => {
    if (userId && movie.movie_id) {
      try {
        await logInteraction(userId, {
          movie_id: movie.movie_id,
          interaction_type: 'click',
        })
        onInteraction?.()
      } catch {
        // non-blocking
      }
    }
    navigate(`/movies/${movie.movie_id}`)
  }, [movie.movie_id, userId, navigate, onInteraction])

  const rating = movie.vote_average ? Number(movie.vote_average).toFixed(1) : null
  const genreList = Array.isArray(movie.genres) ? movie.genres.slice(0, 2).join(' · ') : ''

  return (
    <div onClick={handleClick} className={`group cursor-pointer ${className}`}>
      <div className="relative rounded-lg overflow-hidden bg-card border border-border aspect-[2/3] mb-2">
        {posterSrc ? (
          <img
            src={posterSrc}
            alt={movie.title}
            loading="lazy"
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
            onError={handlePosterError}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center p-3 bg-surface">
            <span className="text-text-secondary text-xs text-center leading-tight font-medium">
              {movie.title}
            </span>
          </div>
        )}

        {rating && (
          <div className="absolute top-2 right-2 flex items-center gap-0.5 bg-black/70 backdrop-blur-sm rounded px-1.5 py-0.5">
            <StarIcon />
            <span className="text-xs font-semibold text-yellow-400">{rating}</span>
          </div>
        )}

        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      </div>

      <div className="px-0.5">
        <h3 className="text-text-primary text-sm font-medium leading-tight truncate group-hover:text-accent transition-colors">
          {movie.title}
        </h3>
        <div className="flex items-center gap-1.5 mt-0.5">
          {movie.year && <span className="text-muted text-xs">{movie.year}</span>}
          {movie.year && genreList && <span className="text-border text-xs">·</span>}
          {genreList && <span className="text-muted text-xs truncate">{genreList}</span>}
        </div>
        {movie.reason && (
          <p className="text-accent text-xs mt-0.5 truncate">{movie.reason}</p>
        )}
      </div>
    </div>
  )
}
