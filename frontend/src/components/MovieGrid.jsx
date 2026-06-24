import MovieCard from './MovieCard'

function SkeletonCard() {
  return (
    <div className="flex-shrink-0 w-40 sm:w-44 animate-pulse">
      <div className="rounded-lg bg-surface aspect-[2/3] mb-2" />
      <div className="h-3 bg-surface rounded w-4/5 mb-1.5" />
      <div className="h-2.5 bg-surface rounded w-3/5" />
    </div>
  )
}

export default function MovieGrid({ movies, loading, userId, skeletonCount = 8, onInteraction }) {
  if (loading) {
    return (
      <div className="flex gap-4 overflow-x-auto scrollbar-hide pb-2">
        {Array.from({ length: skeletonCount }).map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
    )
  }

  if (!movies || movies.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-muted text-sm">
        No movies found.
      </div>
    )
  }

  return (
    <div className="flex gap-4 overflow-x-auto scrollbar-hide pb-2">
      {movies.map((movie) => (
        <MovieCard
          key={movie.movie_id}
          movie={movie}
          userId={userId}
          onInteraction={onInteraction}
        />
      ))}
    </div>
  )
}
