import MovieGrid from './MovieGrid'

export default function RecommendationSection({
  title,
  movies,
  loading,
  userId,
  skeletonCount = 8,
  onInteraction,
}) {
  if (!loading && (!movies || movies.length === 0)) return null

  return (
    <section className="mb-10">
      <h2 className="text-text-primary text-lg font-semibold mb-4 flex items-center gap-2">
        {title}
      </h2>
      <MovieGrid
        movies={movies}
        loading={loading}
        userId={userId}
        skeletonCount={skeletonCount}
        onInteraction={onInteraction}
      />
    </section>
  )
}
