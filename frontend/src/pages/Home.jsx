import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import RecommendationSection from '../components/RecommendationSection'
import {
  getPopularMovies,
  getUserRecommendations,
  getBecauseYouWatched,
  getUserHistory,
} from '../api/client'

// ── Cold-start prompt ──────────────────────────────────────────────────────────
function EmptyState() {
  const navigate = useNavigate()
  return (
    <div className="flex flex-col items-center justify-center py-20 px-4 text-center">
      <div className="w-20 h-20 rounded-full bg-accent/10 flex items-center justify-center mb-6">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-accent">
          <circle cx="11" cy="11" r="8" />
          <path d="m21 21-4.35-4.35" />
        </svg>
      </div>
      <h2 className="text-text-primary text-2xl font-bold mb-3">What are you in the mood for?</h2>
      <p className="text-text-secondary text-base mb-8 max-w-md">
        Search for a movie you enjoy and we'll build personalised recommendations around your taste.
      </p>
      <button
        onClick={() => navigate('/search')}
        className="bg-accent hover:bg-accent-hover text-white font-semibold px-8 py-3 rounded-lg transition-colors text-base"
      >
        Search Movies
      </button>
      <div className="mt-6 flex gap-2 flex-wrap justify-center">
        {['Action', 'Drama', 'Comedy', 'Thriller', 'Sci-Fi'].map((g) => (
          <button
            key={g}
            onClick={() => navigate(`/search?q=${encodeURIComponent(g)}`)}
            className="px-3 py-1.5 text-sm bg-surface border border-border text-text-secondary rounded-full hover:text-text-primary hover:border-accent/50 transition-colors"
          >
            {g}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function Home({ user, showEmptyState, refreshUser, onInteraction }) {
  const [popular, setPopular] = useState([])
  const [popularLoading, setPopularLoading] = useState(true)

  const [forYou, setForYou] = useState([])
  const [forYouLoading, setForYouLoading] = useState(true)

  // "Similar To Your Interests" — content-based from most recent liked/viewed movie
  const [similarMovies, setSimilarMovies] = useState([])
  const [similarLoading, setSimilarLoading] = useState(false)
  const [hasSimilar, setHasSimilar] = useState(false)

  const refreshedRef = useRef(false)

  // Refresh user on mount so interaction_count is authoritative
  useEffect(() => {
    if (user?.id && refreshUser && !refreshedRef.current) {
      refreshedRef.current = true
      refreshUser()
    }
    return () => { refreshedRef.current = false }
  }, [user?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  // Popular movies — always shown
  useEffect(() => {
    getPopularMovies(20)
      .then(setPopular)
      .catch(console.error)
      .finally(() => setPopularLoading(false))
  }, [])

  // Personalised recs — only when user has engaged
  useEffect(() => {
    if (!user?.id) {
      setForYouLoading(false)
      return
    }
    setForYouLoading(true)
    getUserRecommendations(user.id, 20)
      .then((res) => setForYou(res.recommendations || []))
      .catch(console.error)
      .finally(() => setForYouLoading(false))
  }, [user?.id])

  // "Similar To Your Interests" — seeded by most recent liked/viewed movie
  useEffect(() => {
    if (!user?.id) return

    getUserHistory(user.id, 50).then((history) => {
      if (!Array.isArray(history) || history.length === 0) return

      const seed = history.find(
        (h) => ['like', 'view', 'click', 'watch'].includes(h.interaction_type) && h.movie_id
      )
      if (!seed) return

      setHasSimilar(true)
      setSimilarLoading(true)
      getBecauseYouWatched(seed.movie_id, user.id, 15)
        .then((res) => setSimilarMovies(res.recommendations || []))
        .catch(console.error)
        .finally(() => setSimilarLoading(false))
    }).catch(console.error)
  }, [user?.id])

  const userId = user?.id
  const hasEngaged = !showEmptyState

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-12">
      {hasEngaged && (
        <div className="mb-10">
          <h1 className="text-text-primary text-3xl sm:text-4xl font-bold mb-2">
            Your Next Favourite Film
          </h1>
          <p className="text-text-secondary text-base">
            Personalised picks powered by machine learning.
          </p>
        </div>
      )}

      {showEmptyState && <EmptyState />}

      {/* 1 — Recommended For You (personalised SVD/hybrid) */}
      {hasEngaged && (
        <RecommendationSection
          title="Recommended For You"
          movies={forYou}
          loading={forYouLoading}
          userId={userId}
          onInteraction={onInteraction}
        />
      )}

      {/* 2 — Similar To Your Interests (content-based from liked movies) */}
      {hasEngaged && (hasSimilar || similarLoading) && (
        <RecommendationSection
          title="Similar To Your Interests"
          movies={similarMovies}
          loading={similarLoading}
          userId={userId}
          onInteraction={onInteraction}
          skeletonCount={5}
        />
      )}

      {/* 3 — Popular Movies (always visible) */}
      <RecommendationSection
        title="Popular Movies"
        movies={popular}
        loading={popularLoading}
        userId={userId}
        onInteraction={onInteraction}
      />
    </main>
  )
}
