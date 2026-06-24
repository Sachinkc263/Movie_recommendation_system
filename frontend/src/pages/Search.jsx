import { useState, useEffect, useRef, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { searchMovies, logInteraction } from '../api/client'
import MovieCard from '../components/MovieCard'

const PAGE_SIZE = 12
const FETCH_SIZE = 100

function SkeletonCard() {
  return (
    <div className="animate-pulse">
      <div className="rounded-lg bg-surface aspect-[2/3] mb-2" />
      <div className="h-3 bg-surface rounded w-4/5 mb-1.5" />
      <div className="h-2.5 bg-surface rounded w-3/5" />
    </div>
  )
}

export default function Search({ userId, onInteraction }) {
  const [searchParams, setSearchParams] = useSearchParams()
  const query = searchParams.get('q') || ''

  const [allResults, setAllResults] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [inputValue, setInputValue] = useState(query)
  const [page, setPage] = useState(1)

  const debounceRef = useRef(null)
  const loggedRef = useRef('')
  const lastQueryRef = useRef('')

  const doSearch = useCallback(
    async (q) => {
      if (!q.trim()) {
        setAllResults([])
        setTotal(0)
        setPage(1)
        return
      }
      if (lastQueryRef.current === q.trim()) return

      setLoading(true)
      setError(null)
      setPage(1)
      try {
        const data = await searchMovies(q.trim(), FETCH_SIZE)
        const movies = data.movies || []
        setAllResults(movies)
        setTotal(data.total || movies.length)
        lastQueryRef.current = q.trim()

        // Log search interaction once per unique query
        if (userId && loggedRef.current !== q.trim()) {
          loggedRef.current = q.trim()
          logInteraction(userId, {
            interaction_type: 'search',
            search_query: q.trim(),
          })
            .then(() => onInteraction?.())  // update cold-start state after search logged
            .catch(console.error)
        }
      } catch (err) {
        console.error(err)
        setError('Search failed. Please try again.')
      } finally {
        setLoading(false)
      }
    },
    [userId, onInteraction]
  )

  useEffect(() => {
    if (query) {
      setInputValue(query)
      doSearch(query)
    } else {
      setAllResults([])
      setTotal(0)
      setPage(1)
      lastQueryRef.current = ''
    }
  }, [query, doSearch])

  const handleInputChange = (e) => {
    const val = e.target.value
    setInputValue(val)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      if (val.trim()) {
        if (val.trim() !== query) {
          lastQueryRef.current = ''
        }
        setSearchParams({ q: val.trim() })
      } else {
        setSearchParams({})
        setAllResults([])
        setTotal(0)
        setPage(1)
        lastQueryRef.current = ''
      }
    }, 400)
  }

  // Pagination helpers
  const totalPages = Math.ceil(allResults.length / PAGE_SIZE)
  const pageMovies = allResults.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-12">
      <div className="mb-8">
        <h1 className="text-text-primary text-2xl font-bold mb-4">Search Movies</h1>
        <div className="relative max-w-lg">
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted pointer-events-none"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            placeholder="Search by title, genre..."
            autoFocus
            className="w-full bg-card border border-border rounded-lg pl-9 pr-4 py-3 text-text-primary placeholder-muted focus:outline-none focus:border-accent transition-colors"
          />
        </div>
      </div>

      {query && !loading && !error && (
        <p className="text-text-secondary text-sm mb-6">
          {total > 0
            ? `${total} result${total !== 1 ? 's' : ''} for "${query}"`
            : `No results found for "${query}"`}
        </p>
      )}

      {error && (
        <div className="mb-6 px-4 py-3 bg-red-900/30 border border-red-700/50 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-5">
          {Array.from({ length: PAGE_SIZE }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      ) : pageMovies.length > 0 ? (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-5">
            {pageMovies.map((movie) => (
              <MovieCard
                key={movie.movie_id}
                movie={movie}
                userId={userId}
                className="w-full"
                onInteraction={onInteraction}
              />
            ))}
          </div>

          {totalPages > 1 && (
            <div className="mt-8 flex items-center justify-center gap-3">
              <button
                onClick={() => { setPage((p) => Math.max(1, p - 1)); window.scrollTo(0, 0) }}
                disabled={page === 1}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg border border-border text-sm text-text-secondary hover:text-text-primary hover:border-accent/50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Prev
              </button>

              <div className="flex items-center gap-1">
                {Array.from({ length: totalPages }).map((_, i) => {
                  const p = i + 1
                  const near = Math.abs(p - page) <= 1 || p === 1 || p === totalPages
                  if (!near) {
                    return (i === 1 || i === totalPages - 2)
                      ? <span key={p} className="text-muted text-sm px-1">…</span>
                      : null
                  }
                  return (
                    <button
                      key={p}
                      onClick={() => { setPage(p); window.scrollTo(0, 0) }}
                      className={`w-8 h-8 rounded-md text-sm font-medium transition-colors ${
                        p === page
                          ? 'bg-accent text-white'
                          : 'text-text-secondary hover:text-text-primary hover:bg-surface'
                      }`}
                    >
                      {p}
                    </button>
                  )
                })}
              </div>

              <button
                onClick={() => { setPage((p) => Math.min(totalPages, p + 1)); window.scrollTo(0, 0) }}
                disabled={page === totalPages}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg border border-border text-sm text-text-secondary hover:text-text-primary hover:border-accent/50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                Next
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          )}
        </>
      ) : !query ? (
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <svg className="w-12 h-12 text-muted mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <p className="text-text-secondary text-base">Start typing to search for movies</p>
        </div>
      ) : null}
    </main>
  )
}
