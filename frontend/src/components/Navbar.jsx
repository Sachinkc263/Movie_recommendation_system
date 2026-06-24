import { useState, useEffect, useRef, useCallback } from 'react'
import { Link, useNavigate, useLocation, useSearchParams } from 'react-router-dom'
import { getMovieSuggestions } from '../api/client'

const TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w92'

export default function Navbar({ user }) {
  const navigate = useNavigate()
  const location = useLocation()
  const [searchParams] = useSearchParams()
  const isSearchPage = location.pathname === '/search'

  // BUG FIX: sync query input with URL ?q= param when on /search
  const [query, setQuery] = useState(() =>
    isSearchPage ? (searchParams.get('q') || '') : ''
  )
  const [suggestions, setSuggestions] = useState([])
  const [showDropdown, setShowDropdown] = useState(false)
  const [activeSuggestion, setActiveSuggestion] = useState(-1)

  const debounceRef = useRef(null)
  const wrapperRef = useRef(null)
  const inputRef = useRef(null)

  // Keep input in sync with URL param when navigating to /search
  useEffect(() => {
    if (isSearchPage) {
      setQuery(searchParams.get('q') || '')
    } else {
      setQuery('')
    }
    setSuggestions([])
    setShowDropdown(false)
    setActiveSuggestion(-1)
  }, [location.pathname, searchParams, isSearchPage])

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleOutsideClick(e) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setShowDropdown(false)
        setActiveSuggestion(-1)
      }
    }
    document.addEventListener('mousedown', handleOutsideClick)
    return () => document.removeEventListener('mousedown', handleOutsideClick)
  }, [])

  const fetchSuggestions = useCallback(async (q) => {
    if (q.trim().length < 2) {
      setSuggestions([])
      setShowDropdown(false)
      return
    }
    try {
      const data = await getMovieSuggestions(q.trim(), 6)
      setSuggestions(data || [])
      setShowDropdown((data || []).length > 0)
      setActiveSuggestion(-1)
    } catch {
      setSuggestions([])
      setShowDropdown(false)
    }
  }, [])

  const handleInputChange = (e) => {
    const val = e.target.value
    setQuery(val)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => fetchSuggestions(val), 280)
  }

  const commitSearch = useCallback(
    (q) => {
      const trimmed = (q || query).trim()
      if (!trimmed) return
      setShowDropdown(false)
      setSuggestions([])
      setActiveSuggestion(-1)
      navigate(`/search?q=${encodeURIComponent(trimmed)}`)
    },
    [query, navigate]
  )

  const handleKeyDown = (e) => {
    if (!showDropdown || suggestions.length === 0) {
      if (e.key === 'Enter') commitSearch()
      return
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveSuggestion((prev) => Math.min(prev + 1, suggestions.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveSuggestion((prev) => Math.max(prev - 1, -1))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (activeSuggestion >= 0) {
        const s = suggestions[activeSuggestion]
        setShowDropdown(false)
        setSuggestions([])
        setActiveSuggestion(-1)
        setQuery(s.title)
        navigate(`/movies/${s.movie_id}`)
      } else {
        commitSearch()
      }
    } else if (e.key === 'Escape') {
      setShowDropdown(false)
      setActiveSuggestion(-1)
    }
  }

  const handleSuggestionClick = (s) => {
    setShowDropdown(false)
    setSuggestions([])
    setActiveSuggestion(-1)
    setQuery(s.title)
    navigate(`/movies/${s.movie_id}`)
  }

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/95 backdrop-blur-sm border-b border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-4">
          <Link
            to="/"
            className="flex-shrink-0 flex items-center gap-2 text-accent font-bold text-xl tracking-tight"
          >
            <svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M18 3v2h-2V3H8v2H6V3H4v18h2v-2h2v2h8v-2h2v2h2V3h-2zM8 17H6v-2h2v2zm0-4H6v-2h2v2zm0-4H6V7h2v2zm10 8h-2v-2h2v2zm0-4h-2v-2h2v2zm0-4h-2V7h2v2z" />
            </svg>
            CineMatch
          </Link>

          {/* Search box with autocomplete */}
          <div ref={wrapperRef} className="flex-1 max-w-xl relative">
            <div className="relative">
              <svg
                className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted pointer-events-none"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                onFocus={() => suggestions.length > 0 && setShowDropdown(true)}
                placeholder="Search movies..."
                className="w-full bg-surface border border-border rounded-lg pl-9 pr-4 py-2 text-sm text-text-primary placeholder-muted focus:outline-none focus:border-accent transition-colors"
              />
            </div>

            {/* Autocomplete dropdown */}
            {showDropdown && suggestions.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-card border border-border rounded-lg shadow-2xl overflow-hidden z-50">
                {suggestions.map((s, i) => {
                  const posterUrl = s.poster_path
                    ? `${TMDB_IMAGE_BASE}${s.poster_path}`
                    : null
                  return (
                    <button
                      key={s.movie_id}
                      type="button"
                      onClick={() => handleSuggestionClick(s)}
                      className={`w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors ${
                        i === activeSuggestion
                          ? 'bg-accent/20 text-text-primary'
                          : 'hover:bg-surface text-text-secondary hover:text-text-primary'
                      }`}
                    >
                      <div className="flex-shrink-0 w-8 h-11 rounded overflow-hidden bg-surface border border-border">
                        {posterUrl ? (
                          <img
                            src={posterUrl}
                            alt={s.title}
                            className="w-full h-full object-cover"
                            onError={(e) => { e.target.style.display = 'none' }}
                          />
                        ) : null}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate text-text-primary">
                          {s.title}
                        </p>
                        <p className="text-xs text-muted">
                          {[s.year, ...(s.genres || [])].filter(Boolean).join(' · ')}
                        </p>
                      </div>
                    </button>
                  )
                })}
                <button
                  type="button"
                  onClick={() => commitSearch()}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-accent hover:bg-surface transition-colors border-t border-border"
                >
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  Search for "{query}"
                </button>
              </div>
            )}
          </div>

          <div className="flex items-center gap-1 flex-shrink-0">
            <Link
              to="/"
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                location.pathname === '/'
                  ? 'text-text-primary bg-surface'
                  : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              Home
            </Link>
          </div>
        </div>
      </div>
    </nav>
  )
}
