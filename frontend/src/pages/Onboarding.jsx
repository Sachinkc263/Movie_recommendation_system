import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { submitOnboarding } from '../api/client'

const GENRES = [
  'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
  'Documentary', 'Drama', 'Fantasy', 'Horror', 'Mystery',
  'Romance', 'Science Fiction', 'Thriller',
]

const LANGUAGES = [
  { value: 'en', label: 'English' },
  { value: 'es', label: 'Spanish' },
  { value: 'fr', label: 'French' },
  { value: 'de', label: 'German' },
  { value: 'ja', label: 'Japanese' },
  { value: 'ko', label: 'Korean' },
  { value: 'hi', label: 'Hindi' },
  { value: 'it', label: 'Italian' },
  { value: 'pt', label: 'Portuguese' },
  { value: 'zh', label: 'Chinese' },
]

export default function Onboarding({ userId, onComplete }) {
  const navigate = useNavigate()
  const [selectedGenres, setSelectedGenres] = useState([])
  const [selectedLanguages, setSelectedLanguages] = useState([])
  const [actorInput, setActorInput] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  const toggleGenre = (genre) =>
    setSelectedGenres((prev) =>
      prev.includes(genre) ? prev.filter((g) => g !== genre) : [...prev, genre]
    )

  const toggleLanguage = (lang) =>
    setSelectedLanguages((prev) =>
      prev.includes(lang) ? prev.filter((l) => l !== lang) : [...prev, lang]
    )

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (selectedGenres.length === 0) {
      setError('Please select at least one genre.')
      return
    }

    const actors = actorInput
      .split(',')
      .map((a) => a.trim())
      .filter(Boolean)

    setSubmitting(true)
    setError(null)
    try {
      await submitOnboarding(userId, {
        preferred_genres: selectedGenres,
        preferred_languages: selectedLanguages,
        favorite_actors: actors,
      })
      // BUG FIX: await refreshUser so needsOnboarding flips to false BEFORE navigating
      if (onComplete) await onComplete()
      navigate('/', { replace: true })
    } catch (err) {
      setError('Failed to save preferences. Please try again.')
      console.error(err)
    } finally {
      setSubmitting(false)
    }
  }

  const handleSkip = async () => {
    // BUG FIX: save empty preferences so has_onboarding = true on the backend,
    // preventing the redirect loop (needsOnboarding would stay true otherwise).
    try {
      if (userId) {
        await submitOnboarding(userId, {
          preferred_genres: [],
          preferred_languages: [],
          favorite_actors: [],
        })
        // Refresh user so useUser sees has_onboarding=true, needsOnboarding=false
        if (onComplete) await onComplete()
      }
    } catch {
      // Non-blocking — navigate anyway
    }
    navigate('/', { replace: true })
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-2xl">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 text-accent font-bold text-2xl mb-3">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
              <path d="M18 3v2h-2V3H8v2H6V3H4v18h2v-2h2v2h8v-2h2v2h2V3h-2zM8 17H6v-2h2v2zm0-4H6v-2h2v2zm0-4H6V7h2v2zm10 8h-2v-2h2v2zm0-4h-2v-2h2v2zm0-4h-2V7h2v2z" />
            </svg>
            CineMatch
          </div>
          <h1 className="text-text-primary text-3xl font-bold mb-2">
            What do you like to watch?
          </h1>
          <p className="text-text-secondary text-base">
            Help us personalise your recommendations. You can skip for now.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="bg-card border border-border rounded-xl p-6 sm:p-8">
          {/* Genres */}
          <div className="mb-8">
            <label className="block text-text-primary font-semibold mb-4">
              Favourite Genres <span className="text-accent">*</span>
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {GENRES.map((genre) => {
                const active = selectedGenres.includes(genre)
                return (
                  <button
                    key={genre}
                    type="button"
                    onClick={() => toggleGenre(genre)}
                    className={`px-3 py-2 rounded-lg text-sm font-medium border transition-colors text-left ${
                      active
                        ? 'bg-accent border-accent text-white'
                        : 'bg-surface border-border text-text-secondary hover:border-accent/50 hover:text-text-primary'
                    }`}
                  >
                    {active && <span className="mr-1.5">✓</span>}
                    {genre}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Languages */}
          <div className="mb-8">
            <label className="block text-text-primary font-semibold mb-4">
              Preferred Languages{' '}
              <span className="text-muted text-sm font-normal">(optional)</span>
            </label>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {LANGUAGES.map((lang) => {
                const active = selectedLanguages.includes(lang.value)
                return (
                  <button
                    key={lang.value}
                    type="button"
                    onClick={() => toggleLanguage(lang.value)}
                    className={`px-3 py-2 rounded-lg text-sm font-medium border transition-colors text-left ${
                      active
                        ? 'bg-accent border-accent text-white'
                        : 'bg-surface border-border text-text-secondary hover:border-accent/50 hover:text-text-primary'
                    }`}
                  >
                    {active && <span className="mr-1.5">✓</span>}
                    {lang.label}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Actors */}
          <div className="mb-8">
            <label htmlFor="actors" className="block text-text-primary font-semibold mb-2">
              Favourite Actors{' '}
              <span className="text-muted text-sm font-normal">(optional)</span>
            </label>
            <input
              id="actors"
              type="text"
              value={actorInput}
              onChange={(e) => setActorInput(e.target.value)}
              placeholder="e.g. Tom Hanks, Meryl Streep, Leonardo DiCaprio"
              className="w-full bg-surface border border-border rounded-lg px-4 py-2.5 text-sm text-text-primary placeholder-muted focus:outline-none focus:border-accent transition-colors"
            />
            <p className="text-muted text-xs mt-1.5">Separate names with commas</p>
          </div>

          {error && (
            <div className="mb-4 px-4 py-3 bg-red-900/30 border border-red-700/50 rounded-lg text-red-400 text-sm">
              {error}
            </div>
          )}

          <div className="flex gap-3">
            <button
              type="submit"
              disabled={submitting}
              className="flex-1 bg-accent hover:bg-accent-hover text-white font-semibold py-3 rounded-lg transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {submitting ? 'Saving...' : 'Get Recommendations'}
            </button>
            <button
              type="button"
              onClick={handleSkip}
              disabled={submitting}
              className="px-5 py-3 text-text-secondary hover:text-text-primary border border-border hover:border-surface rounded-lg transition-colors text-sm disabled:opacity-50"
            >
              Skip
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
