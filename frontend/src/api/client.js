import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API error:', error?.response?.data || error.message)
    return Promise.reject(error)
  }
)

// ── Users ────────────────────────────────────────────────────────────────────
export async function createUser(sessionId) {
  const { data } = await api.post('/users/', { session_id: sessionId })
  return data
}

export async function getUser(userId) {
  const { data } = await api.get(`/users/${userId}`)
  return data
}

export async function submitOnboarding(userId, preferences) {
  const { data } = await api.post(`/users/${userId}/onboarding`, preferences)
  return data
}

export async function logInteraction(userId, interaction) {
  const { data } = await api.post(`/users/${userId}/interactions`, interaction)
  return data
}

export async function getUserHistory(userId, limit = 100) {
  const { data } = await api.get(`/users/${userId}/history`, { params: { limit } })
  return data
}

// ── Movies ───────────────────────────────────────────────────────────────────
export async function getPopularMovies(n = 20) {
  const { data } = await api.get('/movies/popular', { params: { n } })
  return data
}

export async function searchMovies(q, n = 20) {
  const { data } = await api.get('/movies/search', { params: { q, n } })
  return data
}

export async function getMovieSuggestions(q, n = 6) {
  const { data } = await api.get('/movies/suggestions', { params: { q, n } })
  return data
}

export async function getFreshPosterUrl(movieId) {
  const { data } = await api.get(`/movies/${movieId}/poster-url`)
  return data
}

export async function getMovie(movieId) {
  const { data } = await api.get(`/movies/${movieId}`)
  return data
}

export async function getSimilarMovies(movieId, n = 10) {
  const { data } = await api.get(`/movies/${movieId}/similar`, { params: { n } })
  return data
}

// ── Recommendations ──────────────────────────────────────────────────────────
export async function getUserRecommendations(userId, n = 20) {
  const { data } = await api.get(`/recommendations/user/${userId}`, { params: { n } })
  return data
}

export async function getPopularRecommendations(n = 20) {
  const { data } = await api.get('/recommendations/popular', { params: { n } })
  return data
}

export async function getBecauseYouWatched(movieId, userId, n = 10) {
  const params = { n }
  if (userId) params.user_id = userId
  const { data } = await api.get(`/recommendations/because-you-watched/${movieId}`, { params })
  return data
}

// ── Health ───────────────────────────────────────────────────────────────────
export async function getHealth() {
  const { data } = await api.get('/health')
  return data
}
