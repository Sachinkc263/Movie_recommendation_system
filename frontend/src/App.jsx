import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom'
import { useUser } from './hooks/useUser'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Onboarding from './pages/Onboarding'
import MovieDetail from './pages/MovieDetail'
import Search from './pages/Search'

function AppRoutes() {
  const {
    user,
    loading,
    needsOnboarding,
    showEmptyState,
    refreshUser,
    markInteracted,
  } = useUser()
  const navigate = useNavigate()
  const location = useLocation()

  // Redirect to onboarding only when user has never submitted the form.
  // After skip, has_onboarding=true → needsOnboarding=false → no redirect loop.
  useEffect(() => {
    if (!loading && needsOnboarding && location.pathname !== '/onboarding') {
      navigate('/onboarding', { replace: true })
    }
  }, [loading, needsOnboarding, location.pathname, navigate])

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <p className="text-text-secondary text-sm">Loading CineMatch…</p>
        </div>
      </div>
    )
  }

  return (
    <>
      {location.pathname !== '/onboarding' && <Navbar user={user} />}
      <Routes>
        <Route
          path="/"
          element={
            <Home
              user={user}
              showEmptyState={showEmptyState}
              refreshUser={refreshUser}
              onInteraction={markInteracted}
            />
          }
        />
        <Route
          path="/onboarding"
          element={
            <Onboarding
              userId={user?.id}
              onComplete={refreshUser}
            />
          }
        />
        <Route
          path="/movies/:id"
          element={
            <MovieDetail
              userId={user?.id}
              onInteraction={markInteracted}
            />
          }
        />
        <Route
          path="/search"
          element={
            <Search
              userId={user?.id}
              onInteraction={markInteracted}
            />
          }
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  )
}
