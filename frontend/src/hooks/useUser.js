import { useState, useEffect, useCallback } from 'react'
import { createUser, getUser } from '../api/client'

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

export function useUser() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const initUser = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)

      let sessionId = localStorage.getItem('session_id')
      if (!sessionId) {
        sessionId = generateUUID()
        localStorage.setItem('session_id', sessionId)
      }

      let userId = localStorage.getItem('user_id')
      let userData

      if (userId) {
        try {
          userData = await getUser(userId)
        } catch {
          // Stale user_id — create fresh
          userData = await createUser(sessionId)
          localStorage.setItem('user_id', String(userData.id))
        }
      } else {
        userData = await createUser(sessionId)
        localStorage.setItem('user_id', String(userData.id))
      }

      setUser(userData)
    } catch (err) {
      setError(err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    initUser()
  }, [initUser])

  const refreshUser = useCallback(async () => {
    const userId = localStorage.getItem('user_id')
    if (!userId) return
    try {
      const userData = await getUser(userId)
      setUser(userData)
      return userData
    } catch (err) {
      console.error('Failed to refresh user:', err)
    }
  }, [])

  /**
   * Optimistic interaction counter update.
   *
   * Call this immediately after logging ANY interaction (search, click, view, like).
   * It increments `interaction_count` in local React state so `showEmptyState`
   * flips to false without waiting for a round-trip to the server.
   * The server's real count is authoritative and will sync on the next `refreshUser`.
   */
  const markInteracted = useCallback(() => {
    setUser((prev) =>
      prev ? { ...prev, interaction_count: (prev.interaction_count || 0) + 1 } : prev
    )
  }, [])

  // ── Derived state ──────────────────────────────────────────────────────────

  // needsOnboarding: user exists but has NEVER submitted the form
  // (has_onboarding covers both "genres saved" and "skipped" paths)
  const needsOnboarding = user !== null && !user.has_onboarding

  // showEmptyState: user skipped onboarding AND has no logged interactions
  // → show "What are you in the mood for?" search prompt on home page
  const showEmptyState =
    user !== null &&
    !user.has_preferences &&
    (user.interaction_count || 0) === 0

  return {
    user,
    loading,
    error,
    needsOnboarding,
    showEmptyState,
    refreshUser,
    markInteracted,
  }
}
