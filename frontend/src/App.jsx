import { useEffect, useRef, useState } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './stores/useStore'
import { ThemeProvider, useTheme } from './contexts/ThemeContext'
import ErrorBoundary from './components/ErrorBoundary'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import ForgotPasswordPage from './pages/ForgotPasswordPage'
import VerifyEmailPage from './pages/VerifyEmailPage'
import ChatPage from './pages/ChatPage'
import DashboardPage from './pages/DashboardPage'
import SettingsPage from './pages/SettingsPage'
import { WifiOff } from 'lucide-react'

const SESSION_IDLE_TIMEOUT_MS = 30 * 60 * 1000 // 30 minutes (matches backend)

function ProtectedRoute({ children }) {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)
  if (!isAuthenticated) return <Navigate to="/login" replace />
  return children
}

function OfflineBanner() {
  const [offline, setOffline] = useState(!navigator.onLine)

  useEffect(() => {
    const goOffline = () => setOffline(true)
    const goOnline = () => setOffline(false)
    window.addEventListener('offline', goOffline)
    window.addEventListener('online', goOnline)
    return () => {
      window.removeEventListener('offline', goOffline)
      window.removeEventListener('online', goOnline)
    }
  }, [])

  if (!offline) return null

  return (
    <div className="fixed top-0 inset-x-0 z-50 bg-yellow-900/90 border-b border-yellow-700 px-4 py-2 flex items-center justify-center gap-2 text-yellow-200 text-sm">
      <WifiOff className="w-4 h-4" />
      <span>You are offline. Some features may be unavailable.</span>
    </div>
  )
}

function AppRoutes() {
  const loadUser = useAuthStore((s) => s.loadUser)
  const logout = useAuthStore((s) => s.logout)
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated)
  const [isRestoring, setIsRestoring] = useState(true)
  const idleTimerRef = useRef(null)

  // Restore auth state from localStorage on mount (persist across sessions like ChatGPT/Claude)
  useEffect(() => {
    const token = localStorage.getItem('medai_token')
    if (token) {
      loadUser()
        .catch(() => {
          // Token expired or invalid — clean up
          localStorage.removeItem('medai_token')
          localStorage.removeItem('medai_refresh')
          localStorage.removeItem('medai_role')
          localStorage.removeItem('medai_current_session')
        })
        .finally(() => setIsRestoring(false))
    } else {
      setIsRestoring(false)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Session idle timeout — auto-logout after 30 minutes of inactivity
  useEffect(() => {
    if (!isAuthenticated) return

    const resetTimer = () => {
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current)
      idleTimerRef.current = setTimeout(() => {
        logout()
      }, SESSION_IDLE_TIMEOUT_MS)
    }

    const events = ['mousedown', 'keydown', 'scroll', 'touchstart']
    events.forEach((e) => window.addEventListener(e, resetTimer, { passive: true }))
    resetTimer()

    return () => {
      events.forEach((e) => window.removeEventListener(e, resetTimer))
      if (idleTimerRef.current) clearTimeout(idleTimerRef.current)
    }
  }, [isAuthenticated, logout])

  if (isRestoring) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-950">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-gray-400 text-sm">Restoring session...</p>
        </div>
      </div>
    )
  }

  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/verify-email" element={<VerifyEmailPage />} />
      <Route path="/" element={<ProtectedRoute><ChatPage /></ProtectedRoute>} />
      <Route path="/dashboard" element={<ProtectedRoute><DashboardPage /></ProtectedRoute>} />
      <Route path="/settings" element={<ProtectedRoute><SettingsPage /></ProtectedRoute>} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default function App() {
  return (
    <ThemeProvider>
      <ErrorBoundary>
        <OfflineBanner />
        <AppRoutes />
      </ErrorBoundary>
    </ThemeProvider>
  )
}
