import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'
import { Activity, ArrowLeft, Mail, CheckCircle } from 'lucide-react'

function validatePassword(password) {
  if (password.length < 10) return 'Password must be at least 10 characters.'
  if (!/[A-Z]/.test(password)) return 'Must include at least one uppercase letter.'
  if (!/[a-z]/.test(password)) return 'Must include at least one lowercase letter.'
  if (!/\d/.test(password)) return 'Must include at least one digit.'
  if (!/[^A-Za-z0-9]/.test(password)) return 'Must include at least one special character.'
  return ''
}

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('')
  const [token, setToken] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [step, setStep] = useState('request') // request | reset | done
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleRequestReset = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await api.forgotPassword(email)
      setStep('reset')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleResetPassword = async (e) => {
    e.preventDefault()
    setError('')
    const passwordIssue = validatePassword(newPassword)
    if (passwordIssue) {
      setError(passwordIssue)
      return
    }
    setLoading(true)
    try {
      await api.resetPassword(token, newPassword)
      setStep('done')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Activity className="w-10 h-10 text-medical-500" />
            <h1 className="text-3xl font-bold">MedAI</h1>
          </div>
        </div>

        {step === 'done' ? (
          <div className="card text-center space-y-4">
            <CheckCircle className="w-12 h-12 text-green-400 mx-auto" />
            <h2 className="text-xl font-semibold">Password Reset</h2>
            <p className="text-gray-400">Your password has been reset successfully.</p>
            <Link to="/login" className="btn-primary inline-block">Sign In</Link>
          </div>
        ) : step === 'reset' ? (
          <form onSubmit={handleResetPassword} className="card space-y-4">
            <h2 className="text-xl font-semibold text-center">Reset Password</h2>
            <p className="text-gray-400 text-sm text-center">
              Enter the reset token from your email and your new password.
            </p>

            {error && (
              <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg text-red-300 text-sm">{error}</div>
            )}

            <input
              type="text" placeholder="Reset token" value={token}
              onChange={(e) => setToken(e.target.value)}
              className="input-field" required autoFocus
            />

            <input
              type="password" placeholder="New password (10+ chars)"
              value={newPassword} onChange={(e) => setNewPassword(e.target.value)}
              className="input-field" required minLength={10}
            />

            <button type="submit" className="btn-primary w-full" disabled={loading}>
              {loading ? 'Resetting...' : 'Reset Password'}
            </button>

            <button type="button" onClick={() => setStep('request')} className="w-full text-sm text-gray-400 hover:text-gray-200">
              Back
            </button>
          </form>
        ) : (
          <form onSubmit={handleRequestReset} className="card space-y-4">
            <h2 className="text-xl font-semibold text-center">Forgot Password</h2>
            <p className="text-gray-400 text-sm text-center">
              Enter your email and we'll send you a reset link.
            </p>

            {error && (
              <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg text-red-300 text-sm">{error}</div>
            )}

            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
              <input
                type="email" placeholder="Email" value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="input-field pl-10" required autoFocus
              />
            </div>

            <button type="submit" className="btn-primary w-full" disabled={loading}>
              {loading ? 'Sending...' : 'Send Reset Link'}
            </button>

            <Link to="/login" className="flex items-center justify-center gap-2 text-sm text-gray-400 hover:text-gray-200">
              <ArrowLeft className="w-4 h-4" /> Back to Sign In
            </Link>
          </form>
        )}
      </div>
    </div>
  )
}
