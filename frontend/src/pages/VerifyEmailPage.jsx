import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { api } from '../services/api'
import { Activity, CheckCircle, XCircle, Loader } from 'lucide-react'

export default function VerifyEmailPage() {
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token')
  const [status, setStatus] = useState(token ? 'verifying' : 'no-token') // verifying | success | error | no-token | resend
  const [error, setError] = useState('')
  const [email, setEmail] = useState('')
  const [resendLoading, setResendLoading] = useState(false)
  const [resendSent, setResendSent] = useState(false)

  useEffect(() => {
    if (!token) return
    let cancelled = false

    api.verifyEmail(token)
      .then(() => { if (!cancelled) setStatus('success') })
      .catch((err) => {
        if (!cancelled) {
          setStatus('error')
          setError(err.message)
        }
      })

    return () => { cancelled = true }
  }, [token])

  const handleResend = async (e) => {
    e.preventDefault()
    if (!email) return
    setResendLoading(true)
    setError('')
    try {
      await api.resendVerification(email)
      setResendSent(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setResendLoading(false)
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

        {status === 'verifying' && (
          <div className="card text-center space-y-4">
            <Loader className="w-12 h-12 text-primary-400 mx-auto animate-spin" />
            <h2 className="text-xl font-semibold">Verifying Email</h2>
            <p className="text-gray-400">Please wait...</p>
          </div>
        )}

        {status === 'success' && (
          <div className="card text-center space-y-4">
            <CheckCircle className="w-12 h-12 text-green-400 mx-auto" />
            <h2 className="text-xl font-semibold">Email Verified</h2>
            <p className="text-gray-400">Your email has been verified successfully.</p>
            <Link to="/login" className="btn-primary inline-block">Sign In</Link>
          </div>
        )}

        {status === 'error' && (
          <div className="card text-center space-y-4">
            <XCircle className="w-12 h-12 text-red-400 mx-auto" />
            <h2 className="text-xl font-semibold">Verification Failed</h2>
            <p className="text-gray-400">{error || 'The link may have expired or is invalid.'}</p>
            <button onClick={() => setStatus('resend')} className="btn-primary">
              Resend Verification Email
            </button>
            <Link to="/login" className="block text-sm text-gray-400 hover:text-gray-200">
              Back to Sign In
            </Link>
          </div>
        )}

        {status === 'no-token' && (
          <div className="card text-center space-y-4">
            <h2 className="text-xl font-semibold">Email Verification</h2>
            <p className="text-gray-400">
              Check your email for a verification link, or request a new one below.
            </p>
            <button onClick={() => setStatus('resend')} className="btn-primary">
              Resend Verification Email
            </button>
            <Link to="/login" className="block text-sm text-gray-400 hover:text-gray-200">
              Back to Sign In
            </Link>
          </div>
        )}

        {status === 'resend' && (
          <div className="card space-y-4">
            <h2 className="text-xl font-semibold text-center">Resend Verification</h2>

            {resendSent ? (
              <div className="text-center space-y-3">
                <CheckCircle className="w-10 h-10 text-green-400 mx-auto" />
                <p className="text-gray-400">If that email is registered, a verification link has been sent.</p>
              </div>
            ) : (
              <form onSubmit={handleResend} className="space-y-4">
                {error && (
                  <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg text-red-300 text-sm">{error}</div>
                )}
                <input
                  type="email" placeholder="Email" value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="input-field" required autoFocus
                />
                <button type="submit" className="btn-primary w-full" disabled={resendLoading}>
                  {resendLoading ? 'Sending...' : 'Send Verification Email'}
                </button>
              </form>
            )}

            <Link to="/login" className="block text-center text-sm text-gray-400 hover:text-gray-200">
              Back to Sign In
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}
