import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../services/api'
import { Activity, CheckCircle } from 'lucide-react'

function validatePassword(password) {
  if (password.length < 10) return 'Password must be at least 10 characters.'
  if (!/[A-Z]/.test(password)) return 'Password must include at least one uppercase letter.'
  if (!/[a-z]/.test(password)) return 'Password must include at least one lowercase letter.'
  if (!/\d/.test(password)) return 'Password must include at least one digit.'
  if (!/[^A-Za-z0-9]/.test(password)) return 'Password must include at least one special character.'
  return ''
}

export default function RegisterPage() {
  const [form, setForm] = useState({ email: '', password: '', fullName: '', role: 'patient' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [registered, setRegistered] = useState(false)
  const navigate = useNavigate()

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    const passwordIssue = validatePassword(form.password)
    if (passwordIssue) {
      setError(passwordIssue)
      setLoading(false)
      return
    }

    try {
      await api.register(form.email, form.password, form.fullName, form.role)
      setRegistered(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const update = (field) => (e) => setForm({ ...form, [field]: e.target.value })

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Activity className="w-10 h-10 text-medical-500" />
            <h1 className="text-3xl font-bold">MedAI</h1>
          </div>
        </div>

        {registered ? (
          <div className="card text-center space-y-4">
            <CheckCircle className="w-12 h-12 text-green-400 mx-auto" />
            <h2 className="text-xl font-semibold">Account Created</h2>
            <p className="text-gray-400">
              Check your email for a verification link before signing in.
            </p>
            <div className="flex flex-col gap-2">
              <Link to="/login" className="btn-primary inline-block">Sign In</Link>
              <Link to="/verify-email" className="text-sm text-gray-400 hover:text-gray-200">
                Didn't get the email? Resend verification
              </Link>
            </div>
          </div>
        ) : (
        <form onSubmit={handleSubmit} className="card space-y-4">
          <h2 className="text-xl font-semibold text-center">Create Account</h2>

          {error && (
            <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg text-red-300 text-sm">
              {error}
            </div>
          )}

          <input type="text" placeholder="Full Name" value={form.fullName}
            onChange={update('fullName')} className="input-field" required />

          <input type="email" placeholder="Email" value={form.email}
            onChange={update('email')} className="input-field" required />

          <input type="password" placeholder="Password (10+ chars, upper, lower, digit, special)"
            value={form.password} onChange={update('password')} className="input-field"
            minLength={10} autoComplete="new-password" required />

          <select value={form.role} onChange={update('role')} className="input-field">
            <option value="patient">Patient</option>
            <option value="doctor">Doctor</option>
          </select>

          <button type="submit" className="btn-primary w-full" disabled={loading}>
            {loading ? 'Creating account...' : 'Register'}
          </button>

          <p className="text-center text-sm text-gray-400">
            Already have an account?{' '}
            <Link to="/login" className="text-primary-400 hover:underline">Sign in</Link>
          </p>
        </form>
        )}
      </div>
    </div>
  )
}
