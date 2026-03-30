import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../stores/useStore'
import { api } from '../services/api'
import { ArrowLeft, Shield, ShieldCheck, CreditCard, Lock, AlertCircle, CheckCircle } from 'lucide-react'

const TIERS = [
  { id: 'free', name: 'Free', price: '$0', queries: '5/day', features: ['Basic AI queries', 'Single device'] },
  { id: 'pro', name: 'Pro', price: '$29/mo', queries: 'Unlimited', features: ['Unlimited queries', 'Priority support', 'File uploads', 'Multi-device'] },
  { id: 'enterprise', name: 'Enterprise', price: 'Custom', queries: 'Unlimited', features: ['Everything in Pro', 'Hospital integration', 'Custom models', 'SLA guarantee', 'Audit exports'] },
]

export default function SettingsPage() {
  const user = useAuthStore((s) => s.user)
  const loadUser = useAuthStore((s) => s.loadUser)
  const navigate = useNavigate()

  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  // 2FA state
  const [setupData, setSetupData] = useState(null)
  const [totpCode, setTotpCode] = useState('')
  const [twoFALoading, setTwoFALoading] = useState(false)

  // Password change state
  const [currentPw, setCurrentPw] = useState('')
  const [newPw, setNewPw] = useState('')
  const [pwLoading, setPwLoading] = useState(false)

  // Subscription
  const [subLoading, setSubLoading] = useState('')

  const showSuccess = (msg) => { setSuccess(msg); setError(''); setTimeout(() => setSuccess(''), 4000) }
  const showError = (msg) => { setError(msg); setSuccess('') }

  // ── 2FA ─────────────────────────────────────────────────────
  const handleSetup2FA = async () => {
    setTwoFALoading(true)
    setError('')
    try {
      const data = await api.setup2FA()
      setSetupData(data)
    } catch (err) { showError(err.message) }
    finally { setTwoFALoading(false) }
  }

  const handleEnable2FA = async () => {
    if (!totpCode || totpCode.length !== 6) return
    setTwoFALoading(true)
    try {
      await api.enable2FA(totpCode)
      setSetupData(null)
      setTotpCode('')
      await loadUser()
      showSuccess('2FA enabled successfully')
    } catch (err) { showError(err.message) }
    finally { setTwoFALoading(false) }
  }

  const handleDisable2FA = async () => {
    if (!totpCode || totpCode.length !== 6) return
    setTwoFALoading(true)
    try {
      await api.disable2FA(totpCode)
      setTotpCode('')
      await loadUser()
      showSuccess('2FA disabled')
    } catch (err) { showError(err.message) }
    finally { setTwoFALoading(false) }
  }

  // ── Password Change ─────────────────────────────────────────
  const handleChangePassword = async (e) => {
    e.preventDefault()
    setPwLoading(true)
    try {
      await api.changePassword(currentPw, newPw)
      setCurrentPw('')
      setNewPw('')
      showSuccess('Password changed. You may need to log in again.')
    } catch (err) { showError(err.message) }
    finally { setPwLoading(false) }
  }

  // ── Subscription ────────────────────────────────────────────
  const handleChangeTier = async (tier) => {
    setSubLoading(tier)
    try {
      await api.updateSubscription(tier)
      await loadUser()
      showSuccess(`Subscription updated to ${tier}`)
    } catch (err) { showError(err.message) }
    finally { setSubLoading('') }
  }

  const currentTier = user?.subscription_tier || 'free'

  return (
    <div className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <button onClick={() => navigate('/')} className="p-2 hover:bg-gray-800 rounded-lg" aria-label="Back to chat">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h1 className="text-2xl font-bold">Settings</h1>
        </div>

        {/* Alerts */}
        {error && (
          <div className="mb-6 p-3 bg-red-900/30 border border-red-800 rounded-lg flex items-center gap-2 text-red-300 text-sm">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span className="flex-1">{error}</span>
            <button onClick={() => setError('')} className="text-red-400 hover:text-red-300 text-xs">Dismiss</button>
          </div>
        )}
        {success && (
          <div className="mb-6 p-3 bg-green-900/30 border border-green-800 rounded-lg flex items-center gap-2 text-green-300 text-sm">
            <CheckCircle className="w-4 h-4 shrink-0" />
            <span>{success}</span>
          </div>
        )}

        {/* 2FA Section */}
        <div className="card mb-6">
          <div className="flex items-center gap-3 mb-4">
            {user?.two_factor_enabled ? <ShieldCheck className="w-5 h-5 text-green-400" /> : <Shield className="w-5 h-5 text-gray-400" />}
            <h2 className="text-lg font-semibold">Two-Factor Authentication</h2>
            {user?.two_factor_enabled && <span className="text-xs bg-green-900/50 text-green-400 px-2 py-0.5 rounded-full">Enabled</span>}
          </div>

          {!user?.two_factor_enabled ? (
            <>
              {!setupData ? (
                <div>
                  <p className="text-sm text-gray-400 mb-3">Add an extra layer of security to your account with TOTP-based 2FA.</p>
                  <button onClick={handleSetup2FA} disabled={twoFALoading} className="btn-primary">
                    {twoFALoading ? 'Setting up...' : 'Set Up 2FA'}
                  </button>
                </div>
              ) : (
                <div>
                  <p className="text-sm text-gray-400 mb-3">Scan this code with your authenticator app (Google Authenticator, Authy, etc.):</p>
                  <div className="bg-gray-800 rounded-lg p-4 mb-4">
                    <p className="text-xs text-gray-500 mb-1">Manual entry key:</p>
                    <code className="text-sm text-primary-400 font-mono break-all">{setupData.secret}</code>
                  </div>
                  <div className="flex gap-2">
                    <input
                      type="text" inputMode="numeric" maxLength={6} placeholder="Enter 6-digit code"
                      value={totpCode} onChange={(e) => setTotpCode(e.target.value.replace(/\D/g, ''))}
                      className="input-field w-48"
                    />
                    <button onClick={handleEnable2FA} disabled={twoFALoading || totpCode.length !== 6} className="btn-primary">
                      {twoFALoading ? 'Verifying...' : 'Verify & Enable'}
                    </button>
                    <button onClick={() => { setSetupData(null); setTotpCode('') }} className="btn-secondary">Cancel</button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div>
              <p className="text-sm text-gray-400 mb-3">Enter a TOTP code from your authenticator to disable 2FA.</p>
              <div className="flex gap-2">
                <input
                  type="text" inputMode="numeric" maxLength={6} placeholder="Enter 6-digit code"
                  value={totpCode} onChange={(e) => setTotpCode(e.target.value.replace(/\D/g, ''))}
                  className="input-field w-48"
                />
                <button onClick={handleDisable2FA} disabled={twoFALoading || totpCode.length !== 6} className="btn-secondary text-red-400 border-red-800 hover:bg-red-900/20">
                  {twoFALoading ? 'Disabling...' : 'Disable 2FA'}
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Password Change */}
        <div className="card mb-6">
          <div className="flex items-center gap-3 mb-4">
            <Lock className="w-5 h-5 text-gray-400" />
            <h2 className="text-lg font-semibold">Change Password</h2>
          </div>
          <form onSubmit={handleChangePassword} className="space-y-3">
            <input
              type="password" placeholder="Current password" value={currentPw}
              onChange={(e) => setCurrentPw(e.target.value)}
              className="input-field w-full" required
            />
            <input
              type="password" placeholder="New password (10+ chars)" value={newPw}
              onChange={(e) => setNewPw(e.target.value)}
              className="input-field w-full" minLength={10} required
            />
            <button type="submit" disabled={pwLoading || !currentPw || newPw.length < 10} className="btn-primary">
              {pwLoading ? 'Changing...' : 'Change Password'}
            </button>
          </form>
        </div>

        {/* Subscription Tiers */}
        <div className="card">
          <div className="flex items-center gap-3 mb-4">
            <CreditCard className="w-5 h-5 text-gray-400" />
            <h2 className="text-lg font-semibold">Subscription Plan</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {TIERS.map((tier) => (
              <div
                key={tier.id}
                className={`rounded-xl border p-4 ${
                  currentTier === tier.id
                    ? 'border-primary-500 bg-primary-900/20'
                    : 'border-gray-700 bg-gray-800/30'
                }`}
              >
                <h3 className="text-lg font-bold mb-1">{tier.name}</h3>
                <p className="text-2xl font-bold text-primary-400 mb-1">{tier.price}</p>
                <p className="text-xs text-gray-500 mb-3">{tier.queries} queries</p>
                <ul className="text-sm text-gray-400 space-y-1 mb-4">
                  {tier.features.map((f, i) => (
                    <li key={i} className="flex items-center gap-1.5">
                      <CheckCircle className="w-3 h-3 text-green-500 shrink-0" />
                      {f}
                    </li>
                  ))}
                </ul>
                {currentTier === tier.id ? (
                  <span className="block text-center text-xs text-primary-400 font-medium py-2">Current Plan</span>
                ) : (
                  <button
                    onClick={() => handleChangeTier(tier.id)}
                    disabled={!!subLoading}
                    className="w-full btn-secondary text-sm"
                  >
                    {subLoading === tier.id ? 'Updating...' : `Switch to ${tier.name}`}
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
