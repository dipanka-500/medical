import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuthStore, useChatStore } from '../stores/useStore'
import { api } from '../services/api'
import { ArrowLeft, Users, FileText, BarChart3, Search, UserPlus, AlertCircle } from 'lucide-react'

export default function DashboardPage() {
  const user = useAuthStore((s) => s.user)
  const newChat = useChatStore((s) => s.newChat)
  const navigate = useNavigate()
  const [dashboard, setDashboard] = useState(null)
  const [patients, setPatients] = useState([])
  const [search, setSearch] = useState('')
  const [linkEmail, setLinkEmail] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [linkLoading, setLinkLoading] = useState(false)
  const [searchLoading, setSearchLoading] = useState(false)

  const loadData = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const [dash, pats] = await Promise.all([
        api.getDoctorDashboard(),
        api.getDoctorPatients(),
      ])
      setDashboard(dash)
      setPatients(pats?.patients || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (user && user.role !== 'doctor') {
      navigate('/', { replace: true })
      return
    }
    loadData()
  }, [loadData, navigate, user])

  const handleSearch = async () => {
    setSearchLoading(true)
    try {
      const data = await api.getDoctorPatients(search)
      setPatients(data?.patients || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setSearchLoading(false)
    }
  }

  const handleLink = async () => {
    if (!linkEmail) return
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(linkEmail)) {
      setError('Please enter a valid email address')
      return
    }
    setLinkLoading(true)
    setError('')
    try {
      await api.linkPatient(linkEmail)
      setLinkEmail('')
      await loadData()
    } catch (err) {
      setError(err.message)
    } finally {
      setLinkLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen text-gray-400">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          Loading dashboard...
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <button onClick={() => navigate('/')} className="p-2 hover:bg-gray-800 rounded-lg" aria-label="Back to chat">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h1 className="text-2xl font-bold">Doctor Dashboard</h1>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-6 p-3 bg-red-900/30 border border-red-800 rounded-lg flex items-center gap-2 text-red-300 text-sm">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span className="flex-1">{error}</span>
            <button onClick={() => setError('')} className="text-red-400 hover:text-red-300 text-xs">Dismiss</button>
          </div>
        )}

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="card flex items-center gap-4">
            <div className="p-3 bg-primary-900/40 rounded-xl">
              <Users className="w-6 h-6 text-primary-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">{dashboard?.total_patients || 0}</p>
              <p className="text-sm text-gray-400">Total Patients</p>
            </div>
          </div>

          <div className="card flex items-center gap-4">
            <div className="p-3 bg-medical-700/40 rounded-xl">
              <FileText className="w-6 h-6 text-medical-500" />
            </div>
            <div>
              <p className="text-2xl font-bold">
                {Object.values(dashboard?.record_type_distribution || {}).reduce((a, b) => a + b, 0)}
              </p>
              <p className="text-sm text-gray-400">Total Records</p>
            </div>
          </div>

          <div className="card flex items-center gap-4">
            <div className="p-3 bg-purple-900/40 rounded-xl">
              <BarChart3 className="w-6 h-6 text-purple-400" />
            </div>
            <div>
              <p className="text-2xl font-bold">
                {Object.keys(dashboard?.record_type_distribution || {}).length}
              </p>
              <p className="text-sm text-gray-400">Record Types</p>
            </div>
          </div>
        </div>

        {/* Link Patient */}
        <div className="card mb-6">
          <h3 className="text-lg font-semibold mb-3">Link New Patient</h3>
          <div className="flex gap-2">
            <input
              type="email" placeholder="Patient email" value={linkEmail}
              onChange={(e) => setLinkEmail(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleLink()}
              className="input-field flex-1"
            />
            <button onClick={handleLink} disabled={linkLoading || !linkEmail} className="btn-primary flex items-center gap-2">
              <UserPlus className="w-4 h-4" /> {linkLoading ? 'Linking...' : 'Link'}
            </button>
          </div>
        </div>

        {/* Patient list */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">My Patients</h3>
            <div className="flex gap-2">
              <input
                type="text" placeholder="Search by name..." value={search}
                onChange={(e) => setSearch(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                className="input-field w-64"
              />
              <button onClick={handleSearch} disabled={searchLoading} className="btn-secondary" aria-label="Search patients">
                <Search className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="space-y-2">
            {patients.length === 0 ? (
              <p className="text-gray-500 text-center py-8">No patients linked yet</p>
            ) : (
              patients.map((p) => (
                <div
                  key={p.patient_id}
                  onClick={() => {
                    newChat()
                    navigate(`/?patient=${encodeURIComponent(p.patient_id)}`)
                  }}
                  className="flex items-center gap-4 p-4 bg-gray-800/50 hover:bg-gray-800 rounded-xl cursor-pointer transition-colors"
                >
                  <div className="w-10 h-10 rounded-full bg-primary-900 flex items-center justify-center text-primary-400 font-bold">
                    {(p.full_name || p.name)?.[0]?.toUpperCase() || '?'}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium truncate">{p.full_name || p.name}</p>
                    <p className="text-sm text-gray-400 truncate">{p.email}</p>
                  </div>
                  <div className="text-right text-sm text-gray-500">
                    {p.gender && <p>{p.gender}</p>}
                    {p.blood_type && <p>{p.blood_type}</p>}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Recent Records */}
        {dashboard?.recent_records?.length > 0 && (
          <div className="card mt-6">
            <h3 className="text-lg font-semibold mb-4">Recent Records (All Patients)</h3>
            <div className="space-y-2">
              {dashboard.recent_records.map((r) => (
                <div key={r.id} className="flex items-center gap-3 p-3 bg-gray-800/50 rounded-lg text-sm">
                  <FileText className="w-4 h-4 text-gray-400 shrink-0" />
                  <span className="flex-1 truncate">{r.title}</span>
                  <span className="text-gray-500">{r.record_type}</span>
                  <span className="text-gray-600">{new Date(r.created_at).toLocaleDateString()}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
