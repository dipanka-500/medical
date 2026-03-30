import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useChatStore, useAuthStore } from '../stores/useStore'
import { api } from '../services/api'
import { Plus, MessageSquare, Trash2, LayoutDashboard, LogOut, Search, X, Pencil, Check, Crown, Sparkles, User, Settings } from 'lucide-react'

const TIER_CONFIG = {
  free: { label: 'Free', icon: User, color: 'text-gray-400 bg-gray-800 border-gray-700' },
  pro: { label: 'Pro', icon: Sparkles, color: 'text-primary-400 bg-primary-900/30 border-primary-700/40' },
  enterprise: { label: 'Enterprise', icon: Crown, color: 'text-yellow-400 bg-yellow-900/20 border-yellow-700/40' },
}

function SubscriptionBadge({ tier }) {
  const cfg = TIER_CONFIG[tier] || TIER_CONFIG.free
  const Icon = cfg.icon
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] font-medium rounded-full border ${cfg.color}`}>
      <Icon className="w-2.5 h-2.5" />
      {cfg.label}
    </span>
  )
}

export default function Sidebar({ onClose }) {
  const { sessions, currentSessionId, selectSession, newChat, deleteSession, loadSessions } = useChatStore()
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()

  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState(null)
  const [searching, setSearching] = useState(false)
  const [editingId, setEditingId] = useState(null)
  const [editTitle, setEditTitle] = useState('')
  const editRef = useRef(null)
  const searchTimerRef = useRef(null)

  // Focus edit input when editing starts
  useEffect(() => {
    if (editingId && editRef.current) editRef.current.focus()
  }, [editingId])

  // Clean up search timer on unmount
  useEffect(() => {
    return () => {
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
    }
  }, [])

  // Debounced search
  const handleSearchChange = (value) => {
    setSearchQuery(value)
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current)

    if (!value.trim()) {
      setSearchResults(null)
      return
    }

    searchTimerRef.current = setTimeout(async () => {
      setSearching(true)
      try {
        const results = await api.searchSessions(value.trim())
        setSearchResults(results)
      } catch {
        setSearchResults([])
      } finally {
        setSearching(false)
      }
    }, 300)
  }

  const clearSearch = () => {
    setSearchQuery('')
    setSearchResults(null)
  }

  const startRename = (e, session) => {
    e.stopPropagation()
    setEditingId(session.id)
    setEditTitle(session.title)
  }

  const [renameError, setRenameError] = useState('')

  const saveRename = async (e) => {
    e?.stopPropagation()
    if (!editTitle.trim() || !editingId) return
    setRenameError('')
    try {
      await api.renameSession(editingId, editTitle.trim())
      loadSessions()
    } catch (err) {
      setRenameError(err.message || 'Rename failed')
      // Auto-clear after 3 seconds
      setTimeout(() => setRenameError(''), 3000)
    }
    setEditingId(null)
  }

  const cancelRename = (e) => {
    e?.stopPropagation()
    setEditingId(null)
  }

  const displaySessions = searchResults !== null ? searchResults : sessions

  return (
    <div className="w-72 h-full bg-gray-900 border-r border-gray-800 flex flex-col">
      {/* New chat */}
      <div className="p-3">
        <button onClick={newChat} className="btn-secondary w-full flex items-center gap-2 justify-center">
          <Plus className="w-4 h-4" /> New Chat
        </button>
      </div>

      {/* Search */}
      <div className="px-3 pb-2">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500" />
          <input
            type="text"
            placeholder="Search chats..."
            value={searchQuery}
            onChange={(e) => handleSearchChange(e.target.value)}
            className="w-full pl-8 pr-8 py-1.5 text-sm bg-gray-800 border border-gray-700 rounded-lg text-gray-200 placeholder-gray-500 outline-none focus:border-primary-500"
          />
          {searchQuery && (
            <button onClick={clearSearch} className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300">
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
        {searching && <p className="text-xs text-gray-500 mt-1 px-1">Searching...</p>}
      </div>

      {/* Rename error */}
      {renameError && (
        <p className="mx-3 mb-1 text-xs text-red-400">{renameError}</p>
      )}

      {/* Session list */}
      <div className="flex-1 overflow-y-auto px-2 space-y-0.5">
        {displaySessions.length === 0 && (
          <p className="text-xs text-gray-500 text-center py-4">
            {searchResults !== null ? 'No matching chats' : 'No chats yet'}
          </p>
        )}
        {displaySessions.map((s) => (
          <div
            key={s.id}
            className={`group flex items-center gap-2 px-3 py-2.5 rounded-lg cursor-pointer text-sm transition-colors
              ${s.id === currentSessionId ? 'bg-gray-800 text-white' : 'text-gray-400 hover:bg-gray-800/50 hover:text-gray-200'}`}
            onClick={() => { if (editingId !== s.id) selectSession(s.id) }}
          >
            <MessageSquare className="w-4 h-4 shrink-0" />

            {editingId === s.id ? (
              <input
                ref={editRef}
                value={editTitle}
                onChange={(e) => setEditTitle(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') saveRename(e)
                  if (e.key === 'Escape') cancelRename(e)
                }}
                onBlur={saveRename}
                onClick={(e) => e.stopPropagation()}
                className="flex-1 bg-gray-700 text-white text-sm px-1.5 py-0.5 rounded outline-none"
                maxLength={120}
              />
            ) : (
              <span className="truncate flex-1" title={s.title}>{s.title}</span>
            )}

            {editingId === s.id ? (
              <button onClick={saveRename} className="p-1 text-green-400 hover:text-green-300" aria-label="Save">
                <Check className="w-3.5 h-3.5" />
              </button>
            ) : (
              <div className="flex opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={(e) => startRename(e, s)}
                  className="p-1 hover:text-primary-400"
                  aria-label={`Rename chat: ${s.title}`}
                >
                  <Pencil className="w-3 h-3" />
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); if (window.confirm('Delete this chat?')) deleteSession(s.id) }}
                  className="p-1 hover:text-red-400"
                  aria-label={`Delete chat: ${s.title}`}
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* User info + subscription tier */}
      <div className="px-3 pt-3 pb-1 border-t border-gray-800">
        <div className="flex items-center gap-2.5 px-2 py-1.5">
          <div className="w-8 h-8 rounded-full bg-primary-900 flex items-center justify-center text-primary-400 font-bold text-sm shrink-0">
            {user?.full_name?.[0]?.toUpperCase() || user?.email?.[0]?.toUpperCase() || '?'}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">{user?.full_name || user?.email}</p>
            <SubscriptionBadge tier={user?.subscription_tier} />
          </div>
        </div>
      </div>

      {/* Bottom nav */}
      <div className="px-3 pb-3 space-y-1">
        {user?.role === 'doctor' && (
          <button
            onClick={() => navigate('/dashboard')}
            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded-lg"
          >
            <LayoutDashboard className="w-4 h-4" /> Dashboard
          </button>
        )}
        <button
          onClick={() => navigate('/settings')}
          className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-400 hover:text-gray-200 hover:bg-gray-800 rounded-lg"
        >
          <Settings className="w-4 h-4" /> Settings
        </button>
        <button
          onClick={logout}
          className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-400 hover:text-red-400 hover:bg-gray-800 rounded-lg"
        >
          <LogOut className="w-4 h-4" /> Sign Out
        </button>
      </div>
    </div>
  )
}
