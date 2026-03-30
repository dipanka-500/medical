import { create } from 'zustand'
import { api } from '../services/api'

export const useAuthStore = create((set) => ({
  user: null,
  isAuthenticated: !!localStorage.getItem('medai_token'),

  login: async (email, password, totpCode) => {
    const data = await api.login(email, password, totpCode)
    if (data.access_token) {
      // Backend returns user info in data.user object
      const user = data.user || { id: data.user_id, role: data.role }
      set({ user, isAuthenticated: true })
    }
    return data
  },

  logout: () => {
    set({ user: null, isAuthenticated: false })
    api.logout() // async — clears localStorage, revokes server token, redirects to /login
  },

  loadUser: async () => {
    try {
      const user = await api.getMe()
      set({ user, isAuthenticated: true })
    } catch {
      api.logout({ redirect: false })
      set({ user: null, isAuthenticated: false })
    }
  },
}))

export const useChatStore = create((set, get) => ({
  sessions: [],
  currentSessionId: localStorage.getItem('medai_current_session') || null,
  messages: [],
  isLoading: false,
  sessionsLoaded: false,
  error: null,

  loadSessions: async () => {
    try {
      const sessions = await api.getSessions()
      const sessionList = Array.isArray(sessions) ? sessions : []
      set({ sessions: sessionList, sessionsLoaded: true })

      // Auto-restore last active session if messages are empty
      const { currentSessionId, messages } = get()
      if (currentSessionId && messages.length === 0) {
        // Verify the session still exists in the list
        const sessionExists = sessionList.some((s) => s.id === currentSessionId)
        if (sessionExists) {
          get().selectSession(currentSessionId)
        } else {
          // Session was deleted — clear stale reference
          localStorage.removeItem('medai_current_session')
          set({ currentSessionId: null })
        }
      }
    } catch (err) {
      set({ sessionsLoaded: true })
      console.warn('Failed to load sessions:', err.message)
    }
  },

  selectSession: async (sessionId) => {
    localStorage.setItem('medai_current_session', sessionId)
    set({ currentSessionId: sessionId, isLoading: true, error: null })
    try {
      const messages = await api.getMessages(sessionId)
      set({ messages: Array.isArray(messages) ? messages : [], isLoading: false })
    } catch (err) {
      set({ isLoading: false, error: err.message })
    }
  },

  uploadProgress: null,

  sendMessage: async (query, file = null, patientId = null, mode = 'doctor', features = {}) => {
    const { currentSessionId, messages } = get()
    const prevMessages = [...messages]

    // Optimistic user message
    const userMsg = { id: `tmp-${crypto.randomUUID()}`, role: 'user', content: query, created_at: new Date().toISOString() }
    set({ messages: [...messages, userMsg], isLoading: true, error: null })

    const onUploadProgress = file ? (pct) => set({ uploadProgress: pct }) : null

    try {
      const response = await api.ask(query, file, currentSessionId, patientId, mode, features, onUploadProgress)

      const newSessionId = response.session_id || currentSessionId
      if (newSessionId) {
        localStorage.setItem('medai_current_session', newSessionId)
      }

      const aiMsg = {
        id: response.message_id || `ai-${crypto.randomUUID()}`,
        role: 'assistant',
        content: response.answer || '',
        ai_metadata: {
          confidence: response.confidence,
          routing: response.routing,
          safety: response.safety,
          sources: response.sources,
        },
        created_at: new Date().toISOString(),
      }

      set((state) => ({
        messages: [...state.messages, aiMsg],
        currentSessionId: newSessionId,
        isLoading: false,
        uploadProgress: null,
        error: null,
      }))

      // Refresh session list (fire-and-forget)
      get().loadSessions()
      return response
    } catch (err) {
      // Rollback optimistic message on failure
      set({ messages: prevMessages, isLoading: false, uploadProgress: null, error: err.message })
      throw err
    }
  },

  newChat: () => {
    localStorage.removeItem('medai_current_session')
    set({ currentSessionId: null, messages: [], error: null })
  },

  clearError: () => set({ error: null }),

  deleteSession: async (sessionId) => {
    try {
      await api.deleteSession(sessionId)
      const { currentSessionId } = get()
      if (currentSessionId === sessionId) {
        localStorage.removeItem('medai_current_session')
        set({ currentSessionId: null, messages: [] })
      }
      get().loadSessions()
    } catch (err) {
      set({ error: err.message })
    }
  },
}))
