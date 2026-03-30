/**
 * API service — handles all communication with the MedAI Platform backend.
 *
 * Security:
 *   - Token refresh queue prevents concurrent refresh races
 *   - Tokens stored in localStorage (persistent across sessions, like ChatGPT/Claude)
 *   - Refresh tokens rotated on every use (backend returns new pair)
 */

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1'

let _refreshPromise = null  // single in-flight refresh (prevents race)

class ApiService {
  constructor() {
    this.token = localStorage.getItem('medai_token')
  }

  _applyAuthPayload(data) {
    if (!data?.access_token) return data

    this.token = data.access_token
    localStorage.setItem('medai_token', data.access_token)

    if (data.refresh_token) {
      localStorage.setItem('medai_refresh', data.refresh_token)
    }

    if (data.user?.role) {
      localStorage.setItem('medai_role', data.user.role)
    }

    return data
  }

  _headers(extra = {}) {
    const h = { ...extra }
    if (this.token) h['Authorization'] = `Bearer ${this.token}`
    // CSRF: read token from cookie and send as header
    const csrfToken = this._getCsrfToken()
    if (csrfToken) h['X-CSRF-Token'] = csrfToken
    return h
  }

  _getCsrfToken() {
    try {
      const match = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/)
      return match?.[1] ? decodeURIComponent(match[1]) : null
    } catch {
      return null
    }
  }

  _canAttemptRefresh(path, skipAuthRefresh = false) {
    if (skipAuthRefresh) return false
    return !path.startsWith('/auth/login')
      && !path.startsWith('/auth/register')
      && !path.startsWith('/auth/2fa/verify')
      && !path.startsWith('/auth/refresh')
  }

  async _fetchRaw(path, options = {}) {
    const { skipAuthRefresh = false, ...fetchOptions } = options
    let res
    try {
      res = await fetch(`${API_BASE}${path}`, {
        ...fetchOptions,
        credentials: 'same-origin',
        headers: this._headers(fetchOptions.headers || {}),
      })
    } catch (err) {
      throw new Error('Network error — check your connection')
    }

    if (res.status === 401 && this._canAttemptRefresh(path, skipAuthRefresh)) {
      const refreshed = await this.refreshToken()
      if (refreshed) {
        return this._fetchRaw(path, { ...fetchOptions, skipAuthRefresh: true })
      }
    }

    if (res.status === 429) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.detail || 'Rate limit exceeded. Please wait.')
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      if (res.status === 401 && this.token && !path.startsWith('/auth/') && skipAuthRefresh) {
        // Only force-logout if refresh already failed (skipAuthRefresh=true means we already retried)
        this.logout()
      }
      throw new Error(err.detail || `Request failed (${res.status})`)
    }

    return res
  }

  async _fetch(path, options = {}) {
    const res = await this._fetchRaw(path, options)
    if (res.status === 204) return null
    return res.json()
  }

  // ── Auth ──────────────────────────────────────────────────────

  async register(email, password, fullName, role = 'patient') {
    return this._fetch('/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, full_name: fullName, role }),
    })
  }

  async login(email, password, totpCode = null) {
    // Step 1: Initial login (password auth)
    const data = await this._fetch('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    })

    // If 2FA required, handle it separately
    if (data.requires_2fa && totpCode) {
      return this.verify2fa(data.user_id, totpCode)
    }

    if (data.access_token) {
      this._applyAuthPayload(data)
    }
    return data
  }

  async verify2fa(userId, totpCode) {
    const data = this._applyAuthPayload(await this._fetch('/auth/2fa/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, totp_code: totpCode }),
    }))

    if (data.access_token && !data.user) {
      try {
        data.user = await this.getMe()
        if (data.user?.role) {
          localStorage.setItem('medai_role', data.user.role)
        }
      } catch {
        // The auth store will retry /auth/me on app restore if needed.
      }
    }
    return data
  }

  async refreshToken() {
    // Prevent concurrent refresh calls
    if (_refreshPromise) return _refreshPromise

    const refresh = localStorage.getItem('medai_refresh')
    if (!refresh) return false

    _refreshPromise = (async () => {
      try {
        const headers = { 'Content-Type': 'application/json' }
        const csrfToken = this._getCsrfToken()
        if (csrfToken) headers['X-CSRF-Token'] = csrfToken
        const res = await fetch(`${API_BASE}/auth/refresh`, {
          method: 'POST',
          headers,
          credentials: 'same-origin',
          body: JSON.stringify({ refresh_token: refresh }),
        })
        if (!res.ok) return false
        const data = this._applyAuthPayload(await res.json())

        if (data.access_token) {
          return true
        }
        return false
      } catch {
        return false
      } finally {
        _refreshPromise = null
      }
    })()

    return _refreshPromise
  }

  async logout({ redirect = true } = {}) {
    // Revoke refresh token server-side before clearing local state
    const refreshToken = localStorage.getItem('medai_refresh')
    try {
      if (this.token) {
        await fetch(`${API_BASE}/auth/logout`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.token}`,
          },
          credentials: 'same-origin',
          body: JSON.stringify({
            refresh_token: refreshToken,
            all_devices: false,
          }),
        })
      }
    } catch {
      // Best-effort — still clear local state even if server call fails
    }
    this.token = null
    localStorage.removeItem('medai_token')
    localStorage.removeItem('medai_refresh')
    localStorage.removeItem('medai_role')
    localStorage.removeItem('medai_current_session')
    if (redirect) {
      window.location.href = '/login'
    }
  }

  async getMe() {
    return this._fetch('/auth/me')
  }

  // ── Password Reset & Email Verification ─────────────────────

  async forgotPassword(email) {
    return this._fetch('/auth/forgot-password', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    })
  }

  async resetPassword(token, newPassword) {
    return this._fetch('/auth/reset-password', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token, new_password: newPassword }),
    })
  }

  async verifyEmail(token) {
    return this._fetch('/auth/verify-email', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    })
  }

  async resendVerification(email) {
    return this._fetch('/auth/resend-verification', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    })
  }

  // ── Chat / AI ─────────────────────────────────────────────────

  async ask(query, file = null, sessionId = null, patientId = null, mode = 'doctor', features = {}, onUploadProgress = null) {
    // File upload uses XMLHttpRequest for progress tracking
    if (file) {
      return this._uploadWithProgress(query, file, sessionId, patientId, mode, features, onUploadProgress)
    }

    // Text-only uses JSON body
    return this._fetch('/chat/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        patient_id: patientId,
        mode,
        web_search: !!features.web_search,
        deep_reasoning: !!features.deep_reasoning,
      }),
    })
  }

  _uploadWithProgress(query, file, sessionId, patientId, mode, features = {}, onProgress, skipAuthRefresh = false) {
    return new Promise((resolve, reject) => {
      const form = new FormData()
      form.append('query', query || 'Analyze this medical image')
      form.append('file', file)
      if (sessionId) form.append('session_id', sessionId)
      if (patientId) form.append('patient_id', patientId)
      form.append('mode', mode)
      form.append('web_search', features?.web_search ? 'true' : 'false')
      form.append('deep_reasoning', features?.deep_reasoning ? 'true' : 'false')

      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${API_BASE}/chat/ask-with-file`)
      xhr.withCredentials = true
      if (this.token) xhr.setRequestHeader('Authorization', `Bearer ${this.token}`)
      const csrfToken = this._getCsrfToken()
      if (csrfToken) xhr.setRequestHeader('X-CSRF-Token', csrfToken)

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress((e.loaded / e.total) * 100)
        }
      }

      xhr.onload = async () => {
        if (xhr.status === 401 && this._canAttemptRefresh('/chat/ask-with-file', skipAuthRefresh)) {
          const refreshed = await this.refreshToken()
          if (refreshed) {
            try {
              resolve(await this._uploadWithProgress(
                query,
                file,
                sessionId,
                patientId,
                mode,
                features,
                onProgress,
                true,
              ))
            } catch (error) {
              reject(error)
            }
            return
          }
        }

        if (onProgress) onProgress(null)
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)) }
          catch { reject(new Error('Invalid response')) }
        } else if (xhr.status === 429) {
          try {
            const err = JSON.parse(xhr.responseText)
            reject(new Error(err.detail || 'Rate limit exceeded. Please wait.'))
          } catch {
            reject(new Error('Rate limit exceeded. Please wait.'))
          }
        } else {
          if (xhr.status === 401 && this.token && skipAuthRefresh) {
            this.logout()
            return
          }
          try {
            const err = JSON.parse(xhr.responseText)
            reject(new Error(err.detail || `Upload failed (${xhr.status})`))
          } catch { reject(new Error(`Upload failed (${xhr.status})`)) }
        }
      }
      xhr.onerror = () => {
        if (onProgress) onProgress(null)
        reject(new Error('Network error — check your connection'))
      }
      xhr.send(form)
    })
  }

  askStream(query, sessionId = null, patientId = null, mode = 'doctor', features = {}) {
    // SSE streaming endpoint uses JSON body
    return this._fetchRaw('/chat/ask-stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        patient_id: patientId,
        mode,
        web_search: !!features.web_search,
        deep_reasoning: !!features.deep_reasoning,
      }),
    })
  }

  async getVoiceCapabilities() {
    return this._fetch('/chat/voice/capabilities')
  }

  async transcribeAudio(audioBlob, language = null, filename = 'voice-input.webm') {
    const form = new FormData()
    form.append('audio', audioBlob, filename)
    if (language) form.append('language', language)

    return this._fetch('/chat/transcribe', {
      method: 'POST',
      body: form,
    })
  }

  async synthesizeSpeech(text, language = null) {
    const res = await this._fetchRaw('/chat/speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language }),
    })

    return res.blob()
  }

  async getSessions() {
    const data = await this._fetch('/chat/sessions')
    return Array.isArray(data?.sessions) ? data.sessions : []
  }

  async getMessages(sessionId) {
    const data = await this._fetch(`/chat/sessions/${encodeURIComponent(sessionId)}/messages`)
    return Array.isArray(data?.messages) ? data.messages : []
  }

  async deleteSession(sessionId) {
    return this._fetch(`/chat/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
  }

  async renameSession(sessionId, title) {
    return this._fetch(`/chat/sessions/${encodeURIComponent(sessionId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    })
  }

  async searchSessions(query) {
    const data = await this._fetch(`/chat/sessions?search=${encodeURIComponent(query)}`)
    return Array.isArray(data?.sessions) ? data.sessions : []
  }

  // ── Patients ──────────────────────────────────────────────────

  async getPatientProfile() {
    return this._fetch('/patients/me/profile')
  }

  async getPatientRecords() {
    return this._fetch('/patients/me/records')
  }

  async getTimeline() {
    return this._fetch('/patients/me/timeline')
  }

  // ── Doctors ───────────────────────────────────────────────────

  async getDoctorPatients(search = '') {
    const q = search ? `?search=${encodeURIComponent(search)}` : ''
    return this._fetch(`/doctors/me/patients${q}`)
  }

  async linkPatient(email, notes = '') {
    return this._fetch('/doctors/me/patients/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient_email: email, notes }),
    })
  }

  async getPatientRecordsAsDoctor(patientId) {
    return this._fetch(`/doctors/me/patients/${encodeURIComponent(patientId)}/records`)
  }

  async getDoctorDashboard() {
    return this._fetch('/doctors/me/dashboard')
  }

  // ── 2FA Management ────────────────────────────────────────────

  async setup2FA() {
    return this._fetch('/auth/2fa/setup', { method: 'POST' })
  }

  async enable2FA(totpCode) {
    return this._fetch('/auth/2fa/enable', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ totp_code: totpCode }),
    })
  }

  async disable2FA(totpCode) {
    return this._fetch('/auth/2fa/disable', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ totp_code: totpCode }),
    })
  }

  // ── Subscription ──────────────────────────────────────────────

  async updateSubscription(tier) {
    return this._fetch('/auth/subscription', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tier }),
    })
  }

  // ── Password Change ───────────────────────────────────────────

  async changePassword(currentPassword, newPassword) {
    return this._fetch('/auth/change-password', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    })
  }

}

export const api = new ApiService()
