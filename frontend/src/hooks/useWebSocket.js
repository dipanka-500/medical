import { useCallback, useEffect, useRef, useState } from 'react'

const WS_BASE = (import.meta.env.VITE_WS_URL || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`)

const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000, 30000] // exponential backoff capped at 30s

/**
 * React hook for WebSocket connection with auto-reconnect and heartbeat handling.
 *
 * @param {string} path - WebSocket path (e.g. "/ws/chat")
 * @param {object} options
 * @param {function} options.onMessage - called with parsed JSON messages
 * @param {boolean} options.enabled - whether to connect (default true)
 * @returns {{ send, readyState, connected }}
 */
export function useWebSocket(path, { onMessage, enabled = true } = {}) {
  const wsRef = useRef(null)
  const reconnectAttemptRef = useRef(0)
  const reconnectTimerRef = useRef(null)
  const onMessageRef = useRef(onMessage)
  const [readyState, setReadyState] = useState(WebSocket.CLOSED)

  // Keep callback ref current without re-triggering effect
  useEffect(() => { onMessageRef.current = onMessage }, [onMessage])

  const connect = useCallback(() => {
    const token = localStorage.getItem('medai_token')
    if (!token || !enabled) return

    // Clean up any existing connection
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    // Connect WITHOUT the token in the URL — token is sent as first message
    // to avoid leaking JWTs via browser history, proxy logs, and tracing.
    const url = `${WS_BASE}${path}`
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      // Send auth as the first message (never in the URL)
      ws.send(JSON.stringify({ type: 'auth', token }))
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)

        // Server confirms auth
        if (data.type === 'auth_ok') {
          reconnectAttemptRef.current = 0
          setReadyState(WebSocket.OPEN)
          return
        }

        // Auto-respond to server pings
        if (data.type === 'ping') {
          ws.send(JSON.stringify({ type: 'pong' }))
          return
        }

        onMessageRef.current?.(data)
      } catch {
        // Ignore malformed messages
      }
    }

    ws.onclose = (event) => {
      setReadyState(WebSocket.CLOSED)
      wsRef.current = null

      // Don't reconnect if intentionally closed or auth rejected
      if (event.code === 1000 || event.code === 1008) return
      if (!enabled) return

      // Schedule reconnection
      const attempt = reconnectAttemptRef.current
      const delay = RECONNECT_DELAYS[Math.min(attempt, RECONNECT_DELAYS.length - 1)]
      reconnectAttemptRef.current = attempt + 1

      reconnectTimerRef.current = setTimeout(() => {
        connect()
      }, delay)
    }

    ws.onerror = () => {
      // onclose will fire after onerror, triggering reconnect
    }
  }, [path, enabled])

  // Connect on mount / reconnect when deps change
  useEffect(() => {
    if (enabled) connect()

    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted')
        wsRef.current = null
      }
    }
  }, [connect, enabled])

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(typeof data === 'string' ? data : JSON.stringify(data))
      return true
    }
    return false
  }, [])

  return {
    send,
    readyState,
    connected: readyState === WebSocket.OPEN,
  }
}
