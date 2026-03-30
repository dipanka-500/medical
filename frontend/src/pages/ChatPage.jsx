import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  Activity,
  AlertCircle,
  Menu,
  Volume2,
  VolumeX,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react'
import { useChatStore, useAuthStore } from '../stores/useStore'
import { useWebSocket } from '../hooks/useWebSocket'
import { useVoiceAssistant } from '../hooks/useVoiceAssistant'
import Sidebar from '../components/Sidebar'
import ChatMessage from '../components/ChatMessage'
import ChatInput from '../components/ChatInput'

export default function ChatPage() {
  const {
    messages, isLoading, error, currentSessionId, uploadProgress,
    sendMessage, loadSessions, clearError,
  } = useChatStore()
  const user = useAuthStore((state) => state.user)
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated)
  const messagesEndRef = useRef(null)
  const pendingVoiceReplyRef = useRef(false)
  const draftBaseRef = useRef('')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [searchParams, setSearchParams] = useSearchParams()
  const [streamingText, setStreamingText] = useState('')
  const [draftText, setDraftText] = useState('')
  const [voiceError, setVoiceError] = useState('')
  const [activeSpeechMessageId, setActiveSpeechMessageId] = useState(null)
  const [voiceOutputEnabled, setVoiceOutputEnabled] = useState(() => (
    localStorage.getItem('medai_voice_output') === '1'
  ))

  const selectedPatientId = user?.role === 'doctor'
    ? searchParams.get('patient')
    : null

  const {
    browserSpeechRecognitionSupported,
    isListening,
    isSpeaking,
    isTranscribing,
    serverVoice,
    speakText,
    startListening,
    stopListening,
    stopSpeaking,
  } = useVoiceAssistant({
    language: typeof navigator !== 'undefined' ? navigator.language : 'en-US',
    onListenStart: useCallback(() => {
      draftBaseRef.current = draftText.trimEnd()
      setVoiceError('')
    }, [draftText]),
    onTranscript: useCallback((transcript) => {
      const cleanTranscript = transcript.trim()
      const base = draftBaseRef.current.trim()
      const nextDraft = [base, cleanTranscript].filter(Boolean).join(base ? '\n' : '')
      setDraftText(nextDraft)
    }, []),
    onError: useCallback((message) => {
      setVoiceError(message)
    }, []),
  })

  const handleStopSpeaking = useCallback(() => {
    stopSpeaking()
    setActiveSpeechMessageId(null)
  }, [stopSpeaking])

  const handleSpeakMessage = useCallback(async (message) => {
    if (!message?.content?.trim()) return

    setVoiceError('')
    setActiveSpeechMessageId(message.id)
    const played = await speakText(message.content, {
      language: typeof navigator !== 'undefined' ? navigator.language : 'en-US',
    })

    if (!played) {
      setActiveSpeechMessageId(null)
    }
  }, [speakText])

  useEffect(() => {
    if (!isSpeaking) {
      setActiveSpeechMessageId(null)
    }
  }, [isSpeaking])

  useEffect(() => {
    localStorage.setItem('medai_voice_output', voiceOutputEnabled ? '1' : '0')
    if (!voiceOutputEnabled) {
      pendingVoiceReplyRef.current = false
      handleStopSpeaking()
    }
  }, [handleStopSpeaking, voiceOutputEnabled])

  const { send: wsSend, connected: wsConnected } = useWebSocket('/ws/chat', {
    enabled: isAuthenticated,
    onMessage: useCallback((data) => {
      if (data.type === 'chunk') {
        setStreamingText((prev) => prev + data.text)
      } else if (data.type === 'complete') {
        setStreamingText('')
        if (data.session_id) {
          const store = useChatStore.getState()
          store.selectSession(data.session_id)
          store.loadSessions()
        } else {
          useChatStore.setState({ isLoading: false })
        }
      } else if (data.type === 'error') {
        setStreamingText('')
        pendingVoiceReplyRef.current = false
        useChatStore.setState({ error: data.detail, isLoading: false })
      }
    }, []),
  })

  const stableLoadSessions = useCallback(() => {
    loadSessions()
  }, [loadSessions])

  useEffect(() => {
    stableLoadSessions()
  }, [stableLoadSessions])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingText])

  useEffect(() => {
    if (!voiceOutputEnabled || !pendingVoiceReplyRef.current) return

    const lastMessage = messages[messages.length - 1]
    if (!lastMessage || lastMessage.role !== 'assistant') return

    pendingVoiceReplyRef.current = false
    handleSpeakMessage(lastMessage)
  }, [handleSpeakMessage, messages, voiceOutputEnabled])

  const handleSend = async (query, file, features = {}) => {
    handleStopSpeaking()
    setVoiceError('')
    pendingVoiceReplyRef.current = voiceOutputEnabled

    if (wsConnected && !file) {
      const userMessage = {
        id: `tmp-${crypto.randomUUID()}`,
        role: 'user',
        content: query,
        created_at: new Date().toISOString(),
      }

      useChatStore.setState((state) => ({
        messages: [...state.messages, userMessage],
        isLoading: true,
        error: null,
      }))

      setStreamingText('')
      wsSend({
        type: 'query',
        query,
        session_id: currentSessionId,
        patient_id: selectedPatientId,
        mode: user?.role === 'doctor' ? 'doctor' : 'patient',
        web_search: !!features.web_search,
        deep_reasoning: !!features.deep_reasoning,
      })
      return
    }

    try {
      await sendMessage(
        query,
        file,
        selectedPatientId,
        user?.role === 'doctor' ? 'doctor' : 'patient',
        features,
      )
    } catch {
      pendingVoiceReplyRef.current = false
      // Error already stored in chat store.
    }
  }

  return (
    <div className="flex h-screen">
      <div className={`${sidebarOpen ? 'w-72' : 'w-0'} transition-all duration-300 overflow-hidden`}>
        <Sidebar onClose={() => setSidebarOpen(false)} />
      </div>

      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-14 border-b border-gray-800 flex items-center px-4 gap-3 shrink-0">
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-1.5 hover:bg-gray-800 rounded-lg" aria-label="Toggle sidebar">
            {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
          <Activity className="w-5 h-5 text-medical-500" />
          <span className="font-semibold">MedAI</span>

          {selectedPatientId && (
            <button
              type="button"
              onClick={() => setSearchParams({}, { replace: true })}
              className="rounded-full border border-primary-500/30 bg-primary-500/10 px-2 py-1 text-xs text-primary-300 hover:bg-primary-500/20"
            >
              Patient context active
            </button>
          )}

          <button
            type="button"
            onClick={() => setVoiceOutputEnabled((prev) => !prev)}
            className={`ml-auto flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors ${
              voiceOutputEnabled
                ? 'border-primary-500/30 bg-primary-500/10 text-primary-300'
                : 'border-gray-700 bg-gray-900/70 text-gray-500'
            }`}
            title={voiceOutputEnabled ? 'Voice replies are on' : 'Voice replies are off'}
          >
            {voiceOutputEnabled ? <Volume2 className="w-3.5 h-3.5" /> : <VolumeX className="w-3.5 h-3.5" />}
            Voice replies
          </button>

          <span className="flex items-center gap-1.5 text-xs text-gray-500" title={wsConnected ? 'Real-time connected' : 'Using standard mode'}>
            {wsConnected ? <Wifi className="w-3 h-3 text-green-500" /> : <WifiOff className="w-3 h-3 text-gray-600" />}
            {currentSessionId && <span>Session: {currentSessionId.slice(0, 8)}...</span>}
          </span>
        </header>

        {error && (
          <div className="mx-4 mt-2 p-3 bg-red-900/30 border border-red-800 rounded-lg flex items-center gap-2 text-red-300 text-sm">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span className="flex-1">{error}</span>
            <button onClick={clearError} className="text-red-400 hover:text-red-300">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        <div className="flex-1 overflow-y-auto px-4 py-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <Activity className="w-16 h-16 text-medical-500/30 mb-6" />
              <h2 className="text-2xl font-semibold mb-2">MedAI Platform</h2>
              <p className="text-gray-400 max-w-lg">
                Upload medical images, documents, ask medical questions, or speak naturally into the mic.
                AI-powered analysis runs with streaming, confidence scores, and safety checks.
              </p>
              <div className="grid grid-cols-2 gap-3 mt-8 max-w-lg">
                {[
                  'Analyze my chest X-ray',
                  'What are symptoms of diabetes?',
                  'Extract text from my lab report',
                  'Latest treatment for hypertension',
                ].map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => handleSend(prompt)}
                    className="p-3 text-sm text-left bg-gray-800/50 hover:bg-gray-800 border border-gray-700 rounded-xl transition-colors"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  onSpeak={handleSpeakMessage}
                  onStopSpeak={handleStopSpeaking}
                  isSpeaking={activeSpeechMessageId === message.id && isSpeaking}
                />
              ))}
              {isLoading && (
                <div className="flex gap-3">
                  <div className="w-8 h-8 rounded-full bg-medical-600 flex items-center justify-center shrink-0">
                    <Activity className="w-4 h-4" />
                  </div>
                  <div className="bg-gray-800 rounded-2xl rounded-tl-sm px-4 py-3">
                    {streamingText ? (
                      <p className="text-gray-200 whitespace-pre-wrap">{streamingText}<span className="inline-block w-1.5 h-4 bg-primary-400 animate-pulse ml-0.5 align-text-bottom" /></p>
                    ) : (
                      <div className="flex gap-1">
                        <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                    )}
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <ChatInput
          value={draftText}
          onTextChange={setDraftText}
          onSend={handleSend}
          disabled={isLoading}
          uploadProgress={uploadProgress}
          isListening={isListening}
          isSpeaking={isSpeaking}
          isTranscribing={isTranscribing}
          onVoiceStart={startListening}
          onVoiceStop={stopListening}
          onStopSpeaking={handleStopSpeaking}
          voiceError={voiceError}
          speechRecognitionAvailable={browserSpeechRecognitionSupported}
          serverVoice={serverVoice}
        />
      </div>
    </div>
  )
}
