import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../services/api'

const getSpeechRecognition = () => {
  if (typeof window === 'undefined') return null
  return window.SpeechRecognition || window.webkitSpeechRecognition || null
}

const pickRecordingMimeType = () => {
  if (typeof MediaRecorder === 'undefined') return ''

  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
    'audio/ogg;codecs=opus',
  ]

  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || ''
}

const fileNameForMimeType = (mimeType) => {
  if (mimeType.includes('mp4')) return 'voice-input.m4a'
  if (mimeType.includes('ogg')) return 'voice-input.ogg'
  return 'voice-input.webm'
}

const getBrowserSpeechSupport = () => (
  typeof window !== 'undefined'
  && 'speechSynthesis' in window
  && typeof window.SpeechSynthesisUtterance !== 'undefined'
)

const normalizeServerLanguage = (value) => {
  if (typeof value !== 'string') return null
  const trimmed = value.trim()
  if (!trimmed) return null
  return trimmed.replace('_', '-').split('-')[0].toLowerCase()
}

export function useVoiceAssistant({
  language = 'en-US',
  onTranscript,
  onListenStart,
  onListenEnd,
  onError,
} = {}) {
  const recognitionRef = useRef(null)
  const recorderRef = useRef(null)
  const recorderStreamRef = useRef(null)
  const recordedChunksRef = useRef([])
  const activeAudioRef = useRef(null)
  const activeAudioUrlRef = useRef(null)
  const activeUtteranceRef = useRef(null)
  const lastTranscriptRef = useRef('')
  const capabilitiesRef = useRef({
    asr_available: false,
    tts_available: false,
  })

  const [isListening, setIsListening] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [serverVoice, setServerVoice] = useState({
    asr_available: false,
    tts_available: false,
  })

  const browserSpeechRecognitionSupported = !!getSpeechRecognition()
  const browserSpeechSynthesisSupported = getBrowserSpeechSupport()

  useEffect(() => {
    let cancelled = false

    api.getVoiceCapabilities()
      .then((capabilities) => {
        if (cancelled || !capabilities) return
        capabilitiesRef.current = capabilities
        setServerVoice({
          asr_available: !!capabilities.asr_available,
          tts_available: !!capabilities.tts_available,
        })
      })
      .catch(() => {
        // Voice is optional; browser fallbacks still work.
      })

    return () => {
      cancelled = true
    }
  }, [])

  const stopRecorderTracks = useCallback(() => {
    recorderStreamRef.current?.getTracks?.().forEach((track) => track.stop())
    recorderStreamRef.current = null
  }, [])

  const stopSpeaking = useCallback(() => {
    if (activeAudioRef.current) {
      activeAudioRef.current.pause()
      activeAudioRef.current.src = ''
      activeAudioRef.current = null
    }
    if (activeAudioUrlRef.current) {
      URL.revokeObjectURL(activeAudioUrlRef.current)
      activeAudioUrlRef.current = null
    }
    if (browserSpeechSynthesisSupported) {
      window.speechSynthesis.cancel()
      activeUtteranceRef.current = null
    }
    setIsSpeaking(false)
  }, [browserSpeechSynthesisSupported])

  const finishListening = useCallback(() => {
    setIsListening(false)
    onListenEnd?.()
  }, [onListenEnd])

  const transcribeRecording = useCallback(async (blob) => {
    if (!blob || blob.size === 0) return

    setIsTranscribing(true)
    try {
      const mimeType = blob.type || pickRecordingMimeType() || 'audio/webm'
      const result = await api.transcribeAudio(
        blob,
        normalizeServerLanguage(language),
        fileNameForMimeType(mimeType),
      )
      const text = result?.text?.trim()
      if (text) {
        lastTranscriptRef.current = text
        onTranscript?.(text, { final: true, source: 'server' })
      }
    } catch (error) {
      onError?.(error.message || 'Voice transcription failed.')
    } finally {
      setIsTranscribing(false)
    }
  }, [language, onError, onTranscript])

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop()
      } catch {
        // Browser implementations can throw if already stopping.
      }
      return
    }

    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop()
      return
    }

    finishListening()
    stopRecorderTracks()
  }, [finishListening, stopRecorderTracks])

  const startBrowserRecognition = useCallback(() => {
    const SpeechRecognition = getSpeechRecognition()
    if (!SpeechRecognition) {
      throw new Error('Browser speech recognition is not supported here.')
    }

    const recognition = new SpeechRecognition()
    recognition.lang = language
    recognition.interimResults = true
    recognition.continuous = false
    recognition.maxAlternatives = 1

    lastTranscriptRef.current = ''
    recognitionRef.current = recognition
    setIsListening(true)
    onListenStart?.()

    recognition.onresult = (event) => {
      let finalText = ''
      let interimText = ''

      for (let index = 0; index < event.results.length; index += 1) {
        const result = event.results[index]
        const transcript = result?.[0]?.transcript?.trim() || ''
        if (!transcript) continue
        if (result.isFinal) {
          finalText = `${finalText} ${transcript}`.trim()
        } else {
          interimText = `${interimText} ${transcript}`.trim()
        }
      }

      const combined = [finalText, interimText].filter(Boolean).join(' ').trim()
      if (!combined) return

      lastTranscriptRef.current = finalText || combined
      onTranscript?.(combined, {
        final: false,
        source: 'browser',
      })
    }

    recognition.onerror = (event) => {
      recognitionRef.current = null
      finishListening()
      if (event.error === 'aborted') return
      onError?.(`Voice input failed: ${event.error}`)
    }

    recognition.onend = () => {
      recognitionRef.current = null
      finishListening()
      if (lastTranscriptRef.current) {
        onTranscript?.(lastTranscriptRef.current, {
          final: true,
          source: 'browser',
        })
      }
    }

    recognition.start()
  }, [finishListening, language, onError, onListenStart, onTranscript])

  const startRecordedCapture = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      throw new Error('Audio recording is not supported in this browser.')
    }
    if (!capabilitiesRef.current.asr_available) {
      throw new Error('Server speech recognition is not enabled yet.')
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const mimeType = pickRecordingMimeType()
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)

    recorderStreamRef.current = stream
    recorderRef.current = recorder
    recordedChunksRef.current = []

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordedChunksRef.current.push(event.data)
      }
    }

    recorder.onerror = () => {
      recorderRef.current = null
      stopRecorderTracks()
      finishListening()
      onError?.('Voice recording failed.')
    }

    recorder.onstop = async () => {
      const recordedBlob = new Blob(
        recordedChunksRef.current,
        { type: mimeType || 'audio/webm' },
      )

      recorderRef.current = null
      recordedChunksRef.current = []
      stopRecorderTracks()
      finishListening()
      await transcribeRecording(recordedBlob)
    }

    setIsListening(true)
    onListenStart?.()
    recorder.start()
  }, [finishListening, onError, onListenStart, stopRecorderTracks, transcribeRecording])

  const startListening = useCallback(async () => {
    stopSpeaking()

    try {
      if (serverVoice.asr_available) {
        await startRecordedCapture()
        return
      }

      if (browserSpeechRecognitionSupported) {
        startBrowserRecognition()
        return
      }

      throw new Error('Voice input is not available in this browser.')
    } catch (error) {
      onError?.(error.message || 'Voice input failed to start.')
    }
  }, [
    browserSpeechRecognitionSupported,
    onError,
    serverVoice.asr_available,
    startBrowserRecognition,
    startRecordedCapture,
    stopSpeaking,
  ])

  const speakWithBrowser = useCallback((text, speakLanguage) => {
    if (!browserSpeechSynthesisSupported) {
      throw new Error('Speech output is not supported in this browser.')
    }

    stopSpeaking()

    const utterance = new window.SpeechSynthesisUtterance(text)
    utterance.lang = speakLanguage || language || navigator.language || 'en-US'
    utterance.rate = 1
    utterance.pitch = 1
    utterance.onend = () => {
      activeUtteranceRef.current = null
      setIsSpeaking(false)
    }
    utterance.onerror = () => {
      activeUtteranceRef.current = null
      setIsSpeaking(false)
    }

    activeUtteranceRef.current = utterance
    setIsSpeaking(true)
    window.speechSynthesis.speak(utterance)
  }, [browserSpeechSynthesisSupported, language, stopSpeaking])

  const speakText = useCallback(async (text, { language: speakLanguage } = {}) => {
    const cleanText = text?.trim()
    if (!cleanText) return false

    stopSpeaking()

    if (serverVoice.tts_available) {
      try {
        const audioBlob = await api.synthesizeSpeech(
          cleanText,
          normalizeServerLanguage(speakLanguage || language),
        )
        const audioUrl = URL.createObjectURL(audioBlob)
        const audio = new Audio(audioUrl)

        activeAudioRef.current = audio
        activeAudioUrlRef.current = audioUrl
        setIsSpeaking(true)

        audio.onended = () => {
          stopSpeaking()
        }
        audio.onerror = () => {
          stopSpeaking()
          if (browserSpeechSynthesisSupported) {
            speakWithBrowser(cleanText, speakLanguage)
          }
        }

        await audio.play()
        return true
      } catch (error) {
        if (!browserSpeechSynthesisSupported) {
          onError?.(error.message || 'Speech output failed.')
          return false
        }
      }
    }

    if (browserSpeechSynthesisSupported) {
      speakWithBrowser(cleanText, speakLanguage)
      return true
    }

    onError?.('Speech output is not available in this browser.')
    return false
  }, [
    browserSpeechSynthesisSupported,
    language,
    onError,
    serverVoice.tts_available,
    speakWithBrowser,
    stopSpeaking,
  ])

  useEffect(() => () => {
    try {
      recognitionRef.current?.stop?.()
    } catch {
      // Ignore stop errors during teardown.
    }
    try {
      recorderRef.current?.stop?.()
    } catch {
      // Ignore stop errors during teardown.
    }
    stopRecorderTracks()
    stopSpeaking()
  }, [stopRecorderTracks, stopSpeaking])

  return {
    isListening,
    isSpeaking,
    isTranscribing,
    serverVoice,
    browserSpeechRecognitionSupported,
    browserSpeechSynthesisSupported,
    startListening,
    stopListening,
    speakText,
    stopSpeaking,
  }
}
