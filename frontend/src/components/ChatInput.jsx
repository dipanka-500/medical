import { useRef, useState } from 'react'
import {
  Brain,
  FileImage,
  FileText,
  Globe,
  LoaderCircle,
  Mic,
  Paperclip,
  Plus,
  Send,
  Square,
  Upload,
  VolumeX,
  X,
} from 'lucide-react'

const MAX_FILE_SIZE_MB = 50

const FEATURES = [
  { key: 'web_search', label: 'Web Search', icon: Globe, description: 'Search medical literature & PubMed' },
  { key: 'deep_reasoning', label: 'Deep Reasoning', icon: Brain, description: 'Multi-model consensus analysis' },
]

export default function ChatInput({
  value,
  onTextChange,
  onSend,
  disabled,
  uploadProgress = null,
  isListening = false,
  isTranscribing = false,
  isSpeaking = false,
  onVoiceStart,
  onVoiceStop,
  onStopSpeaking,
  voiceError = '',
  speechRecognitionAvailable = false,
  serverVoice = {},
}) {
  const [file, setFile] = useState(null)
  const [fileError, setFileError] = useState('')
  const [activeFeatures, setActiveFeatures] = useState({})
  const [showFeatureMenu, setShowFeatureMenu] = useState(false)
  const fileRef = useRef(null)
  const textRef = useRef(null)
  const menuRef = useRef(null)

  const voiceInputAvailable = speechRecognitionAvailable || !!serverVoice?.asr_available

  const resetComposer = () => {
    onTextChange?.('')
    setFile(null)
    setFileError('')
    if (fileRef.current) fileRef.current.value = ''
    textRef.current?.focus()
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    const trimmed = value.trim()
    if (!trimmed && !file) return

    try {
      await Promise.resolve(onSend(trimmed, file, activeFeatures))
      resetComposer()
    } catch {
      // Chat errors are surfaced by the store banner.
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleFile = (e) => {
    const selectedFile = e.target.files?.[0]
    if (!selectedFile) return
    if (selectedFile.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setFileError(`File too large. Max ${MAX_FILE_SIZE_MB}MB.`)
      return
    }
    setFileError('')
    setFile(selectedFile)
  }

  const toggleFeature = (key) => {
    setActiveFeatures((prev) => {
      const next = { ...prev }
      if (next[key]) {
        delete next[key]
      } else {
        next[key] = true
      }
      return next
    })
  }

  const activeCount = Object.keys(activeFeatures).length
  const FileIcon = file?.type?.startsWith('image/') ? FileImage : FileText

  return (
    <div className="border-t border-gray-800 p-4">
      <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
        {voiceError && (
          <p className="mb-2 rounded-xl border border-amber-700/50 bg-amber-950/40 px-3 py-2 text-sm text-amber-300">
            {voiceError}
          </p>
        )}

        {fileError && (
          <p className="text-red-400 text-sm mb-2">{fileError}</p>
        )}

        {uploadProgress !== null && (
          <div className="mb-2">
            <div className="flex items-center gap-2 text-xs text-gray-400 mb-1">
              <Upload className="w-3 h-3" />
              <span>Uploading... {Math.round(uploadProgress)}%</span>
            </div>
            <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-500 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        {(isListening || isTranscribing) && (
          <div className="mb-2 flex items-center gap-2 rounded-xl border border-primary-500/20 bg-primary-500/10 px-3 py-2 text-xs text-primary-200">
            {isListening ? <Mic className="w-3.5 h-3.5" /> : <LoaderCircle className="w-3.5 h-3.5 animate-spin" />}
            <span>
              {isListening
                ? 'Listening... speak naturally and tap stop when you are done.'
                : 'Transcribing your audio with the voice pipeline...'}
            </span>
          </div>
        )}

        {activeCount > 0 && (
          <div className="flex flex-wrap gap-2 mb-2">
            {FEATURES.filter((feature) => activeFeatures[feature.key]).map((feature) => (
              <button
                key={feature.key}
                type="button"
                onClick={() => toggleFeature(feature.key)}
                className="flex items-center gap-1.5 px-3 py-1 text-xs font-medium rounded-full bg-primary-600/20 text-primary-400 border border-primary-600/30 hover:bg-primary-600/30 transition-colors"
              >
                <feature.icon className="w-3 h-3" />
                {feature.label}
                <X className="w-3 h-3 ml-0.5 opacity-60" />
              </button>
            ))}
          </div>
        )}

        {file && (
          <div className="flex items-center gap-2 mb-2 px-3 py-2 bg-gray-800 rounded-lg text-sm">
            <FileIcon className="w-4 h-4 text-gray-400" />
            <span className="truncate flex-1 text-gray-300">{file.name}</span>
            <span className="text-gray-500">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
            <button
              type="button"
              onClick={() => {
                setFile(null)
                setFileError('')
                if (fileRef.current) fileRef.current.value = ''
              }}
              className="p-0.5 hover:text-red-400"
              aria-label="Remove file"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        <div className="flex items-end gap-2 bg-gray-800 rounded-2xl border border-gray-700 focus-within:border-primary-500 px-4 py-3">
          <div className="relative" ref={menuRef}>
            <button
              type="button"
              onClick={() => setShowFeatureMenu(!showFeatureMenu)}
              className={`p-1 rounded-md transition-colors ${
                showFeatureMenu || activeCount > 0
                  ? 'text-primary-400 hover:text-primary-300'
                  : 'text-gray-400 hover:text-gray-200'
              }`}
              aria-label="Features menu"
            >
              <Plus className={`w-5 h-5 transition-transform ${showFeatureMenu ? 'rotate-45' : ''}`} />
            </button>

            {showFeatureMenu && (
              <>
                <div className="fixed inset-0 z-10" onClick={() => setShowFeatureMenu(false)} />
                <div className="absolute bottom-full left-0 mb-2 w-64 bg-gray-900 border border-gray-700 rounded-xl shadow-xl z-20 overflow-hidden">
                  <div className="p-2">
                    <button
                      type="button"
                      onClick={() => {
                        fileRef.current?.click()
                        setShowFeatureMenu(false)
                      }}
                      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-300 hover:bg-gray-800 transition-colors"
                    >
                      <Paperclip className="w-4 h-4 text-gray-400" />
                      <div className="text-left">
                        <div className="font-medium">Add photos & files</div>
                        <div className="text-xs text-gray-500">Images, PDFs, DICOM</div>
                      </div>
                    </button>

                    <div className="border-t border-gray-800 my-1" />

                    {FEATURES.map((feature) => {
                      const isActive = !!activeFeatures[feature.key]
                      return (
                        <button
                          key={feature.key}
                          type="button"
                          onClick={() => toggleFeature(feature.key)}
                          className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                            isActive
                              ? 'bg-primary-600/15 text-primary-400'
                              : 'text-gray-300 hover:bg-gray-800'
                          }`}
                        >
                          <feature.icon className={`w-4 h-4 ${isActive ? 'text-primary-400' : 'text-gray-400'}`} />
                          <div className="text-left flex-1">
                            <div className="font-medium">{feature.label}</div>
                            <div className={`text-xs ${isActive ? 'text-primary-500/70' : 'text-gray-500'}`}>
                              {feature.description}
                            </div>
                          </div>
                          {isActive && (
                            <div className="w-2 h-2 rounded-full bg-primary-400" />
                          )}
                        </button>
                      )
                    })}
                  </div>
                </div>
              </>
            )}
          </div>

          <input
            ref={fileRef}
            type="file"
            className="hidden"
            accept=".pdf,.jpg,.jpeg,.png,.tiff,.dcm,.mp4"
            onChange={handleFile}
          />

          <textarea
            ref={textRef}
            value={value}
            onChange={(e) => onTextChange?.(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a medical question, upload a file, or dictate with your mic..."
            rows={1}
            aria-label="Message input"
            className="flex-1 bg-transparent text-gray-100 placeholder-gray-500 resize-none outline-none max-h-32"
            style={{ height: 'auto', minHeight: '24px' }}
            onInput={(e) => {
              e.target.style.height = 'auto'
              e.target.style.height = `${e.target.scrollHeight}px`
            }}
          />

          {isSpeaking && (
            <button
              type="button"
              onClick={onStopSpeaking}
              className="p-1.5 rounded-lg text-amber-300 bg-amber-500/10 hover:bg-amber-500/20 transition-colors"
              aria-label="Stop voice playback"
              title="Stop voice playback"
            >
              <VolumeX className="w-4 h-4" />
            </button>
          )}

          {voiceInputAvailable && (
            <button
              type="button"
              onClick={isListening || isTranscribing ? onVoiceStop : onVoiceStart}
              disabled={disabled || isTranscribing}
              className={`p-1.5 rounded-lg transition-colors ${
                isListening
                  ? 'bg-red-500/15 text-red-300 hover:bg-red-500/20'
                  : 'bg-primary-500/10 text-primary-300 hover:bg-primary-500/20'
              } disabled:bg-gray-700 disabled:text-gray-500`}
              aria-label={isListening ? 'Stop voice input' : 'Start voice input'}
              title={isListening ? 'Stop voice input' : 'Start voice input'}
            >
              {isListening
                ? <Square className="w-4 h-4" />
                : isTranscribing
                  ? <LoaderCircle className="w-4 h-4 animate-spin" />
                  : <Mic className="w-4 h-4" />}
            </button>
          )}

          <button
            type="submit"
            disabled={disabled || (!value.trim() && !file)}
            className="p-1.5 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg transition-colors"
            aria-label="Send message"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>

        <p className="text-center text-xs text-gray-600 mt-2">
          MedAI provides AI-assisted analysis. Always consult a healthcare professional.
        </p>
      </form>
    </div>
  )
}
