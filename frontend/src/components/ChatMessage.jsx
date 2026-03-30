import ReactMarkdown from 'react-markdown'
import {
  Activity,
  AlertTriangle,
  ExternalLink,
  ShieldCheck,
  User,
  Volume2,
  VolumeX,
} from 'lucide-react'

// Only allow safe markdown elements (XSS prevention)
const ALLOWED_ELEMENTS = [
  'p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4',
  'blockquote', 'code', 'pre', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
  'hr', 'a', 'del',
]

// Clamp confidence to [0, 1] — AI engines may return unexpected values
const safeConfidence = (val) => {
  if (val == null || typeof val !== 'number' || !isFinite(val)) return null
  return Math.max(0, Math.min(1, val))
}

export default function ChatMessage({ message, onSpeak, onStopSpeak, isSpeaking = false }) {
  const isUser = message.role === 'user'
  const meta = message.ai_metadata || null
  const confidence = meta ? safeConfidence(meta.confidence) : null

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : ''}`}>
      {/* AI Avatar */}
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-medical-600 flex items-center justify-center shrink-0">
          <Activity className="w-4 h-4" />
        </div>
      )}

      <div className={`max-w-[80%] ${isUser ? 'order-first' : ''}`}>
        {!isUser && (
          <div className="mb-2 flex items-center justify-end">
            <button
              type="button"
              onClick={isSpeaking ? () => onStopSpeak?.(message) : () => onSpeak?.(message)}
              className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[11px] transition-colors ${
                isSpeaking
                  ? 'border-amber-500/30 bg-amber-500/10 text-amber-300'
                  : 'border-gray-700 bg-gray-900/60 text-gray-400 hover:border-primary-500/30 hover:text-primary-300'
              }`}
              aria-label={isSpeaking ? 'Stop speaking this answer' : 'Speak this answer'}
            >
              {isSpeaking ? <VolumeX className="w-3 h-3" /> : <Volume2 className="w-3 h-3" />}
              {isSpeaking ? 'Stop voice' : 'Speak'}
            </button>
          </div>
        )}

        {/* Message bubble */}
        <div className={`rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-primary-600 text-white rounded-tr-sm'
            : 'bg-gray-800 text-gray-100 rounded-tl-sm'
        }`}>
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown
                allowedElements={ALLOWED_ELEMENTS}
                unwrapDisallowed
                components={{
                  // Sanitize links — only allow http/https
                  a: ({ href, children, ...props }) => {
                    const safe = href && (href.startsWith('http://') || href.startsWith('https://'))
                    return safe
                      ? <a href={href} target="_blank" rel="noopener noreferrer" {...props}>{children}</a>
                      : <span>{children}</span>
                  },
                }}
              >
                {message.content || ''}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* AI metadata bar */}
        {!isUser && meta && (
          <div className="flex flex-wrap gap-2 mt-2 text-xs">
            {/* Confidence */}
            {confidence != null && confidence > 0 && (
              <span className={`flex items-center gap-1 px-2 py-0.5 rounded-full ${
                confidence > 0.8 ? 'bg-green-900/40 text-green-400' :
                confidence > 0.5 ? 'bg-yellow-900/40 text-yellow-400' :
                'bg-red-900/40 text-red-400'
              }`}>
                <ShieldCheck className="w-3 h-3" />
                {(confidence * 100).toFixed(0)}% confidence
              </span>
            )}

            {/* Routing info */}
            {meta.routing?.primary_engine && (
              <span className="px-2 py-0.5 rounded-full bg-gray-800 text-gray-400">
                {meta.routing.primary_engine}
              </span>
            )}

            {/* Safety */}
            {meta.safety && !meta.safety.is_safe && (
              <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-900/40 text-red-400">
                <AlertTriangle className="w-3 h-3" /> Safety flag
              </span>
            )}

            {/* Sources */}
            {meta.sources?.length > 0 && (
              <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-blue-900/40 text-blue-400">
                <ExternalLink className="w-3 h-3" /> {meta.sources.length} sources
              </span>
            )}
          </div>
        )}
      </div>

      {/* User avatar */}
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-primary-700 flex items-center justify-center shrink-0">
          <User className="w-4 h-4" />
        </div>
      )}
    </div>
  )
}
