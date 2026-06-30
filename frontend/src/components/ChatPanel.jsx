import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, StopCircle, MessageSquare, Loader2, AlertTriangle, Bot, User, FlaskConical } from 'lucide-react'
import clsx from 'clsx'
import { streamQuery, evalAnswer } from '../api/client.js'
import SourceCitations from './SourceCitations.jsx'

const ROUTE_STYLES = {
  vector: 'bg-blue-500/10 text-blue-400',
  cag:    'bg-purple-500/10 text-purple-400',
  graph:  'bg-emerald-500/10 text-emerald-400',
}

function ScoreChip({ label, value }) {
  if (value == null) return (
    <span className="badge bg-slate-700/40 text-slate-500 text-[10px] font-mono">{label}: N/A</span>
  )
  const color = value >= 0.8
    ? 'text-accent-400 bg-accent-500/10'
    : value >= 0.6
      ? 'text-warn-400 bg-warn-500/10'
      : 'text-red-400 bg-red-500/10'
  return (
    <span className={clsx('badge text-[10px] font-mono', color)}>
      {label}: {value.toFixed(2)}
    </span>
  )
}

function RouteBadge({ route }) {
  if (!route) return null
  const labels = { vector: 'Vector', cag: 'CAG', graph: 'GraphRAG' }
  return (
    <span className={clsx('badge text-[10px]', ROUTE_STYLES[route] ?? 'bg-slate-500/10 text-slate-400')}>
      {labels[route] ?? route}
    </span>
  )
}

function EmptyState() {
  const examples = [
    'What AWS certifications does the candidate have?',
    'Compare the candidate\'s responsibilities at each company.',
    'What machine learning frameworks has the candidate worked with?',
  ]
  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 pb-16 px-8 select-none">
      <div className="w-14 h-14 rounded-2xl bg-brand-500/10 flex items-center justify-center">
        <MessageSquare className="w-7 h-7 text-brand-400" />
      </div>
      <div className="text-center">
        <h2 className="text-base font-semibold text-slate-200 mb-1">Ask anything</h2>
        <p className="text-sm text-slate-500 max-w-xs">
          Questions are answered with cited source passages from your indexed documents.
        </p>
      </div>
      <div className="space-y-2 w-full max-w-sm">
        {examples.map((ex, i) => (
          <div
            key={i}
            className="px-3 py-2 rounded-lg bg-surface-800 border border-white/5 text-xs text-slate-400 cursor-default"
          >
            {ex}
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ChatPanel({ collection, onTraceEvent, onQueryStart, onQueryEnd, isStreaming }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [evalStates, setEvalStates] = useState({})
  // Tracks {question, rewritten_query} for each completed turn — sent to backend for rewriting follow-ups
  const conversationHistoryRef = useRef([])
  const abortRef = useRef(null)
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  // Reset history when the collection changes (different doc set)
  useEffect(() => {
    conversationHistoryRef.current = []
  }, [collection])

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const submit = useCallback(async () => {
    const q = input.trim()
    if (!q || isStreaming) return

    setInput('')
    onQueryStart?.()

    // Append user message
    const userMsg = { id: Date.now(), role: 'user', text: q }
    setMessages(prev => [...prev, userMsg])

    // Append assistant message as streaming placeholder
    const assistantId = Date.now() + 1
    setMessages(prev => [...prev, {
      id: assistantId,
      role: 'assistant',
      question: q,
      text: '',
      streaming: true,
      route: null,
      citations: [],
      contexts: [],
      error: null,
    }])

    const controller = streamQuery(q, collection, conversationHistoryRef.current, (event) => {
      onTraceEvent?.(event)

      if (event.event === 'node_complete' && event.node === 'generate') {
        setMessages(prev => prev.map(m => m.id === assistantId
          ? { ...m, text: event.answer ?? '', citations: event.citations ?? [] }
          : m
        ))
      }

      if (event.event === 'done') {
        setMessages(prev => prev.map(m => m.id === assistantId
          ? {
              ...m,
              streaming: false,
              text: event.answer || m.text,
              route: event.route,
              citations: event.citations ?? m.citations,
              contexts: event.contexts ?? m.contexts,
            }
          : m
        ))
        // Append this turn to history so follow-ups can reference it
        conversationHistoryRef.current = [
          ...conversationHistoryRef.current,
          { question: q, rewritten_query: event.rewritten_query || q },
        ]
        onQueryEnd?.()
      }

      if (event.event === 'error') {
        setMessages(prev => prev.map(m => m.id === assistantId
          ? { ...m, streaming: false, error: event.detail }
          : m
        ))
        onQueryEnd?.()
      }
    })

    abortRef.current = controller
  }, [input, isStreaming, collection, onQueryStart, onQueryEnd, onTraceEvent])

  const stop = useCallback(() => {
    abortRef.current?.abort()
    setMessages(prev => prev.map(m => m.streaming ? { ...m, streaming: false } : m))
    onQueryEnd?.()
  }, [onQueryEnd])

  const handleEval = useCallback(async (msg) => {
    setEvalStates(prev => ({ ...prev, [msg.id]: { status: 'loading' } }))
    try {
      const result = await evalAnswer(msg.question, msg.text, msg.contexts ?? [])
      setEvalStates(prev => ({ ...prev, [msg.id]: { status: 'done', ...result } }))
    } catch (err) {
      setEvalStates(prev => ({ ...prev, [msg.id]: { status: 'error', detail: err.message } }))
    }
  }, [])

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          messages.map(msg => (
            <div key={msg.id} className={clsx('flex gap-3 message-enter', msg.role === 'user' && 'justify-end')}>
              {msg.role === 'assistant' && (
                <div className="w-7 h-7 rounded-lg bg-brand-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Bot className="w-4 h-4 text-brand-400" />
                </div>
              )}

              <div className={clsx(
                'max-w-[80%] min-w-0',
                msg.role === 'user' ? 'flex flex-col items-end' : '',
              )}>
                {msg.role === 'user' ? (
                  <div className="flex items-center gap-2">
                    <div className="px-3 py-2 rounded-2xl rounded-tr-sm bg-brand-500/15 border border-brand-500/20 text-sm text-slate-200">
                      {msg.text}
                    </div>
                    <div className="w-7 h-7 rounded-lg bg-surface-800 flex items-center justify-center flex-shrink-0">
                      <User className="w-4 h-4 text-slate-400" />
                    </div>
                  </div>
                ) : (
                  <div className="px-3.5 py-2.5 rounded-2xl rounded-tl-sm bg-surface-800 border border-white/5">
                    {msg.error ? (
                      <div className="flex items-center gap-2 text-red-400 text-sm">
                        <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                        <span>{msg.error}</span>
                      </div>
                    ) : (
                      <>
                        <div className="flex items-center gap-2 mb-2">
                          {msg.route && <RouteBadge route={msg.route} />}
                          {msg.streaming && (
                            <span className="flex items-center gap-1 text-[11px] text-brand-400">
                              <Loader2 className="w-3 h-3 animate-spin" /> thinking…
                            </span>
                          )}
                        </div>
                        <p className={clsx(
                          'text-sm text-slate-200 leading-relaxed whitespace-pre-wrap',
                          msg.streaming && !msg.text && 'cursor-blink',
                        )}>
                          {msg.text || (msg.streaming ? '' : '—')}
                        </p>
                        {!msg.streaming && <SourceCitations citations={msg.citations} />}
                        {!msg.streaming && msg.text && (() => {
                          const es = evalStates[msg.id]
                          if (!es || es.status === 'idle') return (
                            <button
                              onClick={() => handleEval(msg)}
                              className="mt-2 flex items-center gap-1.5 text-[11px] text-slate-500 hover:text-slate-300 transition-colors"
                            >
                              <FlaskConical className="w-3 h-3" /> Eval
                            </button>
                          )
                          if (es.status === 'loading') return (
                            <div className="mt-2 flex items-center gap-1.5 text-[11px] text-slate-500">
                              <Loader2 className="w-3 h-3 animate-spin" /> evaluating…
                            </div>
                          )
                          if (es.status === 'error') return (
                            <div className="mt-2 text-[11px] text-red-500">eval failed</div>
                          )
                          return (
                            <div className="mt-2 flex items-center gap-2 flex-wrap">
                              <ScoreChip label="Faith" value={es.faithfulness} />
                              <ScoreChip label="Relevancy" value={es.answer_relevancy} />
                              <span className="text-[10px] font-mono text-slate-600">{es.latency_ms}ms</span>
                            </div>
                          )
                        })()}
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex-shrink-0 px-4 pb-4">
        <div className="flex items-end gap-2 p-2 rounded-xl bg-surface-800 border border-white/10 focus-within:border-brand-500/50 transition-colors">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={`Ask about your ${collection} documents…`}
            rows={1}
            disabled={isStreaming}
            className="flex-1 bg-transparent resize-none text-sm text-slate-200 placeholder-slate-500 focus:outline-none leading-relaxed py-1.5 px-1 max-h-32 overflow-y-auto disabled:opacity-50"
            style={{ fieldSizing: 'content' }}
          />
          {isStreaming ? (
            <button onClick={stop} className="flex-shrink-0 p-2 rounded-lg bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors">
              <StopCircle className="w-4 h-4" />
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={!input.trim()}
              className="flex-shrink-0 p-2 rounded-lg bg-brand-500 hover:bg-brand-600 text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            >
              <Send className="w-4 h-4" />
            </button>
          )}
        </div>
        <p className="text-[10px] text-slate-600 text-center mt-1.5">
          Enter to send · Shift+Enter for newline
        </p>
      </div>
    </div>
  )
}
