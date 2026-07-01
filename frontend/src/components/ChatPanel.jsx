import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, StopCircle, MessageSquare, Loader2, AlertTriangle, Bot, User, FlaskConical, Coins } from 'lucide-react'
import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'
import { streamQuery, evalAnswer } from '../api/client.js'
import SourceCitations from './SourceCitations.jsx'

const MD_COMPONENTS = {
  h1: ({ children }) => (
    <h1 className="text-base font-bold text-slate-100 mt-4 mb-2 pb-1 border-b border-white/10">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-sm font-bold text-brand-400 mt-4 mb-1.5 uppercase tracking-wide">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-sm font-semibold text-slate-200 mt-3 mb-1">{children}</h3>
  ),
  p: ({ children }) => (
    <p className="text-sm text-slate-300 leading-relaxed mb-2 last:mb-0">{children}</p>
  ),
  ul: ({ children }) => (
    <ul className="space-y-1 mb-2 pl-1">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="space-y-1 mb-2 pl-1 list-decimal list-inside">{children}</ol>
  ),
  li: ({ children }) => (
    <li className="flex gap-2 text-sm text-slate-300 leading-relaxed">
      <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-brand-500/70 flex-shrink-0" />
      <span>{children}</span>
    </li>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-slate-100">{children}</strong>
  ),
  hr: () => (
    <hr className="border-white/10 my-3" />
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-brand-500/50 pl-3 my-2 text-slate-400 italic text-sm">{children}</blockquote>
  ),
  code: ({ children }) => (
    <code className="px-1 py-0.5 rounded bg-surface-900 text-accent-400 text-xs font-mono">{children}</code>
  ),
  table: ({ children }) => (
    <div className="overflow-x-auto my-2">
      <table className="w-full text-xs border-collapse">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="border-b border-white/10">{children}</thead>,
  tbody: ({ children }) => <tbody>{children}</tbody>,
  tr: ({ children }) => <tr className="border-b border-white/5 hover:bg-white/[0.02]">{children}</tr>,
  th: ({ children }) => (
    <th className="px-2 py-1.5 text-left text-[11px] font-semibold text-slate-300 whitespace-nowrap">{children}</th>
  ),
  td: ({ children }) => (
    <td className="px-2 py-1.5 text-[11px] text-slate-400 font-mono whitespace-nowrap">{children}</td>
  ),
}

const ROUTE_STYLES = {
  vector: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  cag:    'bg-blue-500/10 text-blue-400 border-blue-500/20',
  graph:  'bg-blue-500/10 text-blue-400 border-blue-500/20',
}

function ScoreChip({ label, value }) {
  if (value == null) return (
    <span className="badge bg-slate-800 text-slate-500 text-[10px] font-mono">{label}: N/A</span>
  )
  const color = value >= 0.8
    ? 'text-green-400 bg-green-500/10 border-green-500/20'
    : value >= 0.6
      ? 'text-amber-400 bg-amber-500/10 border-amber-500/20'
      : 'text-red-400 bg-red-500/10 border-red-500/20'
  return (
    <span className={clsx('badge text-[10px] font-mono border', color)}>
      {label}: {value.toFixed(2)}
    </span>
  )
}

function RouteBadge({ route }) {
  if (!route) return null
  const labels = { vector: 'Vector', cag: 'CAG', graph: 'GraphRAG' }
  return (
    <span className={clsx('badge text-[10px] border', ROUTE_STYLES[route] ?? 'bg-amber-500/10 text-amber-400 border-amber-500/20')}>
      {labels[route] ?? route}
    </span>
  )
}

function CollectionsBadge({ collections }) {
  if (!collections?.length) return null
  const label = collections.length >= 2
    ? collections.join(' + ')
    : collections[0]
  return (
    <span className="badge text-[10px] border bg-amber-500/10 text-amber-300 border-amber-500/25 font-mono">
      {collections.length >= 2 ? '⟂ ' : ''}{label}
    </span>
  )
}

const SAMPLE_QUESTIONS = [
  {
    label: 'SEC only — Apple revenue',
    q: "What was Apple's total net revenue in the most recent fiscal year?",
    hint: 'sec-filings · 15 company 10-Ks',
  },
  {
    label: 'Legal only — MSFT contracts',
    q: 'What termination or indemnification clauses are in the Microsoft EX-10 supply agreements?',
    hint: 'legal-docs · MSFT + JPM exhibits',
  },
  {
    label: 'Both collections — Microsoft',
    q: 'For Microsoft, what was total revenue last fiscal year and what termination clauses appear in their supply agreements?',
    hint: 'sec-filings + legal-docs in parallel',
  },
]

function EmptyState({ onPickQuestion }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-5 pb-12 px-6 overflow-y-auto">
      <div className="w-14 h-14 rounded-2xl bg-brand-500/10 flex items-center justify-center">
        <MessageSquare className="w-7 h-7 text-brand-400" />
      </div>
      <div className="text-center max-w-md">
        <h2 className="text-base font-semibold text-slate-200 mb-1">Ask anything</h2>
        <p className="text-sm text-slate-500">
          Auto-routing picks the right collection. Check <span className="text-amber-400">Live Trace → Router</span> to see which namespaces were searched.
        </p>
      </div>

      <div className="w-full max-w-lg space-y-3">
        <p className="text-[11px] text-slate-600 uppercase tracking-wide font-medium">What&apos;s indexed</p>
        <div className="grid gap-2 text-left">
          <div className="px-3 py-2 rounded-lg bg-surface-950 border border-blue-500/20">
            <p className="text-xs font-medium text-blue-300">sec-filings</p>
            <p className="text-[11px] text-slate-500 mt-0.5">10-K annual reports — AAPL, MSFT, NVDA, GOOGL, AMZN, JPM, TSLA, META, JNJ, V, WMT, XOM, PFE, KO, DIS</p>
          </div>
          <div className="px-3 py-2 rounded-lg bg-surface-950 border border-blue-500/20">
            <p className="text-xs font-medium text-blue-300">legal-docs</p>
            <p className="text-[11px] text-slate-500 mt-0.5">Material contracts (EX-10 exhibits) — MSFT officer indemnification agreements (EX-10.7, EX-10.8), JPM credit/license exhibits.</p>
          </div>
        </div>

        <p className="text-[11px] text-slate-600 uppercase tracking-wide font-medium pt-1">Try these</p>
        <div className="space-y-2">
          {SAMPLE_QUESTIONS.map(({ label, q, hint }) => (
            <button
              key={label}
              onClick={() => onPickQuestion?.(q)}
              className="w-full text-left px-3 py-2.5 rounded-lg bg-surface-950 border border-white/5 hover:border-amber-500/30 hover:bg-amber-500/5 transition-colors group"
            >
              <p className="text-[10px] text-amber-500/80 font-medium mb-0.5">{label}</p>
              <p className="text-xs text-slate-300 group-hover:text-slate-200 leading-snug">{q}</p>
              <p className="text-[10px] text-slate-600 mt-1">{hint}</p>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function ChatPanel({ collection, role = 'general', onTraceEvent, onQueryStart, onQueryEnd, isStreaming }) {
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
      activeCollections: [],
      citations: [],
      contexts: [],
      metrics: null,
      error: null,
    }])

    const controller = streamQuery(q, collection, conversationHistoryRef.current, (event) => {
      onTraceEvent?.(event)

      if (event.event === 'node_complete' && event.node === 'router') {
        setMessages(prev => prev.map(m => m.id === assistantId
          ? { ...m, route: event.route, activeCollections: event.active_collections ?? [] }
          : m
        ))
      }

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
              activeCollections: event.active_collections ?? m.activeCollections,
              citations: event.citations ?? m.citations,
              contexts: event.contexts ?? m.contexts,
              metrics: event.metrics ?? null,
              requiresCalculation: event.requires_calculation ?? false,
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
    }, role)

    abortRef.current = controller
  }, [input, isStreaming, collection, role, onQueryStart, onQueryEnd, onTraceEvent])

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
          <EmptyState onPickQuestion={q => setInput(q)} />
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
                        <div className="flex items-center gap-2 mb-2 flex-wrap">
                          {msg.route && <RouteBadge route={msg.route} />}
                          {msg.activeCollections?.length > 0 && (
                            <CollectionsBadge collections={msg.activeCollections} />
                          )}
                          {msg.streaming && (
                            <span className="flex items-center gap-1 text-[11px] text-brand-400">
                              <Loader2 className="w-3 h-3 animate-spin" /> thinking…
                            </span>
                          )}
                        </div>
                        <div className={clsx(
                          'min-w-0',
                          msg.streaming && !msg.text && 'cursor-blink',
                        )}>
                          {msg.text
                            ? <ReactMarkdown components={MD_COMPONENTS}>{msg.text}</ReactMarkdown>
                            : (msg.streaming ? '' : <span className="text-sm text-slate-500">—</span>)}
                        </div>
                        {!msg.streaming && <SourceCitations citations={msg.citations} />}
                        {!msg.streaming && msg.text && (
                          <div className="mt-3 pt-3 border-t border-white/5 space-y-2">
                            <div className="flex items-center gap-2 flex-wrap">
                              {msg.metrics?.total_latency_ms > 0 && (
                                <span className="flex items-center gap-1 px-2 py-0.5 rounded-md bg-surface-950 border border-white/5 text-[10px] text-slate-500 font-mono">
                                  {(msg.metrics.total_latency_ms / 1000).toFixed(1)}s
                                </span>
                              )}
                              {msg.metrics?.estimated_cost_usd > 0 && (
                                <span className="flex items-center gap-1 px-2 py-0.5 rounded-md bg-surface-950 border border-white/5 text-[10px] text-slate-500 font-mono">
                                  <Coins className="w-2.5 h-2.5" />
                                  ${msg.metrics.estimated_cost_usd.toFixed(4)}
                                </span>
                              )}
                              {msg.metrics?.input_tokens > 0 && (
                                <span className="px-2 py-0.5 rounded-md bg-surface-950 border border-white/5 text-[10px] text-slate-600 font-mono">
                                  {(msg.metrics.input_tokens + (msg.metrics.output_tokens ?? 0)).toLocaleString()} tok
                                </span>
                              )}
                            </div>

                            {(() => {
                              const es = evalStates[msg.id]
                              if (!es || es.status === 'idle') return (
                                <button
                                  onClick={() => handleEval(msg)}
                                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-amber-500/10 border border-amber-500/30 hover:bg-amber-500/20 hover:border-amber-500/50 text-sm text-amber-300 hover:text-amber-200 transition-all font-semibold"
                                >
                                  <FlaskConical className="w-4 h-4" />
                                  Evaluate Answer Quality
                                </button>
                              )
                              if (es.status === 'loading') return (
                                <div className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-surface-950 border border-amber-500/20 text-sm text-amber-400/70">
                                  <Loader2 className="w-4 h-4 animate-spin" />
                                  Running RAGAS evaluation…
                                </div>
                              )
                              if (es.status === 'error') return (
                                <div className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-red-500/10 border border-red-500/30 text-sm text-red-400">
                                  <AlertTriangle className="w-4 h-4" />
                                  Evaluation failed
                                </div>
                              )
                              return (
                                <div className="space-y-1.5">
                                  <div className="flex items-center gap-2 flex-wrap">
                                    <ScoreChip label="Faithfulness" value={es.faithfulness} />
                                    <ScoreChip label="Relevancy" value={es.answer_relevancy} />
                                    <span className="text-[10px] font-mono text-slate-600">{es.latency_ms}ms</span>
                                  </div>
                                  {msg.requiresCalculation && (
                                    <p className="text-[10px] text-slate-600 leading-snug">
                                      Faithfulness may be lower for calculation queries — computed values (e.g. % change) are verified by the system but do not appear verbatim in source documents.
                                    </p>
                                  )}
                                </div>
                              )
                            })()}
                          </div>
                        )}
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
            placeholder="Ask anything — auto-routing picks the right collection…"
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
