import { useState, useEffect, useCallback } from 'react'
import { Brain, Activity, ChevronRight, Wifi, WifiOff, RefreshCw } from 'lucide-react'
import clsx from 'clsx'
import { healthCheck } from './api/client.js'
import DocumentLibrary from './components/DocumentLibrary.jsx'
import ChatPanel from './components/ChatPanel.jsx'
import LiveTrace from './components/LiveTrace.jsx'

const COLLECTIONS = ['demo', 'finance', 'legal', 'general']

export default function App() {
  const [collection, setCollection] = useState('demo')
  const [showTrace, setShowTrace] = useState(true)
  const [traceEvents, setTraceEvents] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [backendOk, setBackendOk] = useState(null)

  // Backend health check
  useEffect(() => {
    let alive = true
    const check = async () => {
      const ok = await healthCheck()
      if (alive) setBackendOk(ok)
    }
    check()
    const id = setInterval(check, 10_000)
    return () => { alive = false; clearInterval(id) }
  }, [])

  const handleTraceEvent = useCallback((event) => {
    if (event.event === 'node_complete') {
      setTraceEvents(prev => [...prev, event])
    }
  }, [])

  const handleQueryStart = useCallback(() => {
    setTraceEvents([])
    setIsStreaming(true)
  }, [])

  const handleQueryEnd = useCallback(() => {
    setIsStreaming(false)
  }, [])

  return (
    <div className="flex flex-col h-screen bg-surface-900 overflow-hidden">
      {/* ── Top bar ─────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-4 h-12 border-b border-white/5 bg-surface-950/80 backdrop-blur-sm flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-brand-500/20 flex items-center justify-center">
              <Brain className="w-4 h-4 text-brand-400" />
            </div>
            <span className="font-semibold text-sm text-slate-100">Agentic RAG</span>
          </div>

          <ChevronRight className="w-3.5 h-3.5 text-slate-600" />

          {/* Collection selector */}
          <select
            value={collection}
            onChange={e => setCollection(e.target.value)}
            className="bg-surface-800 border border-white/10 rounded-md px-2.5 py-1 text-xs text-slate-300 focus:outline-none focus:ring-1 focus:ring-brand-500 cursor-pointer"
          >
            {COLLECTIONS.map(c => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-2">
          {/* Backend status */}
          {backendOk === null ? (
            <span className="badge bg-slate-700/60 text-slate-400">
              <RefreshCw className="w-3 h-3 animate-spin mr-1" /> checking
            </span>
          ) : backendOk ? (
            <span className="badge bg-accent-500/10 text-accent-400">
              <Wifi className="w-3 h-3 mr-1" /> backend ok
            </span>
          ) : (
            <span className="badge bg-red-500/10 text-red-400">
              <WifiOff className="w-3 h-3 mr-1" /> backend offline
            </span>
          )}

          {/* Trace toggle */}
          <button
            onClick={() => setShowTrace(v => !v)}
            className={clsx(
              'btn-ghost text-xs gap-1.5',
              showTrace && 'text-brand-400 bg-brand-500/10'
            )}
          >
            <Activity className="w-3.5 h-3.5" />
            {showTrace ? 'Hide Trace' : 'Show Trace'}
          </button>
        </div>
      </header>

      {/* ── Main layout ─────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Document library */}
        <aside className="w-64 flex-shrink-0 flex flex-col border-r border-white/5 bg-surface-900">
          <DocumentLibrary
            collection={collection}
            onCollectionChange={setCollection}
            collections={COLLECTIONS}
          />
        </aside>

        {/* Center: Chat */}
        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          <ChatPanel
            collection={collection}
            onTraceEvent={handleTraceEvent}
            onQueryStart={handleQueryStart}
            onQueryEnd={handleQueryEnd}
            isStreaming={isStreaming}
          />
        </main>

        {/* Right: Live Trace (collapsible) */}
        {showTrace && (
          <aside className="w-80 flex-shrink-0 flex flex-col border-l border-white/5 bg-surface-900 animate-fade-in">
            <LiveTrace events={traceEvents} isStreaming={isStreaming} />
          </aside>
        )}
      </div>
    </div>
  )
}
