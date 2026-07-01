import { useState, useEffect, useCallback } from 'react'
import { Brain, Activity, Wifi, WifiOff, RefreshCw, Shuffle, ShieldCheck } from 'lucide-react'
import clsx from 'clsx'
import { healthCheck } from './api/client.js'
import DocumentLibrary from './components/DocumentLibrary.jsx'
import ChatPanel from './components/ChatPanel.jsx'
import LiveTrace from './components/LiveTrace.jsx'

const COLLECTIONS = ['sec-filings', 'legal-docs']

const ROLES = [
  { value: 'general', label: 'General', color: 'text-slate-400 bg-slate-700/40 border-slate-600/40' },
  { value: 'finance', label: 'Finance', color: 'text-blue-400 bg-blue-500/10 border-blue-500/20' },
  { value: 'legal',   label: 'Legal',   color: 'text-purple-400 bg-purple-500/10 border-purple-500/20' },
  { value: 'admin',   label: 'Admin',   color: 'text-amber-400 bg-amber-500/10 border-amber-500/20' },
]

export default function App() {
  // libraryCollection drives the document browser only; chat always uses 'auto'
  const [libraryCollection, setLibraryCollection] = useState('sec-filings')
  const [showTrace, setShowTrace] = useState(true)
  const [traceEvents, setTraceEvents] = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [backendOk, setBackendOk] = useState(null)
  const [role, setRole] = useState('general')

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
    <div className="flex flex-col h-screen bg-surface-950 overflow-hidden">
      {/* ── Top bar ─────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-4 h-12 border-b border-white/5 bg-surface-950/80 backdrop-blur-sm flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-brand-500/20 flex items-center justify-center">
              <Brain className="w-4 h-4 text-brand-400" />
            </div>
            <span className="font-semibold text-sm text-slate-100">Agentic RAG</span>
          </div>

          {/* Auto-routing badge — replaces the manual collection selector */}
          <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-md bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-400 font-medium">
            <Shuffle className="w-3 h-3" />
            scoped to: auto
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Role picker */}
          <div className="flex items-center gap-1 p-0.5 rounded-lg bg-surface-900 border border-white/5">
            <ShieldCheck className="w-3 h-3 text-slate-600 ml-1.5 flex-shrink-0" />
            {ROLES.map(r => (
              <button
                key={r.value}
                onClick={() => setRole(r.value)}
                className={clsx(
                  'px-2 py-0.5 rounded-md text-[10px] font-medium border transition-all',
                  role === r.value ? r.color : 'text-slate-600 bg-transparent border-transparent hover:text-slate-400',
                )}
              >
                {r.label}
              </button>
            ))}
          </div>

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
        {/* Left: Document library — has its own collection switcher */}
        <aside className="w-64 flex-shrink-0 flex flex-col border-r border-white/5 bg-surface-950">
          <DocumentLibrary
            collection={libraryCollection}
            onCollectionChange={setLibraryCollection}
            collections={COLLECTIONS}
          />
        </aside>

        {/* Center: Chat — always uses auto-routing */}
        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          <ChatPanel
            collection="auto"
            role={role}
            onTraceEvent={handleTraceEvent}
            onQueryStart={handleQueryStart}
            onQueryEnd={handleQueryEnd}
            isStreaming={isStreaming}
          />
        </main>

        {/* Right: Live Trace (collapsible) */}
        {showTrace && (
          <aside className="w-80 flex-shrink-0 flex flex-col border-l border-white/5 bg-surface-950 animate-fade-in">
            <LiveTrace events={traceEvents} isStreaming={isStreaming} />
          </aside>
        )}
      </div>
    </div>
  )
}
