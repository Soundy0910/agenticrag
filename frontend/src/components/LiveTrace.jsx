import { useState, useEffect, useRef } from 'react'
import {
  Activity,
  PenLine,
  Shuffle,
  Search,
  CheckSquare,
  Sparkles,
  ChevronDown,
  ChevronRight,
  Loader2,
  Check,
  RefreshCw,
  GitBranch,
  Layers,
  Network,
  Zap,
} from 'lucide-react'
import clsx from 'clsx'

// ── Node metadata ────────────────────────────────────────────────────────────

const NODE_META = {
  rewrite:         { label: 'Query Rewrite',   short: 'Rewrite',  Icon: PenLine,     color: 'text-violet-400',  bg: 'bg-violet-500/15',   border: 'border-violet-500/30', glow: 'rgba(139,92,246,0.35)' },
  router:          { label: 'Router',           short: 'Route',    Icon: GitBranch,   color: 'text-amber-400',   bg: 'bg-amber-500/15',    border: 'border-amber-500/30',  glow: 'rgba(245,158,11,0.35)' },
  retrieve_vector: { label: 'Vector Retrieval', short: 'Retrieve', Icon: Search,      color: 'text-blue-400',    bg: 'bg-blue-500/15',     border: 'border-blue-500/30',   glow: 'rgba(59,130,246,0.35)' },
  retrieve_cag:    { label: 'CAG Retrieval',    short: 'Retrieve', Icon: Layers,      color: 'text-purple-400',  bg: 'bg-purple-500/15',   border: 'border-purple-500/30', glow: 'rgba(168,85,247,0.35)' },
  retrieve_graph:  { label: 'Graph Retrieval',  short: 'Retrieve', Icon: Network,     color: 'text-teal-400',    bg: 'bg-teal-500/15',     border: 'border-teal-500/30',   glow: 'rgba(20,184,166,0.35)' },
  grade:           { label: 'Grade',            short: 'Grade',    Icon: CheckSquare, color: 'text-yellow-400',  bg: 'bg-yellow-500/15',   border: 'border-yellow-500/30', glow: 'rgba(234,179,8,0.35)'  },
  generate:        { label: 'Generate',         short: 'Generate', Icon: Sparkles,    color: 'text-green-400',   bg: 'bg-green-500/15',    border: 'border-green-500/30',  glow: 'rgba(34,197,94,0.35)'  },
}

// Canonical pipeline shown in the top diagram (retrieve is one logical step)
const PIPELINE_STEPS = ['rewrite', 'router', 'retrieve_vector', 'grade', 'generate']

// All retrieve node names map to the 'retrieve_vector' slot in the diagram
const toStepKey = (node) =>
  ['retrieve_vector', 'retrieve_cag', 'retrieve_graph'].includes(node) ? 'retrieve_vector' : node

const ROUTE_META = {
  vector: { label: 'Hybrid Vector + BM25',  color: 'text-blue-400',   bg: 'bg-blue-500/15',   border: 'border-blue-500/40' },
  cag:    { label: 'Context Stuffing (CAG)', color: 'text-purple-400', bg: 'bg-purple-500/15', border: 'border-purple-500/40' },
  graph:  { label: 'GraphRAG (Neo4j)',       color: 'text-teal-400',   bg: 'bg-teal-500/15',   border: 'border-teal-500/40' },
}

// ── Pipeline progress bar ────────────────────────────────────────────────────

function PipelineBar({ firedSteps, activeStep }) {
  return (
    <div className="flex items-center justify-between px-3 py-2.5 bg-surface-950/50 rounded-lg mx-3 mb-3">
      {PIPELINE_STEPS.map((step, i) => {
        const meta = NODE_META[step]
        const done   = firedSteps.has(step)
        const active = activeStep === step
        const Icon   = meta.Icon

        return (
          <div key={step} className="flex items-center">
            {/* Node circle */}
            <div className="flex flex-col items-center gap-1">
              <div
                className={clsx(
                  'w-7 h-7 rounded-full flex items-center justify-center transition-all duration-300 border',
                  done   && [meta.bg, meta.border, 'step-complete'],
                  active && [meta.bg, meta.border, 'glow-ring'],
                  !done && !active && 'bg-surface-800 border-white/5',
                )}
                style={active ? { '--glow': meta.glow } : undefined}
              >
                {done ? (
                  <Check className={clsx('w-3 h-3', meta.color)} />
                ) : active ? (
                  <Icon className={clsx('w-3 h-3 animate-pulse', meta.color)} />
                ) : (
                  <Icon className="w-3 h-3 text-slate-700" />
                )}
              </div>
              <span className={clsx(
                'text-[9px] font-medium',
                done || active ? meta.color : 'text-slate-700',
              )}>
                {meta.short}
              </span>
            </div>

            {/* Connector line */}
            {i < PIPELINE_STEPS.length - 1 && (
              <div className="relative w-6 h-0.5 mx-0.5 -mt-3 overflow-hidden bg-surface-800 rounded-full">
                {done && (
                  <div
                    className={clsx('absolute inset-y-0 left-0 rounded-full line-fill', meta.bg.replace('/15', '/60'))}
                  />
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── Animated vertical connector between event cards ──────────────────────────

function Connector({ color }) {
  return (
    <div className="flex justify-center my-0.5 relative h-5">
      {/* Background track */}
      <div className="w-px h-full bg-white/5 rounded-full" />
      {/* Filled segment */}
      <div
        className={clsx('absolute top-0 w-px rounded-full line-fill', color ?? 'bg-brand-500/50')}
        style={{ height: '100%' }}
      />
      {/* Travelling dot */}
      <div
        className={clsx('absolute w-1.5 h-1.5 rounded-full -translate-x-[2px] flow-dot', color?.replace('bg-', 'bg-').replace('/50', '') ?? 'bg-brand-400')}
        style={{ left: '50%' }}
      />
    </div>
  )
}

// ── Router branch visualisation ──────────────────────────────────────────────

function RouterBranches({ route }) {
  return (
    <div className="mt-2 space-y-1">
      {Object.entries(ROUTE_META).map(([key, meta]) => {
        const chosen = key === route
        return (
          <div
            key={key}
            className={clsx(
              'flex items-center gap-2 px-2 py-1.5 rounded-lg border transition-all',
              chosen
                ? [meta.bg, meta.border, 'badge-pop']
                : 'bg-surface-900/50 border-white/5 opacity-30',
            )}
          >
            {chosen && <Zap className={clsx('w-3 h-3 flex-shrink-0', meta.color)} />}
            <span className={clsx('text-[11px] font-medium', chosen ? meta.color : 'text-slate-600')}>
              {meta.label}
            </span>
            {chosen && (
              <span className={clsx('ml-auto badge text-[9px]', meta.bg, meta.color)}>selected</span>
            )}
          </div>
        )
      })}
    </div>
  )
}

// ── Node detail content ──────────────────────────────────────────────────────

function NodeDetail({ event }) {
  switch (event.node) {
    case 'rewrite':
      return event.rewritten_query ? (
        <div className="mt-2 p-2 rounded-lg bg-surface-950/70 border border-white/5">
          <p className="text-[10px] text-slate-600 mb-1">Standalone query</p>
          <p className="text-[11px] text-violet-300 font-mono leading-snug italic">
            "{event.rewritten_query}"
          </p>
        </div>
      ) : null

    case 'router':
      return <RouterBranches route={event.route} />

    case 'retrieve_vector':
    case 'retrieve_cag':
    case 'retrieve_graph': {
      const chunks = event.chunks ?? []
      if (!chunks.length) return null
      return (
        <div className="mt-2 space-y-1">
          <p className="text-[10px] text-slate-600">Top chunks</p>
          {chunks.slice(0, 4).map((c, i) => (
            <div key={c.chunk_id ?? i} className="rounded-lg bg-surface-950/70 border border-white/5 px-2 py-1.5">
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-[10px] font-mono font-medium text-slate-400 truncate max-w-[140px]" title={c.filename}>
                  {c.filename}
                </span>
                <span className="text-[9px] font-mono text-slate-600 flex-shrink-0 ml-1">
                  {c.is_parent ? 'parent' : `${c.score}`}
                </span>
              </div>
              <p className="text-[10px] font-mono text-slate-500 leading-snug line-clamp-2">{c.preview}</p>
            </div>
          ))}
          {chunks.length > 4 && (
            <p className="text-[10px] text-slate-600 text-center">+{chunks.length - 4} more</p>
          )}
        </div>
      )
    }

    case 'grade':
      return (
        <div className="mt-2 flex items-center gap-2 px-2 py-1.5 rounded-lg bg-surface-950/70 border border-white/5">
          <div className={clsx(
            'w-2 h-2 rounded-full flex-shrink-0',
            event.grade === 'sufficient' ? 'bg-green-400' : 'bg-yellow-400',
          )} />
          <span className={clsx(
            'text-[11px] font-medium',
            event.grade === 'sufficient' ? 'text-green-400' : 'text-yellow-400',
          )}>
            {event.grade}
          </span>
          {event.retry_count > 0 && (
            <span className="ml-auto flex items-center gap-1 text-[10px] text-yellow-600">
              <RefreshCw className="w-2.5 h-2.5" /> retry {event.retry_count}
            </span>
          )}
        </div>
      )

    case 'generate':
      return (event.citations?.length ?? 0) > 0 ? (
        <div className="mt-2 space-y-1">
          <p className="text-[10px] text-slate-600">{event.citation_count} citation{event.citation_count !== 1 && 's'}</p>
          {event.citations.slice(0, 3).map((c, i) => (
            <div key={c.chunk_id ?? i} className="rounded-lg bg-surface-950/70 border border-white/5 px-2 py-1">
              <span className="text-[10px] font-mono text-slate-400 truncate block">{c.filename}</span>
            </div>
          ))}
        </div>
      ) : null

    default:
      return null
  }
}

// ── Single event card ────────────────────────────────────────────────────────

function NodeCard({ event, index, isLast }) {
  const [open, setOpen] = useState(false)
  // Auto-open router (route decision is the most interesting) and retrieve
  const autoOpen = ['router', 'retrieve_vector', 'retrieve_cag', 'retrieve_graph'].includes(event.node)
  const [expanded, setExpanded] = useState(autoOpen)

  const meta = NODE_META[event.node] ?? {
    label: event.node, short: event.node, Icon: Activity,
    color: 'text-slate-400', bg: 'bg-slate-500/15', border: 'border-white/10', glow: 'rgba(148,163,184,0.3)',
  }
  const { Icon, label, color, bg, border } = meta
  const hasDetail = event.chunks?.length || event.citations?.length || event.rewritten_query || event.route

  const summary = getSummary(event)

  return (
    <div className="node-appear">
      <div
        className={clsx('rounded-xl border overflow-hidden', bg, border)}
      >
        {/* Card header */}
        <button
          onClick={() => hasDetail && setExpanded(v => !v)}
          className={clsx(
            'w-full flex items-center gap-2.5 px-3 py-2.5 text-left transition-colors',
            hasDetail ? 'hover:bg-white/[0.02] cursor-pointer' : 'cursor-default',
          )}
        >
          {/* Step badge */}
          <span className="w-4 h-4 rounded-full bg-surface-900/60 flex items-center justify-center text-[9px] font-mono text-slate-600 flex-shrink-0">
            {index + 1}
          </span>

          {/* Node icon */}
          <div className={clsx('w-5 h-5 rounded-md flex items-center justify-center flex-shrink-0 bg-surface-900/40')}>
            <Icon className={clsx('w-3 h-3', color)} />
          </div>

          {/* Label */}
          <div className="flex-1 min-w-0">
            <span className={clsx('text-xs font-semibold', color)}>{label}</span>
            {summary && (
              <p className="text-[11px] text-slate-400 mt-0.5 truncate leading-snug">{summary}</p>
            )}
          </div>

          {/* Status dot + chevron */}
          <div className="flex items-center gap-1.5 flex-shrink-0">
            <div className={clsx('w-1.5 h-1.5 rounded-full', color.replace('text-', 'bg-'))} />
            {hasDetail && (
              <span className="text-slate-600">
                {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              </span>
            )}
          </div>
        </button>

        {/* Expanded detail */}
        {expanded && hasDetail && (
          <div className="px-3 pb-3 animate-fade-in">
            <NodeDetail event={event} />
          </div>
        )}
      </div>
    </div>
  )
}

function getSummary(event) {
  switch (event.node) {
    case 'rewrite':
      return event.changed ? `→ "${truncate(event.rewritten_query, 45)}"` : 'Query unchanged'
    case 'router':
      return `${event.route?.toUpperCase()} — ${event.reason}`
    case 'retrieve_vector':
    case 'retrieve_cag':
    case 'retrieve_graph':
      return `${event.chunk_count} chunk${event.chunk_count !== 1 ? 's' : ''} retrieved`
    case 'grade':
      return event.grade === 'sufficient'
        ? 'Context sufficient ✓'
        : `Insufficient${event.retry_count > 0 ? ` · retry ${event.retry_count}` : ''}`
    case 'generate':
      return `Answer generated · ${event.citation_count ?? 0} citations`
    default:
      return null
  }
}

// ── Active (pending) node skeleton ──────────────────────────────────────────

function ActiveNodeSkeleton({ nodeName }) {
  const meta = NODE_META[nodeName] ?? NODE_META['rewrite']
  const { Icon, label, color, bg, border, glow } = meta

  return (
    <div
      className={clsx('rounded-xl border overflow-hidden node-appear', bg, border)}
    >
      <div className="flex items-center gap-2.5 px-3 py-2.5">
        <span className="w-4 h-4 rounded-full bg-surface-900/60 flex items-center justify-center text-[9px] text-slate-600 flex-shrink-0">
          •
        </span>

        {/* Pulsing icon with glow ring */}
        <div
          className={clsx('w-5 h-5 rounded-md flex items-center justify-center flex-shrink-0 bg-surface-900/40 glow-ring')}
          style={{ '--glow': glow }}
        >
          <Icon className={clsx('w-3 h-3 animate-pulse', color)} />
        </div>

        <div className="flex-1 min-w-0 space-y-1.5">
          <span className={clsx('text-xs font-semibold', color)}>{label}</span>
          {/* Shimmer skeleton line */}
          <div className="h-2 rounded-full shimmer bg-white/5 w-3/4" />
        </div>

        <Loader2 className={clsx('w-3.5 h-3.5 animate-spin flex-shrink-0', color)} />
      </div>
    </div>
  )
}

// ── Idle empty state ─────────────────────────────────────────────────────────

function IdleState() {
  return (
    <div className="flex flex-col items-center gap-4 px-4 py-6">
      <div className="w-10 h-10 rounded-xl bg-surface-800 flex items-center justify-center">
        <Activity className="w-5 h-5 text-slate-600" />
      </div>
      <div className="text-center">
        <p className="text-xs text-slate-500 font-medium mb-1">Pipeline ready</p>
        <p className="text-[11px] text-slate-700 leading-relaxed">
          Ask a question to watch LangGraph route, retrieve, grade, and generate in real time.
        </p>
      </div>

      {/* Static pipeline preview */}
      <div className="w-full space-y-1">
        {PIPELINE_STEPS.map((step, i) => {
          const meta = NODE_META[step]
          const Icon = meta.Icon
          return (
            <div key={step} className="flex items-center gap-2">
              {/* Track line */}
              <div className="flex flex-col items-center w-4 flex-shrink-0">
                <div className={clsx('w-4 h-4 rounded-full flex items-center justify-center', meta.bg)}>
                  <Icon className={clsx('w-2.5 h-2.5', meta.color)} />
                </div>
                {i < PIPELINE_STEPS.length - 1 && (
                  <div className="w-px h-3 bg-white/5 mt-0.5" />
                )}
              </div>
              <span className="text-[11px] text-slate-600">{meta.label}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Main component ───────────────────────────────────────────────────────────

function truncate(str, n) {
  if (!str) return ''
  return str.length > n ? str.slice(0, n) + '…' : str
}

export default function LiveTrace({ events, isStreaming }) {
  const bottomRef = useRef(null)

  // Auto-scroll to bottom as events arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  const firedNodes = new Set(events.map(e => e.node))
  const firedSteps = new Set([...firedNodes].map(toStepKey))

  // Determine which pipeline step is currently active
  const lastFiredStepIdx = [...PIPELINE_STEPS].map(toStepKey).reduce((acc, step, i) => {
    return firedSteps.has(step) ? i : acc
  }, -1)
  const activeStepKey = isStreaming ? PIPELINE_STEPS[lastFiredStepIdx + 1] : null

  // Connector color based on last fired node
  const lastEvent = events[events.length - 1]
  const connectorMeta = lastEvent ? NODE_META[lastEvent.node] : null
  const connectorBg = connectorMeta ? connectorMeta.bg.replace('/15', '/40') : 'bg-brand-500/40'

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 pt-3 pb-2 flex-shrink-0">
        <Activity className={clsx(
          'w-4 h-4 transition-colors',
          isStreaming ? 'text-brand-400 animate-pulse-slow' : 'text-slate-500',
        )} />
        <span className="text-sm font-semibold text-slate-200">Live Trace</span>
        {isStreaming && (
          <span className="ml-auto flex items-center gap-1.5 text-[11px] text-brand-400">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse" />
            running
          </span>
        )}
        {!isStreaming && events.length > 0 && (
          <span className="ml-auto flex items-center gap-1 text-[11px] text-accent-400">
            <Check className="w-3 h-3" /> done
          </span>
        )}
      </div>

      {/* Pipeline progress bar (only show when there's activity) */}
      {(events.length > 0 || isStreaming) && (
        <PipelineBar firedSteps={firedSteps} activeStep={activeStepKey ?? ''} />
      )}

      <div className="border-t border-white/5 mx-3 mb-0 flex-shrink-0" />

      {/* Scrollable event stream */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-0">
        {events.length === 0 && !isStreaming ? (
          <IdleState />
        ) : (
          <>
            {events.map((event, i) => (
              <div key={`${event.node}-${i}`}>
                <NodeCard event={event} index={i} isLast={i === events.length - 1} />
                {/* Animated connector to next node */}
                {(i < events.length - 1 || isStreaming) && (
                  <Connector color={connectorBg} />
                )}
              </div>
            ))}

            {/* Active (pending) node */}
            {isStreaming && activeStepKey && (
              <ActiveNodeSkeleton nodeName={activeStepKey} />
            )}

            <div ref={bottomRef} />
          </>
        )}
      </div>
    </div>
  )
}
