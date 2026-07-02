import { useState, useEffect, useRef, useMemo } from 'react'
import {
  Activity,
  PenLine,
  Shuffle,
  Search,
  CheckSquare,
  Sparkles,
  ChevronDown,
  ChevronRight,
  Check,
  RefreshCw,
  GitBranch,
  Layers,
  Network,
  Split,
  Tag,
  ShieldCheck,
  Calculator,
} from 'lucide-react'
import clsx from 'clsx'

const C = {
  retrieval: { color: 'text-blue-400',    bg: 'bg-blue-500/15',    border: 'border-blue-500/30',    glow: 'rgba(59,130,246,0.35)' },
  decision:  { color: 'text-amber-400',   bg: 'bg-amber-500/15',   border: 'border-amber-500/30',   glow: 'rgba(245,158,11,0.35)' },
  success:   { color: 'text-green-400',   bg: 'bg-green-500/15',   border: 'border-green-500/30',   glow: 'rgba(34,197,94,0.35)' },
  error:     { color: 'text-red-400',     bg: 'bg-red-500/15',     border: 'border-red-500/30',     glow: 'rgba(239,68,68,0.35)' },
}

const NODE_META = {
  rewrite:          { label: 'Query Rewrite',      short: 'Rewrite',   Icon: PenLine,      ...C.decision },
  classify:         { label: 'Classify',           short: 'Classify',  Icon: Tag,          ...C.decision },
  router:           { label: 'Router',             short: 'Route',     Icon: GitBranch,    ...C.decision },
  access_check:     { label: 'Access Check',       short: 'Access',    Icon: ShieldCheck,  ...C.decision },
  decompose:        { label: 'Decompose',          short: 'Split',     Icon: Split,        ...C.decision },
  retrieve_vector:  { label: 'Vector Retrieval',   short: 'Retrieve',  Icon: Search,       ...C.retrieval },
  retrieve_cag:     { label: 'CAG Retrieval',      short: 'Retrieve',  Icon: Layers,       ...C.retrieval },
  retrieve_graph:   { label: 'Graph Retrieval',    short: 'Retrieve',  Icon: Network,      ...C.retrieval },
  grade:            { label: 'Grade',              short: 'Grade',     Icon: CheckSquare,  ...C.decision },
  validate_numbers: { label: 'Validate Numbers',   short: 'Validate',  Icon: Calculator,   ...C.decision },
  generate:         { label: 'Generate',           short: 'Generate',  Icon: Sparkles,     ...C.success },
}

const BASE_PIPELINE = ['rewrite', 'classify', 'router', 'access_check', 'retrieve_vector', 'grade', 'generate']

const toStepKey = (node) =>
  ['retrieve_vector', 'retrieve_cag', 'retrieve_graph'].includes(node) ? 'retrieve_vector' : node

const QUERY_TYPE_LABELS = {
  factual_lookup:    'Factual',
  comparison:        'Comparison',
  risk_analysis:     'Risk Analysis',
  financial_summary: 'Financial',
  trend_analysis:    'Trend',
  legal_review:      'Legal',
  out_of_scope:      'Out of Scope',
}

const ROUTE_META = {
  vector: { label: 'Hybrid Vector + BM25',   ...C.retrieval },
  cag:    { label: 'Context Stuffing (CAG)', ...C.retrieval },
  graph:  { label: 'GraphRAG (Neo4j)',        ...C.retrieval },
}

function PipelineBar({ steps, firedSteps }) {
  return (
    <div className="flex items-center justify-between px-3 py-2.5 bg-surface-950/80 rounded-lg mx-3 mb-3 overflow-x-auto">
      {steps.map((step, i) => {
        const meta = NODE_META[step]
        const done = firedSteps.has(step)
        const Icon = meta.Icon
        return (
          <div key={step} className="flex items-center flex-shrink-0">
            <div className="flex flex-col items-center gap-1">
              <div className={clsx(
                'w-7 h-7 rounded-full flex items-center justify-center transition-all duration-300 border',
                done ? [meta.bg, meta.border, 'step-complete'] : 'bg-surface-900 border-white/5',
              )}>
                {done ? <Check className={clsx('w-3 h-3', meta.color)} /> : <Icon className="w-3 h-3 text-slate-800" />}
              </div>
              <span className={clsx('text-[9px] font-medium', done ? meta.color : 'text-slate-800')}>{meta.short}</span>
            </div>
            {i < steps.length - 1 && (
              <div className="relative w-5 h-0.5 mx-0.5 -mt-3 overflow-hidden bg-surface-900 rounded-full">
                {done && <div className={clsx('absolute inset-y-0 left-0 rounded-full line-fill', meta.bg.replace('/15', '/50'))} />}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function Connector({ colorClass }) {
  return (
    <div className="flex justify-center my-0.5 relative h-4">
      <div className="w-px h-full bg-white/5 rounded-full" />
      <div className={clsx('absolute top-0 w-px rounded-full line-fill', colorClass ?? 'bg-blue-500/40')} style={{ height: '100%' }} />
    </div>
  )
}

function ChunkList({ chunks, limit = 3 }) {
  if (!chunks?.length) return <p className="text-[10px] text-red-400/80 mt-1">No chunks</p>
  return (
    <div className="mt-1.5 space-y-1">
      {chunks.slice(0, limit).map((c, i) => (
        <div key={c.chunk_id ?? i} className="rounded-md bg-surface-950/80 border border-blue-500/10 px-2 py-1">
          <div className="flex justify-between gap-2">
            <span className="text-[10px] font-mono text-blue-300 truncate">{c.filename}</span>
            <span className="text-[9px] font-mono text-blue-400 flex-shrink-0">{c.is_parent ? 'parent' : c.score}</span>
          </div>
          <p className="text-[10px] text-slate-600 line-clamp-1">{c.preview}</p>
        </div>
      ))}
    </div>
  )
}

function FacetTree({ facets, variant = 'retrieve' }) {
  if (!facets?.length) return null
  return (
    <div className="mt-2 space-y-2">
      <p className="text-[10px] text-slate-600 uppercase tracking-wide">
        {facets.length} facet{facets.length !== 1 && 's'}
      </p>
      {facets.map((facet, i) => (
        <div
          key={`${facet.collection}-${i}`}
          className="node-appear ml-1 border-l-2 border-amber-500/30 pl-3 pb-1"
        >
          <div className="flex items-center gap-1.5 mb-0.5">
            <span className="text-[10px] font-mono font-semibold text-amber-400">{facet.collection}</span>
            {variant === 'retrieve' && (
              <span className="text-[9px] text-blue-400">{facet.chunk_count} chunk{facet.chunk_count !== 1 && 's'}</span>
            )}
            {variant === 'grade' && (
              <span className={clsx(
                'text-[9px] font-semibold uppercase',
                facet.grade === 'sufficient' ? 'text-green-400' : 'text-amber-400',
              )}>
                {facet.grade}
              </span>
            )}
          </div>
          <p className="text-[11px] text-slate-400 leading-snug">{facet.sub_question}</p>
          {variant === 'retrieve' && <ChunkList chunks={facet.chunks} />}
        </div>
      ))}
    </div>
  )
}

function RouterDetail({ event }) {
  const route = event.route
  const cols = event.active_collections ?? []
  return (
    <div className="mt-2 space-y-1.5">
      <p className="text-[10px] text-slate-600">
        Route: <span className="text-amber-400 font-medium">{route?.toUpperCase()}</span>
        {event.reason && <> — {event.reason}</>}
      </p>
      {cols.length > 0 && (
        <p className="text-[10px] font-mono text-slate-500">
          active_collections: [{cols.map(c => `'${c}'`).join(', ')}]
        </p>
      )}
      {event.will_decompose && (
        <p className="text-[11px] text-amber-400 font-medium">→ will decompose into per-collection sub-questions</p>
      )}
      {cols.length >= 2 && !event.will_decompose && (
        <p className="text-[11px] text-blue-400 font-medium">Searching {cols.join(' + ')}</p>
      )}
      <div className="space-y-1">
        {Object.entries(ROUTE_META).map(([key, meta]) => {
          const chosen = key === route
          return (
            <div key={key} className={clsx(
              'flex items-center gap-2 px-2 py-1.5 rounded-lg border transition-all',
              chosen ? [meta.bg, meta.border, 'badge-pop'] : 'bg-surface-950/50 border-white/5 opacity-25',
            )}>
              <span className={clsx('text-[11px] font-medium', chosen ? meta.color : 'text-slate-700')}>{meta.label}</span>
              {chosen && <span className={clsx('ml-auto badge text-[9px]', meta.bg, meta.color)}>selected</span>}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function NodeDetail({ event }) {
  switch (event.node) {
    case 'rewrite':
      return event.rewritten_query ? (
        <div className="mt-2 p-2 rounded-lg bg-surface-950/80 border border-amber-500/20">
          <p className="text-[10px] text-slate-600 mb-1">Rewritten query</p>
          <p className="text-[11px] text-amber-300 font-mono leading-snug">{event.rewritten_query}</p>
        </div>
      ) : null

    case 'classify':
      return (
        <div className="mt-2 space-y-1.5">
          <div className="flex items-center flex-wrap gap-1.5">
            <span className="badge bg-amber-500/10 text-amber-300 border-amber-500/20 text-[10px]">
              {QUERY_TYPE_LABELS[event.query_type] ?? event.query_type}
            </span>
            {event.requires_calculation && (
              <span className="badge bg-blue-500/10 text-blue-300 border-blue-500/20 text-[10px]">calc</span>
            )}
            {event.requires_multi_doc && (
              <span className="badge bg-purple-500/10 text-purple-300 border-purple-500/20 text-[10px]">multi-doc</span>
            )}
            {event.requires_graph && (
              <span className="badge bg-green-500/10 text-green-300 border-green-500/20 text-[10px]">graph</span>
            )}
          </div>
          {event.reason && (
            <p className="text-[11px] text-slate-500 leading-snug">{event.reason}</p>
          )}
          {event.expected_output_format && (
            <p className="text-[10px] font-mono text-slate-600">format: {event.expected_output_format}</p>
          )}
        </div>
      )

    case 'access_check': {
      const denied = event.access_denied
      return (
        <div className="mt-2 space-y-1.5">
          <div className={clsx(
            'flex items-center gap-2 px-2 py-1.5 rounded-lg border',
            denied ? 'bg-red-500/10 border-red-500/20' : 'bg-green-500/10 border-green-500/20',
          )}>
            <ShieldCheck className={clsx('w-3 h-3', denied ? 'text-red-400' : 'text-green-400')} />
            <span className={clsx('text-[11px] font-semibold', denied ? 'text-red-400' : 'text-green-400')}>
              {denied ? 'Access Denied' : 'Access Granted'}
            </span>
            <span className="ml-auto text-[10px] font-mono text-slate-500">role: {event.role}</span>
          </div>
          {denied && event.denial_reason && (
            <p className="text-[11px] text-red-400/80 leading-snug">{event.denial_reason}</p>
          )}
          {event.active_collections?.length > 0 && (
            <p className="text-[10px] font-mono text-slate-600">
              collections: [{event.active_collections.map(c => `'${c}'`).join(', ')}]
            </p>
          )}
        </div>
      )
    }

    case 'router':
      return <RouterDetail event={event} />

    case 'decompose':
      return (
        <div className="mt-2">
          <p className="text-[10px] text-slate-600 mb-2">Sub-questions per collection</p>
          <FacetTree facets={event.facets} variant="decompose" />
        </div>
      )

    case 'retrieve_graph':
      if (event.facets?.length >= 2) return <FacetTree facets={event.facets} variant="retrieve" />
      return (
        <div className="mt-2 space-y-1.5">
          <div className="flex items-center gap-2 flex-wrap">
            {event.graph_query_type && (
              <span className="badge bg-purple-500/10 text-purple-300 border-purple-500/20 text-[10px]">
                {event.graph_query_type.replace(/_/g, ' ')}
              </span>
            )}
            {event.graph_chunk_count != null && (
              <span className="badge bg-green-500/10 text-green-300 border-green-500/20 text-[10px]">
                {event.graph_chunk_count} graph fact{event.graph_chunk_count !== 1 ? 's' : ''}
              </span>
            )}
            {event.vector_chunk_count != null && (
              <span className="badge bg-blue-500/10 text-blue-300 border-blue-500/20 text-[10px]">
                {event.vector_chunk_count} vector chunk{event.vector_chunk_count !== 1 ? 's' : ''}
              </span>
            )}
            {event.unsupported_count > 0 && (
              <span className="badge bg-amber-500/10 text-amber-400 border-amber-500/20 text-[10px]">
                {event.unsupported_count} unsupported
              </span>
            )}
            {event.graph_chunk_count == null && (
              <p className="text-[10px] text-slate-600">{event.chunk_count ?? 0} chunks merged</p>
            )}
          </div>
          {event.chunks?.length > 0 && <ChunkList chunks={event.chunks} limit={5} />}
        </div>
      )

    case 'retrieve_vector':
    case 'retrieve_cag':
      if (event.facets?.length >= 2) return <FacetTree facets={event.facets} variant="retrieve" />
      if (!event.chunks?.length) return <p className="mt-2 text-[11px] text-red-400">No chunks retrieved</p>
      return (
        <div className="mt-2 space-y-1">
          <p className="text-[10px] text-slate-600">{event.chunk_count} chunks · top scores</p>
          <ChunkList chunks={event.chunks} limit={5} />
        </div>
      )

    case 'grade':
      if (event.facets?.length >= 2) {
        return (
          <div className="mt-2 space-y-1">
            <div className="flex items-center gap-2 px-2 py-1 rounded-lg bg-surface-950/80 border border-amber-500/20">
              <span className={clsx(
                'text-[11px] font-semibold uppercase',
                event.grade === 'sufficient' ? 'text-green-400' : 'text-amber-400',
              )}>
                Overall: {event.grade}
              </span>
              {event.retry_count > 0 && (
                <span className="ml-auto text-[10px] text-amber-500 flex items-center gap-1">
                  <RefreshCw className="w-2.5 h-2.5" /> retry {event.retry_count}
                </span>
              )}
            </div>
            <FacetTree facets={event.facets} variant="grade" />
          </div>
        )
      }
      return (
        <div className="mt-2 px-2 py-1.5 rounded-lg bg-surface-950/80 border border-amber-500/20">
          <span className={clsx('text-[11px] font-semibold uppercase', event.grade === 'sufficient' ? 'text-green-400' : 'text-amber-400')}>
            {event.grade}
          </span>
        </div>
      )

    case 'validate_numbers':
      return (
        <div className="mt-2 space-y-1.5">
          {event.metric && (
            <p className="text-[11px] text-slate-400">
              <span className="text-slate-600">metric: </span>
              <span className="font-mono text-blue-300">{event.metric}</span>
              {event.company && <span className="text-slate-600"> · {event.company}</span>}
            </p>
          )}
          {event.formula && (
            <p className="text-[10px] font-mono text-slate-600 break-words">{event.formula}</p>
          )}
          {event.result && (
            <div className={clsx(
              'px-2 py-1 rounded-lg border text-[11px] font-semibold',
              event.validated
                ? 'bg-green-500/10 border-green-500/20 text-green-400'
                : 'bg-amber-500/10 border-amber-500/20 text-amber-400',
            )}>
              {event.result}
            </div>
          )}
        </div>
      )

    case 'generate':
      return (event.citations?.length ?? 0) > 0 ? (
        <div className="mt-2 space-y-1">
          <p className="text-[10px] text-green-400">{event.citation_count} citations attached</p>
          {event.citations.slice(0, 3).map((c, i) => (
            <div key={c.chunk_id ?? i} className="rounded-lg bg-surface-950/80 border border-green-500/15 px-2 py-1">
              <span className="text-[10px] font-mono text-green-300 truncate block">
                {c.display_name || c.filename}
              </span>
            </div>
          ))}
        </div>
      ) : null

    default:
      return null
  }
}

function getSummary(event) {
  switch (event.node) {
    case 'rewrite':
      return event.changed ? `→ ${truncate(event.rewritten_query, 60)}` : 'Query unchanged'
    case 'classify': {
      const label = QUERY_TYPE_LABELS[event.query_type] ?? event.query_type ?? 'Unknown'
      const flags = [
        event.requires_calculation && 'calc',
        event.requires_multi_doc && 'multi-doc',
        event.requires_graph && 'graph',
      ].filter(Boolean)
      return flags.length ? `${label} · ${flags.join(', ')}` : label
    }
    case 'access_check':
      return event.access_denied
        ? `Denied · role: ${event.role}`
        : `Allowed · role: ${event.role}`
    case 'router': {
      const cols = event.active_collections ?? []
      if (event.will_decompose) return `${event.route?.toUpperCase()} → decompose ${cols.join(' + ')}`
      const colLabel = cols.length >= 2 ? cols.join(' + ') : cols[0] ?? 'auto'
      return `${event.route?.toUpperCase()} → ${colLabel}`
    }
    case 'decompose':
      return `${event.facet_count ?? event.facets?.length ?? 0} sub-question${(event.facet_count ?? 0) !== 1 ? 's' : ''} scoped to collections`
    case 'retrieve_vector':
    case 'retrieve_cag':
      if (event.facets?.length >= 2) {
        const total = event.facets.reduce((n, f) => n + (f.chunk_count ?? 0), 0)
        return `${total} chunks across ${event.facets.length} facets`
      }
      return `${event.chunk_count} chunk${event.chunk_count !== 1 ? 's' : ''} retrieved`
    case 'retrieve_graph':
      if (event.graph_chunk_count != null) {
        const vCount = event.vector_chunk_count ?? 0
        return `${event.graph_chunk_count} graph fact${event.graph_chunk_count !== 1 ? 's' : ''} + ${vCount} vector chunk${vCount !== 1 ? 's' : ''}`
      }
      return `${event.chunk_count ?? 0} chunks retrieved (hybrid)`
    case 'grade':
      if (event.facets?.length >= 2) {
        const ok = event.facets.filter(f => f.grade === 'sufficient').length
        return `${ok}/${event.facets.length} facets sufficient`
      }
      return event.grade === 'sufficient' ? 'Context sufficient' : `Insufficient${event.retry_count > 0 ? ` · retry ${event.retry_count}` : ''}`
    case 'validate_numbers':
      return event.result ? event.result : event.metric ? `Validating ${event.metric}` : 'Numeric validation'
    case 'generate':
      return `Answer generated · ${event.citation_count ?? 0} citations`
    default:
      return null
  }
}

function NodeCard({ event, index }) {
  const [expanded, setExpanded] = useState(true)
  const meta = NODE_META[event.node] ?? { label: event.node, short: event.node, Icon: Activity, ...C.decision }
  const { Icon, label, color, bg, border } = meta
  const hasDetail = Boolean(
    event.facets?.length || event.chunks?.length || event.citations?.length
    || event.rewritten_query || event.route || event.grade
    || event.query_type || event.access_denied != null || event.result || event.formula,
  )
  const summary = getSummary(event)

  return (
    <div className="node-appear">
      <div className={clsx('rounded-xl border overflow-hidden', bg, border)}>
        <button
          onClick={() => hasDetail && setExpanded(v => !v)}
          className={clsx(
            'w-full flex items-center gap-2.5 px-3 py-2.5 text-left transition-colors',
            hasDetail ? 'hover:bg-white/[0.02] cursor-pointer' : 'cursor-default',
          )}
        >
          <span className="w-4 h-4 rounded-full bg-surface-950/60 flex items-center justify-center text-[9px] font-mono text-slate-600 flex-shrink-0">
            {index + 1}
          </span>
          <div className="w-5 h-5 rounded-md flex items-center justify-center flex-shrink-0 bg-surface-950/40">
            <Icon className={clsx('w-3 h-3', color)} />
          </div>
          <div className="flex-1 min-w-0">
            <span className={clsx('text-xs font-semibold', color)}>{label}</span>
            {summary && <p className="text-[11px] text-slate-400 mt-0.5 leading-snug break-words">{summary}</p>}
          </div>
          <div className="flex items-center gap-1.5 flex-shrink-0">
            <div className={clsx('w-1.5 h-1.5 rounded-full', color.replace('text-', 'bg-'))} />
            {hasDetail && (
              <span className="text-slate-600">
                {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              </span>
            )}
          </div>
        </button>
        {expanded && hasDetail && (
          <div className="px-3 pb-3 animate-fade-in">
            <NodeDetail event={event} />
          </div>
        )}
      </div>
    </div>
  )
}

function IdleState() {
  return (
    <div className="flex flex-col items-center gap-4 px-4 py-8">
      <div className="w-10 h-10 rounded-xl bg-surface-950 flex items-center justify-center border border-white/5">
        <Activity className="w-5 h-5 text-slate-700" />
      </div>
      <div className="text-center">
        <p className="text-xs text-slate-500 font-medium mb-1">Pipeline ready</p>
        <p className="text-[11px] text-slate-700 leading-relaxed">
          Multi-collection queries show a decompose step with per-collection sub-questions.
        </p>
      </div>
    </div>
  )
}

function truncate(str, n) {
  if (!str) return ''
  return str.length > n ? str.slice(0, n) + '…' : str
}

export default function LiveTrace({ events, isStreaming }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events.length])

  const hasDecompose = events.some(e => e.node === 'decompose')
  const hasValidate = events.some(e => e.node === 'validate_numbers')
  const pipelineSteps = useMemo(() => {
    let steps = [...BASE_PIPELINE]
    if (hasDecompose) {
      const routerIdx = steps.indexOf('router')
      steps.splice(routerIdx + 1, 0, 'decompose')
    }
    if (hasValidate) {
      const gradeIdx = steps.indexOf('grade')
      steps.splice(gradeIdx + 1, 0, 'validate_numbers')
    }
    return steps
  }, [hasDecompose, hasValidate])

  const firedSteps = new Set(events.map(e => toStepKey(e.node)))
  if (hasDecompose) firedSteps.add('decompose')
  if (hasValidate) firedSteps.add('validate_numbers')

  const lastEvent = events[events.length - 1]
  const connectorMeta = lastEvent ? NODE_META[lastEvent.node] : null
  const connectorColor = connectorMeta?.bg.replace('/15', '/40') ?? 'bg-blue-500/40'

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-center gap-2 px-3 pt-3 pb-2 flex-shrink-0">
        <Activity className={clsx('w-4 h-4 transition-colors', isStreaming ? 'text-amber-400 animate-pulse-slow' : 'text-slate-600')} />
        <span className="text-sm font-semibold text-slate-200">Live Trace</span>
        {isStreaming && (
          <span className="ml-auto flex items-center gap-1.5 text-[11px] text-amber-400">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" /> streaming
          </span>
        )}
        {!isStreaming && events.length > 0 && (
          <span className="ml-auto flex items-center gap-1 text-[11px] text-green-400">
            <Check className="w-3 h-3" /> done
          </span>
        )}
      </div>

      {events.length > 0 && <PipelineBar steps={pipelineSteps} firedSteps={firedSteps} />}

      <div className="border-t border-white/5 mx-3 mb-0 flex-shrink-0" />

      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-0">
        {events.length === 0 && !isStreaming ? (
          <IdleState />
        ) : events.length === 0 && isStreaming ? (
          <p className="text-[11px] text-slate-600 text-center py-6 animate-fade-in">Waiting for first node…</p>
        ) : (
          <>
            {events.map((event, i) => (
              <div key={`${event.node}-${i}`}>
                <NodeCard event={event} index={i} />
                {i < events.length - 1 && <Connector colorClass={connectorColor} />}
              </div>
            ))}
            {isStreaming && (
              <div className="flex items-center justify-center gap-2 py-3 text-[11px] text-slate-600 animate-fade-in">
                <Shuffle className="w-3 h-3 animate-pulse text-amber-500/70" /> next step…
              </div>
            )}
            <div ref={bottomRef} />
          </>
        )}
      </div>
    </div>
  )
}
