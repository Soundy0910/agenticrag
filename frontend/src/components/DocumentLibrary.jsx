import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { Library, Trash2, RefreshCw, AlertTriangle, Search, X, ChevronDown } from 'lucide-react'
import clsx from 'clsx'
import { listDocuments, deleteDocument } from '../api/client.js'
import UploadDropzone from './UploadDropzone.jsx'

const COLLAPSE_THRESHOLD = 15
const ROW_HEIGHT = 52
const OVERSCAN = 4

const TYPE_COLORS = {
  pdf:  'bg-orange-500/10 text-orange-400',
  docx: 'bg-blue-500/10 text-blue-400',
  txt:  'bg-slate-500/10 text-slate-400',
  md:   'bg-green-500/10 text-green-400',
  csv:  'bg-yellow-500/10 text-yellow-400',
  xlsx: 'bg-emerald-500/10 text-emerald-400',
  pptx: 'bg-red-500/10 text-red-400',
}

function DocRow({ doc, isDeleting, onDelete }) {
  const ext = doc.file_type || doc.filename.split('.').pop()?.toLowerCase()
  const typeClass = TYPE_COLORS[ext] ?? 'bg-slate-500/10 text-slate-400'

  return (
    <li
      className={clsx(
        'group flex items-start gap-2 px-2 py-2 rounded-lg hover:bg-white/[0.03] transition-colors',
        isDeleting && 'opacity-50 pointer-events-none',
      )}
      style={{ minHeight: ROW_HEIGHT }}
    >
      <div className={clsx('badge mt-0.5 flex-shrink-0', typeClass)}>
        {ext?.toUpperCase() ?? '?'}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-slate-300 truncate leading-snug" title={doc.filename}>
          {doc.filename}
        </p>
        <p className="text-[10px] text-slate-600 mt-0.5">
          {doc.vector_count?.toLocaleString()} vectors
        </p>
      </div>
      <button
        onClick={() => onDelete(doc)}
        className="opacity-0 group-hover:opacity-100 p-1 text-slate-600 hover:text-red-400 transition-all"
        title="Delete"
      >
        <Trash2 className="w-3 h-3" />
      </button>
    </li>
  )
}

/** Windowed list — only mounts visible rows for large libraries. */
function VirtualDocList({ items, deleting, onDelete }) {
  const containerRef = useRef(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewportH, setViewportH] = useState(320)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const measure = () => setViewportH(el.clientHeight || 320)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const startIdx = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN)
  const endIdx = Math.min(items.length, startIdx + Math.ceil(viewportH / ROW_HEIGHT) + OVERSCAN * 2)
  const paddingTop = startIdx * ROW_HEIGHT
  const paddingBottom = Math.max(0, (items.length - endIdx) * ROW_HEIGHT)

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto px-2"
      onScroll={e => setScrollTop(e.currentTarget.scrollTop)}
    >
      <ul style={{ paddingTop, paddingBottom }} className="space-y-0.5">
        {items.slice(startIdx, endIdx).map(doc => (
          <DocRow
            key={doc.doc_id}
            doc={doc}
            isDeleting={deleting === doc.doc_id}
            onDelete={onDelete}
          />
        ))}
      </ul>
    </div>
  )
}

export default function DocumentLibrary({ collection, onCollectionChange, collections }) {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [deleting, setDeleting] = useState(null)
  const [search, setSearch] = useState('')
  const [expanded, setExpanded] = useState(false)

  const fetchDocs = useCallback(async () => {
    setLoading(true)
    setError(null)
    setExpanded(false)
    try {
      const data = await listDocuments(collection)
      setDocs(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [collection])

  useEffect(() => {
    setSearch('')
    fetchDocs()
  }, [fetchDocs])

  const handleDelete = async (doc) => {
    if (!confirm(`Delete "${doc.filename}" from ${collection}?`)) return
    setDeleting(doc.doc_id)
    try {
      await deleteDocument(doc.doc_id, collection)
      setDocs(prev => prev.filter(d => d.doc_id !== doc.doc_id))
    } catch (err) {
      alert(`Delete failed: ${err.message}`)
    } finally {
      setDeleting(null)
    }
  }

  const filtered = useMemo(() => {
    if (!search.trim()) return docs
    const q = search.toLowerCase()
    return docs.filter(d => d.filename.toLowerCase().includes(q))
  }, [docs, search])

  const isLarge = docs.length > COLLAPSE_THRESHOLD
  const isCollapsed = isLarge && !expanded && !search.trim()
  const showList = !isCollapsed && filtered.length > 0

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between px-3 pt-3 pb-2 flex-shrink-0">
        <div className="flex items-center gap-2 text-sm font-medium text-slate-300">
          <Library className="w-4 h-4 text-brand-400" />
          Library
        </div>
        <button
          onClick={fetchDocs}
          disabled={loading}
          className="btn-ghost p-1"
          title="Refresh"
        >
          <RefreshCw className={clsx('w-3.5 h-3.5', loading && 'animate-spin')} />
        </button>
      </div>

      {collections && collections.length > 1 && (
        <div className="flex gap-1 px-3 pb-2 flex-shrink-0">
          {collections.map(c => (
            <button
              key={c}
              onClick={() => onCollectionChange?.(c)}
              className={clsx(
                'flex-1 px-2 py-1 rounded-md text-[11px] font-medium transition-colors truncate',
                c === collection
                  ? 'bg-brand-500/15 text-brand-300 border border-brand-500/25'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-white/5',
              )}
              title={c}
            >
              {c.replace('sec-filings', 'SEC').replace('legal-docs', 'Legal')}
            </button>
          ))}
        </div>
      )}

      <div className="px-3 pb-2 flex-shrink-0">
        <UploadDropzone
          collection={collection}
          onUploaded={() => setTimeout(fetchDocs, 500)}
        />
      </div>

      <div className="border-t border-white/5 mx-3 mb-2 flex-shrink-0" />

      <div className="px-3 mb-2 flex-shrink-0 space-y-1.5">
        <div className="flex items-center justify-between">
          <span className="text-[11px] text-slate-500 uppercase tracking-wide">{collection}</span>
          {docs.length > 0 && (
            <span className="badge bg-surface-800 text-slate-500 text-[10px]">
              {filtered.length}{filtered.length !== docs.length && `/${docs.length}`} doc{docs.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        {(isLarge || docs.length > 0) && (
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-slate-600" />
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search filenames…"
              className="w-full pl-6 pr-6 py-1 text-[11px] bg-surface-800 border border-white/8 rounded-md text-slate-300 placeholder-slate-600 focus:outline-none focus:border-blue-500/40"
            />
            {search && (
              <button
                onClick={() => setSearch('')}
                className="absolute right-1.5 top-1/2 -translate-y-1/2 text-slate-600 hover:text-slate-400"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
        )}

        {isCollapsed && (
          <button
            onClick={() => setExpanded(true)}
            className="w-full flex items-center justify-center gap-1.5 text-[11px] text-slate-500 hover:text-slate-300 py-2 rounded-lg border border-white/5 hover:border-white/10 hover:bg-white/[0.02] transition-colors"
          >
            <ChevronDown className="w-3.5 h-3.5" />
            Browse {docs.length} documents
          </button>
        )}
      </div>

      {error ? (
        <div className="flex items-start gap-2 mx-3 p-2 rounded-lg bg-red-500/10 text-red-400 text-xs">
          <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      ) : loading && docs.length === 0 ? (
        <div className="flex items-center gap-2 px-3 py-4 text-slate-500 text-xs">
          <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Loading…
        </div>
      ) : filtered.length === 0 && !isCollapsed ? (
        <p className="text-xs text-slate-600 px-3 py-4">
          {search ? 'No documents match your search.' : 'No documents in this collection. Upload one above.'}
        </p>
      ) : showList ? (
        <VirtualDocList
          items={filtered}
          deleting={deleting}
          onDelete={handleDelete}
        />
      ) : null}

      <div className="pb-3 flex-shrink-0" />
    </div>
  )
}
