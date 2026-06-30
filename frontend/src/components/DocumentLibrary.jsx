import { useState, useEffect, useCallback } from 'react'
import { Library, Trash2, RefreshCw, FileText, File, AlertTriangle } from 'lucide-react'
import clsx from 'clsx'
import { listDocuments, deleteDocument } from '../api/client.js'
import UploadDropzone from './UploadDropzone.jsx'

const TYPE_COLORS = {
  pdf:  'bg-orange-500/10 text-orange-400',
  docx: 'bg-blue-500/10 text-blue-400',
  txt:  'bg-slate-500/10 text-slate-400',
  md:   'bg-green-500/10 text-green-400',
  csv:  'bg-yellow-500/10 text-yellow-400',
  xlsx: 'bg-emerald-500/10 text-emerald-400',
  pptx: 'bg-red-500/10 text-red-400',
}

export default function DocumentLibrary({ collection, onCollectionChange, collections }) {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [deleting, setDeleting] = useState(null)

  const fetchDocs = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await listDocuments(collection)
      setDocs(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [collection])

  useEffect(() => { fetchDocs() }, [fetchDocs])

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

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
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

      {/* Upload */}
      <div className="px-3 pb-2 flex-shrink-0">
        <UploadDropzone
          collection={collection}
          onUploaded={() => setTimeout(fetchDocs, 500)}
        />
      </div>

      {/* Divider */}
      <div className="border-t border-white/5 mx-3 mb-2 flex-shrink-0" />

      {/* Doc count badge */}
      <div className="flex items-center justify-between px-3 mb-1 flex-shrink-0">
        <span className="text-[11px] text-slate-500 uppercase tracking-wide">
          {collection}
        </span>
        {docs.length > 0 && (
          <span className="badge bg-surface-800 text-slate-500 text-[10px]">
            {docs.length} doc{docs.length !== 1 && 's'}
          </span>
        )}
      </div>

      {/* Document list */}
      <div className="flex-1 overflow-y-auto px-2 pb-3">
        {error ? (
          <div className="flex items-start gap-2 mx-1 mt-1 p-2 rounded-lg bg-red-500/10 text-red-400 text-xs">
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        ) : loading && docs.length === 0 ? (
          <div className="flex items-center gap-2 px-3 py-4 text-slate-500 text-xs">
            <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Loading…
          </div>
        ) : docs.length === 0 ? (
          <p className="text-xs text-slate-600 px-3 py-4">
            No documents in this collection. Upload one above.
          </p>
        ) : (
          <ul className="space-y-0.5">
            {docs.map(doc => {
              const ext = doc.file_type || doc.filename.split('.').pop()?.toLowerCase()
              const typeClass = TYPE_COLORS[ext] ?? 'bg-slate-500/10 text-slate-400'
              const isDeleting = deleting === doc.doc_id

              return (
                <li
                  key={doc.doc_id}
                  className={clsx(
                    'group flex items-start gap-2 px-2 py-2 rounded-lg hover:bg-white/[0.03] transition-colors',
                    isDeleting && 'opacity-50 pointer-events-none',
                  )}
                >
                  <div className={clsx('badge mt-0.5 flex-shrink-0', typeClass)}>
                    {ext?.toUpperCase() ?? '?'}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-slate-300 truncate leading-snug" title={doc.filename}>
                      {doc.filename}
                    </p>
                    <p className="text-[10px] text-slate-600 mt-0.5">
                      {doc.vector_count} vectors
                    </p>
                  </div>
                  <button
                    onClick={() => handleDelete(doc)}
                    className="opacity-0 group-hover:opacity-100 p-1 text-slate-600 hover:text-red-400 transition-all"
                    title="Delete"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </div>
    </div>
  )
}
