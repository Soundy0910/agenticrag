import { useState, useRef, useCallback } from 'react'
import { Upload, X, CheckCircle, AlertCircle, Loader2, FileText, File } from 'lucide-react'
import clsx from 'clsx'
import { uploadDocument } from '../api/client.js'

const ACCEPTED = '.pdf,.docx,.pptx,.txt,.md,.csv,.xlsx'
const ACCEPTED_LABELS = ['PDF', 'DOCX', 'PPTX', 'TXT', 'MD', 'CSV', 'XLSX']

const ICON_MAP = {
  pdf:  FileText,
  docx: FileText,
  txt:  FileText,
  md:   FileText,
}

function FileIcon({ ext }) {
  const Icon = ICON_MAP[ext?.toLowerCase()] ?? File
  return <Icon className="w-4 h-4" />
}

export default function UploadDropzone({ collection, onUploaded }) {
  const [dragging, setDragging] = useState(false)
  const [uploads, setUploads] = useState([])  // [{file, status, error, result}]
  const inputRef = useRef(null)

  const addFiles = useCallback(async (files) => {
    const newItems = Array.from(files).map(f => ({
      id: Math.random().toString(36).slice(2),
      file: f,
      status: 'pending',
      error: null,
      result: null,
    }))
    setUploads(prev => [...prev, ...newItems])

    for (const item of newItems) {
      setUploads(prev => prev.map(u => u.id === item.id ? { ...u, status: 'uploading' } : u))
      try {
        const result = await uploadDocument(item.file, collection)
        setUploads(prev => prev.map(u => u.id === item.id ? { ...u, status: 'done', result } : u))
        onUploaded?.(result)
      } catch (err) {
        setUploads(prev => prev.map(u => u.id === item.id ? { ...u, status: 'error', error: err.message } : u))
      }
    }
  }, [collection, onUploaded])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    addFiles(e.dataTransfer.files)
  }, [addFiles])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const remove = (id) => setUploads(prev => prev.filter(u => u.id !== id))

  return (
    <div className="space-y-2">
      {/* Drop zone */}
      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => inputRef.current?.click()}
        className={clsx(
          'border border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors',
          dragging
            ? 'border-brand-500 bg-brand-500/10'
            : 'border-white/10 hover:border-white/20 hover:bg-white/[0.02]',
        )}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          multiple
          className="hidden"
          onChange={e => addFiles(e.target.files)}
        />
        <Upload className={clsx('w-5 h-5 mx-auto mb-1.5', dragging ? 'text-brand-400' : 'text-slate-500')} />
        <p className="text-xs text-slate-400">
          {dragging ? 'Drop to upload' : 'Drop files or click'}
        </p>
        <p className="text-[10px] text-slate-600 mt-0.5">{ACCEPTED_LABELS.join(' · ')}</p>
        <p className="text-[10px] text-blue-400/80 mt-1.5 font-medium">
          Uploading to: <span className="text-blue-300">{collection}</span>
        </p>
      </div>

      {/* Upload queue */}
      {uploads.length > 0 && (
        <div className="space-y-1">
          {uploads.map(item => {
            const ext = item.file.name.split('.').pop()
            return (
              <div
                key={item.id}
                className={clsx(
                  'flex items-center gap-2 px-2.5 py-1.5 rounded-md text-xs',
                  item.status === 'error' ? 'bg-red-500/10' : 'bg-surface-800',
                )}
              >
                <FileIcon ext={ext} />
                <span className="flex-1 truncate text-slate-300" title={item.file.name}>
                  {item.file.name}
                </span>
                <span className="flex-shrink-0">
                  {item.status === 'uploading' && (
                    <Loader2 className="w-3.5 h-3.5 text-brand-400 animate-spin" />
                  )}
                  {item.status === 'done' && (
                    <CheckCircle className="w-3.5 h-3.5 text-accent-400" />
                  )}
                  {item.status === 'error' && (
                    <span title={item.error}>
                      <AlertCircle className="w-3.5 h-3.5 text-red-400" />
                    </span>
                  )}
                  {item.status === 'pending' && (
                    <span className="text-slate-600">•</span>
                  )}
                </span>
                <button
                  onClick={(e) => { e.stopPropagation(); remove(item.id) }}
                  className="text-slate-600 hover:text-slate-400"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
