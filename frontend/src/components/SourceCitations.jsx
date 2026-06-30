import { useState } from 'react'
import { BookOpen, ChevronDown, ChevronUp } from 'lucide-react'
import clsx from 'clsx'

export default function SourceCitations({ citations = [] }) {
  const [expanded, setExpanded] = useState(false)

  if (!citations.length) return null

  const visible = expanded ? citations : citations.slice(0, 2)

  return (
    <div className="mt-2.5 space-y-1.5">
      <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
        <BookOpen className="w-3 h-3" />
        <span>{citations.length} source{citations.length !== 1 && 's'}</span>
      </div>

      <div className="space-y-1">
        {visible.map((c, i) => (
          <div
            key={c.chunk_id ?? i}
            className="px-2.5 py-2 rounded-md bg-surface-800 border border-white/5"
          >
            <div className="flex items-center justify-between gap-2 mb-0.5">
              <span className="text-[11px] font-mono font-medium text-slate-400 truncate" title={c.filename}>
                {c.filename}
              </span>
              <span className="badge bg-brand-500/10 text-brand-400 text-[10px] font-mono flex-shrink-0">
                [{i + 1}]
              </span>
            </div>
            <p className="text-[11px] font-mono text-slate-500 leading-relaxed line-clamp-2">
              {c.preview}
            </p>
          </div>
        ))}
      </div>

      {citations.length > 2 && (
        <button
          onClick={() => setExpanded(v => !v)}
          className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-300 transition-colors"
        >
          {expanded ? (
            <><ChevronUp className="w-3 h-3" /> Show fewer</>
          ) : (
            <><ChevronDown className="w-3 h-3" /> {citations.length - 2} more</>
          )}
        </button>
      )}
    </div>
  )
}
