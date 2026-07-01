import { useState } from 'react'
import { BookOpen, ChevronDown, ChevronUp, Building2, FileText, Calendar } from 'lucide-react'
import clsx from 'clsx'

function CitationBadge({ icon: Icon, text, className }) {
  if (!text) return null
  return (
    <span className={clsx('inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-medium border', className)}>
      {Icon && <Icon className="w-2.5 h-2.5" />}
      {text}
    </span>
  )
}

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
        {visible.map((c, i) => {
          const title = c.display_name || c.filename || `Source ${i + 1}`
          return (
            <div
              key={c.chunk_id ?? i}
              className="px-2.5 py-2 rounded-md bg-surface-800 border border-white/5"
            >
              <div className="flex items-center justify-between gap-2 mb-1">
                <span
                  className="text-[11px] font-medium text-blue-300 truncate leading-snug"
                  title={c.filename}
                >
                  {title}
                </span>
                <span className="badge bg-blue-500/10 text-blue-400 text-[10px] font-mono flex-shrink-0">
                  [{i + 1}]
                </span>
              </div>

              {/* Metadata badges */}
              {(c.company || c.filing_type || c.fiscal_year) && (
                <div className="flex items-center flex-wrap gap-1 mb-1">
                  {c.company && (
                    <CitationBadge
                      icon={Building2}
                      text={c.company}
                      className="bg-slate-700/60 text-slate-300 border-slate-600/40"
                    />
                  )}
                  {c.filing_type && (
                    <CitationBadge
                      icon={FileText}
                      text={c.filing_type}
                      className="bg-blue-500/10 text-blue-400 border-blue-500/20"
                    />
                  )}
                  {c.fiscal_year && (
                    <CitationBadge
                      icon={Calendar}
                      text={c.fiscal_year}
                      className="bg-amber-500/10 text-amber-400 border-amber-500/20"
                    />
                  )}
                </div>
              )}

              <p className="text-[11px] font-mono text-slate-500 leading-relaxed line-clamp-2">
                {c.preview ?? c.source_text?.slice(0, 160)}
              </p>
            </div>
          )
        })}
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
