import { useState } from 'react'
import type { ProfileRecommendation } from '../api/types'

interface Props {
  item: ProfileRecommendation | null
  onClose: () => void
}

export function WhyDrawer({ item, onClose }: Props) {
  const [techOpen, setTechOpen] = useState(false)

  if (!item) return null

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end"
      role="dialog"
      aria-modal="true"
      aria-label={`Why this was picked: ${item.prod_name}`}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <div className="relative z-10 flex h-full w-full max-w-md flex-col bg-white shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4">
          <div>
            <h2 className="text-base font-semibold text-gray-900">
              Why this was picked for you
            </h2>
            <p className="text-sm text-gray-500">{item.prod_name}</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            aria-label="Close panel"
          >
            <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {/* Plain reason list */}
          {item.reasons.length > 0 ? (
            <ul className="space-y-3" aria-label="Reasons for this recommendation">
              {item.reasons.map((reason, i) => (
                <li key={i} className="flex items-start gap-3">
                  <span
                    className="mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-indigo-100 text-xs font-bold text-indigo-700"
                    aria-hidden="true"
                  >
                    {i + 1}
                  </span>
                  <span className="text-sm text-gray-700">{reason}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-500">
              This item is popular with shoppers right now.
            </p>
          )}

          {/* Source */}
          {item.source && (
            <div className="rounded-lg bg-gray-50 px-4 py-3">
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                Discovery source
              </p>
              <p className="mt-1 text-sm text-gray-700">{item.source}</p>
            </div>
          )}

          {/* Technical details (collapsed) */}
          <div className="border-t border-gray-100 pt-4">
            <button
              onClick={() => setTechOpen((v) => !v)}
              className="flex w-full items-center justify-between text-xs font-medium text-gray-500 hover:text-gray-700 focus:outline-none"
              aria-expanded={techOpen}
            >
              <span>Technical details</span>
              <svg
                className={`h-4 w-4 transition-transform ${techOpen ? 'rotate-180' : ''}`}
                viewBox="0 0 20 20"
                fill="currentColor"
                aria-hidden="true"
              >
                <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
            {techOpen && (
              <div className="mt-3 space-y-2 text-xs font-mono text-gray-500">
                <div className="flex justify-between">
                  <span>article_idx</span>
                  <span>{item.article_idx}</span>
                </div>
                <div className="flex justify-between">
                  <span>article_id</span>
                  <span>{item.article_id}</span>
                </div>
                <div className="flex justify-between">
                  <span>score</span>
                  <span>{item.score.toFixed(6)}</span>
                </div>
                <div className="flex justify-between">
                  <span>source</span>
                  <span>{item.source}</span>
                </div>
                <div className="flex justify-between">
                  <span>cold_start</span>
                  <span>{String(item.cold_start)}</span>
                </div>
                <div className="flex justify-between">
                  <span>latency_ms</span>
                  <span>{item.latency_ms.toFixed(1)}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
