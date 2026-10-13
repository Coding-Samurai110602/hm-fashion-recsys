import { useState } from 'react'
import { getImageUrl } from '../api/client'
import type { RecommendedItem } from '../api/types'

interface Props {
  item: RecommendedItem
  isActualPurchase?: boolean
  onClick?: () => void
}

export function ProductCard({ item, isActualPurchase = false, onClick }: Props) {
  const [imgError, setImgError] = useState(false)

  return (
    <div
      className={`group relative cursor-pointer rounded-lg border bg-white p-3 shadow-sm transition-shadow hover:shadow-md focus-within:ring-2 focus-within:ring-indigo-500 ${
        isActualPurchase ? 'border-green-400 ring-1 ring-green-300' : 'border-gray-200'
      }`}
      onClick={onClick}
    >
      {isActualPurchase && (
        <div className="mb-2 rounded bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800">
          Actual purchase — shown for evaluation only
        </div>
      )}

      <div className="mb-2 flex h-32 items-center justify-center overflow-hidden rounded bg-gray-100">
        {imgError ? (
          <div className="flex h-full w-full flex-col items-center justify-center text-gray-400">
            <svg
              className="h-8 w-8"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              aria-hidden="true"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <span className="mt-1 text-xs">No image</span>
          </div>
        ) : (
          <img
            src={getImageUrl(item.article_id)}
            alt={`${item.prod_name} in ${item.colour}`}
            className="h-full w-full object-cover"
            onError={() => setImgError(true)}
          />
        )}
      </div>

      <div className="flex items-start justify-between gap-1">
        <span className="text-xs font-bold text-indigo-600">#{item.rank}</span>
        <span className="text-right text-xs font-medium text-gray-800 leading-tight">
          {item.prod_name}
        </span>
      </div>

      <p className="mt-0.5 text-xs text-gray-500">
        {item.product_type} · {item.colour}
      </p>

      {item.reasons.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {item.reasons.slice(0, 3).map((reason, i) => (
            <span
              key={i}
              className="inline-block rounded-full bg-indigo-50 px-2 py-0.5 text-xs text-indigo-700"
            >
              {reason}
            </span>
          ))}
        </div>
      )}

      {onClick && (
        <button
          className="mt-2 w-full rounded text-xs text-indigo-600 hover:text-indigo-800 focus:outline-none focus:underline"
          onClick={(e) => {
            e.stopPropagation()
            onClick()
          }}
        >
          Why this item →
        </button>
      )}
    </div>
  )
}
