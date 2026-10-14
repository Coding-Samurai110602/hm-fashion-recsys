import { useState } from 'react'
import { getImageUrl } from '../api/client'
import type { ProfileRecommendation } from '../api/types'

interface Props {
  item: ProfileRecommendation
  rank: number
  onWhyClick?: () => void
}

export function RecommendationCard({ item, rank, onWhyClick }: Props) {
  const [imgError, setImgError] = useState(false)

  return (
    <div className="group relative rounded-xl border border-gray-100 bg-white p-4 shadow-sm transition-shadow hover:shadow-md focus-within:ring-2 focus-within:ring-indigo-500">
      {/* Image */}
      <div className="mb-3 flex h-48 items-center justify-center overflow-hidden rounded-lg bg-gray-100">
        {imgError ? (
          <div
            className="flex h-full w-full items-center justify-center rounded-lg text-2xl font-bold text-gray-400 select-none"
            style={{ background: `hsl(${(item.article_idx * 37) % 360} 40% 85%)` }}
            aria-label={`${item.prod_name} placeholder`}
          >
            {item.prod_name
              .split(' ')
              .slice(0, 2)
              .map((w) => w[0])
              .join('')
              .toUpperCase()}
          </div>
        ) : (
          <img
            src={item.image_url || getImageUrl(item.article_id)}
            alt={`${item.prod_name} in ${item.colour}`}
            className="h-full w-full object-cover"
            onError={() => setImgError(true)}
          />
        )}
      </div>

      {/* Rank badge */}
      <span className="absolute left-5 top-5 rounded-full bg-indigo-600 px-2 py-0.5 text-xs font-bold text-white">
        #{rank}
      </span>

      {/* Source chip */}
      {item.source && (
        <span className="mb-2 inline-block rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-500">
          {item.source === 'trending'
            ? 'Trending now'
            : item.source === 'repurchase'
            ? 'In your style'
            : item.source === 'segment_popular'
            ? 'Popular with shoppers like you'
            : item.source}
        </span>
      )}

      <p className="font-medium leading-snug text-gray-900">{item.prod_name}</p>
      <p className="mt-0.5 text-sm text-gray-500">
        {item.product_group} · {item.colour}
      </p>

      {/* Reason chips */}
      {item.reasons.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {item.reasons.slice(0, 3).map((reason, i) => (
            <span
              key={i}
              className="rounded-full bg-indigo-50 px-2 py-0.5 text-xs text-indigo-700"
            >
              {reason}
            </span>
          ))}
        </div>
      )}

      {onWhyClick && (
        <button
          onClick={onWhyClick}
          className="mt-3 w-full rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-600 transition hover:border-indigo-300 hover:text-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          aria-label={`Why ${item.prod_name} was recommended`}
        >
          Why this?
        </button>
      )}
    </div>
  )
}
