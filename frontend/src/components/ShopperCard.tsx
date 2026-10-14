import type { DemoShopper } from '../api/types'

interface Props {
  shopper: DemoShopper
  onView?: () => void
  onCompare?: () => void
}

export function ShopperCard({ shopper, onView, onCompare }: Props) {
  const ageLabel = shopper.age_range === '<25' ? 'Under 25' : shopper.age_range
  const channelLabel = shopper.channel === 'online' ? 'Online shopper' : 'In-store shopper'

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm transition-shadow hover:shadow-md">
      {/* Header */}
      <div className="mb-3 flex items-center gap-3">
        <div
          className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full text-sm font-bold text-white select-none"
          style={{
            background: `hsl(${(shopper.customer_idx * 61) % 360} 60% 50%)`,
          }}
          aria-hidden="true"
        >
          {ageLabel.slice(0, 2)}
        </div>
        <div>
          <p className="font-semibold text-gray-900">{ageLabel}</p>
          <p className="text-xs text-gray-500">{channelLabel}</p>
        </div>
      </div>

      {/* Stats */}
      <div className="mb-3 grid grid-cols-2 gap-2">
        <div className="rounded-lg bg-gray-50 px-3 py-2">
          <p className="text-xs text-gray-500">Purchases</p>
          <p className="font-semibold text-gray-800">{shopper.purchase_count.toLocaleString()}</p>
        </div>
        {shopper.top_colour && (
          <div className="rounded-lg bg-gray-50 px-3 py-2">
            <p className="text-xs text-gray-500">Favourite colour</p>
            <p className="font-semibold text-gray-800">{shopper.top_colour}</p>
          </div>
        )}
      </div>

      {/* Top categories */}
      {shopper.top_product_groups.length > 0 && (
        <div className="mb-3">
          <p className="mb-1 text-xs text-gray-500">Top categories</p>
          <div className="flex flex-wrap gap-1">
            {shopper.top_product_groups.slice(0, 3).map((pg) => (
              <span
                key={pg}
                className="rounded-full bg-indigo-50 px-2 py-0.5 text-xs text-indigo-700"
              >
                {pg}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2">
        {onView && (
          <button
            onClick={onView}
            className="flex-1 rounded-lg bg-indigo-600 px-3 py-2 text-xs font-semibold text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            aria-label={`View recommendations for ${ageLabel} shopper`}
          >
            View picks
          </button>
        )}
        {onCompare && (
          <button
            onClick={onCompare}
            className="flex-1 rounded-lg border border-gray-200 px-3 py-2 text-xs font-semibold text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            aria-label={`Compare with ${ageLabel} shopper`}
          >
            Compare
          </button>
        )}
      </div>
    </div>
  )
}
