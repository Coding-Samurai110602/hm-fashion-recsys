import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router'
import {
  fetchSampleCustomers,
  fetchCustomerProfile,
  fetchRecommendations,
  fetchBaseline,
  fetchActualPurchases,
} from '../api/endpoints'
import { ErrorState } from '../components/ErrorState'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { ProductCard } from '../components/ProductCard'
import { ExplainDrawer } from '../components/ExplainDrawer'
import type { RecommendedItem } from '../api/types'

const SEGMENTS = ['cold-start', 'light', 'medium', 'heavy'] as const

function truncateId(id: string): string {
  if (id.length <= 16) return id
  return `${id.slice(0, 8)}...${id.slice(-8)}`
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => {
        void navigator.clipboard.writeText(text)
        setCopied(true)
        setTimeout(() => setCopied(false), 1500)
      }}
      className="ml-2 rounded px-2 py-0.5 text-xs text-indigo-600 hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
      aria-label="Copy customer ID"
    >
      {copied ? 'Copied!' : 'Copy'}
    </button>
  )
}

export default function CustomerExplorer() {
  const [segment, setSegment] = useState<string>('medium')
  const [customerId, setCustomerId] = useState<string>('')
  const [activeCustomerId, setActiveCustomerId] = useState<string | null>(null)
  const [showBaseline, setShowBaseline] = useState(false)
  const [explainItem, setExplainItem] = useState<RecommendedItem | null>(null)

  // Sample customers by segment
  const sampleQuery = useQuery({
    queryKey: ['customers', 'sample', segment],
    queryFn: () => fetchSampleCustomers(segment, 5),
    staleTime: 5 * 60 * 1000,
  })

  // Profile, recommendations, baseline, actuals for activeCustomerId
  const profileQuery = useQuery({
    queryKey: ['customer', 'profile', activeCustomerId],
    queryFn: () => fetchCustomerProfile(activeCustomerId!),
    enabled: !!activeCustomerId,
    staleTime: 30 * 1000,
  })

  const recsQuery = useQuery({
    queryKey: ['customer', 'recs', activeCustomerId, true],
    queryFn: () => fetchRecommendations(activeCustomerId!, 12, true),
    enabled: !!activeCustomerId,
    staleTime: 30 * 1000,
  })

  const baselineQuery = useQuery({
    queryKey: ['customer', 'baseline', activeCustomerId],
    queryFn: () => fetchBaseline(activeCustomerId!),
    enabled: !!activeCustomerId && showBaseline,
    staleTime: 30 * 1000,
  })

  const actualsQuery = useQuery({
    queryKey: ['customer', 'actuals', activeCustomerId],
    queryFn: () => fetchActualPurchases(activeCustomerId!),
    enabled: !!activeCustomerId,
    staleTime: 30 * 1000,
  })

  const actualIds = new Set(actualsQuery.data?.map((a) => a.article_id) ?? [])

  const rankerHits =
    recsQuery.data?.filter((r) => actualIds.has(r.article_id)).length ?? 0
  const baselineHits =
    baselineQuery.data?.filter((r) => actualIds.has(r.article_id)).length ?? 0

  function selectCustomer(cid: string) {
    setActiveCustomerId(cid)
    setExplainItem(null)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Customer Explorer</h1>
        <p className="mt-1 text-gray-500">
          Browse recommendations for real or sample customers.
        </p>
      </div>

      {/* Segment picker */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <h2 className="mb-3 text-sm font-semibold text-gray-700">
          Sample by segment
        </h2>
        <div className="flex flex-wrap gap-2">
          {SEGMENTS.map((s) => (
            <button
              key={s}
              onClick={() => setSegment(s)}
              className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                segment === s
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {s}
            </button>
          ))}
        </div>

        {sampleQuery.isLoading && <LoadingSpinner label="Loading samples..." />}
        {sampleQuery.error && (
          <ErrorState error={sampleQuery.error as Error} title="Could not load samples" />
        )}
        {sampleQuery.data && sampleQuery.data.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {sampleQuery.data.map((c) => (
              <button
                key={c.customer_id}
                onClick={() => selectCustomer(c.customer_id)}
                className={`rounded-lg border px-3 py-1.5 text-left text-xs transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                  activeCustomerId === c.customer_id
                    ? 'border-indigo-400 bg-indigo-50 text-indigo-700'
                    : 'border-gray-200 bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                <span className="font-mono">{truncateId(c.customer_id)}</span>
                <span className="ml-1 text-gray-400">· {c.age_bucket}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Manual ID input */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <h2 className="mb-2 text-sm font-semibold text-gray-700">
          Or enter a customer ID
        </h2>
        <div className="flex gap-2">
          <input
            type="text"
            value={customerId}
            onChange={(e) => setCustomerId(e.target.value)}
            placeholder="64-character hex string"
            className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm font-mono focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
          <button
            onClick={() => {
              if (customerId.trim()) selectCustomer(customerId.trim())
            }}
            disabled={!customerId.trim()}
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-50"
          >
            Load
          </button>
        </div>
      </div>

      {/* Customer profile */}
      {activeCustomerId && (
        <>
          {profileQuery.isLoading && (
            <LoadingSpinner label="Loading customer profile..." />
          )}
          {profileQuery.error && (
            <ErrorState
              error={profileQuery.error as Error}
              title="Could not load profile"
            />
          )}
          {profileQuery.data && (
            <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
              <div className="flex items-start justify-between gap-4 flex-wrap">
                <div>
                  <p className="text-sm font-semibold text-gray-700">
                    Customer{' '}
                    <span className="font-mono text-xs text-gray-500">
                      {truncateId(profileQuery.data.customer_id)}
                    </span>
                    <CopyButton text={profileQuery.data.customer_id} />
                  </p>
                  <div className="mt-2 flex flex-wrap gap-4 text-sm text-gray-600">
                    <span>Age: {profileQuery.data.age_bucket}</span>
                    <span>
                      All-time purchases:{' '}
                      {profileQuery.data.n_purchases_all_time}
                    </span>
                    <span>
                      Last 4 weeks: {profileQuery.data.n_purchases_last_4w}
                    </span>
                  </div>
                </div>
                <Link
                  to={`/customers/${activeCustomerId}/funnel`}
                  className="rounded-lg border border-gray-200 px-3 py-1.5 text-sm text-indigo-600 hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  View candidate funnel →
                </Link>
              </div>
              {profileQuery.data.recent_history.length > 0 && (
                <div className="mt-3">
                  <p className="mb-2 text-xs font-medium text-gray-500">
                    Recent purchases
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {profileQuery.data.recent_history.slice(0, 8).map((a) => (
                      <span
                        key={a.article_id}
                        className="rounded bg-gray-100 px-2 py-0.5 text-xs text-gray-700"
                      >
                        {a.prod_name}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Recommendations */}
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-base font-semibold text-gray-800">
                  Recommendations
                </h2>
                {actualsQuery.data && (
                  <p className="text-sm text-gray-500">
                    Ranker hits: {rankerHits}/12
                    {showBaseline && ` · Baseline hits: ${baselineHits}/12`}
                  </p>
                )}
              </div>
              <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showBaseline}
                  onChange={(e) => setShowBaseline(e.target.checked)}
                  className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                />
                Compare with heuristic baseline
              </label>
            </div>

            {recsQuery.isLoading && (
              <div className="flex h-40 items-center justify-center">
                <LoadingSpinner label="Fetching personalized recommendations..." size="lg" />
              </div>
            )}
            {recsQuery.error && (
              <ErrorState
                error={recsQuery.error as Error}
                title="Could not load recommendations"
              />
            )}

            {recsQuery.data && recsQuery.data.length === 0 && (
              <p className="rounded-lg border border-gray-200 bg-gray-50 p-8 text-center text-gray-400">
                No recommendations available for this customer.
              </p>
            )}

            {recsQuery.data && recsQuery.data.length > 0 && (
              <div
                className={`grid gap-4 ${
                  showBaseline
                    ? 'grid-cols-1 lg:grid-cols-2'
                    : 'grid-cols-2 sm:grid-cols-3 md:grid-cols-4'
                }`}
              >
                {/* Ranker column */}
                <div>
                  {showBaseline && (
                    <h3 className="mb-3 font-semibold text-indigo-700">
                      Ranker
                    </h3>
                  )}
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                    {recsQuery.data.map((item) => (
                      <ProductCard
                        key={item.article_id}
                        item={item}
                        isActualPurchase={actualIds.has(item.article_id)}
                        onClick={() => setExplainItem(item)}
                      />
                    ))}
                  </div>
                </div>

                {/* Baseline column */}
                {showBaseline && (
                  <div>
                    <h3 className="mb-3 font-semibold text-gray-600">
                      Heuristic baseline
                    </h3>
                    {baselineQuery.isLoading && (
                      <LoadingSpinner label="Loading baseline..." />
                    )}
                    {baselineQuery.error && (
                      <ErrorState
                        error={baselineQuery.error as Error}
                        title="Could not load baseline"
                      />
                    )}
                    {baselineQuery.data && (
                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                        {baselineQuery.data.map((item) => (
                          <div
                            key={item.article_id}
                            className={`rounded-lg border p-3 text-xs shadow-sm ${
                              actualIds.has(item.article_id)
                                ? 'border-green-300 bg-green-50'
                                : 'border-gray-200 bg-white'
                            }`}
                          >
                            {actualIds.has(item.article_id) && (
                              <p className="mb-1 text-green-700">
                                Actual purchase
                              </p>
                            )}
                            <p className="font-medium text-gray-800">
                              #{item.rank} {item.prod_name}
                            </p>
                            <p className="text-gray-500">
                              {item.product_type} · {item.colour}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}

      {!activeCustomerId && (
        <div className="rounded-xl border border-dashed border-gray-300 p-12 text-center text-gray-400">
          Select a sample customer above or enter a customer ID to see
          recommendations.
        </div>
      )}

      {/* Explain drawer */}
      {explainItem && activeCustomerId && (
        <ExplainDrawer
          customerId={activeCustomerId}
          articleId={explainItem.article_id}
          articleName={explainItem.prod_name}
          onClose={() => setExplainItem(null)}
        />
      )}
    </div>
  )
}
