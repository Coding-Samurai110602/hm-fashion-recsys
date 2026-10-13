import { useQuery } from '@tanstack/react-query'
import { fetchExplain } from '../api/endpoints'
import { ErrorState } from './ErrorState'
import { LoadingSpinner } from './LoadingSpinner'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from 'recharts'

// Feature label mapping
const FEATURE_LABELS: Record<string, string> = {
  a_sales_1w: 'Sales last week',
  a_days_since_last_sale: 'Days since last sale',
  i_channel_gap: 'Channel preference match',
  i_days_since_bought_product_code: 'Days since similar purchase',
  i_age_gap: 'Age group fit',
  i_price_ratio: 'Price match to budget',
  i_customer_share_garment_group: 'Category affinity',
  a_section_name: 'Section',
  a_sales_2w: 'Sales last 2 weeks',
  a_colour_group_name: 'Colour group',
  i_days_since_bought_article: 'Days since last purchase',
  a_trend_ratio: 'Trend ratio',
  i_customer_share_product_group: 'Product group affinity',
  a_repurchase_rate: 'Repurchase rate',
  i_bought_same_product_code: 'Bought same style',
}

function featureLabel(key: string): string {
  if (key in FEATURE_LABELS) return FEATURE_LABELS[key] as string
  // snake_case → Title Case fallback
  return key
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}

interface Props {
  customerId: string
  articleId: string
  articleName: string
  onClose: () => void
}

export function ExplainDrawer({
  customerId,
  articleId,
  articleName,
  onClose,
}: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['explain', customerId, articleId],
    queryFn: () => fetchExplain(customerId, articleId),
    staleTime: 5 * 60 * 1000,
  })

  // Sort top 10 contrast features by absolute value
  const chartData = data
    ? Object.entries(data.contrast)
        .map(([feat, val]) => ({ feature: featureLabel(feat), value: val }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 10)
    : []

  return (
    <div
      className="fixed inset-0 z-50 flex justify-end"
      role="dialog"
      aria-modal="true"
      aria-label={`Why this item: ${articleName}`}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <div className="relative z-10 flex h-full w-full max-w-lg flex-col bg-white shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b px-6 py-4">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">
              Why this item?
            </h2>
            <p className="text-sm text-gray-500">{articleName}</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            aria-label="Close panel"
          >
            <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {isLoading && (
            <div className="flex h-40 items-center justify-center">
              <LoadingSpinner label="Fetching personalized recommendations..." size="lg" />
            </div>
          )}

          {error && <ErrorState error={error as Error} title="Could not load explanation" />}

          {data && (
            <>
              {/* Reason chips */}
              {data.reasons.length > 0 && (
                <div>
                  <h3 className="mb-2 text-sm font-semibold text-gray-700">
                    Top reasons
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {data.reasons.map((r, i) => (
                      <span
                        key={i}
                        className="rounded-full bg-indigo-100 px-3 py-1 text-sm text-indigo-700"
                      >
                        {r}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* SHAP contrast chart */}
              {chartData.length > 0 && (
                <div>
                  <h3 className="mb-2 text-sm font-semibold text-gray-700">
                    Feature influence (SHAP contrast vs. average)
                  </h3>
                  <div
                    aria-label="SHAP contrast bar chart showing positive and negative feature drivers"
                    className="h-64"
                  >
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={chartData}
                        layout="vertical"
                        margin={{ left: 16, right: 16, top: 4, bottom: 4 }}
                      >
                        <XAxis type="number" tick={{ fontSize: 11 }} />
                        <YAxis
                          type="category"
                          dataKey="feature"
                          width={160}
                          tick={{ fontSize: 11 }}
                        />
                        <Tooltip
                          formatter={(v: number) => v.toFixed(4)}
                          labelStyle={{ fontSize: 12 }}
                        />
                        <ReferenceLine x={0} stroke="#9ca3af" />
                        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                          {chartData.map((entry, index) => (
                            <Cell
                              key={index}
                              fill={entry.value >= 0 ? '#16a34a' : '#dc2626'}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="mt-1 text-xs text-gray-400">
                    Green = positive driver, Red = negative driver relative to
                    average candidate.
                  </p>
                </div>
              )}

              {/* Score */}
              <div className="rounded-lg bg-gray-50 p-3 text-sm">
                <span className="font-medium text-gray-700">Model score: </span>
                <span className="font-mono text-gray-900">
                  {data.score.toFixed(4)}
                </span>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
