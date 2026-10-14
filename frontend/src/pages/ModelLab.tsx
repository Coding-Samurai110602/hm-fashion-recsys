import { useQuery } from '@tanstack/react-query'
import { fetchAblations, fetchImportance, fetchSegments } from '../api/endpoints'
import { ErrorState } from '../components/ErrorState'
import { LoadingSpinner } from '../components/LoadingSpinner'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ErrorBar,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
} from 'recharts'

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
}

function featureLabel(key: string): string {
  if (key in FEATURE_LABELS) return FEATURE_LABELS[key] as string
  return key
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}

export default function ModelLab() {
  const ablationsQuery = useQuery({
    queryKey: ['insights', 'ablations'],
    queryFn: fetchAblations,
    staleTime: 5 * 60 * 1000,
  })

  const importanceQuery = useQuery({
    queryKey: ['insights', 'importance'],
    queryFn: fetchImportance,
    staleTime: 5 * 60 * 1000,
  })

  const segmentsQuery = useQuery({
    queryKey: ['insights', 'segments'],
    queryFn: fetchSegments,
    staleTime: 5 * 60 * 1000,
  })

  // Prepare ablation chart data
  const ablationChartData = ablationsQuery.data?.map((a) => ({
    name: a.group,
    delta: a.map12_delta,
    errorBars: [a.map12_delta - a.ci_lo, a.ci_hi - a.map12_delta],
    n_features: a.n_features,
    ci_excludes_zero: a.ci_excludes_zero,
  }))

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Lab</h1>
        <p className="mt-1 text-gray-500">
          Feature ablations, SHAP importance, and segment analysis.
        </p>
      </div>

      {/* Feature group ablations */}
      <section>
        <h2 className="mb-4 text-base font-semibold text-gray-800">
          Feature Group Ablations
        </h2>
        <p className="mb-4 text-sm text-gray-500">
          MAP@12 drop when each feature group is removed. Negative means the
          group contributes positively to ranking.
        </p>

        {ablationsQuery.isLoading && <LoadingSpinner label="Loading ablations..." />}
        {ablationsQuery.error && (
          <ErrorState
            error={ablationsQuery.error as Error}
            title="Could not load ablations"
          />
        )}
        {ablationChartData && ablationChartData.length > 0 && (
          <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
            <div
              aria-label="Bar chart of MAP@12 delta for each feature group ablation"
              className="h-64"
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={ablationChartData}
                  margin={{ top: 16, right: 16, left: 8, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 12 }}
                    tickFormatter={(v: string) =>
                      v.charAt(0).toUpperCase() + v.slice(1)
                    }
                  />
                  <YAxis tick={{ fontSize: 12 }} />
                  <ReferenceLine y={0} stroke="#9ca3af" />
                  <Tooltip
                    formatter={(v: number) => v.toFixed(6)}
                    labelFormatter={(l: string) =>
                      `${l.charAt(0).toUpperCase() + l.slice(1)} group`
                    }
                  />
                  <Bar
                    dataKey="delta"
                    name="MAP@12 delta"
                    fill="#4f46e5"
                    radius={[4, 4, 0, 0]}
                  >
                    <ErrorBar
                      dataKey="errorBars"
                      width={4}
                      strokeWidth={2}
                      stroke="#1e1b4b"
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-100 text-left text-xs text-gray-500">
                    <th className="pb-2 pr-4">Group</th>
                    <th className="pb-2 pr-4">Features</th>
                    <th className="pb-2 pr-4">MAP@12 delta</th>
                    <th className="pb-2 pr-4">95% CI</th>
                    <th className="pb-2">Significant</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-50">
                  {ablationsQuery.data?.map((a) => (
                    <tr key={a.group}>
                      <td className="py-2 pr-4 font-medium capitalize text-gray-800">
                        {a.group}
                      </td>
                      <td className="py-2 pr-4 text-gray-600">
                        {a.n_features}
                      </td>
                      <td className="py-2 pr-4 font-mono text-red-600">
                        {a.map12_delta.toFixed(6)}
                      </td>
                      <td className="py-2 pr-4 font-mono text-xs text-gray-500">
                        [{a.ci_lo.toFixed(6)}, {a.ci_hi.toFixed(6)}]
                      </td>
                      <td className="py-2">
                        {a.ci_excludes_zero ? (
                          <span className="rounded-full bg-green-100 px-2 py-0.5 text-xs text-green-700">
                            Yes
                          </span>
                        ) : (
                          <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-500">
                            No
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      {/* SHAP vs Gain importance */}
      <section>
        <h2 className="mb-4 text-base font-semibold text-gray-800">
          SHAP vs. Gain Importance (Top 10)
        </h2>
        {importanceQuery.isLoading && (
          <LoadingSpinner label="Loading importance..." />
        )}
        {importanceQuery.error && (
          <ErrorState
            error={importanceQuery.error as Error}
            title="Could not load importance"
          />
        )}
        {importanceQuery.data && importanceQuery.data.length > 0 && (
          <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-gray-100 bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    SHAP Rank
                  </th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    Feature
                  </th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    Mean |SHAP| (raw)
                  </th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    Contrast Rank
                  </th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    Mean |Contrast SHAP|
                  </th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    Gain Rank
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {importanceQuery.data.map((row) => (
                  <tr key={row.rank_shap} className="bg-white">
                    <td className="px-4 py-2 font-mono text-xs text-indigo-600">
                      #{row.rank_shap}
                    </td>
                    <td className="px-4 py-2 font-medium text-gray-800">
                      {featureLabel(row.feature)}
                      <span className="ml-2 font-mono text-xs text-gray-400">
                        ({row.feature})
                      </span>
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-gray-700">
                      {row.mean_abs_shap.toFixed(4)}
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-violet-600">
                      {row.rank_contrast != null ? `#${row.rank_contrast}` : '—'}
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-violet-700">
                      {row.mean_abs_contrast_shap != null
                        ? row.mean_abs_contrast_shap.toFixed(4)
                        : '—'}
                    </td>
                    <td className="px-4 py-2 text-xs text-gray-500">
                      {row.rank_gain != null ? `#${row.rank_gain}` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* Segment results */}
      <section>
        <h2 className="mb-4 text-base font-semibold text-gray-800">
          Segment Analysis (Fold 103, Holm-adjusted)
        </h2>
        {segmentsQuery.isLoading && (
          <LoadingSpinner label="Loading segment data..." />
        )}
        {segmentsQuery.error && (
          <ErrorState
            error={segmentsQuery.error as Error}
            title="Could not load segments"
          />
        )}
        {segmentsQuery.data && segmentsQuery.data.length === 0 && (
          <p className="text-gray-400">No segment data available.</p>
        )}
        {segmentsQuery.data && segmentsQuery.data.length > 0 && (
          <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-gray-100 bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    Family
                  </th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    Segment
                  </th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    n
                  </th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    Ranker MAP@12
                  </th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    Heuristic MAP@12
                  </th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    Lift
                  </th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    p-adj
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {segmentsQuery.data.map((seg, i) => (
                  <tr key={i} className="bg-white">
                    <td className="px-4 py-2 text-xs capitalize text-gray-500">
                      {seg.family}
                    </td>
                    <td className="px-4 py-2 font-medium text-gray-800">
                      {seg.segment}
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-gray-600">
                      {seg.n.toLocaleString()}
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-indigo-600">
                      {seg.ranker_map12.toFixed(4)}
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-gray-600">
                      {seg.heuristic_map12.toFixed(4)}
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-green-600">
                      +{seg.lift_pct.toFixed(1)}%
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-gray-500">
                      {seg.p_ttest_adj < 0.001
                        ? '<0.001'
                        : seg.p_ttest_adj.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
