import { useParams, Link } from 'react-router'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { fetchFunnel } from '../api/endpoints'
import { ErrorState } from '../components/ErrorState'
import { LoadingSpinner } from '../components/LoadingSpinner'
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts'

const SOURCE_COLORS: Record<string, string> = {
  repurchase: '#4f46e5',
  product_code: '#0891b2',
  copurchase: '#16a34a',
  popularity_last_week: '#d97706',
  popularity_decayed: '#ea580c',
  segment_popular: '#9333ea',
}

function getSourceColor(source: string): string {
  return SOURCE_COLORS[source] ?? '#6b7280'
}

type SortKey = 'final_rank' | 'model_score' | 'source_rank' | 'source'

export default function CandidateFunnel() {
  const { customerId } = useParams<{ customerId: string }>()
  const [sortKey, setSortKey] = useState<SortKey>('final_rank')
  const [sortAsc, setSortAsc] = useState(true)

  const { data, isLoading, error } = useQuery({
    queryKey: ['funnel', customerId],
    queryFn: () => fetchFunnel(customerId!),
    enabled: !!customerId,
    staleTime: 30 * 1000,
  })

  // Count by source
  const sourceCounts: Record<string, number> = {}
  data?.forEach((item) => {
    sourceCounts[item.source] = (sourceCounts[item.source] ?? 0) + 1
  })

  // Scatter data per source
  const scatterGroups: Record<
    string,
    { source_rank: number | null; final_rank: number; top12: boolean }[]
  > = {}
  data?.forEach((item) => {
    const g = item.source
    if (!scatterGroups[g]) scatterGroups[g] = []
    scatterGroups[g]!.push({
      source_rank: item.source_rank,
      final_rank: item.final_rank,
      top12: item.final_rank <= 12,
    })
  })

  // Sorted table
  const sortedData = data
    ? [...data].sort((a, b) => {
        let av: string | number | null = a[sortKey]
        let bv: string | number | null = b[sortKey]
        if (av === null) av = Infinity
        if (bv === null) bv = Infinity
        const cmp = av < bv ? -1 : av > bv ? 1 : 0
        return sortAsc ? cmp : -cmp
      })
    : []

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortAsc(!sortAsc)
    } else {
      setSortKey(key)
      setSortAsc(true)
    }
  }

  const truncateId = (id: string) =>
    id.length <= 16 ? id : `${id.slice(0, 8)}...${id.slice(-8)}`

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link
          to="/customers"
          className="text-sm text-indigo-600 hover:underline focus:outline-none focus:underline"
        >
          ← Back to Customer Explorer
        </Link>
      </div>

      <div>
        <h1 className="text-2xl font-bold text-gray-900">Candidate Funnel</h1>
        <p className="mt-1 font-mono text-xs text-gray-500">
          {truncateId(customerId ?? '')}
        </p>
      </div>

      {isLoading && (
        <div className="flex h-40 items-center justify-center">
          <LoadingSpinner label="Loading funnel..." size="lg" />
        </div>
      )}
      {error && (
        <ErrorState error={error as Error} title="Could not load funnel" />
      )}

      {data && data.length === 0 && (
        <p className="rounded-xl border border-dashed border-gray-300 p-12 text-center text-gray-400">
          No funnel data available.
        </p>
      )}

      {data && data.length > 0 && (
        <>
          {/* Source counts */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {Object.entries(sourceCounts).map(([src, count]) => (
              <div
                key={src}
                className="rounded-lg border border-gray-200 bg-white p-3 text-center shadow-sm"
              >
                <p
                  className="text-lg font-bold"
                  style={{ color: getSourceColor(src) }}
                >
                  {count}
                </p>
                <p className="text-xs text-gray-500">
                  {src.replace(/_/g, ' ')}
                </p>
              </div>
            ))}
          </div>

          {/* Scatter chart */}
          <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
            <h2 className="mb-4 text-sm font-semibold text-gray-700">
              Source rank vs. final rank (top-12 highlighted)
            </h2>
            <div
              aria-label="Scatter chart showing candidate source rank vs final rank, colored by source"
              className="h-72"
            >
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart
                  margin={{ top: 8, right: 16, left: 0, bottom: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    type="number"
                    dataKey="source_rank"
                    name="Source Rank"
                    label={{
                      value: 'Source Rank',
                      position: 'insideBottom',
                      offset: -4,
                      fontSize: 11,
                    }}
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis
                    type="number"
                    dataKey="final_rank"
                    name="Final Rank"
                    label={{
                      value: 'Final Rank',
                      angle: -90,
                      position: 'insideLeft',
                      fontSize: 11,
                    }}
                    tick={{ fontSize: 11 }}
                  />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    content={({ payload }) => {
                      if (!payload?.length) return null
                      const d = payload[0]?.payload as {
                        source_rank: number | null
                        final_rank: number
                        top12: boolean
                      }
                      return (
                        <div className="rounded bg-white border border-gray-200 p-2 text-xs shadow">
                          <p>Source rank: {d.source_rank ?? 'N/A'}</p>
                          <p>Final rank: {d.final_rank}</p>
                          {d.top12 && (
                            <p className="font-semibold text-indigo-600">
                              Top-12
                            </p>
                          )}
                        </div>
                      )
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {Object.entries(scatterGroups).map(([src, pts]) => (
                    <Scatter
                      key={src}
                      name={src.replace(/_/g, ' ')}
                      data={pts}
                      fill={getSourceColor(src)}
                      opacity={0.7}
                    />
                  ))}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Table */}
          <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-gray-100 bg-gray-50">
                <tr>
                  {(
                    [
                      ['final_rank', 'Final Rank'],
                      ['source', 'Source'],
                      ['source_rank', 'Source Rank'],
                      ['model_score', 'Model Score'],
                    ] as [SortKey, string][]
                  ).map(([key, label]) => (
                    <th
                      key={key}
                      className="cursor-pointer px-4 py-3 text-left font-semibold text-gray-600 hover:text-gray-900 focus:outline-none"
                      onClick={() => handleSort(key)}
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleSort(key)
                      }}
                      aria-sort={
                        sortKey === key
                          ? sortAsc
                            ? 'ascending'
                            : 'descending'
                          : 'none'
                      }
                    >
                      {label} {sortKey === key ? (sortAsc ? '↑' : '↓') : ''}
                    </th>
                  ))}
                  <th className="px-4 py-3 text-left font-semibold text-gray-600">
                    Top-12
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {sortedData.slice(0, 100).map((item, i) => (
                  <tr
                    key={i}
                    className={
                      item.final_rank <= 12 ? 'bg-indigo-50' : 'bg-white'
                    }
                  >
                    <td className="px-4 py-2 font-mono text-xs text-gray-700">
                      {item.final_rank}
                    </td>
                    <td className="px-4 py-2">
                      <span
                        className="rounded-full px-2 py-0.5 text-xs font-medium text-white"
                        style={{ backgroundColor: getSourceColor(item.source) }}
                      >
                        {item.source.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-gray-700">
                      {item.source_rank ?? '—'}
                    </td>
                    <td className="px-4 py-2 font-mono text-xs text-gray-700">
                      {item.model_score.toFixed(4)}
                    </td>
                    <td className="px-4 py-2">
                      {item.final_rank <= 12 && (
                        <span className="rounded-full bg-indigo-100 px-2 py-0.5 text-xs text-indigo-700">
                          Yes
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {sortedData.length > 100 && (
              <p className="px-4 py-2 text-xs text-gray-400">
                Showing first 100 of {sortedData.length} candidates.
              </p>
            )}
          </div>
        </>
      )}
    </div>
  )
}
