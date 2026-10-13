import { useQuery } from '@tanstack/react-query'
import { fetchVersion, fetchReady } from '../api/endpoints'
import { ErrorState } from '../components/ErrorState'
import { LoadingSpinner } from '../components/LoadingSpinner'

const LATENCY_TABLE = [
  {
    endpoint: 'GET /api/v1/customers/{id}/recommendations (no reasons)',
    p50: '56ms',
    p95: '58ms',
    budget: '150ms',
    status: 'PASS',
  },
  {
    endpoint: 'GET /api/v1/customers/{id}/recommendations (with reasons)',
    p50: '409ms',
    p95: '480ms',
    budget: '700ms',
    status: 'PASS',
  },
  {
    endpoint: 'GET /api/v1/customers/{id}/explain/{article_id}',
    p50: '2ms',
    p95: '3ms',
    budget: '700ms',
    status: 'PASS (warm)',
  },
  {
    endpoint: 'POST /api/v1/recommendations/custom',
    p50: '427ms',
    p95: '458ms',
    budget: '700ms',
    status: 'PASS',
  },
  {
    endpoint: 'GET /api/v1/insights/summary',
    p50: '<1ms',
    p95: '<1ms',
    budget: '50ms',
    status: 'PASS',
  },
  {
    endpoint: 'POST /api/v1/experiment/power',
    p50: '2ms',
    p95: '2ms',
    budget: '50ms',
    status: 'PASS',
  },
  {
    endpoint: 'POST /api/v1/experiment/simulate (50 sims)',
    p50: '223ms',
    p95: '225ms',
    budget: '500ms',
    status: 'PASS',
  },
]

const ML_METRICS = [
  { label: 'Ranker MAP@12 (fold 103)', value: '0.035832' },
  { label: 'Holdout MAP@12 (week 104)', value: '0.036904' },
  { label: 'Heuristic MAP@12 (holdout)', value: '0.024999' },
  { label: 'Ranker vs heuristic lift (holdout)', value: '+47.6%' },
  { label: 'Bootstrap 95% CI (fold 103)', value: '[+0.010563, +0.011758]' },
  { label: 'Oracle MAP@12 (fold 103)', value: '0.200527' },
  { label: 'Features', value: '68' },
  { label: 'Model', value: 'LightGBM LambdaRank' },
  { label: 'Rounds', value: '421' },
]

export default function About() {
  const versionQuery = useQuery({
    queryKey: ['version'],
    queryFn: fetchVersion,
    staleTime: 5 * 60 * 1000,
  })

  const readyQuery = useQuery({
    queryKey: ['ready'],
    queryFn: fetchReady,
    staleTime: 30 * 1000,
  })

  const v = versionQuery.data

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">About / System</h1>
        <p className="mt-1 text-gray-500">
          API version info, measured latencies, and key ML metrics.
        </p>
      </div>

      {/* System status */}
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
          <h2 className="mb-3 text-sm font-semibold text-gray-700">
            API Version
          </h2>
          {versionQuery.isLoading && <LoadingSpinner label="Loading..." />}
          {versionQuery.error && (
            <ErrorState
              error={versionQuery.error as Error}
              title="Could not fetch version"
            />
          )}
          {v && (
            <dl className="space-y-1 text-sm">
              <div className="flex justify-between">
                <dt className="text-gray-500">API version</dt>
                <dd className="font-mono text-gray-800">{v.api_version}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Git commit</dt>
                <dd className="font-mono text-gray-800">
                  {v.git_commit.slice(0, 8)}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Bundle week</dt>
                <dd className="font-mono text-gray-800">
                  {v.bundle_as_of_week}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">MAP@12 fold 103</dt>
                <dd className="font-mono text-gray-800">
                  {v.model_map12_fold103}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">MAP@12 holdout</dt>
                <dd className="font-mono text-gray-800">
                  {v.model_map12_holdout}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-gray-500">Features</dt>
                <dd className="font-mono text-gray-800">{v.n_features}</dd>
              </div>
            </dl>
          )}
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
          <h2 className="mb-3 text-sm font-semibold text-gray-700">
            Readiness
          </h2>
          {readyQuery.isLoading && <LoadingSpinner label="Checking..." />}
          {readyQuery.error && (
            <ErrorState
              error={readyQuery.error as Error}
              title="Could not check readiness"
            />
          )}
          {readyQuery.data && (
            <div className="flex items-center gap-2">
              <div
                className={`h-3 w-3 rounded-full ${
                  readyQuery.data.ready ? 'bg-green-500' : 'bg-red-500'
                }`}
              />
              <span className="text-sm font-medium text-gray-800">
                {readyQuery.data.ready ? 'Ready' : 'Not ready'}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Key ML Metrics */}
      <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-base font-semibold text-gray-800">
          Key ML Metrics
        </h2>
        <dl className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {ML_METRICS.map((m) => (
            <div
              key={m.label}
              className="flex items-center justify-between rounded-lg bg-gray-50 px-3 py-2"
            >
              <dt className="text-sm text-gray-600">{m.label}</dt>
              <dd className="font-mono text-sm font-semibold text-indigo-700">
                {m.value}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      {/* Latency table */}
      <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="mb-2 text-base font-semibold text-gray-800">
          Measured API Latencies
        </h2>
        <p className="mb-4 text-sm text-gray-500">
          Measured over HTTP, 30 calls, 5 warm-ups, real bundle (MacBook Air).
          See docs/API.md for full details.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-gray-100 bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">
                  Endpoint
                </th>
                <th className="px-4 py-3 text-right font-semibold text-gray-600">
                  p50
                </th>
                <th className="px-4 py-3 text-right font-semibold text-gray-600">
                  p95
                </th>
                <th className="px-4 py-3 text-right font-semibold text-gray-600">
                  Budget
                </th>
                <th className="px-4 py-3 text-left font-semibold text-gray-600">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {LATENCY_TABLE.map((row, i) => (
                <tr key={i} className="bg-white">
                  <td className="px-4 py-2 font-mono text-xs text-gray-700">
                    {row.endpoint}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-xs text-gray-700">
                    {row.p50}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-xs text-gray-700">
                    {row.p95}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-xs text-gray-500">
                    {row.budget}
                  </td>
                  <td className="px-4 py-2">
                    <span
                      className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                        row.status.startsWith('PASS')
                          ? 'bg-green-100 text-green-700'
                          : 'bg-red-100 text-red-700'
                      }`}
                    >
                      {row.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
