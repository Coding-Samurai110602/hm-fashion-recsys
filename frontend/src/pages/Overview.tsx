import { useQuery } from '@tanstack/react-query'
import { fetchInsightsSummary, fetchWeeklyLift } from '../api/endpoints'
import { ErrorState } from '../components/ErrorState'
import { LoadingSpinner } from '../components/LoadingSpinner'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts'

function MetricCard({
  label,
  value,
  sub,
  highlight,
}: {
  label: string
  value: string
  sub?: string
  highlight?: boolean
}) {
  return (
    <div
      className={`rounded-xl border p-5 shadow-sm ${
        highlight ? 'border-indigo-200 bg-indigo-50' : 'border-gray-200 bg-white'
      }`}
    >
      <p className="text-sm text-gray-500">{label}</p>
      <p
        className={`mt-1 text-3xl font-bold ${
          highlight ? 'text-indigo-700' : 'text-gray-900'
        }`}
      >
        {value}
      </p>
      {sub && <p className="mt-1 text-xs text-gray-400">{sub}</p>}
    </div>
  )
}

function PipelineDiagram() {
  const steps = [
    { label: 'Candidates (~200)', sub: '6 sources' },
    { label: 'Feature Store', sub: '68 features' },
    { label: 'LightGBM LambdaRank', sub: '421 rounds' },
    { label: 'Top-12 + Reasons', sub: 'SHAP contrast' },
  ]

  return (
    <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
      <h2 className="mb-4 text-base font-semibold text-gray-800">
        Pipeline Architecture
      </h2>
      <div className="flex flex-wrap items-center gap-2">
        {steps.map((step, i) => (
          <div key={i} className="flex items-center gap-2">
            <div className="rounded-lg bg-indigo-50 px-4 py-3 text-center">
              <p className="text-sm font-medium text-indigo-800">{step.label}</p>
              <p className="text-xs text-indigo-500">{step.sub}</p>
            </div>
            {i < steps.length - 1 && (
              <span className="text-gray-400 text-lg" aria-hidden="true">
                →
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default function Overview() {
  const summaryQuery = useQuery({
    queryKey: ['insights', 'summary'],
    queryFn: fetchInsightsSummary,
    staleTime: 5 * 60 * 1000,
  })

  const liftQuery = useQuery({
    queryKey: ['insights', 'weekly-lift'],
    queryFn: fetchWeeklyLift,
    staleTime: 5 * 60 * 1000,
  })

  const s = summaryQuery.data

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">
          Fashion Recommendation System
        </h1>
        <p className="mt-1 text-gray-500">
          LightGBM LambdaRank with SHAP explainability — H&amp;M Personalized
          Fashion dataset
        </p>
      </div>

      {/* Headline metrics */}
      {summaryQuery.isLoading && (
        <div className="flex h-32 items-center justify-center">
          <LoadingSpinner label="Loading metrics..." size="lg" />
        </div>
      )}
      {summaryQuery.error && (
        <ErrorState
          error={summaryQuery.error as Error}
          title="Could not load metrics"
        />
      )}
      {s && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <MetricCard
            label="Holdout MAP@12 (Ranker)"
            value={s.holdout_map12_ranker.toFixed(4)}
            sub="Week 104 holdout"
            highlight
          />
          <MetricCard
            label="Holdout MAP@12 (Heuristic)"
            value={s.holdout_map12_heuristic.toFixed(4)}
            sub="Baseline"
          />
          <MetricCard
            label="Lift vs Heuristic"
            value={`+${s.holdout_lift_pct.toFixed(1)}%`}
            sub="Holdout"
            highlight
          />
          <MetricCard
            label="Bootstrap 95% CI (Fold 103)"
            value={`[+${s.fold103_ci_lo.toFixed(4)}, +${s.fold103_ci_hi.toFixed(4)}]`}
            sub="CI excludes zero"
          />
        </div>
      )}
      {!s && !summaryQuery.isLoading && !summaryQuery.error && (
        <p className="text-gray-400">No metrics available.</p>
      )}

      {/* Weekly lift chart */}
      <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
        <h2 className="mb-4 text-base font-semibold text-gray-800">
          MAP@12 by Fold (Rolling Origin Validation)
        </h2>
        {liftQuery.isLoading && (
          <div className="flex h-48 items-center justify-center">
            <LoadingSpinner label="Loading chart..." />
          </div>
        )}
        {liftQuery.error && (
          <ErrorState
            error={liftQuery.error as Error}
            title="Could not load chart data"
          />
        )}
        {liftQuery.data && liftQuery.data.length > 0 && (
          <div
            aria-label="Bar chart showing MAP@12 for ranker and heuristic across folds 101, 102, 103"
            className="h-64"
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={liftQuery.data.map((d) => ({
                  ...d,
                  name: `Fold ${d.fold}`,
                }))}
                margin={{ top: 8, right: 16, left: 0, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} domain={[0, 'auto']} />
                <Tooltip
                  formatter={(v: number) => v.toFixed(4)}
                  labelStyle={{ fontSize: 12 }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar
                  dataKey="ranker_map12"
                  name="Ranker MAP@12"
                  fill="#4f46e5"
                  radius={[4, 4, 0, 0]}
                />
                <Bar
                  dataKey="heuristic_map12"
                  name="Heuristic MAP@12"
                  fill="#d1d5db"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        {liftQuery.data && liftQuery.data.length === 0 && (
          <p className="text-center text-gray-400">No fold data available.</p>
        )}
      </div>

      {/* Pipeline diagram */}
      <PipelineDiagram />
    </div>
  )
}
