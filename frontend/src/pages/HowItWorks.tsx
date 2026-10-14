import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router'
import { fetchInsightsSummary } from '../api/endpoints'
import { useProfile } from '../contexts/ProfileContext'
import { LoadingSpinner } from '../components/LoadingSpinner'

const FUNNEL_SOURCES = [
  { label: 'Your past buys', count: 42, color: '#6366f1' },
  { label: 'Different colours & sizes', count: 38, color: '#8b5cf6' },
  { label: 'Bought together', count: 31, color: '#a78bfa' },
  { label: 'Trending now', count: 45, color: '#c4b5fd' },
  { label: 'Recently popular', count: 28, color: '#ddd6fe' },
  { label: 'Popular with your age group', count: 22, color: '#ede9fe' },
]

const EXAMPLE_FEATURES = [
  { name: 'Days since last sale', value: '3', plain: 'This item sold recently' },
  { name: 'Your purchase history length', value: '47', plain: 'You\'ve bought 47 items' },
  { name: 'Item trend (recently vs average)', value: '1.8×', plain: 'Selling faster than usual' },
  { name: 'Typical buyer age', value: '35-44', plain: 'Popular with your age group' },
  { name: 'Times you\'ve bought from this category', value: '12', plain: 'You often buy from this category' },
]

const GLOSSARY = [
  {
    term: 'Recommender system',
    def: 'Software that suggests items based on patterns in what people buy.',
  },
  {
    term: 'Personalisation',
    def: 'Tailoring suggestions to each individual based on their history.',
  },
  {
    term: 'Cold start',
    def: 'When the system has no information about a new shopper.',
  },
  {
    term: 'Candidate generation',
    def: 'Finding a shortlist of possibilities from a large catalog.',
  },
  {
    term: 'Ranking',
    def: 'Sorting the shortlist to put the best matches first.',
  },
  {
    term: 'A/B test',
    def: 'A controlled experiment that compares two approaches with real users.',
  },
]

const MOCK_SCORES = [
  { name: 'Black Jeans', score: 0.91 },
  { name: 'White Sweater', score: 0.84 },
  { name: 'Blue Blouse', score: 0.76 },
  { name: 'Brown Boots', score: 0.61 },
  { name: 'Grey Jacket', score: 0.45 },
]

export default function HowItWorks() {
  const { profile } = useProfile()
  const [activeStep, setActiveStep] = useState(0)

  const summaryQuery = useQuery({
    queryKey: ['insights', 'summary'],
    queryFn: fetchInsightsSummary,
    staleTime: 5 * 60 * 1000,
  })

  const summary = summaryQuery.data

  // Compute "1 in X" hit rate
  const rankerRate = summary?.holdout_map12_ranker ?? 0.037
  const heuristicRate = summary?.holdout_map12_heuristic ?? 0.025
  const rankerOneIn = Math.round(1 / rankerRate)
  const heuristicOneIn = Math.round(1 / heuristicRate)

  const STEPS = [
    {
      number: 1,
      title: 'Gather possibilities',
      subtitle: `About ${FUNNEL_SOURCES.reduce((a, s) => a + s.count, 0)} possibilities from 6 sources`,
      content: (
        <div className="space-y-3">
          <p className="text-sm text-gray-600">
            We start by finding a shortlist of items that might interest you from across the entire catalog.
          </p>
          <ul className="space-y-2" aria-label="Candidate sources">
            {FUNNEL_SOURCES.map((src) => (
              <li key={src.label} className="flex items-center gap-3">
                <div
                  className="h-2 rounded-full"
                  style={{ width: `${(src.count / 50) * 100}%`, maxWidth: '120px', background: src.color }}
                  aria-hidden="true"
                />
                <span className="text-sm text-gray-700">{src.label}</span>
                <span className="ml-auto text-xs font-mono text-gray-400">~{src.count}</span>
              </li>
            ))}
          </ul>
        </div>
      ),
    },
    {
      number: 2,
      title: 'Describe each item',
      subtitle: 'Each possibility is described with facts about the item and your history',
      content: (
        <div>
          <p className="mb-3 text-sm text-gray-600">
            For every candidate item, we compute facts that describe how well it fits you.
          </p>
          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="w-full text-sm" aria-label="Example item features">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-gray-500">Feature</th>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-gray-500">Value</th>
                  <th className="px-4 py-2 text-left text-xs font-semibold text-gray-500">Plain meaning</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {EXAMPLE_FEATURES.map((f) => (
                  <tr key={f.name}>
                    <td className="px-4 py-2 font-mono text-xs text-gray-600">{f.name}</td>
                    <td className="px-4 py-2 font-mono text-xs font-bold text-indigo-700">{f.value}</td>
                    <td className="px-4 py-2 text-xs text-gray-600">{f.plain}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ),
    },
    {
      number: 3,
      title: 'Score with a trained model',
      subtitle: 'A model trained on purchase history scores each item',
      content: (
        <div>
          <p className="mb-3 text-sm text-gray-600">
            Each item gets a score — higher means a better match for you.
          </p>
          <ul className="space-y-2" aria-label="Example item scores">
            {MOCK_SCORES.map((item) => (
              <li key={item.name} className="flex items-center gap-3">
                <span className="w-32 text-sm text-gray-700 truncate">{item.name}</span>
                <div className="flex-1 rounded-full bg-gray-100 h-3" aria-hidden="true">
                  <div
                    className="h-3 rounded-full bg-indigo-500 transition-all"
                    style={{ width: `${item.score * 100}%` }}
                  />
                </div>
                <span className="w-10 text-right font-mono text-xs text-gray-600">
                  {item.score.toFixed(2)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ),
    },
    {
      number: 4,
      title: 'Pick the top 12',
      subtitle: 'The 12 highest-scoring items become your recommendations',
      content: (
        <div>
          <p className="mb-3 text-sm text-gray-600">
            From the scored shortlist, the top 12 items are shown to you, ranked best first.
          </p>
          <div
            className="grid grid-cols-3 gap-2"
            aria-label="Example top 3 recommendations"
          >
            {MOCK_SCORES.slice(0, 3).map((item, i) => (
              <div
                key={item.name}
                className="rounded-lg border border-indigo-100 bg-indigo-50 p-3 text-center"
              >
                <span className="text-xs font-bold text-indigo-600">#{i + 1}</span>
                <p className="mt-1 text-xs text-gray-700">{item.name}</p>
              </div>
            ))}
          </div>
          {profile ? (
            <p className="mt-3 text-sm text-gray-500">
              <span className="font-medium text-indigo-700">Your picks</span> are based on your actual purchase history.{' '}
              <a
                href="/for-you"
                className="text-indigo-600 underline hover:text-indigo-800 focus:outline-none"
              >
                See your recommendations.
              </a>
            </p>
          ) : (
            <p className="mt-3 text-sm text-gray-500">
              <Link
                to="/"
                className="text-indigo-600 underline hover:text-indigo-800 focus:outline-none"
              >
                Create a profile and shop
              </Link>{' '}
              to see your own personalised picks.
            </p>
          )}
        </div>
      ),
    },
  ]

  return (
    <div className="space-y-12">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">How it works</h1>
        <p className="mt-1 text-gray-500">
          A plain-language walkthrough of how we make recommendations.
        </p>
      </div>

      {/* Step navigation */}
      <div className="flex gap-2 overflow-x-auto pb-1" role="tablist" aria-label="How it works steps">
        {STEPS.map((step, i) => (
          <button
            key={step.number}
            role="tab"
            aria-selected={activeStep === i}
            aria-controls={`step-panel-${i}`}
            onClick={() => setActiveStep(i)}
            className={`flex-shrink-0 rounded-xl px-4 py-2 text-sm font-medium transition focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
              activeStep === i
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            Step {step.number}: {step.title}
          </button>
        ))}
      </div>

      {/* Step content */}
      {STEPS.map((step, i) => (
        <div
          key={step.number}
          id={`step-panel-${i}`}
          role="tabpanel"
          aria-labelledby={`tab-${i}`}
          hidden={activeStep !== i}
          className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm"
        >
          <div className="mb-4 flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-full bg-indigo-600 text-base font-bold text-white">
              {step.number}
            </span>
            <div>
              <h2 className="text-lg font-bold text-gray-900">{step.title}</h2>
              <p className="text-sm text-gray-500">{step.subtitle}</p>
            </div>
          </div>
          {step.content}
        </div>
      ))}

      {/* Accuracy section */}
      <section aria-labelledby="accuracy-heading" className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">
        <h2 id="accuracy-heading" className="mb-2 text-lg font-bold text-gray-900">
          Does it really work?
        </h2>

        {summaryQuery.isLoading && <LoadingSpinner label="Loading accuracy data..." />}

        {summary && (
          <div className="space-y-4">
            <p className="text-gray-700">
              For about{' '}
              <strong>1 in {rankerOneIn} shoppers</strong>, our suggestions included something they actually bought next.
            </p>
            <p className="text-gray-600">
              Compare that to{' '}
              <strong>1 in {heuristicOneIn}</strong> for simple rules that just show what's popular.
            </p>

            <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div className="rounded-xl bg-indigo-50 px-4 py-3">
                <p className="text-xs text-indigo-600 font-medium uppercase tracking-wide">Our system</p>
                <p className="mt-1 text-2xl font-bold text-indigo-700">
                  {(summary.holdout_map12_ranker * 100).toFixed(1)}%
                </p>
                <p className="text-xs text-indigo-600">MAP@12 accuracy</p>
              </div>
              <div className="rounded-xl bg-gray-50 px-4 py-3">
                <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Simple rules</p>
                <p className="mt-1 text-2xl font-bold text-gray-700">
                  {(summary.holdout_map12_heuristic * 100).toFixed(1)}%
                </p>
                <p className="text-xs text-gray-500">MAP@12 accuracy</p>
              </div>
            </div>

            <details className="text-xs text-gray-400">
              <summary className="cursor-pointer hover:text-gray-600">Exact metrics</summary>
              <div className="mt-2 space-y-1 font-mono">
                <p>Ranker MAP@12: {summary.holdout_map12_ranker.toFixed(6)}</p>
                <p>Heuristic MAP@12: {summary.holdout_map12_heuristic.toFixed(6)}</p>
                {summary.holdout_ci_lo != null && summary.holdout_ci_hi != null && (
                  <p>
                    95% CI: [{summary.holdout_ci_lo.toFixed(6)}, {summary.holdout_ci_hi.toFixed(6)}]
                  </p>
                )}
                <p>Lift: +{summary.holdout_lift_pct.toFixed(1)}%</p>
              </div>
            </details>
          </div>
        )}
      </section>

      {/* Glossary */}
      <section aria-labelledby="glossary-heading">
        <h2 id="glossary-heading" className="mb-4 text-lg font-bold text-gray-900">
          Glossary
        </h2>
        <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          {GLOSSARY.map((item) => (
            <div
              key={item.term}
              className="rounded-xl border border-gray-100 bg-white p-4 shadow-sm"
            >
              <dt className="font-semibold text-gray-900">{item.term}</dt>
              <dd className="mt-1 text-sm text-gray-600">{item.def}</dd>
            </div>
          ))}
        </dl>
      </section>
    </div>
  )
}
