import { Link } from 'react-router'

const CARDS = [
  {
    to: '/engineering/lab',
    title: 'Model Lab',
    description:
      'Ablation study, feature importance (SHAP vs gain), and 21-segment breakdown showing +45% MAP@12 lift holds across every customer group.',
  },
  {
    to: '/engineering/simulator',
    title: 'A/B Simulator',
    description:
      'Power calculator and permutation simulation: how many users to detect a real lift, and why peeking at interim results inflates false-positive rate 4×.',
  },
  {
    to: '/engineering/system',
    title: 'System',
    description:
      'Serving latency by stage, throughput capacity, and API contract for all 33 endpoints.',
  },
]

export default function Engineering() {
  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Engineering</h1>
        <p className="mt-2 text-gray-500">
          Engineering pages show the technical evidence behind the demo. Each page links back here.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-1 md:grid-cols-3">
        {CARDS.map((card) => (
          <Link
            key={card.to}
            to={card.to}
            className="block rounded-xl border border-gray-200 bg-white p-6 shadow-sm hover:shadow-md transition-shadow"
          >
            <div className="flex items-start justify-between gap-2">
              <h2 className="text-lg font-bold text-gray-900">{card.title}</h2>
              <span className="flex-shrink-0 text-indigo-600 text-lg font-medium" aria-hidden="true">
                →
              </span>
            </div>
            <p className="mt-2 text-sm text-gray-600">{card.description}</p>
          </Link>
        ))}
      </div>
    </div>
  )
}
