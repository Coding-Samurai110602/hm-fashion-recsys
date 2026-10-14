import { useState, useEffect } from 'react'

const TOUR_KEY = 'fashion_demo_tour_seen'

const STEPS = [
  {
    id: 'profile',
    title: 'Create your profile',
    description:
      'Tell us your age range and how you usually shop. This is all the system needs to get started.',
    targetLabel: 'profile button in header',
  },
  {
    id: 'shop',
    title: 'Browse and shop',
    description:
      'Explore the catalog, filter by category or colour, and add items to your cart.',
    targetLabel: 'Shop nav link',
  },
  {
    id: 'cart',
    title: 'Check out',
    description:
      "When you're done browsing, check out. Your purchases teach the system about your taste.",
    targetLabel: 'cart icon',
  },
  {
    id: 'for-you',
    title: 'See your picks',
    description:
      'After checkout, visit "For You" to see 12 personalised recommendations ranked just for you.',
    targetLabel: 'For You nav link',
  },
  {
    id: 'how-it-works',
    title: 'Explore how it works',
    description:
      'Curious how we make recommendations? "How It Works" walks through each step in plain language.',
    targetLabel: 'How It Works nav link',
  },
]

interface Props {
  onDismiss?: () => void
}

export function GuidedTour({ onDismiss }: Props) {
  const [step, setStep] = useState(0)

  function dismiss() {
    try {
      localStorage.setItem(TOUR_KEY, 'true')
    } catch {
      // ignore
    }
    onDismiss?.()
  }

  function next() {
    if (step < STEPS.length - 1) {
      setStep((s) => s + 1)
    } else {
      dismiss()
    }
  }

  function back() {
    setStep((s) => Math.max(0, s - 1))
  }

  const current = STEPS[step]!

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center p-4 sm:items-center"
      role="dialog"
      aria-modal="true"
      aria-label="Getting started tour"
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50" aria-hidden="true" />

      {/* Tour card */}
      <div className="relative z-10 w-full max-w-sm rounded-2xl bg-white p-6 shadow-2xl">
        {/* Step indicator */}
        <div className="mb-4 flex items-center gap-2" aria-label={`Step ${step + 1} of ${STEPS.length}`}>
          {STEPS.map((_, i) => (
            <div
              key={i}
              className={`h-1.5 flex-1 rounded-full transition-colors ${
                i === step ? 'bg-indigo-600' : i < step ? 'bg-indigo-200' : 'bg-gray-200'
              }`}
              aria-hidden="true"
            />
          ))}
        </div>

        <p className="mb-1 text-xs font-medium text-indigo-600 uppercase tracking-wide">
          Step {step + 1} of {STEPS.length}
        </p>

        <h2 className="text-lg font-bold text-gray-900">{current.title}</h2>
        <p className="mt-2 text-sm text-gray-600">{current.description}</p>

        <div className="mt-6 flex items-center gap-3">
          {step > 0 && (
            <button
              onClick={back}
              className="flex-1 rounded-xl border border-gray-200 px-4 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              Back
            </button>
          )}

          <button
            onClick={next}
            className="flex-1 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {step === STEPS.length - 1 ? 'Done' : 'Next'}
          </button>
        </div>

        <button
          onClick={dismiss}
          className="mt-3 w-full text-center text-xs text-gray-400 hover:text-gray-600 focus:outline-none focus:underline"
        >
          Skip tour
        </button>
      </div>
    </div>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export function useTourSeen() {
  const [seen, setSeen] = useState(true) // default true to avoid flash

  useEffect(() => {
    try {
      const stored = localStorage.getItem(TOUR_KEY)
      setSeen(stored === 'true')
    } catch {
      setSeen(true)
    }
  }, [])

  return seen
}

// eslint-disable-next-line react-refresh/only-export-components
export function markTourSeen() {
  try {
    localStorage.setItem(TOUR_KEY, 'true')
  } catch {
    // ignore
  }
}
