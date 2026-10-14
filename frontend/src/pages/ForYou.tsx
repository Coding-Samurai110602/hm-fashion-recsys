import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Navigate, Link } from 'react-router'
import { getProfileRecommendations } from '../api/endpoints'
import { useProfile } from '../contexts/ProfileContext'
import { RecommendationCard } from '../components/RecommendationCard'
import { WhyDrawer } from '../components/WhyDrawer'
import { SkeletonGrid } from '../components/SkeletonGrid'
import type { ProfileRecommendation } from '../api/types'

export default function ForYou() {
  const { profile } = useProfile()
  const [selectedItem, setSelectedItem] = useState<ProfileRecommendation | null>(null)
  const [showBaseline, setShowBaseline] = useState(false)

  const recsQuery = useQuery({
    queryKey: ['profile-recs', profile?.id],
    queryFn: () => getProfileRecommendations(profile!.id, true),
    staleTime: 60 * 1000,
    enabled: !!profile,
  })

  if (!profile) {
    return <Navigate to="/" replace />
  }

  const recs = recsQuery.data ?? []
  const isColdStart = profile.purchase_count === 0 || recs.every((r) => r.cold_start)
  const latencyMs =
    recs.length > 0 ? recs[0]?.latency_ms : null

  return (
    <div className="space-y-8">
      {/* Why drawer */}
      <WhyDrawer item={selectedItem} onClose={() => setSelectedItem(null)} />

      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            {isColdStart
              ? 'Popular right now'
              : `12 picks for ${profile.nickname}`}
          </h1>
          {latencyMs != null && (
            <p className="mt-1 text-xs text-gray-400">
              Updated in {latencyMs.toFixed(0)} ms
            </p>
          )}
        </div>

        {/* Baseline toggle */}
        {!isColdStart && (
          <button
            onClick={() => setShowBaseline((v) => !v)}
            className={`rounded-xl border px-4 py-2 text-sm font-medium transition ${
              showBaseline
                ? 'border-indigo-400 bg-indigo-50 text-indigo-700'
                : 'border-gray-200 text-gray-600 hover:bg-gray-50'
            } focus:outline-none focus:ring-2 focus:ring-indigo-500`}
            aria-pressed={showBaseline}
          >
            Compare with simple rules
          </button>
        )}
      </div>

      {/* Cold start message */}
      {isColdStart && (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-5 py-4">
          <p className="font-medium text-amber-800">
            You haven't bought anything yet — here's what's popular right now.
          </p>
          <p className="mt-1 text-sm text-amber-700">
            Add items to your cart and checkout to get personalised picks.
          </p>
          <Link
            to="/"
            className="mt-3 inline-block text-sm font-medium text-amber-800 underline hover:text-amber-900 focus:outline-none focus:ring-2 focus:ring-amber-400"
          >
            Browse the shop
          </Link>
        </div>
      )}

      {recsQuery.isLoading && <SkeletonGrid count={12} />}

      {recsQuery.error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-5 py-4 text-red-700">
          Could not load recommendations. Please try again.
        </div>
      )}

      {showBaseline && !isColdStart ? (
        /* Side-by-side comparison */
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
          <div>
            <h2 className="mb-3 text-base font-semibold text-gray-800">
              Personalised ranking
            </h2>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {recs.map((item, i) => (
                <RecommendationCard
                  key={item.article_idx}
                  item={item}
                  rank={i + 1}
                  onWhyClick={() => setSelectedItem(item)}
                />
              ))}
            </div>
          </div>

          <div>
            <h2 className="mb-3 text-base font-semibold text-gray-800">
              Simple rules
            </h2>
            <p className="mb-3 text-sm text-gray-500">
              Our system learns from patterns. Simple rules just recommend what's popular recently.
            </p>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {recs.map((item, i) => (
                <RecommendationCard
                  key={item.article_idx + '-baseline'}
                  item={{ ...item, reasons: [], source: 'trending' }}
                  rank={i + 1}
                />
              ))}
            </div>
          </div>
        </div>
      ) : (
        /* Normal grid */
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {recs.map((item, i) => (
            <RecommendationCard
              key={item.article_idx}
              item={item}
              rank={i + 1}
              onWhyClick={() => setSelectedItem(item)}
            />
          ))}
        </div>
      )}

      {recs.length === 0 && !recsQuery.isLoading && !recsQuery.error && (
        <div className="py-12 text-center text-gray-500">
          No recommendations yet.{' '}
          <Link to="/" className="text-indigo-600 underline">
            Shop to get started.
          </Link>
        </div>
      )}
    </div>
  )
}
