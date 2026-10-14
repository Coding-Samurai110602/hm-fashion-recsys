import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router'
import {
  listProfiles,
  getDemoShoppers,
  getProfileSummary,
  getProfileRecommendations,
  fetchActualPurchases,
  fetchBaseline,
} from '../api/endpoints'
import { useProfile } from '../contexts/ProfileContext'
import { ShopperCard } from '../components/ShopperCard'
import { SkeletonGrid } from '../components/SkeletonGrid'
import { LoadingSpinner } from '../components/LoadingSpinner'
import type { ProfileSummary, DemoShopper } from '../api/types'
import type { ArticleMeta } from '../api/types'

// ── Profile compare view ─────────────────────────────────────────────────────

function CompareView({
  profileAId,
  profileBId,
  onBack,
}: {
  profileAId: string
  profileBId: string
  onBack: () => void
}) {
  const aRecsQuery = useQuery({
    queryKey: ['profile-recs', profileAId],
    queryFn: () => getProfileRecommendations(profileAId, false),
    staleTime: 60 * 1000,
  })

  const bRecsQuery = useQuery({
    queryKey: ['profile-recs', profileBId],
    queryFn: () => getProfileRecommendations(profileBId, false),
    staleTime: 60 * 1000,
  })

  const aSummaryQuery = useQuery({
    queryKey: ['profile-summary', profileAId],
    queryFn: () => getProfileSummary(profileAId),
    staleTime: 60 * 1000,
  })

  const bSummaryQuery = useQuery({
    queryKey: ['profile-summary', profileBId],
    queryFn: () => getProfileSummary(profileBId),
    staleTime: 60 * 1000,
  })

  const aIds = new Set(aRecsQuery.data?.map((r) => r.article_id) ?? [])
  const overlapCount = bRecsQuery.data?.filter((r) => aIds.has(r.article_id)).length ?? 0

  function SummaryCard({ summary }: { summary: ProfileSummary }) {
    return (
      <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <p className="font-semibold text-gray-900">{summary.nickname}</p>
        <p className="text-sm text-gray-500">{summary.age_range} · {summary.channel}</p>
        <p className="mt-1 text-sm text-gray-600">{summary.purchase_count} purchases</p>
        {summary.top_product_groups.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {summary.top_product_groups.slice(0, 3).map((pg) => (
              <span key={pg} className="rounded-full bg-indigo-50 px-2 py-0.5 text-xs text-indigo-700">
                {pg}
              </span>
            ))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <button
          onClick={onBack}
          className="rounded-lg border border-gray-200 px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          ← Back
        </button>
        <h2 className="text-lg font-bold text-gray-900">Comparing profiles</h2>
      </div>

      {overlapCount > 0 && (
        <div className="rounded-xl border border-indigo-200 bg-indigo-50 px-5 py-3 text-sm text-indigo-700">
          These shoppers have <strong>{overlapCount}</strong> item{overlapCount !== 1 ? 's' : ''} in common.
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Column A */}
        <div className="space-y-4">
          {aSummaryQuery.data && <SummaryCard summary={aSummaryQuery.data} />}
          {aRecsQuery.isLoading && <SkeletonGrid count={4} />}
          <div className="grid grid-cols-2 gap-3">
            {aRecsQuery.data?.slice(0, 6).map((item) => (
              <div
                key={item.article_idx}
                className={`rounded-lg border p-3 text-xs ${
                  aIds.has(item.article_id) && bRecsQuery.data?.some((b) => b.article_id === item.article_id)
                    ? 'border-green-300 bg-green-50'
                    : 'border-gray-100 bg-white'
                }`}
              >
                <p className="font-medium text-gray-800 truncate">{item.prod_name}</p>
                <p className="text-gray-500">{item.colour}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Column B */}
        <div className="space-y-4">
          {bSummaryQuery.data && <SummaryCard summary={bSummaryQuery.data} />}
          {bRecsQuery.isLoading && <SkeletonGrid count={4} />}
          <div className="grid grid-cols-2 gap-3">
            {bRecsQuery.data?.slice(0, 6).map((item) => (
              <div
                key={item.article_idx}
                className={`rounded-lg border p-3 text-xs ${
                  aIds.has(item.article_id)
                    ? 'border-green-300 bg-green-50'
                    : 'border-gray-100 bg-white'
                }`}
              >
                <p className="font-medium text-gray-800 truncate">{item.prod_name}</p>
                <p className="text-gray-500">{item.colour}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Real shopper reveal modal ─────────────────────────────────────────────────

function RevealModal({
  shopper,
  onClose,
}: {
  shopper: DemoShopper
  onClose: () => void
}) {
  const actualQuery = useQuery({
    queryKey: ['actual-purchases', shopper.customer_id],
    queryFn: () => fetchActualPurchases(shopper.customer_id),
    staleTime: 5 * 60 * 1000,
  })

  const rankerQuery = useQuery({
    queryKey: ['customer-recs', shopper.customer_id],
    queryFn: () =>
      import('../api/endpoints').then((m) =>
        m.fetchRecommendations(shopper.customer_id, 12, false),
      ),
    staleTime: 5 * 60 * 1000,
  })

  const baselineQuery = useQuery({
    queryKey: ['baseline', shopper.customer_id],
    queryFn: () => fetchBaseline(shopper.customer_id),
    staleTime: 5 * 60 * 1000,
  })

  const actualIds = new Set(actualQuery.data?.map((a: ArticleMeta) => a.article_id) ?? [])
  const rankerMatches = rankerQuery.data?.filter((r) => actualIds.has(r.article_id)).length ?? 0
  const baselineMatches = baselineQuery.data?.filter((r) => actualIds.has(r.article_id)).length ?? 0

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="reveal-modal-title"
    >
      <div className="absolute inset-0 bg-black/50" onClick={onClose} aria-hidden="true" />
      <div className="relative z-10 w-full max-w-lg rounded-2xl bg-white p-6 shadow-2xl">
        <div className="mb-4 flex items-center justify-between">
          <h2 id="reveal-modal-title" className="text-base font-bold text-gray-900">
            What happened next?
          </h2>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-gray-400 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            aria-label="Close"
          >
            <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>

        {(actualQuery.isLoading || rankerQuery.isLoading) && (
          <LoadingSpinner label="Loading purchases..." />
        )}

        {actualQuery.data && rankerQuery.data && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-xl bg-indigo-50 px-4 py-3 text-center">
                <p className="text-2xl font-bold text-indigo-700">{rankerMatches}</p>
                <p className="text-xs text-indigo-600">
                  Our system matched {rankerMatches} of their purchases
                </p>
              </div>
              <div className="rounded-xl bg-gray-50 px-4 py-3 text-center">
                <p className="text-2xl font-bold text-gray-700">{baselineMatches}</p>
                <p className="text-xs text-gray-500">
                  Simple rules matched {baselineMatches}
                </p>
              </div>
            </div>

            <p className="text-xs text-gray-400">
              This is used only to check accuracy, never to make recommendations.
            </p>

            {actualQuery.data.length > 0 && (
              <div>
                <p className="mb-2 text-sm font-medium text-gray-700">They actually bought:</p>
                <ul className="space-y-1">
                  {actualQuery.data.map((a: ArticleMeta, i: number) => (
                    <li key={i} className="flex items-center gap-2 text-sm text-gray-600">
                      <span
                        className={`h-2 w-2 rounded-full flex-shrink-0 ${
                          actualIds.has(a.article_id) && rankerQuery.data?.some((r) => r.article_id === a.article_id)
                            ? 'bg-green-500'
                            : 'bg-gray-300'
                        }`}
                        aria-hidden="true"
                      />
                      {a.prod_name}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Main Shoppers page ────────────────────────────────────────────────────────

export default function Shoppers() {
  const { profile } = useProfile()
  const [compareIds, setCompareIds] = useState<[string, string] | null>(null)
  const [compareTarget, setCompareTarget] = useState<string | null>(null)
  const [revealShopper, setRevealShopper] = useState<DemoShopper | null>(null)

  const ownSummaryQuery = useQuery({
    queryKey: ['profile-summary', profile?.id],
    queryFn: () => getProfileSummary(profile!.id),
    staleTime: 60 * 1000,
    enabled: !!profile,
  })

  const profilesQuery = useQuery({
    queryKey: ['profiles'],
    queryFn: () => listProfiles(20),
    staleTime: 30 * 1000,
  })

  const demoQuery = useQuery({
    queryKey: ['demo-shoppers'],
    queryFn: getDemoShoppers,
    staleTime: 10 * 60 * 1000,
  })

  const otherProfiles = profilesQuery.data?.profiles.filter(
    (p) => p.id !== profile?.id,
  ) ?? []

  if (compareIds) {
    return (
      <CompareView
        profileAId={compareIds[0]}
        profileBId={compareIds[1]}
        onBack={() => setCompareIds(null)}
      />
    )
  }

  return (
    <div className="space-y-10">
      {/* Reveal modal */}
      {revealShopper && (
        <RevealModal shopper={revealShopper} onClose={() => setRevealShopper(null)} />
      )}

      <div>
        <h1 className="text-2xl font-bold text-gray-900">Shoppers</h1>
        <p className="mt-1 text-gray-500">
          Your profile, other visitors, and real shoppers from the dataset.
        </p>
      </div>

      {/* Your profile section */}
      <section aria-labelledby="your-profile-heading">
        <h2 id="your-profile-heading" className="mb-4 text-lg font-semibold text-gray-800">
          Your profile
        </h2>

        {!profile && (
          <div className="rounded-xl border border-gray-200 bg-gray-50 px-6 py-8 text-center">
            <p className="text-gray-500">No profile yet.</p>
            <Link
              to="/"
              className="mt-2 inline-block text-sm text-indigo-600 underline hover:text-indigo-800"
            >
              Go to Shop to create one
            </Link>
          </div>
        )}

        {profile && (
          <div className="rounded-xl border border-indigo-100 bg-white p-5 shadow-sm">
            <div className="mb-3 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-indigo-600 text-lg font-bold text-white select-none">
                {profile.nickname.slice(0, 2).toUpperCase()}
              </div>
              <div>
                <p className="font-bold text-gray-900">{profile.nickname}</p>
                <p className="text-sm text-gray-500">
                  {profile.age_range} · {profile.channel === 'online' ? 'Online shopper' : 'In-store shopper'} · {profile.purchase_count} purchases
                </p>
              </div>
            </div>

            {ownSummaryQuery.data && (
              <div className="mb-3">
                <p className="mb-1 text-xs text-gray-500">Top categories</p>
                <div className="flex flex-wrap gap-1">
                  {ownSummaryQuery.data.top_product_groups.map((pg) => (
                    <span key={pg} className="rounded-full bg-indigo-50 px-2 py-0.5 text-xs text-indigo-700">
                      {pg}
                    </span>
                  ))}
                </div>
                {ownSummaryQuery.data.top_colour && (
                  <p className="mt-2 text-xs text-gray-500">
                    Favourite colour: <strong>{ownSummaryQuery.data.top_colour}</strong>
                  </p>
                )}
              </div>
            )}

            <Link
              to="/for-you"
              className="inline-block rounded-xl bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              See your picks
            </Link>
          </div>
        )}
      </section>

      {/* Other visitors section */}
      {otherProfiles.length > 0 && (
        <section aria-labelledby="visitors-heading">
          <h2 id="visitors-heading" className="mb-4 text-lg font-semibold text-gray-800">
            Other visitors
          </h2>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {otherProfiles.map((p) => (
              <div key={p.id} className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
                <p className="font-semibold text-gray-900">{p.nickname}</p>
                <p className="text-sm text-gray-500">
                  {p.age_range} · {p.purchase_count} purchases
                </p>
                {profile && (
                  <button
                    onClick={() => {
                      if (compareTarget) {
                        setCompareIds([compareTarget, p.id])
                        setCompareTarget(null)
                      } else {
                        setCompareTarget(p.id)
                      }
                    }}
                    className="mt-3 w-full rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-600 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                    aria-label={`Compare with ${p.nickname}`}
                  >
                    {compareTarget === p.id ? 'Cancel' : 'Compare'}
                  </button>
                )}
              </div>
            ))}
          </div>
          {compareTarget && (
            <p className="mt-3 text-sm text-indigo-600">
              Now select another profile to compare with.
            </p>
          )}
        </section>
      )}

      {/* Real shoppers section */}
      <section aria-labelledby="real-shoppers-heading">
        <h2 id="real-shoppers-heading" className="mb-4 text-lg font-semibold text-gray-800">
          Real shoppers
        </h2>
        <p className="mb-4 text-sm text-gray-500">
          A sample of real shoppers from the dataset. Use "Reveal" to see how our system performed.
        </p>

        {demoQuery.isLoading && <SkeletonGrid count={6} />}

        {demoQuery.data && (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {demoQuery.data.map((shopper) => (
              <ShopperCard
                key={shopper.customer_id}
                shopper={shopper}
                onView={() => setRevealShopper(shopper)}
                onCompare={
                  profile
                    ? () => setCompareIds([profile.id, shopper.customer_id])
                    : undefined
                }
              />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
