import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getTrends } from '../api/endpoints'
import { getImageUrl } from '../api/client'
import { SkeletonGrid } from '../components/SkeletonGrid'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts'

const AGE_TABS = ['<25', '25-34', '35-44', '45-54', '55+'] as const

function TrendCard({ item, rank }: { item: { article_idx: number; article_id: string; prod_name: string; product_group: string; colour: string; image_url: string }; rank: number }) {
  const [imgError, setImgError] = useState(false)

  return (
    <div className="rounded-xl border border-gray-100 bg-white p-4 shadow-sm">
      <div className="mb-3 flex h-40 items-center justify-center overflow-hidden rounded-lg bg-gray-100">
        {imgError ? (
          <div
            className="flex h-full w-full select-none items-center justify-center rounded-lg text-xl font-bold"
            style={{
              background: `hsl(${(item.article_idx * 37) % 360} 40% 85%)`,
              color: `hsl(${(item.article_idx * 37) % 360} 60% 35%)`,
            }}
            aria-label={`${item.prod_name} placeholder`}
          >
            {item.prod_name.split(' ').slice(0, 2).map((w) => w[0]).join('').toUpperCase()}
          </div>
        ) : (
          <img
            src={item.image_url || getImageUrl(item.article_id)}
            alt={`${item.prod_name} in ${item.colour}`}
            className="h-full w-full object-cover"
            onError={() => setImgError(true)}
          />
        )}
      </div>
      <div className="flex items-start gap-2">
        <span className="flex-shrink-0 rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-bold text-indigo-700">
          #{rank}
        </span>
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-gray-900">{item.prod_name}</p>
          <p className="text-xs text-gray-500">{item.product_group} · {item.colour}</p>
        </div>
      </div>
    </div>
  )
}

export default function StoreTrends() {
  const [activeAgeTab, setActiveAgeTab] = useState<typeof AGE_TABS[number]>('<25')

  const trendsQuery = useQuery({
    queryKey: ['insights', 'trends'],
    queryFn: getTrends,
    staleTime: 5 * 60 * 1000,
  })

  const data = trendsQuery.data

  const decayChartData = data?.popularity_decay.map((bin) => ({
    name: bin.age_label,
    share: +(bin.share * 100).toFixed(1),
  }))

  const groupChartData = data?.top_product_groups.slice(0, 8).map((pg) => ({
    name: pg.product_group,
    share: +(pg.share * 100).toFixed(1),
  }))

  return (
    <div className="space-y-10">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Store Trends</h1>
        <p className="mt-1 text-gray-500">
          What's selling across the store right now.
        </p>
      </div>

      {/* Trending now */}
      <section aria-labelledby="trending-now-heading">
        <h2 id="trending-now-heading" className="mb-4 text-lg font-semibold text-gray-800">
          Trending now
        </h2>
        {trendsQuery.isLoading && <SkeletonGrid count={8} />}
        {data && (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
            {data.trending_now.slice(0, 20).map((item, i) => (
              <TrendCard key={item.article_idx} item={item} rank={i + 1} />
            ))}
          </div>
        )}
      </section>

      {/* Popular by shopper type */}
      <section aria-labelledby="by-age-heading">
        <h2 id="by-age-heading" className="mb-4 text-lg font-semibold text-gray-800">
          Popular by shopper age
        </h2>

        {/* Tabs */}
        <div className="mb-4 flex gap-2 overflow-x-auto pb-1" role="tablist" aria-label="Age group tabs">
          {AGE_TABS.map((age) => (
            <button
              key={age}
              role="tab"
              aria-selected={activeAgeTab === age}
              aria-controls={`age-panel-${age}`}
              onClick={() => setActiveAgeTab(age)}
              className={`flex-shrink-0 rounded-full px-4 py-1.5 text-sm font-medium transition focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                activeAgeTab === age
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {age === '<25' ? 'Under 25' : age}
            </button>
          ))}
        </div>

        {AGE_TABS.map((age) => (
          <div
            key={age}
            id={`age-panel-${age}`}
            role="tabpanel"
            hidden={activeAgeTab !== age}
          >
            {trendsQuery.isLoading && <SkeletonGrid count={5} />}
            {data?.trending_by_age[age] && (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
                {(data.trending_by_age[age] ?? []).slice(0, 10).map((item, i) => (
                  <TrendCard key={item.article_idx} item={item} rank={i + 1} />
                ))}
              </div>
            )}
            {data && !data.trending_by_age[age] && (
              <p className="text-sm text-gray-400">No data for this group.</p>
            )}
          </div>
        ))}
      </section>

      {/* Top categories chart */}
      <section aria-labelledby="categories-heading">
        <h2 id="categories-heading" className="mb-4 text-lg font-semibold text-gray-800">
          Top categories
        </h2>
        {trendsQuery.isLoading && (
          <div className="h-64 animate-pulse rounded-xl bg-gray-100" aria-hidden="true" />
        )}
        {groupChartData && groupChartData.length > 0 && (
          <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
            <div
              aria-label="Bar chart of top product group shares"
              className="h-64"
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={groupChartData} margin={{ top: 8, right: 8, left: 0, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 11 }}
                    angle={-30}
                    textAnchor="end"
                    interval={0}
                  />
                  <YAxis
                    tick={{ fontSize: 11 }}
                    tickFormatter={(v: number) => `${v}%`}
                  />
                  <Tooltip formatter={(v: number) => [`${v}%`, 'Share of sales']} />
                  <Bar dataKey="share" fill="#6366f1" radius={[4, 4, 0, 0]} name="Share of sales" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </section>

      {/* Popularity decay chart */}
      <section aria-labelledby="decay-heading">
        <h2 id="decay-heading" className="mb-1 text-lg font-semibold text-gray-800">
          How long do items stay popular?
        </h2>
        <p className="mb-4 text-sm text-gray-500">
          Fashion moves fast: most items sell best soon after launch.
        </p>
        {trendsQuery.isLoading && (
          <div className="h-56 animate-pulse rounded-xl bg-gray-100" aria-hidden="true" />
        )}
        {decayChartData && decayChartData.length > 0 && (
          <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
            <div
              aria-label="Bar chart showing share of sales by item age"
              className="h-56"
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={decayChartData} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => `${v}%`} />
                  <Tooltip formatter={(v: number) => [`${v}%`, 'Share of sales']} />
                  <Bar dataKey="share" fill="#a78bfa" radius={[4, 4, 0, 0]} name="Share of sales" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
