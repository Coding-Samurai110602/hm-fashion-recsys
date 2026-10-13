import { useState, useEffect, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import { searchArticles, fetchCustomRecommendations } from '../api/endpoints'
import { ErrorState } from '../components/ErrorState'
import { LoadingSpinner } from '../components/LoadingSpinner'
import { ProductCard } from '../components/ProductCard'
import type { ArticleMeta, CustomHistoryItem, RecommendedItem } from '../api/types'

const MAX_DATE = '2020-09-22'

function useDebounce<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay)
    return () => clearTimeout(t)
  }, [value, delay])
  return debounced
}

export default function CustomCustomer() {
  const [searchQ, setSearchQ] = useState('')
  const [searchResults, setSearchResults] = useState<ArticleMeta[]>([])
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [history, setHistory] = useState<
    { article: ArticleMeta; date: string }[]
  >([])
  const [selectedDate, setSelectedDate] = useState(MAX_DATE)
  const [dateError, setDateError] = useState<string | null>(null)
  const [recs, setRecs] = useState<RecommendedItem[] | null>(null)
  const searchRef = useRef<AbortController | null>(null)

  const debouncedQ = useDebounce(searchQ, 300)

  // Search articles on debounced query change
  useEffect(() => {
    if (!debouncedQ.trim()) {
      setSearchResults([])
      return
    }
    if (searchRef.current) searchRef.current.abort()
    searchRef.current = new AbortController()

    setSearchLoading(true)
    setSearchError(null)

    searchArticles(debouncedQ, 10)
      .then((res) => {
        setSearchResults(res)
        setSearchLoading(false)
      })
      .catch((err: Error) => {
        if (err.name !== 'AbortError') {
          setSearchError(err.message)
          setSearchLoading(false)
        }
      })
  }, [debouncedQ])

  const recsMutation = useMutation({
    mutationFn: (items: CustomHistoryItem[]) =>
      fetchCustomRecommendations({ history: items }),
    onSuccess: (data) => setRecs(data),
  })

  function addToHistory(article: ArticleMeta) {
    if (selectedDate > MAX_DATE) {
      setDateError(`Date must be on or before ${MAX_DATE}`)
      return
    }
    setDateError(null)
    if (history.find((h) => h.article.article_id === article.article_id)) return
    setHistory((prev) => [...prev, { article, date: selectedDate }])
    setSearchQ('')
    setSearchResults([])
  }

  function removeFromHistory(articleId: string) {
    setHistory((prev) =>
      prev.filter((h) => h.article.article_id !== articleId),
    )
  }

  function handleGetRecs() {
    if (history.length === 0) return
    const items: CustomHistoryItem[] = history.map((h) => ({
      article_id: h.article.article_id,
      t_dat: h.date,
    }))
    recsMutation.mutate(items)
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">
          Build Your Own Customer
        </h1>
        <p className="mt-1 text-gray-500">
          Create a custom purchase history and get personalized recommendations.
        </p>
      </div>

      {/* Article search */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm space-y-3">
        <h2 className="text-sm font-semibold text-gray-700">Add articles to history</h2>

        <div className="flex flex-col gap-2 sm:flex-row">
          <div className="relative flex-1">
            <input
              type="text"
              value={searchQ}
              onChange={(e) => setSearchQ(e.target.value)}
              placeholder="Search article name..."
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              aria-label="Search articles"
            />
            {searchLoading && (
              <div className="absolute right-2 top-2">
                <LoadingSpinner size="sm" label="" />
              </div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-gray-500 whitespace-nowrap">
              Purchase date:
            </label>
            <input
              type="date"
              value={selectedDate}
              max={MAX_DATE}
              onChange={(e) => {
                setSelectedDate(e.target.value)
                if (e.target.value > MAX_DATE) {
                  setDateError(`Date must be on or before ${MAX_DATE}`)
                } else {
                  setDateError(null)
                }
              }}
              className="rounded-lg border border-gray-300 px-2 py-1.5 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
          </div>
        </div>

        {dateError && (
          <p className="text-xs text-red-600" role="alert">
            {dateError}
          </p>
        )}
        {searchError && (
          <p className="text-xs text-red-600" role="alert">
            {searchError}
          </p>
        )}

        {searchResults.length > 0 && (
          <ul className="rounded-lg border border-gray-200 divide-y divide-gray-100 max-h-48 overflow-y-auto">
            {searchResults.map((article) => (
              <li key={article.article_id}>
                <button
                  onClick={() => addToHistory(article)}
                  className="w-full px-3 py-2 text-left text-sm hover:bg-indigo-50 focus:outline-none focus:bg-indigo-50"
                >
                  <span className="font-medium text-gray-800">
                    {article.prod_name}
                  </span>
                  <span className="ml-2 text-xs text-gray-500">
                    {article.product_type} · {article.colour}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* History list */}
      {history.length > 0 && (
        <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
          <h2 className="mb-3 text-sm font-semibold text-gray-700">
            Purchase history ({history.length} items)
          </h2>
          <ul className="space-y-2">
            {history.map((h) => (
              <li
                key={h.article.article_id}
                className="flex items-center justify-between gap-2 rounded-lg bg-gray-50 px-3 py-2 text-sm"
              >
                <div>
                  <span className="font-medium text-gray-800">
                    {h.article.prod_name}
                  </span>
                  <span className="ml-2 text-xs text-gray-400">
                    {h.article.product_type} · {h.date}
                  </span>
                </div>
                <button
                  onClick={() => removeFromHistory(h.article.article_id)}
                  className="text-xs text-red-500 hover:text-red-700 focus:outline-none focus:underline"
                  aria-label={`Remove ${h.article.prod_name}`}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>

          <button
            onClick={handleGetRecs}
            disabled={recsMutation.isPending}
            className="mt-4 w-full rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-60"
          >
            {recsMutation.isPending
              ? 'Fetching personalized recommendations...'
              : 'Get Recommendations'}
          </button>

          {recsMutation.isPending && (
            <div className="mt-2 flex justify-center">
              <LoadingSpinner label="Fetching personalized recommendations..." />
            </div>
          )}
          {recsMutation.error && (
            <div className="mt-2">
              <ErrorState
                error={recsMutation.error as Error}
                title="Could not get recommendations"
              />
            </div>
          )}
        </div>
      )}

      {history.length === 0 && (
        <div className="rounded-xl border border-dashed border-gray-300 p-8 text-center text-gray-400">
          Search for articles and add them to your history to get
          recommendations.
        </div>
      )}

      {/* Recommendations */}
      {recs && recs.length > 0 && (
        <div>
          <h2 className="mb-4 text-base font-semibold text-gray-800">
            Your Recommendations
          </h2>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
            {recs.map((item) => (
              <ProductCard key={item.article_id} item={item} />
            ))}
          </div>
        </div>
      )}

      {recs && recs.length === 0 && (
        <p className="rounded-lg border border-gray-200 bg-gray-50 p-8 text-center text-gray-400">
          No recommendations returned. Try adding more items to your history.
        </p>
      )}
    </div>
  )
}
