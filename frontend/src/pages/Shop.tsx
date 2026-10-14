import { useState, useEffect, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getTrending, browseCatalog, getCatalogFilters } from '../api/endpoints'
import { getImageUrl } from '../api/client'
import { useProfile } from '../contexts/ProfileContext'
import { useCart } from '../contexts/CartContext'
import { ProfileModal } from '../components/ProfileModal'
import { CartDrawer } from '../components/CartDrawer'
import { SkeletonGrid } from '../components/SkeletonGrid'
import { GuidedTour, useTourSeen } from '../components/GuidedTour'
import type { CatalogArticle } from '../api/endpoints'

function ProductCard({
  item,
  onAddToCart,
}: {
  item: CatalogArticle
  onAddToCart: () => void
}) {
  const [imgError, setImgError] = useState(false)

  return (
    <div className="group rounded-xl border border-gray-100 bg-white p-4 shadow-sm transition-shadow hover:shadow-md">
      {/* Image */}
      <div className="mb-3 flex h-48 items-center justify-center overflow-hidden rounded-lg bg-gray-100">
        {imgError ? (
          <div
            className="flex h-full w-full select-none items-center justify-center rounded-lg text-2xl font-bold"
            style={{
              background: `hsl(${(item.article_idx * 37) % 360} 40% 85%)`,
              color: `hsl(${(item.article_idx * 37) % 360} 60% 35%)`,
            }}
            aria-label={`${item.prod_name} placeholder`}
          >
            {item.prod_name
              .split(' ')
              .slice(0, 2)
              .map((w) => w[0])
              .join('')
              .toUpperCase()}
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

      <p className="font-medium leading-snug text-gray-900">{item.prod_name}</p>
      <p className="mt-0.5 text-sm text-gray-500">
        {item.product_group} · {item.colour}
      </p>

      <button
        onClick={onAddToCart}
        className="mt-3 w-full rounded-xl bg-indigo-600 px-3 py-2 text-sm font-semibold text-white transition hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        aria-label={`Add ${item.prod_name} to cart`}
      >
        Add to Cart
      </button>
    </div>
  )
}

export default function Shop() {
  const { profile } = useProfile()
  const { addItem, totalItems } = useCart()
  const [profileModalOpen, setProfileModalOpen] = useState(false)
  const [cartOpen, setCartOpen] = useState(false)
  const [pendingItem, setPendingItem] = useState<CatalogArticle | null>(null)
  const [page, setPage] = useState(1)
  const [productGroup, setProductGroup] = useState('')
  const [colour, setColour] = useState('')
  const [searchInput, setSearchInput] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')

  const tourSeen = useTourSeen()
  const [showTour, setShowTour] = useState(false)

  // Show tour on first visit if not seen
  useEffect(() => {
    if (!tourSeen) {
      setShowTour(true)
    }
  }, [tourSeen])

  // Debounce search 300ms
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(searchInput), 300)
    return () => clearTimeout(timer)
  }, [searchInput])

  // Reset to page 1 on filter change
  useEffect(() => {
    setPage(1)
  }, [productGroup, colour, debouncedSearch])

  const filtersQuery = useQuery({
    queryKey: ['catalog', 'filters'],
    queryFn: getCatalogFilters,
    staleTime: 10 * 60 * 1000,
  })

  const trendingQuery = useQuery({
    queryKey: ['catalog', 'trending', 8],
    queryFn: () => getTrending(8),
    staleTime: 5 * 60 * 1000,
    enabled: !profile || profile.purchase_count === 0,
  })

  const browseQuery = useQuery({
    queryKey: ['catalog', 'browse', productGroup, colour, debouncedSearch, page],
    queryFn: () =>
      browseCatalog({
        product_group: productGroup || undefined,
        colour: colour || undefined,
        q: debouncedSearch || undefined,
        page,
        page_size: 24,
      }),
    staleTime: 2 * 60 * 1000,
  })

  const handleAddToCart = useCallback(
    (item: CatalogArticle) => {
      if (!profile) {
        setPendingItem(item)
        setProfileModalOpen(true)
        return
      }
      addItem({
        article_idx: item.article_idx,
        article_id: item.article_id,
        prod_name: item.prod_name,
        product_group: item.product_group,
        colour: item.colour,
        image_url: item.image_url || getImageUrl(item.article_id),
      })
    },
    [profile, addItem],
  )

  // After profile created, add the pending item
  useEffect(() => {
    if (profile && pendingItem) {
      addItem({
        article_idx: pendingItem.article_idx,
        article_id: pendingItem.article_id,
        prod_name: pendingItem.prod_name,
        product_group: pendingItem.product_group,
        colour: pendingItem.colour,
        image_url: pendingItem.image_url || getImageUrl(pendingItem.article_id),
      })
      setPendingItem(null)
    }
  }, [profile, pendingItem, addItem])

  const totalPages = browseQuery.data
    ? Math.ceil(browseQuery.data.total / browseQuery.data.page_size)
    : 1

  return (
    <div>
      {/* Tour */}
      {showTour && <GuidedTour onDismiss={() => setShowTour(false)} />}

      {/* Profile modal (forced if no profile on first visit) */}
      <ProfileModal
        open={profileModalOpen}
        canDismiss={!!profile}
        onClose={() => setProfileModalOpen(false)}
      />

      {/* Cart drawer */}
      <CartDrawer open={cartOpen} onClose={() => setCartOpen(false)} />

      {/* Floating cart button */}
      {totalItems > 0 && (
        <button
          onClick={() => setCartOpen(true)}
          className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full bg-indigo-600 px-4 py-3 text-sm font-semibold text-white shadow-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          aria-label={`Open cart, ${totalItems} item${totalItems !== 1 ? 's' : ''}`}
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
          <span>{totalItems} item{totalItems !== 1 ? 's' : ''}</span>
        </button>
      )}

      {/* Trending section (shown when no purchases or no profile) */}
      {(!profile || profile.purchase_count === 0) && (
        <section className="mb-10" aria-labelledby="trending-heading">
          <div className="mb-4">
            <h2 id="trending-heading" className="text-lg font-bold text-gray-900">
              Trending now
            </h2>
            <p className="text-sm text-gray-500">
              New shoppers see what's popular — the system knows nothing about you yet.
            </p>
          </div>

          {trendingQuery.isLoading && <SkeletonGrid count={8} />}
          {trendingQuery.data && (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {trendingQuery.data.map((item) => (
                <ProductCard
                  key={item.article_idx}
                  item={{
                    article_idx: item.article_idx,
                    article_id: item.article_id,
                    prod_name: item.prod_name,
                    product_group: item.product_group,
                    colour: item.colour,
                    image_url: item.image_url,
                    popularity_rank: item.popularity_rank,
                  }}
                  onAddToCart={() =>
                    handleAddToCart({
                      article_idx: item.article_idx,
                      article_id: item.article_id,
                      prod_name: item.prod_name,
                      product_group: item.product_group,
                      colour: item.colour,
                      image_url: item.image_url,
                      popularity_rank: item.popularity_rank,
                    })
                  }
                />
              ))}
            </div>
          )}
        </section>
      )}

      {/* Catalog section */}
      <section aria-labelledby="catalog-heading">
        <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <h2 id="catalog-heading" className="text-lg font-bold text-gray-900">
            Browse catalog
          </h2>

          {/* Filters */}
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
            {/* Search */}
            <div className="relative">
              <input
                type="search"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                placeholder="Search products…"
                className="w-full rounded-lg border border-gray-300 px-3 py-2 pl-8 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:w-48"
                aria-label="Search products"
              />
              <svg
                className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-400"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                aria-hidden="true"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>

            {/* Product group */}
            <select
              value={productGroup}
              onChange={(e) => setProductGroup(e.target.value)}
              className="rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              aria-label="Filter by product group"
            >
              <option value="">All categories</option>
              {filtersQuery.data?.product_groups.map((pg) => (
                <option key={pg} value={pg}>
                  {pg}
                </option>
              ))}
            </select>

            {/* Colour */}
            <select
              value={colour}
              onChange={(e) => setColour(e.target.value)}
              className="rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              aria-label="Filter by colour"
            >
              <option value="">All colours</option>
              {filtersQuery.data?.colours.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        </div>

        {browseQuery.isLoading && <SkeletonGrid count={24} />}

        {browseQuery.data && browseQuery.data.items.length === 0 && (
          <div className="py-16 text-center text-gray-500">
            No products found. Try different filters.
          </div>
        )}

        {browseQuery.data && browseQuery.data.items.length > 0 && (
          <>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {browseQuery.data.items.map((item) => (
                <ProductCard
                  key={item.article_idx}
                  item={item}
                  onAddToCart={() => handleAddToCart(item)}
                />
              ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="mt-8 flex items-center justify-center gap-3">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="rounded-lg border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-40 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  Previous
                </button>
                <span className="text-sm text-gray-600">
                  Page {page} of {totalPages}
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                  className="rounded-lg border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-40 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </section>
    </div>
  )
}
