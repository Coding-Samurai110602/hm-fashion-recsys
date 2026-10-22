import { Suspense, lazy, useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { LoadingSpinner } from './components/LoadingSpinner'
import { ErrorBoundary } from './components/ErrorBoundary'
import { ProfileProvider, useProfile } from './contexts/ProfileContext'
import { CartProvider, useCart } from './contexts/CartContext'
import { ProfileModal } from './components/ProfileModal'
import { CartDrawer } from './components/CartDrawer'

// Route-level code splitting
const Shop = lazy(() => import('./pages/Shop'))
const ForYou = lazy(() => import('./pages/ForYou'))
const HowItWorks = lazy(() => import('./pages/HowItWorks'))
const StoreTrends = lazy(() => import('./pages/StoreTrends'))
const Shoppers = lazy(() => import('./pages/Shoppers'))
const Engineering = lazy(() => import('./pages/Engineering'))
const ModelLab = lazy(() => import('./pages/ModelLab'))
const ABSimulator = lazy(() => import('./pages/ABSimulator'))
const System = lazy(() => import('./pages/System'))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

const NAV_LINKS = [
  { to: '/', label: 'Shop', end: true },
  { to: '/for-you', label: 'For You' },
  { to: '/how-it-works', label: 'How It Works' },
  { to: '/trends', label: 'Store Trends' },
  { to: '/shoppers', label: 'Shoppers' },
  { to: '/engineering', label: 'Engineering' },
]

const MOBILE_NAV_LINKS = [
  ...NAV_LINKS,
]

function PageFallback() {
  return (
    <div className="flex h-64 items-center justify-center">
      <LoadingSpinner label="Loading page..." size="lg" />
    </div>
  )
}

function HeaderProfile() {
  const { profile } = useProfile()
  const [profileModalOpen, setProfileModalOpen] = useState(false)

  return (
    <>
      <ProfileModal
        open={profileModalOpen}
        canDismiss={true}
        onClose={() => setProfileModalOpen(false)}
      />

      {profile ? (
        <div className="flex items-center gap-2">
          <span className="hidden text-sm font-medium text-gray-700 sm:block">
            {profile.nickname}
          </span>
          <button
            onClick={() => setProfileModalOpen(true)}
            className="rounded-lg border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-600 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            aria-label={`Switch profile (currently ${profile.nickname})`}
          >
            Switch
          </button>
        </div>
      ) : (
        <button
          onClick={() => setProfileModalOpen(true)}
          className="rounded-xl bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
          aria-label="Create profile"
        >
          Create Profile
        </button>
      )}
    </>
  )
}

function HeaderCart() {
  const { totalItems } = useCart()
  const [cartOpen, setCartOpen] = useState(false)

  return (
    <>
      <CartDrawer open={cartOpen} onClose={() => setCartOpen(false)} />
      <button
        onClick={() => setCartOpen(true)}
        className="relative rounded-lg p-2 text-gray-600 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        aria-label={`Cart, ${totalItems} item${totalItems !== 1 ? 's' : ''}`}
      >
        <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" />
        </svg>
        {totalItems > 0 && (
          <span
            className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-indigo-600 text-xs font-bold text-white"
            aria-hidden="true"
          >
            {totalItems > 9 ? '9+' : totalItems}
          </span>
        )}
      </button>
    </>
  )
}

function Shell({ children }: { children: React.ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false)
  const { pathname } = useLocation()

  // Close mobile menu on navigation
  useEffect(() => {
    setMobileOpen(false)
  }, [pathname])

  return (
    <div className="min-h-screen bg-[#f9fafb]">
      {/* Top nav */}
      <nav
        className="sticky top-0 z-40 border-b border-gray-200 bg-white"
        aria-label="Main navigation"
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <div className="flex h-14 items-center gap-4">
            {/* Brand */}
            <span className="flex-shrink-0 text-sm font-semibold text-indigo-700">
              Fashion RecSys
            </span>

            {/* Desktop nav links — hidden on small screens */}
            <div className="hidden sm:flex flex-1 gap-1 overflow-x-auto">
              {NAV_LINKS.map((link) => (
                <NavLink
                  key={link.to}
                  to={link.to}
                  end={'end' in link ? link.end : undefined}
                  className={({ isActive }) =>
                    `flex-shrink-0 rounded-md px-3 py-1.5 text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                      isActive
                        ? 'bg-indigo-50 font-medium text-indigo-700'
                        : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                    }`
                  }
                >
                  {link.label}
                </NavLink>
              ))}
            </div>

            {/* Spacer on mobile */}
            <div className="flex-1 sm:hidden" />

            {/* Right side: profile + cart + hamburger */}
            <div className="flex flex-shrink-0 items-center gap-2">
              <HeaderProfile />
              <HeaderCart />

              {/* Hamburger — only on small screens */}
              <button
                className="flex sm:hidden rounded-lg p-2 text-gray-600 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                onClick={() => setMobileOpen((v) => !v)}
                aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
                aria-expanded={mobileOpen}
                aria-controls="mobile-nav"
              >
                {mobileOpen ? (
                  <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile nav panel */}
        {mobileOpen && (
          <div
            id="mobile-nav"
            className="sm:hidden border-t border-gray-200 bg-white px-4 pb-4 pt-2"
          >
            <nav className="flex flex-col gap-1" aria-label="Mobile navigation">
              {MOBILE_NAV_LINKS.map((link) => (
                <NavLink
                  key={link.to}
                  to={link.to}
                  end={'end' in link ? link.end : undefined}
                  className={({ isActive }) =>
                    `rounded-md px-3 py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                      isActive
                        ? 'bg-indigo-50 text-indigo-700'
                        : 'text-gray-700 hover:bg-gray-100'
                    }`
                  }
                >
                  {link.label}
                </NavLink>
              ))}
            </nav>
          </div>
        )}
      </nav>

      {/* Main content */}
      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        {children}
      </main>
    </div>
  )
}

export default function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <ProfileProvider>
          <CartProvider>
            <BrowserRouter>
              <Shell>
                <Suspense fallback={<PageFallback />}>
                  <Routes>
                    <Route path="/" element={<ErrorBoundary><Shop /></ErrorBoundary>} />
                    <Route path="/for-you" element={<ErrorBoundary><ForYou /></ErrorBoundary>} />
                    <Route path="/how-it-works" element={<ErrorBoundary><HowItWorks /></ErrorBoundary>} />
                    <Route path="/trends" element={<ErrorBoundary><StoreTrends /></ErrorBoundary>} />
                    <Route path="/shoppers" element={<ErrorBoundary><Shoppers /></ErrorBoundary>} />
                    <Route path="/engineering" element={<ErrorBoundary><Engineering /></ErrorBoundary>} />
                    <Route path="/engineering/lab" element={<ErrorBoundary><ModelLab /></ErrorBoundary>} />
                    <Route path="/engineering/simulator" element={<ErrorBoundary><ABSimulator /></ErrorBoundary>} />
                    <Route path="/engineering/system" element={<ErrorBoundary><System /></ErrorBoundary>} />
                  </Routes>
                </Suspense>
              </Shell>
            </BrowserRouter>
          </CartProvider>
        </ProfileProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  )
}
