import { Suspense, lazy } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { LoadingSpinner } from './components/LoadingSpinner'

// Route-level code splitting
const Overview = lazy(() => import('./pages/Overview'))
const CustomerExplorer = lazy(() => import('./pages/CustomerExplorer'))
const CandidateFunnel = lazy(() => import('./pages/CandidateFunnel'))
const CustomCustomer = lazy(() => import('./pages/CustomCustomer'))
const ModelLab = lazy(() => import('./pages/ModelLab'))
const ABSimulator = lazy(() => import('./pages/ABSimulator'))
const About = lazy(() => import('./pages/About'))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

const NAV_LINKS = [
  { to: '/', label: 'Overview', end: true },
  { to: '/customers', label: 'Customer Explorer' },
  { to: '/custom', label: 'Build Your Own' },
  { to: '/lab', label: 'Model Lab' },
  { to: '/simulator', label: 'A/B Simulator' },
  { to: '/about', label: 'About' },
]

function PageFallback() {
  return (
    <div className="flex h-64 items-center justify-center">
      <LoadingSpinner label="Loading page..." size="lg" />
    </div>
  )
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top nav */}
      <nav
        className="sticky top-0 z-40 border-b border-gray-200 bg-white"
        aria-label="Main navigation"
      >
        <div className="mx-auto max-w-7xl px-4 sm:px-6">
          <div className="flex h-14 items-center gap-6 overflow-x-auto">
            <span className="flex-shrink-0 text-sm font-semibold text-indigo-700">
              Fashion RecSys
            </span>
            <div className="flex gap-1">
              {NAV_LINKS.map((link) => (
                <NavLink
                  key={link.to}
                  to={link.to}
                  end={link.end}
                  className={({ isActive }) =>
                    `rounded-md px-3 py-1.5 text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
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
          </div>
        </div>
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
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Shell>
          <Suspense fallback={<PageFallback />}>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/customers" element={<CustomerExplorer />} />
              <Route
                path="/customers/:customerId/funnel"
                element={<CandidateFunnel />}
              />
              <Route path="/custom" element={<CustomCustomer />} />
              <Route path="/lab" element={<ModelLab />} />
              <Route path="/simulator" element={<ABSimulator />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </Suspense>
        </Shell>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
