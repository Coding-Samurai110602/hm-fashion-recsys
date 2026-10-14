import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Shoppers from './Shoppers'
import { ProfileProvider } from '../contexts/ProfileContext'
import { CartProvider } from '../contexts/CartContext'

const PROFILE_ID = 'prof-test-001'

function makeQC() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
}

function Wrapper({
  children,
  withProfile = false,
}: {
  children: React.ReactNode
  withProfile?: boolean
}) {
  if (withProfile) {
    localStorage.setItem(
      'fashion_demo_profile',
      JSON.stringify({
        id: PROFILE_ID,
        nickname: 'TestUser',
        age_range: '25-34',
        channel: 'online',
        purchase_count: 3,
      }),
    )
  }
  return (
    <QueryClientProvider client={makeQC()}>
      <ProfileProvider>
        <CartProvider>
          <MemoryRouter>{children}</MemoryRouter>
        </CartProvider>
      </ProfileProvider>
    </QueryClientProvider>
  )
}

beforeEach(() => {
  localStorage.clear()
})

describe('Shoppers page', () => {
  it('renders the page heading', () => {
    render(<Shoppers />, { wrapper: Wrapper })
    expect(screen.getByText('Shoppers')).toBeInTheDocument()
  })

  it('shows "Your profile" section', () => {
    render(<Shoppers />, { wrapper: Wrapper })
    expect(screen.getByText('Your profile')).toBeInTheDocument()
  })

  it('shows message when no profile', () => {
    render(<Shoppers />, { wrapper: Wrapper })
    expect(screen.getByText('No profile yet.')).toBeInTheDocument()
  })

  it('shows profile card when profile exists', async () => {
    localStorage.setItem(
      'fashion_demo_profile',
      JSON.stringify({
        id: PROFILE_ID,
        nickname: 'TestUser',
        age_range: '25-34',
        channel: 'online',
        purchase_count: 3,
      }),
    )
    render(<Shoppers />, {
      wrapper: ({ children }) => (
        <QueryClientProvider client={makeQC()}>
          <ProfileProvider>
            <CartProvider>
              <MemoryRouter>{children}</MemoryRouter>
            </CartProvider>
          </ProfileProvider>
        </QueryClientProvider>
      ),
    })
    expect(await screen.findByText('TestUser')).toBeInTheDocument()
  })

  it('shows real shoppers section', () => {
    render(<Shoppers />, { wrapper: Wrapper })
    expect(screen.getByText('Real shoppers')).toBeInTheDocument()
  })

  it('loads demo shopper cards from the API', async () => {
    render(<Shoppers />, { wrapper: Wrapper })
    // Wait for demo shoppers to load — MSW returns 5 shoppers
    await waitFor(
      () => {
        // The ShopperCard renders age-range text like "Under 25" or "25-34"
        expect(screen.getByText('Under 25')).toBeInTheDocument()
      },
      { timeout: 5000 },
    )
  })

  it('shows "See your picks" link when profile exists', async () => {
    localStorage.setItem(
      'fashion_demo_profile',
      JSON.stringify({
        id: PROFILE_ID,
        nickname: 'TestUser',
        age_range: '25-34',
        channel: 'online',
        purchase_count: 3,
      }),
    )
    render(<Shoppers />, {
      wrapper: ({ children }) => (
        <QueryClientProvider client={makeQC()}>
          <ProfileProvider>
            <CartProvider>
              <MemoryRouter>{children}</MemoryRouter>
            </CartProvider>
          </ProfileProvider>
        </QueryClientProvider>
      ),
    })
    expect(await screen.findByRole('link', { name: /see your picks/i })).toBeInTheDocument()
  })
})
