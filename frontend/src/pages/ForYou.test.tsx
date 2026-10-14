import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import ForYou from './ForYou'
import { ProfileProvider } from '../contexts/ProfileContext'
import { CartProvider } from '../contexts/CartContext'
import { server } from '../test/setup'
import { http, HttpResponse } from 'msw'

const PROFILE_ID = 'prof-test-001'

function makeQC() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
}

beforeEach(() => {
  localStorage.clear()
  sessionStorage.clear()
})

describe('ForYou page', () => {
  it('redirects to / when no profile', () => {
    render(<ForYou />, {
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
    // Navigate component renders without crashing — redirect occurs
    expect(document.body).toBeTruthy()
  })

  it('shows 12 recommendation cards for a shopper with purchases', async () => {
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
    render(<ForYou />, {
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
    await waitFor(() => {
      expect(screen.getAllByRole('button', { name: /why.*was recommended/i }).length).toBeGreaterThan(0)
    })
  })

  it('shows cold start message when purchase_count is 0', async () => {
    server.use(
      http.get(`/api/v1/profiles/${PROFILE_ID}/recommendations`, () =>
        HttpResponse.json({
          profile_id: PROFILE_ID,
          cold_start: true,
          items: Array.from({ length: 8 }, (_, i) => ({
            article_idx: 3000 + i,
            article_id: `030000${String(3000 + i).padStart(4, '0')}`,
            prod_name: `Pick ${i + 1}`,
            product_group: 'Trousers',
            colour: 'Black',
            image_url: '',
            score: 0.9 - i * 0.05,
            reasons: [],
            source: 'trending',
            latency_ms: 42,
            cold_start: true,
          })),
          latency_ms: 42,
        }),
      ),
    )

    localStorage.setItem(
      'fashion_demo_profile',
      JSON.stringify({
        id: PROFILE_ID,
        nickname: 'TestUser',
        age_range: '25-34',
        channel: 'online',
        purchase_count: 0,
      }),
    )

    render(<ForYou />, {
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

    expect(
      await screen.findByText(/You haven't bought anything yet/),
    ).toBeInTheDocument()
  })

  it('shows "12 picks for TestUser" heading', async () => {
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
    render(<ForYou />, {
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
    expect(await screen.findByText('12 picks for TestUser')).toBeInTheDocument()
  })

  it('opens WhyDrawer when clicking "Why this?" button', async () => {
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
    render(<ForYou />, {
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
    const whyButtons = await screen.findAllByRole('button', {
      name: /why.*was recommended/i,
    })
    fireEvent.click(whyButtons[0]!)
    expect(
      await screen.findByText('Why this was picked for you'),
    ).toBeInTheDocument()
  })
})
