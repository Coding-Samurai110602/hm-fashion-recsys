import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import StoreTrends from './StoreTrends'
import { ProfileProvider } from '../contexts/ProfileContext'
import { CartProvider } from '../contexts/CartContext'

function makeQC() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
}

function Wrapper({ children }: { children: React.ReactNode }) {
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

describe('StoreTrends page', () => {
  it('renders the page heading', () => {
    render(<StoreTrends />, { wrapper: Wrapper })
    expect(screen.getByText('Store Trends')).toBeInTheDocument()
  })

  it('shows trending now section heading', () => {
    render(<StoreTrends />, { wrapper: Wrapper })
    expect(screen.getByText('Trending now')).toBeInTheDocument()
  })

  it('loads trending items from the API', async () => {
    render(<StoreTrends />, { wrapper: Wrapper })
    await waitFor(
      () => {
        // The trending_now section renders TrendCard items with rank badges
        // At least one rank badge should be visible
        const rankBadges = screen.getAllByText(/#1/)
        expect(rankBadges.length).toBeGreaterThan(0)
      },
      { timeout: 5000 },
    )
  })

  it('shows age group tab buttons', () => {
    render(<StoreTrends />, { wrapper: Wrapper })
    expect(screen.getByRole('tab', { name: /under 25/i })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /25-34/i })).toBeInTheDocument()
  })

  it('switches age tab when clicked', async () => {
    render(<StoreTrends />, { wrapper: Wrapper })
    const tab2534 = screen.getByRole('tab', { name: '25-34' })
    fireEvent.click(tab2534)
    expect(tab2534).toHaveAttribute('aria-selected', 'true')
  })

  it('shows popularity decay chart section', async () => {
    render(<StoreTrends />, { wrapper: Wrapper })
    expect(screen.getByText(/How long do items stay popular/)).toBeInTheDocument()
    expect(screen.getByText(/Fashion moves fast/)).toBeInTheDocument()
  })

  it('shows top categories section', () => {
    render(<StoreTrends />, { wrapper: Wrapper })
    expect(screen.getByText('Top categories')).toBeInTheDocument()
  })

  it('does not show any week numbers or calendar dates', async () => {
    render(<StoreTrends />, { wrapper: Wrapper })
    await waitFor(() => {
      expect(screen.queryByText(/week 10\d/i)).toBeNull()
    })
    const body = document.body.textContent ?? ''
    expect(body).not.toMatch(/\b20\d{2}\b/)
  })
})
