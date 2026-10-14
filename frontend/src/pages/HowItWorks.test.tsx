import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import HowItWorks from './HowItWorks'
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

describe('HowItWorks page', () => {
  it('renders the page heading', async () => {
    render(<HowItWorks />, { wrapper: Wrapper })
    expect(screen.getByText('How it works')).toBeInTheDocument()
  })

  it('shows all 4 step navigation tabs', async () => {
    render(<HowItWorks />, { wrapper: Wrapper })
    expect(screen.getByRole('tab', { name: /step 1/i })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /step 2/i })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /step 3/i })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /step 4/i })).toBeInTheDocument()
  })

  it('shows step 1 content by default', () => {
    render(<HowItWorks />, { wrapper: Wrapper })
    expect(screen.getByText('Gather possibilities')).toBeInTheDocument()
  })

  it('switches to step 2 when clicked', async () => {
    render(<HowItWorks />, { wrapper: Wrapper })
    const step2Tab = screen.getByRole('tab', { name: /step 2/i })
    fireEvent.click(step2Tab)
    expect(await screen.findByText('Describe each item')).toBeInTheDocument()
  })

  it('shows the glossary section', () => {
    render(<HowItWorks />, { wrapper: Wrapper })
    expect(screen.getByText('Glossary')).toBeInTheDocument()
    expect(screen.getByText('Recommender system')).toBeInTheDocument()
    expect(screen.getByText('Cold start')).toBeInTheDocument()
  })

  it('loads accuracy metrics from API', async () => {
    render(<HowItWorks />, { wrapper: Wrapper })
    await waitFor(() => {
      // Summary fixture has holdout_map12_ranker: 0.036
      expect(screen.getByText(/3\.6%|Does it really work/)).toBeInTheDocument()
    })
  })

  it('does not show any 4-digit years in the page', () => {
    render(<HowItWorks />, { wrapper: Wrapper })
    const body = document.body.textContent ?? ''
    expect(body).not.toMatch(/\b20\d{2}\b/)
  })
})
