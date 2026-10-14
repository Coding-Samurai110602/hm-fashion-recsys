import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router'
import { server } from '../test/setup'
import { errorHandlers } from '../test/handlers'
import { http } from 'msw'
import Overview from './Overview'

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return (
    <QueryClientProvider client={qc}>
      <MemoryRouter>{children}</MemoryRouter>
    </QueryClientProvider>
  )
}

describe('Overview page', () => {
  it('renders loading state', () => {
    server.use(
      http.get('/api/v1/insights/summary', () => new Promise(() => {})),
      http.get('/api/v1/insights/weekly-lift', () => new Promise(() => {})),
    )
    render(<Overview />, { wrapper })
    // Loading spinners (multiple possible for summary + lift)
    const spinners = screen.getAllByRole('status')
    expect(spinners.length).toBeGreaterThan(0)
  })

  it('renders success state with fixture data', async () => {
    render(<Overview />, { wrapper })
    await waitFor(() => {
      // holdout_map12_ranker 0.036 → displayed as 0.036 (fmtNum with 4 decimals)
      const cards = screen.getAllByText(/0\.036/)
      expect(cards.length).toBeGreaterThan(0)
    })
    // holdout_lift_pct 45 → +45.0%
    expect(screen.getAllByText(/\+45/).length).toBeGreaterThan(0)
  })

  it('renders error state with API message and request_id', async () => {
    server.use(errorHandlers.summary)
    render(<Overview />, { wrapper })
    await waitFor(() => {
      expect(screen.getByText(/server error/i)).toBeTruthy()
    })
    expect(screen.getByText(/req-test-123/i)).toBeTruthy()
  })

  it('renders pipeline diagram', () => {
    render(<Overview />, { wrapper })
    // Multiple elements may have "LightGBM LambdaRank"
    const elements = screen.getAllByText(/LightGBM LambdaRank/i)
    expect(elements.length).toBeGreaterThan(0)
    expect(screen.getByText(/Pipeline Architecture/i)).toBeTruthy()
  })
})
