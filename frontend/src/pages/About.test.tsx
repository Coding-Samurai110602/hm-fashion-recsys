import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router'
import { server } from '../test/setup'
import { http, HttpResponse } from 'msw'
import About from './About'

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

describe('About page', () => {
  it('renders loading state', () => {
    server.use(http.get('/version', () => new Promise(() => {})))
    render(<About />, { wrapper })
    // Multiple spinners possible (version + ready queries)
    const spinners = screen.getAllByRole('status')
    expect(spinners.length).toBeGreaterThan(0)
  })

  it('renders version info on success', async () => {
    render(<About />, { wrapper })
    await waitFor(() => {
      expect(screen.getByText('v1')).toBeTruthy()
    })
    expect(screen.getByText('abc1234d')).toBeTruthy() // truncated git commit
    expect(screen.getByText('104')).toBeTruthy() // bundle week
  })

  it('renders ready status', async () => {
    render(<About />, { wrapper })
    await waitFor(() => {
      // "Ready" appears in readiness section
      const readyText = screen.getAllByText(/ready/i)
      expect(readyText.length).toBeGreaterThan(0)
    })
  })

  it('renders error state for version fetch', async () => {
    server.use(
      http.get('/version', () =>
        HttpResponse.json(
          { error: { code: 'ERR', message: 'Version unavailable', request_id: 'req-ver' } },
          { status: 500 },
        ),
      ),
    )
    render(<About />, { wrapper })
    await waitFor(() => {
      expect(screen.getByText(/version unavailable/i)).toBeTruthy()
    })
    expect(screen.getByText(/req-ver/)).toBeTruthy()
  })

  it('renders static ML metrics table', () => {
    render(<About />, { wrapper })
    expect(screen.getByText(/holdout map@12/i)).toBeTruthy()
    expect(screen.getByText('0.036904')).toBeTruthy()
    expect(screen.getByText(/LightGBM LambdaRank/i)).toBeTruthy()
  })

  it('renders latency table', () => {
    render(<About />, { wrapper })
    expect(screen.getByText(/measured api latencies/i)).toBeTruthy()
    expect(screen.getByText('56ms')).toBeTruthy()
  })
})
