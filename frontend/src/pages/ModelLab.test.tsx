import { describe, it, expect } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router'
import { server } from '../test/setup'
import { http, HttpResponse } from 'msw'
import ModelLab from './ModelLab'

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

describe('ModelLab page', () => {
  it('renders loading state for ablations', () => {
    server.use(
      http.get('/api/v1/insights/ablations', () => new Promise(() => {})),
      http.get('/api/v1/insights/importance', () => new Promise(() => {})),
      http.get('/api/v1/insights/segments', () => new Promise(() => {})),
    )
    render(<ModelLab />, { wrapper })
    // Multiple spinners for multiple queries
    const spinners = screen.getAllByRole('status')
    expect(spinners.length).toBeGreaterThan(0)
  })

  it('renders ablation table on success', async () => {
    render(<ModelLab />, { wrapper })
    await waitFor(() => {
      // The "candidate" group appears in the table
      const rows = screen.getAllByRole('row')
      const hasCandidate = rows.some((r) => r.textContent?.toLowerCase().includes('candidate'))
      expect(hasCandidate).toBe(true)
    })
    // article and interaction groups also present
    const allText = document.body.textContent ?? ''
    expect(allText).toContain('article')
    expect(allText).toContain('interaction')
  })

  it('renders importance table on success', async () => {
    render(<ModelLab />, { wrapper })
    await waitFor(() => {
      expect(screen.getByText(/Sales last week/i)).toBeTruthy()
    })
  })

  it('renders error state for ablations', async () => {
    server.use(
      http.get('/api/v1/insights/ablations', () =>
        HttpResponse.json(
          { error: { code: 'ERR', message: 'Ablation data unavailable', request_id: 'req-abl' } },
          { status: 500 },
        ),
      ),
    )
    render(<ModelLab />, { wrapper })
    await waitFor(() => {
      expect(screen.getByText(/ablation data unavailable/i)).toBeTruthy()
    })
    expect(screen.getByText(/req-abl/)).toBeTruthy()
  })

  it('renders empty segment state', async () => {
    server.use(
      http.get('/api/v1/insights/segments', () => HttpResponse.json([])),
    )
    render(<ModelLab />, { wrapper })
    await waitFor(() => {
      expect(screen.getByText(/no segment data available/i)).toBeTruthy()
    })
  })
})
