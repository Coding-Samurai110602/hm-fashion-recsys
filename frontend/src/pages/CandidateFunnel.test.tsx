import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Route, Routes } from 'react-router'
import { server } from '../test/setup'
import { http, HttpResponse } from 'msw'
import CandidateFunnel from './CandidateFunnel'

const CUSTOMER_ID = 'a73eaaa3b5e5c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2'

function renderFunnel(customerId = CUSTOMER_ID) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <MemoryRouter initialEntries={[`/customers/${customerId}/funnel`]}>
        <Routes>
          <Route path="/customers/:customerId/funnel" element={<CandidateFunnel />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('CandidateFunnel page', () => {
  it('renders loading state', () => {
    renderFunnel()
    // The spinner renders two elements with the same text (sr-only + visible); use getAllBy
    expect(screen.getAllByText(/loading funnel/i).length).toBeGreaterThan(0)
  })

  it('renders success state with source counts', async () => {
    renderFunnel()
    // Wait for the page heading first
    expect(await screen.findByText(/candidate funnel/i)).toBeInTheDocument()
    // Source count cards and table both show "repurchase" — wait for at least one
    expect((await screen.findAllByText('repurchase')).length).toBeGreaterThan(0)
    expect((await screen.findAllByText('popularity last week')).length).toBeGreaterThan(0)
  })

  it('renders sortable table with final_rank column', async () => {
    renderFunnel()
    // "Final Rank" appears in the th and also in the chart heading; getAllByText is safe
    expect(await screen.findByRole('columnheader', { name: /final rank/i })).toBeInTheDocument()
    // The fixture has final_rank 1 and 2 shown in the table rows
    const cells = await screen.findAllByRole('cell')
    const cellTexts = cells.map((c) => c.textContent)
    expect(cellTexts).toContain('1')
    expect(cellTexts).toContain('2')
  })

  it('renders error state with API message and request_id', async () => {
    server.use(
      http.get(`/api/v1/customers/${CUSTOMER_ID}/funnel`, () =>
        HttpResponse.json(
          { error: { code: 'NOT_FOUND', message: 'Funnel data not found', request_id: 'req-funnel-err' } },
          { status: 404 },
        ),
      ),
    )
    renderFunnel()
    expect(await screen.findByText(/funnel data not found/i)).toBeInTheDocument()
    expect(await screen.findByText(/req-funnel-err/i)).toBeInTheDocument()
  })

  it('renders empty state when no candidates', async () => {
    server.use(
      http.get(`/api/v1/customers/${CUSTOMER_ID}/funnel`, () =>
        HttpResponse.json([]),
      ),
    )
    renderFunnel()
    expect(await screen.findByText(/no funnel data available/i)).toBeInTheDocument()
  })

  it('truncates customer ID in heading', async () => {
    renderFunnel()
    // The truncateId function shows first 8 + ... + last 8 for 64-char IDs
    expect(await screen.findByText(/a73eaaa3\.\.\.a9b0c1d2/i)).toBeInTheDocument()
  })
})
