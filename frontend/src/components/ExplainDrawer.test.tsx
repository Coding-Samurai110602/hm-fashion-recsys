import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router'
import { server } from '../test/setup'
import { http, HttpResponse } from 'msw'
import { ExplainDrawer } from './ExplainDrawer'

const CUSTOMER_ID = 'a73eaaa3b5e5c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2'
const ARTICLE_ID = '0108775015'

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

describe('ExplainDrawer', () => {
  it('renders loading state', () => {
    server.use(
      http.get(
        `/api/v1/customers/${CUSTOMER_ID}/explain/${ARTICLE_ID}`,
        () => new Promise(() => {}),
      ),
    )
    render(
      <ExplainDrawer
        customerId={CUSTOMER_ID}
        articleId={ARTICLE_ID}
        articleName="Strindberg Boot"
        onClose={vi.fn()}
      />,
      { wrapper },
    )
    // Loading spinner has role=status
    expect(screen.getByRole('status')).toBeTruthy()
  })

  it('renders reason chips from ExplainResult fixture', async () => {
    render(
      <ExplainDrawer
        customerId={CUSTOMER_ID}
        articleId={ARTICLE_ID}
        articleName="Strindberg Boot"
        onClose={vi.fn()}
      />,
      { wrapper },
    )
    await waitFor(() => {
      expect(
        screen.getByText(/you bought this exact item 9 days ago/i),
      ).toBeTruthy()
    })
    expect(
      screen.getByText(/garment category you've bought from before/i),
    ).toBeTruthy()
  })

  it('renders SHAP contrast chart with positive and negative values', async () => {
    render(
      <ExplainDrawer
        customerId={CUSTOMER_ID}
        articleId={ARTICLE_ID}
        articleName="Strindberg Boot"
        onClose={vi.fn()}
      />,
      { wrapper },
    )
    await waitFor(() => {
      expect(
        screen.getByLabelText(/shap contrast bar chart/i),
      ).toBeTruthy()
    })
  })

  it('shows model score', async () => {
    render(
      <ExplainDrawer
        customerId={CUSTOMER_ID}
        articleId={ARTICLE_ID}
        articleName="Strindberg Boot"
        onClose={vi.fn()}
      />,
      { wrapper },
    )
    await waitFor(() => {
      expect(screen.getByText(/model score/i)).toBeTruthy()
    })
    expect(screen.getByText('2.8340')).toBeTruthy()
  })

  it('renders error state', async () => {
    server.use(
      http.get(
        `/api/v1/customers/${CUSTOMER_ID}/explain/${ARTICLE_ID}`,
        () =>
          HttpResponse.json(
            { error: { code: 'ERR', message: 'Explain failed', request_id: 'req-exp' } },
            { status: 500 },
          ),
      ),
    )
    render(
      <ExplainDrawer
        customerId={CUSTOMER_ID}
        articleId={ARTICLE_ID}
        articleName="Strindberg Boot"
        onClose={vi.fn()}
      />,
      { wrapper },
    )
    await waitFor(() => {
      expect(screen.getByText(/explain failed/i)).toBeTruthy()
    })
    expect(screen.getByText(/req-exp/)).toBeTruthy()
  })
})
