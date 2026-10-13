import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router'
import { server } from '../test/setup'
import { errorHandlers } from '../test/handlers'
import { http, HttpResponse } from 'msw'
import CustomerExplorer from './CustomerExplorer'

const CUSTOMER_ID = 'a73eaaa3b5e5c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2'

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

describe('CustomerExplorer page', () => {
  it('renders loading state for sample customers', () => {
    server.use(http.get('/api/v1/customers/sample', () => new Promise(() => {})))
    render(<CustomerExplorer />, { wrapper })
    // spinner has role=status
    expect(screen.getByRole('status')).toBeTruthy()
  })

  it('renders sample customer list on success', async () => {
    render(<CustomerExplorer />, { wrapper })
    await waitFor(() => {
      // Look for the truncated customer ID button text (a73eaaa3 prefix)
      const btns = screen.getAllByRole('button')
      const found = btns.some((b) => b.textContent?.includes('a73eaaa3'))
      expect(found).toBe(true)
    })
  })

  it('renders error state for sample customers', async () => {
    server.use(
      http.get('/api/v1/customers/sample', () =>
        HttpResponse.json(
          { error: { code: 'ERR', message: 'Failed to sample', request_id: 'req-err' } },
          { status: 500 },
        ),
      ),
    )
    render(<CustomerExplorer />, { wrapper })
    await waitFor(() => {
      expect(screen.getByText(/failed to sample/i)).toBeTruthy()
    })
    expect(screen.getByText(/req-err/)).toBeTruthy()
  })

  it('shows empty state when no customer selected', () => {
    render(<CustomerExplorer />, { wrapper })
    expect(screen.getByText(/select a sample customer/i)).toBeTruthy()
  })

  it('customer ID truncation: 64-char hex → 8+...+8 display', async () => {
    render(<CustomerExplorer />, { wrapper })
    await waitFor(() => {
      const btns = screen.getAllByRole('button')
      const truncatedBtn = btns.find((b) => b.textContent?.includes('a73eaaa3'))
      expect(truncatedBtn).toBeTruthy()
      // The full ID is 64 chars, truncated form should have "..."
      expect(truncatedBtn?.textContent).toContain('...')
    })
  })

  it('shows recommendations after selecting customer', async () => {
    render(<CustomerExplorer />, { wrapper })
    // Wait for samples to load then click one
    await waitFor(() => {
      const btns = screen.getAllByRole('button')
      const sampleBtn = btns.find((b) => b.textContent?.includes('a73eaaa3'))
      expect(sampleBtn).toBeTruthy()
      fireEvent.click(sampleBtn!)
    })
    await waitFor(() => {
      // "Strindberg Boot" may appear multiple times (profile + recommendation)
      const items = screen.getAllByText(/Strindberg Boot/i)
      expect(items.length).toBeGreaterThan(0)
    })
  })

  it('shows error state for recommendations', async () => {
    server.use(errorHandlers.recommendations)
    render(<CustomerExplorer />, { wrapper })
    // Wait for samples then click
    let clicked = false
    await waitFor(() => {
      const btns = screen.getAllByRole('button')
      const sampleBtn = btns.find((b) => b.textContent?.includes('a73eaaa3'))
      if (sampleBtn && !clicked) {
        fireEvent.click(sampleBtn)
        clicked = true
      }
      expect(clicked).toBe(true)
    })
    await waitFor(() => {
      expect(screen.getByText(/customer not found/i)).toBeTruthy()
    }, { timeout: 3000 })
    expect(screen.getByText(/req-404/)).toBeTruthy()
  })

  it('has customer ID in input field', () => {
    render(<CustomerExplorer />, { wrapper })
    const input = screen.getByPlaceholderText(/64-character hex/i)
    expect(input).toBeTruthy()
    fireEvent.change(input, { target: { value: CUSTOMER_ID } })
    expect((input as HTMLInputElement).value).toBe(CUSTOMER_ID)
  })
})
