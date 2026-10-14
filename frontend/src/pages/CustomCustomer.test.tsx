import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router'
import { server } from '../test/setup'
import { http, HttpResponse } from 'msw'
import CustomCustomer from './CustomCustomer'

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

describe('CustomCustomer page', () => {
  it('renders initial empty state', () => {
    render(<CustomCustomer />, { wrapper })
    expect(screen.getByText(/search for articles/i)).toBeTruthy()
  })

  it('shows search input and date picker', () => {
    render(<CustomCustomer />, { wrapper })
    expect(screen.getByPlaceholderText(/search article name/i)).toBeTruthy()
    expect(screen.getByDisplayValue('2020-09-22')).toBeTruthy()
  })

  it('shows search results after debounce', async () => {
    render(<CustomCustomer />, { wrapper })
    const input = screen.getByPlaceholderText(/search article name/i)
    fireEvent.change(input, { target: { value: 'boot' } })
    // Wait for debounce (300ms) + fetch
    await act(async () => {
      await new Promise((r) => setTimeout(r, 350))
    })
    await waitFor(() => {
      expect(screen.getByText(/Strindberg Boot/i)).toBeTruthy()
    }, { timeout: 2000 })
  })

  it('shows history section after adding article', async () => {
    render(<CustomCustomer />, { wrapper })
    const input = screen.getByPlaceholderText(/search article name/i)
    fireEvent.change(input, { target: { value: 'boot' } })
    await act(async () => {
      await new Promise((r) => setTimeout(r, 350))
    })
    await waitFor(() => {
      const items = screen.getAllByText(/Strindberg Boot/i)
      expect(items.length).toBeGreaterThan(0)
    }, { timeout: 2000 })
    fireEvent.click(screen.getAllByText(/Strindberg Boot/i)[0]!)
    await waitFor(() => {
      const histItems = screen.getAllByText(/purchase history/i)
      expect(histItems.length).toBeGreaterThan(0)
    })
  })

  it('shows date validation error for future date', async () => {
    render(<CustomCustomer />, { wrapper })
    const dateInput = screen.getByDisplayValue('2020-09-22')
    fireEvent.change(dateInput, { target: { value: '2021-01-01' } })
    await waitFor(() => {
      expect(screen.getByText(/must be on or before/i)).toBeTruthy()
    })
  })

  it('shows recommendations after submitting history', async () => {
    render(<CustomCustomer />, { wrapper })
    const input = screen.getByPlaceholderText(/search article name/i)
    fireEvent.change(input, { target: { value: 'boot' } })
    await act(async () => {
      await new Promise((r) => setTimeout(r, 350))
    })
    await waitFor(() => {
      expect(screen.getByText(/Strindberg Boot/i)).toBeTruthy()
    }, { timeout: 2000 })
    fireEvent.click(screen.getByText(/Strindberg Boot/i))
    await waitFor(() => {
      expect(screen.getByText(/get recommendations/i)).toBeTruthy()
    })
    fireEvent.click(screen.getByText(/get recommendations/i))
    await waitFor(() => {
      // Synthetic fixture returns "Article X" product names
      const items = screen.getAllByText(/Article \d+/i)
      expect(items.length).toBeGreaterThan(0)
    }, { timeout: 3000 })
  })

  it('shows API error for custom recommendations', async () => {
    server.use(
      http.post('/api/v1/recommendations/custom', () =>
        HttpResponse.json(
          { error: { code: 'ERR', message: 'Custom recs failed', request_id: 'req-cust' } },
          { status: 500 },
        ),
      ),
    )
    render(<CustomCustomer />, { wrapper })
    const input = screen.getByPlaceholderText(/search article name/i)
    fireEvent.change(input, { target: { value: 'boot' } })
    await act(async () => {
      await new Promise((r) => setTimeout(r, 350))
    })
    await waitFor(() => {
      expect(screen.getByText(/Strindberg Boot/i)).toBeTruthy()
    }, { timeout: 2000 })
    fireEvent.click(screen.getByText(/Strindberg Boot/i))
    await waitFor(() => {
      expect(screen.getByText(/get recommendations/i)).toBeTruthy()
    })
    fireEvent.click(screen.getByText(/get recommendations/i))
    await waitFor(() => {
      expect(screen.getByText(/custom recs failed/i)).toBeTruthy()
    }, { timeout: 3000 })
    expect(screen.getByText(/req-cust/)).toBeTruthy()
  })
})
