import { describe, it, expect } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router'
import { server } from '../test/setup'
import { errorHandlers } from '../test/handlers'
import { http, HttpResponse } from 'msw'
import ABSimulator from './ABSimulator'

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

describe('ABSimulator page', () => {
  it('renders initial state with controls', () => {
    render(<ABSimulator />, { wrapper })
    expect(screen.getByText(/power calculator/i)).toBeTruthy()
    expect(screen.getByText(/simulation runner/i)).toBeTruthy()
    expect(screen.getByText(/offline holdout/i)).toBeTruthy()
  })

  it('renders power result on success', async () => {
    render(<ABSimulator />, { wrapper })
    const calcBtn = screen.getByText(/calculate power/i)
    fireEvent.click(calcBtn)
    await waitFor(() => {
      // n_per_arm: 6755 (synthetic fixture value)
      expect(screen.getByText('6,755')).toBeTruthy()
    })
  })

  it('renders error state for power calculation', async () => {
    server.use(errorHandlers.power)
    render(<ABSimulator />, { wrapper })
    fireEvent.click(screen.getByText(/calculate power/i))
    await waitFor(() => {
      expect(screen.getByText(/invalid parameters/i)).toBeTruthy()
    })
    expect(screen.getByText(/req-422/)).toBeTruthy()
  })

  it('renders simulation result on success', async () => {
    render(<ABSimulator />, { wrapper })
    const runBtn = screen.getByText(/run simulation/i)
    fireEvent.click(runBtn)
    await waitFor(() => {
      // false_positive_rate: 0 → 0.0%
      expect(screen.getByText(/0\.0%/)).toBeTruthy()
    })
  })

  it('renders simulation error state', async () => {
    server.use(
      http.post('/api/v1/experiment/simulate', () =>
        HttpResponse.json(
          { error: { code: 'INTERNAL_ERROR', message: 'Simulation failed', request_id: 'req-500' } },
          { status: 500 },
        ),
      ),
    )
    render(<ABSimulator />, { wrapper })
    fireEvent.click(screen.getByText(/run simulation/i))
    await waitFor(() => {
      // "Simulation failed" may appear in button text + error message
      const items = screen.getAllByText(/simulation failed/i)
      expect(items.length).toBeGreaterThan(0)
    })
    expect(screen.getByText(/req-500/)).toBeTruthy()
  })

  it('peeking checkbox toggles peeking mode', () => {
    render(<ABSimulator />, { wrapper })
    const checkbox = screen.getByRole('checkbox', { name: /peeking simulation/i })
    expect((checkbox as HTMLInputElement).checked).toBe(false)
    fireEvent.click(checkbox)
    expect((checkbox as HTMLInputElement).checked).toBe(true)
  })
})
