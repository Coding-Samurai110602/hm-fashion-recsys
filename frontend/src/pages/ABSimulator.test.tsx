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

  it('lift slider label shows percentage and default is +10%', () => {
    render(<ABSimulator />, { wrapper })
    // Default relativeLift = 0.1 → label shows "+10%"
    expect(screen.getByText(/\+10%/)).toBeTruthy()
  })

  it('calcPower sends relative_lift as fraction (0.1), not multiplier', async () => {
    const capturedBodies: unknown[] = []
    server.use(
      http.post('/api/v1/experiment/power', async ({ request }) => {
        capturedBodies.push(await request.json())
        return HttpResponse.json({
          n_per_arm: 6755,
          n_total: 13510,
          n_per_arm_statsmodels: 6747,
          n_per_arm_cuped: null,
          weeks_needed: null,
          cohen_h: 0.03,
        })
      }),
    )
    render(<ABSimulator />, { wrapper })
    fireEvent.click(screen.getByText(/calculate power/i))
    await waitFor(() => expect(capturedBodies.length).toBe(1))
    const body = capturedBodies[0] as { relative_lift: number }
    expect(body.relative_lift).toBe(0.1)
    expect(body.relative_lift).toBeLessThanOrEqual(1.0)
  })

  it('renders total n in power result', async () => {
    render(<ABSimulator />, { wrapper })
    fireEvent.click(screen.getByText(/calculate power/i))
    await waitFor(() => {
      // n_total: 13510 from fixture
      expect(screen.getByText('13,510')).toBeTruthy()
    })
  })

  it('inline validation error shows code and request_id when API returns 422', async () => {
    server.use(errorHandlers.power)
    render(<ABSimulator />, { wrapper })
    fireEvent.click(screen.getByText(/calculate power/i))
    await waitFor(() => {
      expect(screen.getByText(/invalid parameters/i)).toBeTruthy()
    })
    // ErrorState renders code and request_id
    expect(screen.getByText(/req-422/)).toBeTruthy()
    // No generic "Internal Server Error" shown
    expect(screen.queryByText(/internal server error/i)).toBeNull()
  })
})
