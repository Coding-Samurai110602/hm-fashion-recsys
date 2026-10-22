import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import Engineering from './Engineering'

function wrapper({ children }: { children: React.ReactNode }) {
  return <MemoryRouter>{children}</MemoryRouter>
}

describe('Engineering hub page', () => {
  it('renders without crashing', () => {
    render(<Engineering />, { wrapper })
  })

  it('shows the heading "Engineering"', () => {
    render(<Engineering />, { wrapper })
    expect(screen.getByRole('heading', { name: /engineering/i })).toBeTruthy()
  })

  it('shows all three card titles', () => {
    render(<Engineering />, { wrapper })
    expect(screen.getByText('Model Lab')).toBeTruthy()
    expect(screen.getByText('A/B Simulator')).toBeTruthy()
    expect(screen.getByText('System')).toBeTruthy()
  })

  it('Model Lab card links to /engineering/lab', () => {
    render(<Engineering />, { wrapper })
    const link = screen.getByRole('link', { name: /model lab/i })
    expect(link.getAttribute('href')).toBe('/engineering/lab')
  })

  it('A/B Simulator card links to /engineering/simulator', () => {
    render(<Engineering />, { wrapper })
    const link = screen.getByRole('link', { name: /a\/b simulator/i })
    expect(link.getAttribute('href')).toBe('/engineering/simulator')
  })

  it('System card links to /engineering/system', () => {
    render(<Engineering />, { wrapper })
    const link = screen.getByRole('link', { name: /system/i })
    expect(link.getAttribute('href')).toBe('/engineering/system')
  })
})
