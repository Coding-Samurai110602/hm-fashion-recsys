import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, beforeEach } from 'vitest'
import { GuidedTour } from './GuidedTour'

const TOUR_KEY = 'fashion_demo_tour_seen'

beforeEach(() => {
  localStorage.clear()
})

describe('GuidedTour', () => {
  it('renders the first step', () => {
    render(<GuidedTour />)
    expect(screen.getByText('Create your profile')).toBeInTheDocument()
  })

  it('shows 5 step indicator dots', () => {
    render(<GuidedTour />)
    // Step indicators (aria-hidden divs) + text "Step 1 of 5"
    expect(screen.getByText(/step 1 of 5/i)).toBeInTheDocument()
  })

  it('advances to next step on Next click', () => {
    render(<GuidedTour />)
    fireEvent.click(screen.getByRole('button', { name: /next/i }))
    expect(screen.getByText('Browse and shop')).toBeInTheDocument()
    expect(screen.getByText(/step 2 of 5/i)).toBeInTheDocument()
  })

  it('goes back to previous step on Back click', () => {
    render(<GuidedTour />)
    fireEvent.click(screen.getByRole('button', { name: /next/i }))
    fireEvent.click(screen.getByRole('button', { name: /back/i }))
    expect(screen.getByText('Create your profile')).toBeInTheDocument()
  })

  it('shows "Done" button on last step', () => {
    render(<GuidedTour />)
    // Click Next 4 times to reach step 5
    for (let i = 0; i < 4; i++) {
      fireEvent.click(screen.getByRole('button', { name: /next/i }))
    }
    expect(screen.getByRole('button', { name: /done/i })).toBeInTheDocument()
  })

  it('stores tour seen in localStorage on dismiss', () => {
    const onDismiss = vi.fn()
    render(<GuidedTour onDismiss={onDismiss} />)
    fireEvent.click(screen.getByRole('button', { name: /skip tour/i }))
    expect(localStorage.getItem(TOUR_KEY)).toBe('true')
    expect(onDismiss).toHaveBeenCalledOnce()
  })

  it('stores tour seen when Done is clicked on last step', () => {
    const onDismiss = vi.fn()
    render(<GuidedTour onDismiss={onDismiss} />)
    for (let i = 0; i < 4; i++) {
      fireEvent.click(screen.getByRole('button', { name: /next/i }))
    }
    fireEvent.click(screen.getByRole('button', { name: /done/i }))
    expect(localStorage.getItem(TOUR_KEY)).toBe('true')
  })
})
