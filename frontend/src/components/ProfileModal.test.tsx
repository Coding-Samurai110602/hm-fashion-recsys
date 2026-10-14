import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ProfileModal } from './ProfileModal'
import { ProfileProvider } from '../contexts/ProfileContext'

function makeQC() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } })
}

function Wrapper({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={makeQC()}>
      <ProfileProvider>
        <MemoryRouter>{children}</MemoryRouter>
      </ProfileProvider>
    </QueryClientProvider>
  )
}

beforeEach(() => {
  localStorage.clear()
})

describe('ProfileModal', () => {
  it('renders the modal when open=true', () => {
    render(<ProfileModal open={true} />, { wrapper: Wrapper })
    expect(screen.getByText('Create your profile')).toBeInTheDocument()
  })

  it('does not render when open=false', () => {
    render(<ProfileModal open={false} />, { wrapper: Wrapper })
    expect(screen.queryByText('Create your profile')).toBeNull()
  })

  it('shows validation error for empty nickname', async () => {
    render(<ProfileModal open={true} />, { wrapper: Wrapper })
    fireEvent.click(screen.getByRole('button', { name: /start shopping/i }))
    expect(await screen.findByText(/please enter a nickname/i)).toBeInTheDocument()
  })

  it('shows validation error for nickname > 30 chars', async () => {
    render(<ProfileModal open={true} />, { wrapper: Wrapper })
    const input = screen.getByLabelText(/nickname/i)
    fireEvent.change(input, { target: { value: 'A'.repeat(31) } })
    fireEvent.click(screen.getByRole('button', { name: /start shopping/i }))
    expect(await screen.findByText(/30 characters or fewer/i)).toBeInTheDocument()
  })

  it('shows age range select with all options', () => {
    render(<ProfileModal open={true} />, { wrapper: Wrapper })
    const select = screen.getByLabelText(/age range/i)
    expect(select).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Under 25' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: '55 and over' })).toBeInTheDocument()
  })

  it('shows shopping preference radio buttons', () => {
    render(<ProfileModal open={true} />, { wrapper: Wrapper })
    expect(screen.getByLabelText(/I usually shop online/i)).toBeInTheDocument()
    expect(screen.getByLabelText(/I prefer shopping in store/i)).toBeInTheDocument()
  })

  it('shows close button when canDismiss=true', () => {
    render(<ProfileModal open={true} canDismiss={true} onClose={() => {}} />, { wrapper: Wrapper })
    expect(screen.getByRole('button', { name: /close dialog/i })).toBeInTheDocument()
  })

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn()
    render(<ProfileModal open={true} canDismiss={true} onClose={onClose} />, { wrapper: Wrapper })
    fireEvent.click(screen.getByRole('button', { name: /close dialog/i }))
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('submits profile when valid nickname entered', async () => {
    const onClose = vi.fn()
    render(<ProfileModal open={true} canDismiss={true} onClose={onClose} />, { wrapper: Wrapper })
    const input = screen.getByLabelText(/nickname/i)
    fireEvent.change(input, { target: { value: 'Alex' } })
    fireEvent.click(screen.getByRole('button', { name: /start shopping/i }))
    await waitFor(() => {
      expect(onClose).toHaveBeenCalled()
    })
  })
})
