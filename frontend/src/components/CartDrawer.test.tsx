import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, beforeEach } from 'vitest'
import { MemoryRouter } from 'react-router'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { CartDrawer } from './CartDrawer'
import { ProfileProvider } from '../contexts/ProfileContext'
import { CartProvider } from '../contexts/CartContext'

const PROFILE_ID = 'prof-test-001'

function makeQC() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } })
}

function WrapperWithItems({ children }: { children: React.ReactNode }) {
  localStorage.setItem(
    'fashion_demo_profile',
    JSON.stringify({
      id: PROFILE_ID,
      nickname: 'TestUser',
      age_range: '25-34',
      channel: 'online',
      purchase_count: 3,
    }),
  )
  sessionStorage.setItem(
    'fashion_demo_cart',
    JSON.stringify([
      {
        article_idx: 2000,
        article_id: '0200002000',
        prod_name: 'Test Trousers',
        product_group: 'Trousers',
        colour: 'Black',
        image_url: '',
      },
    ]),
  )
  return (
    <QueryClientProvider client={makeQC()}>
      <ProfileProvider>
        <CartProvider>
          <MemoryRouter>{children}</MemoryRouter>
        </CartProvider>
      </ProfileProvider>
    </QueryClientProvider>
  )
}

function EmptyWrapper({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={makeQC()}>
      <ProfileProvider>
        <CartProvider>
          <MemoryRouter>{children}</MemoryRouter>
        </CartProvider>
      </ProfileProvider>
    </QueryClientProvider>
  )
}

beforeEach(() => {
  localStorage.clear()
  sessionStorage.clear()
})

describe('CartDrawer', () => {
  it('renders when open=true', () => {
    render(<CartDrawer open={true} onClose={() => {}} />, { wrapper: EmptyWrapper })
    expect(screen.getByRole('dialog', { name: /shopping cart/i })).toBeInTheDocument()
  })

  it('does not render when open=false', () => {
    render(<CartDrawer open={false} onClose={() => {}} />, { wrapper: EmptyWrapper })
    expect(screen.queryByRole('dialog', { name: /shopping cart/i })).toBeNull()
  })

  it('shows empty state when cart is empty', () => {
    render(<CartDrawer open={true} onClose={() => {}} />, { wrapper: EmptyWrapper })
    expect(screen.getByText(/your cart is empty/i)).toBeInTheDocument()
  })

  it('shows cart item when item is in cart', () => {
    render(<CartDrawer open={true} onClose={() => {}} />, { wrapper: WrapperWithItems })
    expect(screen.getByText('Test Trousers')).toBeInTheDocument()
  })

  it('removes item when remove button clicked', async () => {
    render(<CartDrawer open={true} onClose={() => {}} />, { wrapper: WrapperWithItems })
    const removeBtn = screen.getByRole('button', { name: /remove test trousers from cart/i })
    fireEvent.click(removeBtn)
    await waitFor(() => {
      expect(screen.queryByText('Test Trousers')).toBeNull()
    })
  })

  it('disables checkout button when cart is empty', () => {
    render(<CartDrawer open={true} onClose={() => {}} />, { wrapper: EmptyWrapper })
    const checkoutBtn = screen.getByRole('button', { name: /checkout/i })
    expect(checkoutBtn).toBeDisabled()
  })

  it('closes when backdrop is clicked', () => {
    const onClose = vi.fn()
    render(<CartDrawer open={true} onClose={onClose} />, { wrapper: EmptyWrapper })
    // Backdrop has aria-hidden=true, click it
    const backdrop = document.querySelector('[aria-hidden="true"]') as HTMLElement
    if (backdrop) fireEvent.click(backdrop)
    expect(onClose).toHaveBeenCalled()
  })

  it('triggers checkout API call when checkout clicked with items', async () => {
    render(<CartDrawer open={true} onClose={() => {}} />, { wrapper: WrapperWithItems })
    const checkoutBtn = screen.getByRole('button', { name: /checkout/i })
    fireEvent.click(checkoutBtn)
    // Should show loading state
    await waitFor(() => {
      expect(
        screen.queryByText(/saving purchases/i) || screen.queryByText(/purchases saved/i),
      ).toBeTruthy()
    })
  })
})
