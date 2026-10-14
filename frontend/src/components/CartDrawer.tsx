import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router'
import { useCart } from '../contexts/CartContext'
import { useProfile } from '../contexts/ProfileContext'
import { checkout } from '../api/endpoints'
import { getImageUrl } from '../api/client'

interface Props {
  open: boolean
  onClose: () => void
}

export function CartDrawer({ open, onClose }: Props) {
  const { items, removeItem, clearCart } = useCart()
  const { profile, setProfile } = useProfile()
  const navigate = useNavigate()
  const [toast, setToast] = useState('')

  const checkoutMutation = useMutation({
    mutationFn: () => {
      if (!profile) throw new Error('No profile')
      return checkout(
        profile.id,
        items.map((i) => i.article_id),
      )
    },
    onSuccess: (data) => {
      if (profile) {
        setProfile({ ...profile, purchase_count: data.purchase_count })
      }
      clearCart()
      setToast('Purchases saved! Your recommendations are updated.')
      setTimeout(() => {
        setToast('')
        onClose()
        navigate('/for-you')
      }, 1800)
    },
  })

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true" aria-label="Shopping cart">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <div className="relative z-10 flex h-full w-full max-w-sm flex-col bg-white shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4">
          <h2 className="text-base font-semibold text-gray-900">
            Cart ({items.length})
          </h2>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            aria-label="Close cart"
          >
            <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>

        {/* Items */}
        <div className="flex-1 overflow-y-auto p-6">
          {items.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <div className="mb-3 text-4xl" aria-hidden="true">🛍</div>
              <p className="text-sm text-gray-500">Your cart is empty.</p>
              <p className="mt-1 text-xs text-gray-400">Browse the shop and add items.</p>
            </div>
          ) : (
            <ul className="space-y-3" aria-label="Cart items">
              {items.map((item) => (
                <li
                  key={item.article_idx}
                  className="flex items-center gap-3 rounded-xl border border-gray-100 bg-gray-50 p-3"
                >
                  <div className="h-14 w-14 flex-shrink-0 overflow-hidden rounded-lg bg-gray-200">
                    <img
                      src={getImageUrl(item.article_id)}
                      alt={`${item.prod_name} in ${item.colour}`}
                      className="h-full w-full object-cover"
                      onError={(e) => {
                        const target = e.currentTarget
                        target.style.display = 'none'
                        target.parentElement!.style.display = 'flex'
                        target.parentElement!.style.alignItems = 'center'
                        target.parentElement!.style.justifyContent = 'center'
                        target.parentElement!.textContent = item.prod_name.slice(0, 2).toUpperCase()
                      }}
                    />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-gray-800">
                      {item.prod_name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {item.product_group} · {item.colour}
                    </p>
                  </div>
                  <button
                    onClick={() => removeItem(item.article_idx)}
                    className="flex-shrink-0 rounded p-1 text-gray-400 hover:bg-red-50 hover:text-red-500 focus:outline-none focus:ring-2 focus:ring-red-400"
                    aria-label={`Remove ${item.prod_name} from cart`}
                  >
                    <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                      <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 p-6 space-y-3">
          {toast && (
            <div
              role="status"
              aria-live="polite"
              className="rounded-lg bg-green-50 px-4 py-3 text-sm text-green-700"
            >
              {toast}
            </div>
          )}

          {checkoutMutation.error && (
            <p className="text-sm text-red-600" role="alert">
              Checkout failed. Please try again.
            </p>
          )}

          <button
            onClick={() => checkoutMutation.mutate()}
            disabled={items.length === 0 || checkoutMutation.isPending || !profile}
            className="w-full rounded-xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-60"
          >
            {checkoutMutation.isPending ? 'Saving purchases…' : 'Checkout'}
          </button>

          <p className="text-center text-xs text-gray-400">
            Only purchases count — recommender systems learn from what people buy.
          </p>
        </div>
      </div>
    </div>
  )
}
