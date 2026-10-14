import { createContext, useContext, useState, useEffect, type ReactNode } from 'react'

export interface CartItem {
  article_idx: number
  article_id: string
  prod_name: string
  product_group: string
  colour: string
  image_url: string
}

interface CartContextValue {
  items: CartItem[]
  addItem: (item: CartItem) => void
  removeItem: (article_idx: number) => void
  clearCart: () => void
  totalItems: number
}

const STORAGE_KEY = 'fashion_demo_cart'

const CartContext = createContext<CartContextValue | null>(null)

export function CartProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<CartItem[]>(() => {
    try {
      const stored = sessionStorage.getItem(STORAGE_KEY)
      if (stored) {
        return JSON.parse(stored) as CartItem[]
      }
    } catch {
      // ignore parse errors
    }
    return []
  })

  useEffect(() => {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(items))
    } catch {
      // ignore storage errors
    }
  }, [items])

  const addItem = (item: CartItem) => {
    setItems((prev) => {
      if (prev.some((i) => i.article_idx === item.article_idx)) return prev
      return [...prev, item]
    })
  }

  const removeItem = (article_idx: number) => {
    setItems((prev) => prev.filter((i) => i.article_idx !== article_idx))
  }

  const clearCart = () => {
    setItems([])
  }

  return (
    <CartContext.Provider
      value={{ items, addItem, removeItem, clearCart, totalItems: items.length }}
    >
      {children}
    </CartContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export function useCart(): CartContextValue {
  const ctx = useContext(CartContext)
  if (!ctx) throw new Error('useCart must be used inside CartProvider')
  return ctx
}
