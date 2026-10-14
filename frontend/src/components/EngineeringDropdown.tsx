import { useState, useRef, useEffect, useCallback } from 'react'
import { NavLink, useLocation } from 'react-router'

const ENG_ITEMS = [
  { label: 'Model Lab', to: '/engineering/lab' },
  { label: 'A/B Simulator', to: '/engineering/simulator' },
  { label: 'System', to: '/engineering/system' },
]

export function EngineeringDropdown() {
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const itemRefs = useRef<(HTMLAnchorElement | null)[]>([])
  const { pathname } = useLocation()

  // Close when route changes (after navigation completes, not during click)
  useEffect(() => {
    setOpen(false)
  }, [pathname])

  useEffect(() => {
    function onClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    if (open) {
      document.addEventListener('mousedown', onClickOutside)
    }
    return () => document.removeEventListener('mousedown', onClickOutside)
  }, [open])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setOpen(false)
      return
    }
    if (!open) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      const focused = document.activeElement
      const idx = itemRefs.current.findIndex((el) => el === focused)
      const next = idx < ENG_ITEMS.length - 1 ? idx + 1 : 0
      itemRefs.current[next]?.focus()
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault()
      const focused = document.activeElement
      const idx = itemRefs.current.findIndex((el) => el === focused)
      const prev = idx > 0 ? idx - 1 : ENG_ITEMS.length - 1
      itemRefs.current[prev]?.focus()
    }
  }, [open])

  const isEngRoute =
    typeof window !== 'undefined' &&
    window.location.pathname.startsWith('/engineering')

  return (
    <div ref={containerRef} className="relative" onKeyDown={handleKeyDown}>
      <button
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="true"
        aria-expanded={open}
        aria-label="Engineering menu"
        className={`flex items-center gap-1 rounded-md px-3 py-1.5 text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
          isEngRoute
            ? 'bg-indigo-50 font-medium text-indigo-700'
            : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
        }`}
      >
        Engineering
        <svg
          className={`h-4 w-4 transition-transform ${open ? 'rotate-180' : ''}`}
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full z-50 mt-1 w-44 rounded-xl border border-gray-200 bg-white py-1 shadow-lg"
          aria-label="Engineering sub-pages"
        >
          {ENG_ITEMS.map((item, i) => (
            <NavLink
              key={item.to}
              to={item.to}
              role="menuitem"
              ref={(el) => { itemRefs.current[i] = el }}
              className={({ isActive }) =>
                `block px-4 py-2 text-sm transition-colors ${
                  isActive
                    ? 'bg-indigo-50 font-medium text-indigo-700'
                    : 'text-gray-700 hover:bg-gray-50'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </div>
      )}
    </div>
  )
}
