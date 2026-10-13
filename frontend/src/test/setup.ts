import { setupServer } from 'msw/node'
import { handlers } from './handlers'
import '@testing-library/jest-dom'

// Polyfill ResizeObserver for jsdom (used by Recharts ResponsiveContainer)
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// Set up MSW node server
export const server = setupServer(...handlers)

// Start before each test
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))

// Reset handlers after each test so tests don't bleed into each other
afterEach(() => server.resetHandlers())

// Close after all tests
afterAll(() => server.close())
