import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    exclude: ['**/node_modules/**', '**/e2e/**'],
    pool: 'forks',
    poolOptions: {
      forks: {
        maxForks: 2,
      },
    },
    testTimeout: 15000,
  },
})
