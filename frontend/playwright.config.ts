import { defineConfig, devices } from '@playwright/test'

/**
 * E2E smoke tests against the synthetic-bundle API + Vite dev server.
 *
 * Setup (one-time):
 *   npm install && npx playwright install --with-deps chromium
 *
 * Run (requires both servers):
 *   # Terminal 1: start API with synthetic bundle
 *   BUNDLE_PATH=artifacts/bundle_synthetic make api
 *   # Terminal 2: start Vite dev server
 *   make frontend-dev
 *   # Terminal 3: run tests
 *   cd frontend && npm run e2e
 *
 * These tests join CI in Part 2B.
 */
export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env['CI'],
  retries: process.env['CI'] ? 1 : 0,
  workers: process.env['CI'] ? 1 : undefined,
  timeout: 90000,
  reporter: [['list'], ['html', { outputFolder: 'playwright-report', open: 'never' }]],
  use: {
    // Override with BASE_URL env var when running against the containerised stack
    // (e.g. BASE_URL=http://localhost:8080 npm run e2e)
    baseURL: process.env['BASE_URL'] ?? 'http://localhost:5173',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium-desktop',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1440, height: 900 },
      },
    },
    {
      name: 'chromium-mobile',
      use: {
        ...devices['Pixel 5'],
        viewport: { width: 390, height: 844 },
      },
    },
  ],
  // outputDir is for Playwright failure artifacts (traces, failure screenshots).
  // Explicit page.screenshot() calls use docs/screenshots/ — keep separate.
  outputDir: 'playwright-output',
})
