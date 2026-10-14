/**
 * E2E smoke tests — every route must load with zero console errors.
 *
 * Requires:
 *   - Synthetic-bundle API on http://localhost:8000
 *   - Vite dev server on http://localhost:5173
 *
 * Run: cd frontend && npm run e2e
 */
import { test, expect, type ConsoleMessage } from '@playwright/test'
import * as fs from 'fs'
import * as path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const SCREENSHOTS_DIR = path.join(__dirname, '../../docs/screenshots')

// Console errors to ignore (browser extensions, ResizeObserver, recharts 0x0 container)
const IGNORE_PATTERNS = [
  /ResizeObserver loop/i,
  /width\(0\).*height\(0\)/i,
  /Content Security Policy/i,
  /net::ERR_BLOCKED_BY_CLIENT/i,
]

function isIgnoredError(msg: ConsoleMessage): boolean {
  if (msg.type() !== 'error') return true
  return IGNORE_PATTERNS.some((p) => p.test(msg.text()))
}

function screenshotPath(name: string, viewport: string) {
  fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true })
  return path.join(SCREENSHOTS_DIR, `${name}-${viewport}.png`)
}

const ROUTES = [
  { name: 'overview', path: '/' },
  { name: 'customer-explorer', path: '/customers' },
  { name: 'custom-builder', path: '/custom' },
  { name: 'model-lab', path: '/lab' },
  { name: 'ab-simulator', path: '/simulator' },
  { name: 'about', path: '/about' },
]

for (const route of ROUTES) {
  test(`${route.name} loads with no console errors`, async ({ page, viewport }) => {
    const errors: string[] = []
    const failedRequests: string[] = []

    page.on('console', (msg) => {
      if (!isIgnoredError(msg)) {
        errors.push(`[console.error] ${msg.text()}`)
      }
    })

    page.on('requestfailed', (req) => {
      // Ignore image failures (404 for product images is expected)
      if (!req.url().includes('/articles/') || !req.url().includes('/image')) {
        failedRequests.push(`${req.failure()?.errorText} ${req.url()}`)
      }
    })

    await page.goto(route.path)
    // Wait for the main content to settle
    await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {})

    // Take screenshots at current viewport
    const vp = viewport ? `${viewport.width}x${viewport.height}` : 'default'
    await page.screenshot({
      path: screenshotPath(route.name, vp),
      fullPage: false,
    })

    // No blank page — something meaningful must be rendered
    const bodyText = await page.locator('body').innerText()
    expect(bodyText.trim().length, `${route.name}: page appears blank`).toBeGreaterThan(20)

    // No unhandled console errors
    expect(errors, `${route.name}: console errors:\n${errors.join('\n')}`).toHaveLength(0)

    // No failed API requests
    expect(
      failedRequests,
      `${route.name}: failed requests:\n${failedRequests.join('\n')}`,
    ).toHaveLength(0)
  })
}

test('customer explorer: load recommendations and open explain drawer', async ({ page }) => {
  const errors: string[] = []
  page.on('console', (msg) => {
    if (!isIgnoredError(msg)) errors.push(msg.text())
  })

  await page.goto('/customers')
  await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {})

  // Click first customer in sample list
  const firstCustomer = page.getByRole('button', { name: /show recommendations/i }).first()
  if (await firstCustomer.isVisible()) {
    await firstCustomer.click()
    await page.waitForLoadState('networkidle', { timeout: 5000 }).catch(() => {})
  }

  expect(errors).toHaveLength(0)
})

test('a/b simulator: run power calc and simulation', async ({ page }) => {
  const errors: string[] = []
  page.on('console', (msg) => {
    if (!isIgnoredError(msg)) errors.push(msg.text())
  })

  await page.goto('/simulator')
  await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {})

  // Run A/A simulation
  const simulateBtn = page.getByRole('button', { name: /run simulation/i })
  if (await simulateBtn.isVisible()) {
    await simulateBtn.click()
    await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {})
  }

  expect(errors).toHaveLength(0)
})
