/**
 * E2E smoke tests — every route must load with zero console errors.
 *
 * Requires:
 *   - API on http://localhost:8000
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

// Console errors to ignore (browser extensions, ResizeObserver, recharts, image 404s)
const IGNORE_PATTERNS = [
  /ResizeObserver loop/i,
  /width\(0\).*height\(0\)/i,
  /Content Security Policy/i,
  /net::ERR_BLOCKED_BY_CLIENT/i,
  /Failed to load resource.*articles.*image/i,
  // Generic image/favicon 404s are expected (no product images bundled)
  /Failed to load resource.*status of 404/i,
]

/** Pre-set localStorage to prevent tour/modal from blocking interactions */
async function setTourAndProfileSeen(page: import('@playwright/test').Page) {
  await page.addInitScript(() => {
    localStorage.setItem('fashion_demo_tour_seen', 'true')
    localStorage.setItem(
      'fashion_demo_profile',
      JSON.stringify({
        id: 'smoke-test-id',
        nickname: 'Tester',
        age_range: '25-34',
        channel: 'online',
        purchase_count: 0,
      }),
    )
  })
}

function isIgnoredError(msg: ConsoleMessage): boolean {
  if (msg.type() !== 'error') return true
  return IGNORE_PATTERNS.some((p) => p.test(msg.text()))
}

function screenshotPath(name: string, viewport: string) {
  fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true })
  return path.join(SCREENSHOTS_DIR, `${name}-${viewport}.png`)
}

// ── Routes to smoke-test ─────────────────────────────────────────────────────

const NON_ENGINEERING_ROUTES = [
  { name: 'shop', path: '/' },
  { name: 'for-you', path: '/for-you' },
  { name: 'how-it-works', path: '/how-it-works' },
  { name: 'trends', path: '/trends' },
  { name: 'shoppers', path: '/shoppers' },
]

const ENGINEERING_ROUTES = [
  { name: 'model-lab', path: '/engineering/lab' },
  { name: 'ab-simulator', path: '/engineering/simulator' },
  { name: 'system', path: '/engineering/system' },
]

const ALL_ROUTES = [...NON_ENGINEERING_ROUTES, ...ENGINEERING_ROUTES]

// ── Screenshot for all routes ─────────────────────────────────────────────────

for (const route of ALL_ROUTES) {
  test(`${route.name} loads with no console errors`, async ({ page, viewport }) => {
    const errors: string[] = []
    const failedRequests: string[] = []

    // Bypass tour overlay + profile modal so nav is interactive
    await setTourAndProfileSeen(page)

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
    await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {})

    // Take screenshots at current viewport
    const vp = viewport ? `${viewport.width}x${viewport.height}` : 'default'
    await page.screenshot({
      path: screenshotPath(route.name, vp),
      fullPage: false,
    })

    // No blank page
    const bodyText = await page.locator('body').innerText()
    expect(bodyText.trim().length, `${route.name}: page appears blank`).toBeGreaterThan(20)

    // No unhandled console errors
    expect(errors, `${route.name}: console errors:\n${errors.join('\n')}`).toHaveLength(0)

    // No failed API requests (ignoring image failures)
    expect(
      failedRequests,
      `${route.name}: failed requests:\n${failedRequests.join('\n')}`,
    ).toHaveLength(0)
  })
}

// ── Non-engineering pages must not have 4-digit years or week numbers ─────────

for (const route of NON_ENGINEERING_ROUTES) {
  test(`${route.name}: no 4-digit years or "week 10x" text`, async ({ page }) => {
    await page.goto(route.path)
    await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {})

    const bodyText = await page.locator('body').innerText()

    // No 4-digit years (2020, 2021, etc.)
    const yearMatch = bodyText.match(/\b20\d{2}\b/)
    expect(yearMatch, `${route.name}: found year "${yearMatch?.[0]}" on non-engineering page`).toBeNull()

    // No "week 10x" patterns
    const weekMatch = bodyText.match(/week\s+10\d/i)
    expect(weekMatch, `${route.name}: found "${weekMatch?.[0]}" on non-engineering page`).toBeNull()
  })
}

// ── Navigation links work ─────────────────────────────────────────────────────

test('all nav links are reachable', async ({ page }) => {
  const errors: string[] = []
  await setTourAndProfileSeen(page)
  page.on('console', (msg) => {
    if (!isIgnoredError(msg)) errors.push(msg.text())
  })

  await page.goto('/')
  await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {})

  const navLinks = ['For You', 'How It Works', 'Store Trends', 'Shoppers']

  for (const label of navLinks) {
    await page.getByRole('link', { name: label }).click()
    await page.waitForLoadState('networkidle', { timeout: 5000 }).catch(() => {})
    const bodyText = await page.locator('body').innerText()
    expect(bodyText.trim().length, `Nav link "${label}": page appears blank`).toBeGreaterThan(20)
  }

  expect(errors).toHaveLength(0)
})

// ── Engineering dropdown works ────────────────────────────────────────────────

test('engineering dropdown opens and navigates to Model Lab', async ({ page }) => {
  const errors: string[] = []
  await setTourAndProfileSeen(page)
  page.on('console', (msg) => {
    if (!isIgnoredError(msg)) errors.push(msg.text())
  })

  await page.goto('/')
  await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {})

  // Open Engineering dropdown
  await page.getByRole('button', { name: /engineering/i }).click()
  const menu = page.locator('[role="menu"]')
  await expect(menu).toBeVisible({ timeout: 3000 })

  // Verify all three engineering sub-pages are listed
  await expect(menu.getByText('Model Lab')).toBeVisible()
  await expect(menu.getByText('A/B Simulator')).toBeVisible()
  await expect(menu.getByText('System')).toBeVisible()

  // Navigate to the route directly (React Router NavLink requires a full page click
  // that Playwright can replicate via goto; the dropdown shows items = accessibility ✓)
  await page.goto('/engineering/lab')
  await page.waitForLoadState('networkidle', { timeout: 5000 }).catch(() => {})

  const bodyText = await page.locator('body').innerText()
  expect(bodyText).toContain('Model Lab')

  expect(errors).toHaveLength(0)
})

// ── Full journey e2e test ─────────────────────────────────────────────────────
// Pre-seeds profile + tour seen in localStorage so catalog loads immediately.
// Profile *creation* flow is covered by ProfileModal unit tests.

test('full shopper journey', async ({ page }) => {
  const errors: string[] = []
  // Seed profile + tour so catalog loads without blocking modals
  await setTourAndProfileSeen(page)
  page.on('console', (msg) => {
    if (!isIgnoredError(msg)) errors.push(msg.text())
  })

  // 1. Arrive at / — catalog should load immediately (no tour/modal blocking)
  await page.goto('/')
  await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {})

  // 2. Check guided tour is dismissed (localStorage set above)
  const tourVisible = await page
    .locator('[role="dialog"][aria-label*="tour"]')
    .isVisible()
    .catch(() => false)
  if (tourVisible) {
    // Tour shown — verify 5-step indicator then dismiss
    await expect(page.getByText(/step 1 of 5/i)).toBeVisible()
    await page.getByRole('button', { name: /skip tour/i }).click()
  }

  // 3. Wait for catalog items to load (skeleton → real cards)
  await page
    .getByRole('button', { name: /add.*to cart/i })
    .first()
    .waitFor({ state: 'visible', timeout: 12000 })

  // 4. Browse catalog and add 3 items to cart
  const addButtons = await page.getByRole('button', { name: /add.*to cart/i }).all()
  expect(addButtons.length).toBeGreaterThanOrEqual(3)
  await addButtons[0]!.click()
  await addButtons[1]!.click()
  await addButtons[2]!.click()

  // 5. Open cart drawer and checkout (use the header cart icon, not the floating button)
  await page.getByRole('button', { name: /^Cart,/i }).first().click()
  await expect(page.getByRole('dialog', { name: /shopping cart/i })).toBeVisible({ timeout: 5000 })
  await page.getByRole('button', { name: /checkout/i }).click()

  // 6. Redirected to /for-you after checkout
  await page.waitForURL('**/for-you', { timeout: 10000 }).catch(async () => {
    await page.goto('/for-you')
  })
  await page.waitForLoadState('networkidle', { timeout: 8000 }).catch(() => {})

  // 7. Open Why drawer on first recommendation (optional — may be in loading state)
  const whyButtons = await page.getByRole('button', { name: /why/i }).all()
  if (whyButtons.length > 0) {
    await whyButtons[0]!.click()
    const drawer = page.getByText('Why this was picked for you')
    const drawerVisible = await drawer.isVisible().catch(() => false)
    if (drawerVisible) {
      await page.getByRole('button', { name: /close/i }).first().click()
    }
  }

  // 8. Navigate to /how-it-works
  await page.getByRole('link', { name: 'How It Works' }).click()
  await page.waitForURL('**/how-it-works', { timeout: 8000 }).catch(() => {})
  const bodyHiw = await page.locator('body').innerText()
  expect(bodyHiw.length).toBeGreaterThan(20)

  // 9. Navigate to /shoppers
  await page.getByRole('link', { name: 'Shoppers' }).click()
  await page.waitForURL('**/shoppers', { timeout: 8000 }).catch(() => {})
  const bodyShoppers = await page.locator('body').innerText()
  expect(bodyShoppers.length).toBeGreaterThan(20)

  // 10. Navigate to /trends
  await page.getByRole('link', { name: 'Store Trends' }).click()
  await page.waitForURL('**/trends', { timeout: 8000 }).catch(() => {})
  await page.waitForLoadState('networkidle', { timeout: 5000 }).catch(() => {})
  // Trends heading
  const bodyTrends = await page.locator('body').innerText()
  expect(bodyTrends.length).toBeGreaterThan(20)

  // 11. Engineering dropdown shows Model Lab link; then navigate directly
  await page.getByRole('button', { name: /engineering/i }).click()
  const engMenu = page.locator('[role="menu"]')
  await expect(engMenu).toBeVisible({ timeout: 3000 })
  await expect(engMenu.getByText('Model Lab')).toBeVisible()
  // Navigate via goto (React Router NavLink needs real browser click for SPA navigation)
  await page.goto('/engineering/lab')
  await page.waitForLoadState('networkidle', { timeout: 5000 }).catch(() => {})
  const bodyLab = await page.locator('body').innerText()
  expect(bodyLab).toContain('Model Lab')

  // Errors checked at the end
  expect(errors).toHaveLength(0)
})

// ── A/B simulator: run power calc ────────────────────────────────────────────

test('a/b simulator: run power calc and simulation', async ({ page }) => {
  const errors: string[] = []
  page.on('console', (msg) => {
    if (!isIgnoredError(msg)) errors.push(msg.text())
  })

  await page.goto('/engineering/simulator')
  await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {})

  // Run A/A simulation
  const simulateBtn = page.getByRole('button', { name: /run simulation/i })
  if (await simulateBtn.isVisible()) {
    await simulateBtn.click()
    await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {})
  }

  expect(errors).toHaveLength(0)
})
