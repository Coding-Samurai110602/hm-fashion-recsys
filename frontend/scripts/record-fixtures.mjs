/**
 * Record MSW fixtures from the synthetic-bundle API.
 *
 * Usage:
 *   1. Start the API: BUNDLE_PATH=artifacts/bundle_synthetic make api
 *   2. In another terminal: cd frontend && npm run record-fixtures
 *
 * The script calls each endpoint, saves the response as a fixture JSON, and
 * reports any HTTP errors. Never commit real customer IDs or production data.
 *
 * The synthetic bundle is always used (API_BASE defaults to localhost:8000).
 * Real bundle data must NOT be recorded (it contains real customer IDs).
 */

import { writeFileSync, mkdirSync } from 'fs'
import { resolve, dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const FIXTURES_DIR = resolve(__dirname, '../src/test/fixtures')
const API_BASE = process.env.API_BASE ?? 'http://localhost:8000'

// A stable synthetic customer ID from the synthetic bundle.
// This is always the first customer in bundle_synthetic/state/customers.parquet.
// Run: python3 -c "import polars as pl; print(pl.read_parquet('artifacts/bundle_synthetic/state/customers.parquet')['customer_id'][0])"
// to get the current value.
const SYNTH_CUSTOMER_ID = process.env.SYNTH_CUSTOMER_ID ?? 'SYNTHETIC_PLACEHOLDER'

async function fetchJson(path) {
  const url = `${API_BASE}${path}`
  const resp = await fetch(url)
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText} for ${url}`)
  }
  return resp.json()
}

async function postJson(path, body) {
  const url = `${API_BASE}${path}`
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok) {
    throw new Error(`${resp.status} ${resp.statusText} for POST ${url}`)
  }
  return resp.json()
}

function save(name, data) {
  const path = resolve(FIXTURES_DIR, `${name}.json`)
  writeFileSync(path, JSON.stringify(data, null, 2) + '\n')
  console.log(`  ✓ ${name}.json`)
}

mkdirSync(FIXTURES_DIR, { recursive: true })

console.log(`Recording fixtures from ${API_BASE}…`)
console.log('(Requires synthetic-bundle API to be running)\n')

const errors = []

async function record(name, fn) {
  try {
    const data = await fn()
    save(name, data)
  } catch (err) {
    console.error(`  ✗ ${name}: ${err.message}`)
    errors.push({ name, error: err.message })
  }
}

await record('summary', () => fetchJson('/api/v1/insights/summary'))
await record('weekly_lift', () => fetchJson('/api/v1/insights/weekly-lift'))
await record('ablations', () => fetchJson('/api/v1/insights/ablations'))
await record('importance', () => fetchJson('/api/v1/insights/importance'))

// Power result — use realistic defaults that return stable synthetic output
await record('power_result', () =>
  postJson('/api/v1/experiment/power', {
    baseline_rate: 0.086,
    mde: 0.05,
    alpha: 0.05,
    power: 0.8,
  }),
)

// Simulate result — small synthetic A/A test
await record('simulate_result', () =>
  postJson('/api/v1/experiment/simulate', {
    n_per_arm: 500,
    mode: 'aa',
    n_sims: 100,
    seed: 42,
    peeking: false,
  }),
)

if (SYNTH_CUSTOMER_ID !== 'SYNTHETIC_PLACEHOLDER') {
  await record('customer_profile', () =>
    fetchJson(`/api/v1/customers/${SYNTH_CUSTOMER_ID}`),
  )
  await record('recommendations', () =>
    fetchJson(
      `/api/v1/customers/${SYNTH_CUSTOMER_ID}/recommendations?k=12&include_reasons=true`,
    ),
  )
  await record('explain', () => {
    // Explain requires an article_id; use first recommendation's article_id
    return fetchJson(`/api/v1/customers/${SYNTH_CUSTOMER_ID}/recommendations?k=1&include_reasons=false`)
      .then((recs) => {
        const articleId = recs[0]?.article_id ?? '0000000001'
        return fetchJson(`/api/v1/customers/${SYNTH_CUSTOMER_ID}/explain/${articleId}`)
      })
  })
} else {
  console.log(
    '\n  ⚠ Skipping customer/explain fixtures (set SYNTH_CUSTOMER_ID env var).\n' +
    '    Run: python3 -c "import polars as pl; print(pl.read_parquet(' +
    "'artifacts/bundle_synthetic/state/customers.parquet')['customer_id'][0])\"\n",
  )
}

if (errors.length > 0) {
  console.error(`\n${errors.length} endpoint(s) failed. Is the synthetic-bundle API running?`)
  process.exit(1)
} else {
  console.log('\nAll fixtures recorded successfully.')
  console.log('Review the output before committing — never include real customer IDs.')
}
