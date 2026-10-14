import { http, HttpResponse } from 'msw'
import summaryFixture from './fixtures/summary.json'
import weeklyLiftFixture from './fixtures/weekly_lift.json'
import recsFixture from './fixtures/recommendations.json'
import explainFixture from './fixtures/explain.json'
import customerProfileFixture from './fixtures/customer_profile.json'
import ablationsFixture from './fixtures/ablations.json'
import importanceFixture from './fixtures/importance.json'
import powerResultFixture from './fixtures/power_result.json'
import simulateResultFixture from './fixtures/simulate_result.json'

const CUSTOMER_ID = 'a73eaaa3b5e5c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2'
const PROFILE_ID = 'prof-test-001'

// Synthetic trending items
const TRENDING_ITEMS = Array.from({ length: 20 }, (_, i) => ({
  article_idx: 1000 + i,
  article_id: `010000${String(1000 + i).padStart(4, '0')}`,
  prod_name: `Trending Item ${i + 1}`,
  product_group: ['Trousers', 'Sweater', 'Blouse', 'Shoes', 'Bag'][i % 5] ?? 'Trousers',
  colour: ['Black', 'White', 'Blue', 'Green', 'Pink'][i % 5] ?? 'Black',
  image_url: '',
  popularity_rank: i + 1,
}))

// Synthetic catalog items
const CATALOG_ITEMS = Array.from({ length: 24 }, (_, i) => ({
  article_idx: 2000 + i,
  article_id: `020000${String(2000 + i).padStart(4, '0')}`,
  prod_name: `Product ${i + 1}`,
  product_group: ['Trousers', 'Sweater', 'Blouse', 'Shoes', 'Bag'][i % 5] ?? 'Trousers',
  colour: ['Black', 'White', 'Blue', 'Green', 'Pink'][i % 5] ?? 'Black',
  image_url: '',
  popularity_rank: i + 1,
}))

// Synthetic profile recommendations
const PROFILE_RECS = Array.from({ length: 12 }, (_, i) => ({
  article_idx: 3000 + i,
  article_id: `030000${String(3000 + i).padStart(4, '0')}`,
  prod_name: `Pick ${i + 1}`,
  product_group: ['Trousers', 'Sweater', 'Blouse'][i % 3] ?? 'Trousers',
  colour: ['Black', 'White', 'Blue'][i % 3] ?? 'Black',
  image_url: '',
  score: 0.95 - i * 0.05,
  reasons: i === 0 ? ['Popular with shoppers your age', 'Trending now'] : ['Trending now'],
  source: i < 3 ? 'repurchase' : 'trending',
  latency_ms: 42.5,
  cold_start: false,
}))

// Synthetic demo shoppers
const DEMO_SHOPPERS = Array.from({ length: 5 }, (_, i) => ({
  customer_id: `demo-customer-${i + 1}`,
  customer_idx: 100 + i,
  age_range: ['<25', '25-34', '35-44', '45-54', '55+'][i] ?? '<25',
  channel: i % 2 === 0 ? 'online' : 'store',
  purchase_count: 10 + i * 5,
  top_product_groups: ['Trousers', 'Sweater'],
  top_colour: 'Black',
  summary: `Shopper ${i + 1}`,
}))

// Synthetic trends
const TRENDS_RESPONSE = {
  trending_now: TRENDING_ITEMS,
  trending_by_age: {
    '<25': TRENDING_ITEMS.slice(0, 10),
    '25-34': TRENDING_ITEMS.slice(2, 12),
    '35-44': TRENDING_ITEMS.slice(4, 14),
    '45-54': TRENDING_ITEMS.slice(6, 16),
    '55+': TRENDING_ITEMS.slice(8, 18),
  },
  top_product_groups: [
    { product_group: 'Trousers', share: 0.22, count: 4200 },
    { product_group: 'Sweater', share: 0.18, count: 3400 },
    { product_group: 'Blouse', share: 0.15, count: 2900 },
    { product_group: 'Shoes', share: 0.12, count: 2300 },
    { product_group: 'Bag', share: 0.09, count: 1700 },
  ],
  popularity_decay: [
    { age_label: 'Just launched', age_days_max: 7, share: 0.38 },
    { age_label: 'Recently launched', age_days_max: 30, share: 0.29 },
    { age_label: 'Established', age_days_max: 90, share: 0.22 },
    { age_label: 'Mature', age_days_max: null, share: 0.11 },
  ],
}

export const handlers = [
  // Ops
  http.get('/health', () =>
    HttpResponse.json({ status: 'ok' }),
  ),
  http.get('/ready', () =>
    HttpResponse.json({ ready: true }),
  ),
  http.get('/version', () =>
    HttpResponse.json({
      api_version: 'v1',
      git_commit: 'abc1234def5678',
      bundle_as_of_week: 104,
      model_map12_fold103: 0.035832,
      model_map12_holdout: 0.036904,
      n_features: 68,
    }),
  ),

  // Insights
  http.get('/api/v1/insights/summary', () =>
    HttpResponse.json(summaryFixture),
  ),
  http.get('/api/v1/insights/weekly-lift', () =>
    HttpResponse.json(weeklyLiftFixture),
  ),
  http.get('/api/v1/insights/ablations', () =>
    HttpResponse.json(ablationsFixture),
  ),
  http.get('/api/v1/insights/importance', () =>
    HttpResponse.json(importanceFixture),
  ),
  http.get('/api/v1/insights/segments', () =>
    HttpResponse.json([
      {
        family: 'recency',
        segment: '<=7 days',
        n: 13958,
        ranker_map12: 0.099,
        heuristic_map12: 0.083,
        ci_lo: 0.015,
        ci_hi: 0.018,
        p_ttest_adj: 0.000,
        lift_pct: 19.3,
      },
    ]),
  ),

  // Customers
  http.get('/api/v1/customers/sample', () =>
    HttpResponse.json([
      {
        customer_idx: 896118,
        customer_id: CUSTOMER_ID,
        segment: 'medium',
        age_bucket: '45-54',
      },
    ]),
  ),
  http.get(`/api/v1/customers/${CUSTOMER_ID}`, () =>
    HttpResponse.json(customerProfileFixture),
  ),
  http.get(`/api/v1/customers/${CUSTOMER_ID}/recommendations`, () =>
    HttpResponse.json(recsFixture),
  ),
  http.get(`/api/v1/customers/${CUSTOMER_ID}/baseline`, () =>
    HttpResponse.json(
      recsFixture.map((r) => ({ ...r, reasons: undefined })),
    ),
  ),
  http.get(`/api/v1/customers/${CUSTOMER_ID}/actual-purchases`, () =>
    HttpResponse.json([
      {
        article_idx: 42,
        article_id: '0108775015',
        prod_name: 'Strindberg Boot',
        product_type: 'Boots',
        colour: 'Black',
        department: 'Shoes',
      },
    ]),
  ),
  http.get(`/api/v1/customers/${CUSTOMER_ID}/funnel`, () =>
    HttpResponse.json([
      {
        article_idx: 42,
        source: 'repurchase',
        all_sources: ['repurchase'],
        source_rank: 1,
        model_score: 2.83,
        final_rank: 1,
      },
      {
        article_idx: 99,
        source: 'popularity_last_week',
        all_sources: ['popularity_last_week'],
        source_rank: 5,
        model_score: 0.95,
        final_rank: 2,
      },
    ]),
  ),
  http.get(
    `/api/v1/customers/${CUSTOMER_ID}/explain/:articleId`,
    () => HttpResponse.json(explainFixture),
  ),

  // Articles
  http.get('/api/v1/articles/search', () =>
    HttpResponse.json([
      {
        article_idx: 42,
        article_id: '0108775015',
        prod_name: 'Strindberg Boot',
        product_type: 'Boots',
        colour: 'Black',
        department: 'Shoes',
      },
    ]),
  ),
  http.get('/api/v1/articles/:articleId/image', () =>
    new HttpResponse(null, { status: 404 }),
  ),

  // Custom recommendations
  http.post('/api/v1/recommendations/custom', () =>
    HttpResponse.json(recsFixture),
  ),

  // Experiment
  http.post('/api/v1/experiment/power', () =>
    HttpResponse.json(powerResultFixture),
  ),
  http.post('/api/v1/experiment/simulate', () =>
    HttpResponse.json(simulateResultFixture),
  ),

  // ── New profile endpoints ────────────────────────────────────────────────

  http.post('/api/v1/profiles', () =>
    HttpResponse.json({
      id: PROFILE_ID,
      nickname: 'TestUser',
      age_range: '25-34',
      channel: 'online',
      purchase_count: 0,
      created_at: new Date().toISOString(),
    }),
  ),

  http.get('/api/v1/profiles', () =>
    HttpResponse.json({
      profiles: [
        {
          id: PROFILE_ID,
          nickname: 'TestUser',
          age_range: '25-34',
          channel: 'online',
          purchase_count: 3,
          created_at: new Date().toISOString(),
        },
      ],
      total: 1,
    }),
  ),

  http.get(`/api/v1/profiles/${PROFILE_ID}`, () =>
    HttpResponse.json({
      id: PROFILE_ID,
      nickname: 'TestUser',
      age_range: '25-34',
      channel: 'online',
      purchase_count: 3,
      created_at: new Date().toISOString(),
    }),
  ),

  http.post(`/api/v1/profiles/${PROFILE_ID}/checkout`, () =>
    HttpResponse.json({ purchase_count: 3 }),
  ),

  http.get(`/api/v1/profiles/${PROFILE_ID}/recommendations`, () =>
    HttpResponse.json(PROFILE_RECS),
  ),

  http.get(`/api/v1/profiles/${PROFILE_ID}/summary`, () =>
    HttpResponse.json({
      id: PROFILE_ID,
      nickname: 'TestUser',
      age_range: '25-34',
      channel: 'online',
      purchase_count: 3,
      top_product_groups: ['Trousers', 'Sweater'],
      top_colour: 'Black',
    }),
  ),

  // ── Catalog endpoints ────────────────────────────────────────────────────

  http.get('/catalog/trending', () =>
    HttpResponse.json(TRENDING_ITEMS.slice(0, 8)),
  ),

  http.get('/catalog/browse', () =>
    HttpResponse.json({
      items: CATALOG_ITEMS,
      total: 48,
      page: 1,
      page_size: 24,
    }),
  ),

  http.get('/catalog/filters', () =>
    HttpResponse.json({
      product_groups: ['Trousers', 'Sweater', 'Blouse', 'Shoes', 'Bag', 'Dress', 'Jacket'],
      colours: ['Black', 'White', 'Blue', 'Green', 'Pink', 'Red', 'Grey', 'Brown'],
    }),
  ),

  // ── Demo shoppers ────────────────────────────────────────────────────────

  http.get('/demo/shoppers', () =>
    HttpResponse.json(DEMO_SHOPPERS),
  ),

  // ── Trends ──────────────────────────────────────────────────────────────

  http.get('/api/v1/insights/trends', () =>
    HttpResponse.json(TRENDS_RESPONSE),
  ),

  // ── Compare ─────────────────────────────────────────────────────────────

  http.get('/api/v1/profiles/compare', () =>
    HttpResponse.json({
      a_summary: {
        id: PROFILE_ID,
        nickname: 'TestUser',
        age_range: '25-34',
        channel: 'online',
        purchase_count: 3,
        top_product_groups: ['Trousers'],
        top_colour: 'Black',
      },
      b_summary: {
        id: 'prof-test-002',
        nickname: 'OtherUser',
        age_range: '35-44',
        channel: 'store',
        purchase_count: 7,
        top_product_groups: ['Sweater'],
        top_colour: 'White',
      },
      a_recommendations: PROFILE_RECS,
      b_recommendations: PROFILE_RECS.slice(3),
      overlap: [PROFILE_RECS[3]?.article_id ?? ''],
      overlap_count: 1,
    }),
  ),
]

// Error handlers for testing error states
export const errorHandlers = {
  summary: http.get('/api/v1/insights/summary', () =>
    HttpResponse.json(
      { error: { code: 'INTERNAL_ERROR', message: 'Server error', request_id: 'req-test-123' } },
      { status: 500 },
    ),
  ),
  recommendations: http.get(
    `/api/v1/customers/${CUSTOMER_ID}/recommendations`,
    () =>
      HttpResponse.json(
        { error: { code: 'NOT_FOUND', message: 'Customer not found', request_id: 'req-404' } },
        { status: 404 },
      ),
  ),
  power: http.post('/api/v1/experiment/power', () =>
    HttpResponse.json(
      { error: { code: 'VALIDATION_ERROR', message: 'Invalid parameters', request_id: 'req-422' } },
      { status: 422 },
    ),
  ),
  simulate: http.post('/api/v1/experiment/simulate', () =>
    HttpResponse.json(
      { error: { code: 'INTERNAL_ERROR', message: 'Simulation failed', request_id: 'req-500' } },
      { status: 500 },
    ),
  ),
}
