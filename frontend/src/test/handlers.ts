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
