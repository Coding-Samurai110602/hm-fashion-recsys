// Manual types for endpoints that return schema: {} in openapi.json
// plus convenient re-exports from generated types.

export type { components } from './types.generated'
import type { components } from './types.generated'

// Convenience aliases for schema types
export type ArticleMeta = components['schemas']['ArticleMeta']
export type CustomerProfile = components['schemas']['CustomerProfile']
export type SampleCustomer = components['schemas']['SampleCustomer']
export type FunnelItem = components['schemas']['FunnelItem']
export type ExplainResult = components['schemas']['ExplainResult']
export type RecommendedItemNoReasons = components['schemas']['RecommendedItemNoReasons']
export type CustomHistoryItem = components['schemas']['CustomHistoryItem']
export type CustomRecommendRequest = components['schemas']['CustomRecommendRequest']
export type PowerRequest = components['schemas']['PowerRequest']
export type PowerResult = components['schemas']['PowerResult']
export type SimulateRequest = components['schemas']['SimulateRequest']
export type SimulateResult = components['schemas']['SimulateResult']
export type HealthResponse = components['schemas']['HealthResponse']
export type ReadyResponse = components['schemas']['ReadyResponse']

// Manual types for schema: {} endpoints
export interface RecommendedItem extends RecommendedItemNoReasons {
  reasons: string[]
}

// Insights API types (manually defined because openapi has schema: {})
export interface InsightsSummary {
  holdout_map12_ranker: number
  holdout_map12_heuristic: number
  holdout_lift_pct: number
  holdout_ci_lo: number | null
  holdout_ci_hi: number | null
  holdout_map12_baseline_b: number | null
  holdout_lift_vs_baseline_b_pct: number | null
  holdout_baseline_b_ci_lo: number | null
  holdout_baseline_b_ci_hi: number | null
  fold103_ci_lo: number
  fold103_ci_hi: number
  fold103_ranker_map12: number
  fold103_heuristic_map12: number
}

export interface WeeklyLiftEntry {
  fold: number
  ranker_map12: number
  heuristic_map12: number
  lift_pct: number
  optimistic?: boolean
}

export interface AblationEntry {
  group: string
  n_features: number
  map12_delta: number
  ci_lo: number
  ci_hi: number
  ci_excludes_zero: boolean
}

export interface ImportanceEntry {
  rank_shap: number
  feature: string
  mean_abs_shap: number
  rank_contrast: number | null
  mean_abs_contrast_shap: number | null
  rank_gain: number | null
}

export interface SegmentEntry {
  family: string
  segment: string
  n: number
  ranker_map12: number
  heuristic_map12: number
  ci_lo: number
  ci_hi: number
  p_ttest_adj: number
  lift_pct: number
}

export interface VersionResponse {
  api_version: string
  git_commit: string
  bundle_as_of_week: number
  model_map12_fold103: number
  model_map12_holdout: number
  n_features: number
}

// ── Profile endpoints ────────────────────────────────────────────────────────

export interface Profile {
  id: string
  nickname: string
  age_range: string
  channel: string
  purchase_count: number
  created_at: string
}

export interface ProfileSummary {
  profile_id: string
  nickname: string
  age_range: string
  channel: string
  purchase_count: number
  top_product_groups: string[]
  top_colour: string | null
}

export interface ProfileRecommendation {
  article_idx: number
  article_id: string
  prod_name: string
  product_group: string
  colour: string
  image_url: string
  score: number
  reasons: string[]
  source: string
  latency_ms: number
  cold_start: boolean
}

export interface ProfileRecsResponse {
  profile_id: string
  cold_start: boolean
  items: ProfileRecommendation[]
  latency_ms: number
}

// ── Catalog endpoints ────────────────────────────────────────────────────────

export interface CatalogArticle {
  article_idx: number
  article_id: string
  prod_name: string
  product_group: string
  colour: string
  image_url: string
  popularity_rank: number | null
}

export interface CatalogBrowseResponse {
  items: CatalogArticle[]
  total: number
  page: number
  page_size: number
}

export interface CatalogFilters {
  product_groups: string[]
  colours: string[]
}

// ── Demo shoppers ────────────────────────────────────────────────────────────

export interface DemoShopper {
  customer_id: string
  customer_idx: number
  age_range: string
  channel: string
  purchase_count: number
  top_product_groups: string[]
  top_colour: string | null
  summary: string
}

// ── Trends ───────────────────────────────────────────────────────────────────

export interface TrendingItem {
  article_idx: number
  article_id: string
  prod_name: string
  product_group: string
  colour: string
  image_url: string
  popularity_rank: number
}

export interface ProductGroupShare {
  product_group: string
  share: number
  count: number
}

export interface DecayBin {
  age_label: string
  age_days_max: number | null
  share: number
}

export interface TrendsResponse {
  trending_now: TrendingItem[]
  trending_by_age: Record<string, TrendingItem[]>
  top_product_groups: ProductGroupShare[]
  popularity_decay: DecayBin[]
}

// ── Compare ──────────────────────────────────────────────────────────────────

export interface CompareResult {
  a_summary: ProfileSummary
  b_summary: ProfileSummary
  a_recommendations: ProfileRecommendation[]
  b_recommendations: ProfileRecommendation[]
  overlap: string[]
  overlap_count: number
}
