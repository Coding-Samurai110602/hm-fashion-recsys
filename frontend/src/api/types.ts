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
