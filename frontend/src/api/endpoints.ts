import { get, post } from './client'
import type {
  ArticleMeta,
  AblationEntry,
  CustomerProfile,
  ExplainResult,
  FunnelItem,
  ImportanceEntry,
  InsightsSummary,
  PowerRequest,
  PowerResult,
  RecommendedItem,
  RecommendedItemNoReasons,
  SampleCustomer,
  SegmentEntry,
  SimulateRequest,
  SimulateResult,
  VersionResponse,
  WeeklyLiftEntry,
  CustomRecommendRequest,
  HealthResponse,
  ReadyResponse,
  Profile,
  ProfileSummary,
  ProfileRecommendation,
  CatalogArticle,
  CatalogBrowseResponse,
  CatalogFilters,
  DemoShopper,
  TrendingItem,
  TrendsResponse,
  CompareResult,
} from './types'

// Ops
export const fetchHealth = () => get<HealthResponse>('/health')
export const fetchReady = () => get<ReadyResponse>('/ready')
export const fetchVersion = () => get<VersionResponse>('/version')

// Customers
export const fetchSampleCustomers = (segment?: string, n = 5) =>
  get<SampleCustomer[]>(
    `/api/v1/customers/sample?${segment ? `segment=${encodeURIComponent(segment)}&` : ''}n=${n}`,
  )

export const fetchCustomerProfile = (customerId: string) =>
  get<CustomerProfile>(`/api/v1/customers/${customerId}`)

export const fetchRecommendations = (
  customerId: string,
  k = 12,
  includeReasons = true,
) =>
  get<RecommendedItem[]>(
    `/api/v1/customers/${customerId}/recommendations?k=${k}&include_reasons=${includeReasons}`,
  )

export const fetchBaseline = (customerId: string, k = 12) =>
  get<RecommendedItemNoReasons[]>(
    `/api/v1/customers/${customerId}/baseline?k=${k}`,
  )

export const fetchActualPurchases = (customerId: string) =>
  get<ArticleMeta[]>(`/api/v1/customers/${customerId}/actual-purchases`)

export const fetchFunnel = (customerId: string) =>
  get<FunnelItem[]>(`/api/v1/customers/${customerId}/funnel`)

export const fetchExplain = (customerId: string, articleId: string) =>
  get<ExplainResult>(
    `/api/v1/customers/${customerId}/explain/${articleId}`,
  )

export const fetchCustomRecommendations = (
  body: CustomRecommendRequest,
  k = 12,
) => post<RecommendedItem[]>(`/api/v1/recommendations/custom?k=${k}`, body)

// Articles
export const searchArticles = (q: string, limit = 20) =>
  get<ArticleMeta[]>(
    `/api/v1/articles/search?q=${encodeURIComponent(q)}&limit=${limit}`,
  )

export const fetchArticle = (articleId: string) =>
  get<ArticleMeta>(`/api/v1/articles/${articleId}`)

// Insights
export const fetchInsightsSummary = () =>
  get<InsightsSummary>('/api/v1/insights/summary')

export const fetchWeeklyLift = () =>
  get<WeeklyLiftEntry[]>('/api/v1/insights/weekly-lift')

export const fetchSegments = () =>
  get<SegmentEntry[]>('/api/v1/insights/segments')

export const fetchAblations = () =>
  get<AblationEntry[]>('/api/v1/insights/ablations')

export const fetchImportance = () =>
  get<ImportanceEntry[]>('/api/v1/insights/importance')

// Experiment
export const calcPower = (body: PowerRequest) =>
  post<PowerResult>('/api/v1/experiment/power', body)

export const runSimulation = (body: SimulateRequest) =>
  post<SimulateResult>('/api/v1/experiment/simulate', body)

// ── Profiles (new) ───────────────────────────────────────────────────────────

export const createProfile = (data: {
  nickname: string
  age_range: string
  channel: string
}) => post<Profile>('/api/v1/profiles', data)

export const getProfile = (id: string) =>
  get<Profile>(`/api/v1/profiles/${id}`)

export const listProfiles = (limit = 20) =>
  get<{ profiles: Profile[]; total: number }>(`/api/v1/profiles?limit=${limit}`)

export const deleteProfile = (id: string) =>
  get<void>(`/api/v1/profiles/${id}/delete`)

export const checkout = (profileId: string, articleIds: string[]) =>
  post<{ purchase_count: number }>(`/api/v1/profiles/${profileId}/checkout`, {
    article_ids: articleIds,
  })

export const getProfileRecommendations = (
  profileId: string,
  includeReasons = true,
) =>
  get<ProfileRecommendation[]>(
    `/api/v1/profiles/${profileId}/recommendations?include_reasons=${includeReasons}`,
  )

export const getProfileSummary = (profileId: string) =>
  get<ProfileSummary>(`/api/v1/profiles/${profileId}/summary`)

// ── Catalog (new) ────────────────────────────────────────────────────────────

export const getTrending = (k = 20) =>
  get<TrendingItem[]>(`/catalog/trending?k=${k}`)

export const browseCatalog = (params: {
  product_group?: string
  colour?: string
  q?: string
  page?: number
  page_size?: number
}) => {
  const qs = new URLSearchParams()
  if (params.product_group) qs.set('product_group', params.product_group)
  if (params.colour) qs.set('colour', params.colour)
  if (params.q) qs.set('q', params.q)
  if (params.page != null) qs.set('page', String(params.page))
  if (params.page_size != null) qs.set('page_size', String(params.page_size))
  return get<CatalogBrowseResponse>(`/catalog/browse?${qs.toString()}`)
}

export const getCatalogFilters = () =>
  get<CatalogFilters>('/catalog/filters')

// ── Demo (new) ───────────────────────────────────────────────────────────────

export const getDemoShoppers = () =>
  get<DemoShopper[]>('/demo/shoppers')

// ── Trends (new) ─────────────────────────────────────────────────────────────

export const getTrends = () =>
  get<TrendsResponse>('/api/v1/insights/trends')

// ── Compare (new) ────────────────────────────────────────────────────────────

export const compare = (a: string, b: string) =>
  get<CompareResult>(`/api/v1/profiles/compare?a=${a}&b=${b}`)

// ── Catalog article re-export ─────────────────────────────────────────────────
export type { CatalogArticle }
