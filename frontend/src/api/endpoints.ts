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
