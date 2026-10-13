/// <reference types="vite/client" />
const BASE_URL = (import.meta.env['VITE_API_BASE_URL'] as string | undefined) ?? ''

export interface ApiError {
  code: string
  message: string
  request_id: string
}

export class ApiClientError extends Error {
  constructor(
    public readonly error: ApiError,
    public readonly status: number,
  ) {
    super(error.message)
    this.name = 'ApiClientError'
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${BASE_URL}${path}`
  const resp = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })

  if (!resp.ok) {
    let errorBody: { error?: ApiError } = {}
    try {
      errorBody = (await resp.json()) as { error?: ApiError }
    } catch {
      // ignore parse error
    }
    const err: ApiError = errorBody.error ?? {
      code: `HTTP_${resp.status}`,
      message: resp.statusText || `HTTP ${resp.status}`,
      request_id: '',
    }
    throw new ApiClientError(err, resp.status)
  }

  return resp.json() as Promise<T>
}

export function get<T>(path: string): Promise<T> {
  return request<T>(path)
}

export function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function getImageUrl(articleId: string): string {
  return `${BASE_URL}/api/v1/articles/${articleId}/image`
}
