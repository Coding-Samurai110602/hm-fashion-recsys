import type { ApiClientError } from '../api/client'

interface Props {
  error: Error | ApiClientError | null
  title?: string
}

function isApiClientError(e: Error): e is ApiClientError {
  return e.name === 'ApiClientError'
}

export function ErrorState({ error, title = 'Something went wrong' }: Props) {
  if (!error) return null

  const message = error.message
  const requestId = isApiClientError(error) ? error.error.request_id : null
  const code = isApiClientError(error) ? error.error.code : null

  return (
    <div
      role="alert"
      className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm"
    >
      <p className="font-semibold text-red-800">{title}</p>
      <p className="mt-1 text-red-700">{message}</p>
      {code && (
        <p className="mt-1 font-mono text-xs text-red-500">Code: {code}</p>
      )}
      {requestId && (
        <p className="mt-1 font-mono text-xs text-red-500">
          Request ID: {requestId}
        </p>
      )}
    </div>
  )
}
