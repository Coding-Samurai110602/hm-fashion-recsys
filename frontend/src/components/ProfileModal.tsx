import { useState, useEffect, useRef } from 'react'
import { useMutation } from '@tanstack/react-query'
import { createProfile } from '../api/endpoints'
import { useProfile, type Profile } from '../contexts/ProfileContext'

const AGE_OPTIONS = [
  { label: 'Under 25', value: '<25' as const },
  { label: '25–34', value: '25-34' as const },
  { label: '35–44', value: '35-44' as const },
  { label: '45–54', value: '45-54' as const },
  { label: '55 and over', value: '55+' as const },
]

interface Props {
  open: boolean
  canDismiss?: boolean
  onClose?: () => void
}

export function ProfileModal({ open, canDismiss = true, onClose }: Props) {
  const { setProfile } = useProfile()
  const [nickname, setNickname] = useState('')
  const [ageRange, setAgeRange] = useState<Profile['age_range']>('25-34')
  const [channel, setChannel] = useState<'online' | 'store'>('online')
  const [nicknameError, setNicknameError] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [open])

  const mutation = useMutation({
    mutationFn: () =>
      createProfile({ nickname: nickname.trim(), age_range: ageRange, channel }),
    onSuccess: (data) => {
      setProfile({
        id: data.id,
        nickname: data.nickname,
        age_range: data.age_range as Profile['age_range'],
        channel: data.channel as 'online' | 'store',
        purchase_count: data.purchase_count,
      })
      onClose?.()
    },
  })

  if (!open) return null

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const trimmed = nickname.trim()
    if (trimmed.length < 1) {
      setNicknameError('Please enter a nickname.')
      return
    }
    if (trimmed.length > 30) {
      setNicknameError('Nickname must be 30 characters or fewer.')
      return
    }
    setNicknameError('')
    mutation.mutate()
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Escape' && canDismiss) {
      onClose?.()
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="profile-modal-title"
      onKeyDown={handleKeyDown}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={canDismiss ? onClose : undefined}
        aria-hidden="true"
      />

      {/* Panel */}
      <div className="relative z-10 w-full max-w-md rounded-2xl bg-white p-8 shadow-2xl">
        {canDismiss && (
          <button
            onClick={onClose}
            className="absolute right-4 top-4 rounded-lg p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            aria-label="Close dialog"
          >
            <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        )}

        <h2 id="profile-modal-title" className="text-xl font-bold text-gray-900">
          Create your profile
        </h2>
        <p className="mt-1 text-sm text-gray-500">
          Your age range is one of the signals the model can use; for new shoppers with few purchases, what's popular right now matters most.
        </p>

        <form onSubmit={handleSubmit} className="mt-6 space-y-5" noValidate>
          {/* Nickname */}
          <div>
            <label
              htmlFor="pm-nickname"
              className="block text-sm font-medium text-gray-700"
            >
              Nickname
            </label>
            <input
              ref={inputRef}
              id="pm-nickname"
              type="text"
              value={nickname}
              onChange={(e) => {
                setNickname(e.target.value)
                setNicknameError('')
              }}
              maxLength={30}
              placeholder="e.g. Alex"
              className={`mt-1.5 w-full rounded-lg border px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                nicknameError
                  ? 'border-red-400 focus:ring-red-400'
                  : 'border-gray-300'
              }`}
              aria-describedby={nicknameError ? 'pm-nickname-error' : undefined}
              aria-invalid={nicknameError ? 'true' : 'false'}
            />
            {nicknameError && (
              <p id="pm-nickname-error" className="mt-1 text-xs text-red-600" role="alert">
                {nicknameError}
              </p>
            )}
          </div>

          {/* Age range */}
          <div>
            <label
              htmlFor="pm-age-range"
              className="block text-sm font-medium text-gray-700"
            >
              Age range
            </label>
            <select
              id="pm-age-range"
              value={ageRange}
              onChange={(e) => setAgeRange(e.target.value as Profile['age_range'])}
              className="mt-1.5 w-full rounded-lg border border-gray-300 px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {AGE_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* Shopping preference */}
          <fieldset>
            <legend className="text-sm font-medium text-gray-700">
              Shopping preference
            </legend>
            <div className="mt-2 space-y-2">
              <label className="flex cursor-pointer items-center gap-3 rounded-lg border border-gray-200 px-4 py-3 hover:bg-gray-50 has-[:checked]:border-indigo-400 has-[:checked]:bg-indigo-50">
                <input
                  type="radio"
                  name="pm-channel"
                  value="online"
                  checked={channel === 'online'}
                  onChange={() => setChannel('online')}
                  className="accent-indigo-600"
                />
                <span className="text-sm text-gray-700">I usually shop online</span>
              </label>
              <label className="flex cursor-pointer items-center gap-3 rounded-lg border border-gray-200 px-4 py-3 hover:bg-gray-50 has-[:checked]:border-indigo-400 has-[:checked]:bg-indigo-50">
                <input
                  type="radio"
                  name="pm-channel"
                  value="store"
                  checked={channel === 'store'}
                  onChange={() => setChannel('store')}
                  className="accent-indigo-600"
                />
                <span className="text-sm text-gray-700">I prefer shopping in store</span>
              </label>
            </div>
          </fieldset>

          {mutation.error && (
            <p className="text-sm text-red-600" role="alert">
              Something went wrong. Please try again.
            </p>
          )}

          <button
            type="submit"
            disabled={mutation.isPending}
            className="w-full rounded-xl bg-indigo-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:opacity-60"
          >
            {mutation.isPending ? 'Creating profile…' : 'Start Shopping'}
          </button>
        </form>
      </div>
    </div>
  )
}
