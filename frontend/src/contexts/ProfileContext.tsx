import { createContext, useContext, useState, useEffect, type ReactNode } from 'react'

export interface Profile {
  id: string
  nickname: string
  age_range: '<25' | '25-34' | '35-44' | '45-54' | '55+'
  channel: 'online' | 'store'
  purchase_count: number
}

interface ProfileContextValue {
  profile: Profile | null
  setProfile: (p: Profile | null) => void
  clearProfile: () => void
}

const STORAGE_KEY = 'fashion_demo_profile'

const ProfileContext = createContext<ProfileContextValue | null>(null)

export function ProfileProvider({ children }: { children: ReactNode }) {
  const [profile, setProfileState] = useState<Profile | null>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        return JSON.parse(stored) as Profile
      }
    } catch {
      // ignore parse errors
    }
    return null
  })

  const setProfile = (p: Profile | null) => {
    setProfileState(p)
    if (p) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(p))
      } catch {
        // ignore storage errors
      }
    } else {
      localStorage.removeItem(STORAGE_KEY)
    }
  }

  const clearProfile = () => setProfile(null)

  // Keep localStorage in sync when profile changes externally
  useEffect(() => {
    if (profile) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(profile))
      } catch {
        // ignore
      }
    }
  }, [profile])

  // On mount: verify the stored profile still exists on the server.
  // If it returns 404, clear it so the user isn't stuck with a stale profile.
  useEffect(() => {
    if (profile) {
      fetch(`/api/v1/profiles/${profile.id}`)
        .then((r) => { if (!r.ok) clearProfile() })
        .catch(() => {}) // network error — don't clear, may be offline
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // only on mount

  return (
    <ProfileContext.Provider value={{ profile, setProfile, clearProfile }}>
      {children}
    </ProfileContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export function useProfile(): ProfileContextValue {
  const ctx = useContext(ProfileContext)
  if (!ctx) throw new Error('useProfile must be used inside ProfileProvider')
  return ctx
}
