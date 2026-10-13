import { describe, it, expect } from 'vitest'

function truncateId(id: string): string {
  if (id.length <= 16) return id
  return `${id.slice(0, 8)}...${id.slice(-8)}`
}

describe('truncateId', () => {
  it('truncates 64-char hex to 8+...+8 display', () => {
    const id = 'a73eaaa3b5e5c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2'
    const result = truncateId(id)
    expect(result).toBe('a73eaaa3...a9b0c1d2')
    expect(result.length).toBe(19) // 8 + 3 + 8
  })

  it('does not truncate short IDs', () => {
    const id = 'abc123'
    expect(truncateId(id)).toBe('abc123')
  })

  it('handles exactly 16-char ID without truncation', () => {
    const id = 'abcdef1234567890'
    expect(truncateId(id)).toBe('abcdef1234567890')
  })
})
