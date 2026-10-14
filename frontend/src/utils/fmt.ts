/**
 * Safe number formatter — returns "—" for null/undefined instead of throwing.
 * Use in all JSX number display to prevent blank-page crashes when API fields are missing.
 */
export function fmtNum(
  value: number | null | undefined,
  decimals = 4,
): string {
  if (value == null || !Number.isFinite(value)) return '—'
  return value.toFixed(decimals)
}

/** Format as a percentage with a + prefix (e.g. "+47.6%"). */
export function fmtPct(
  value: number | null | undefined,
  decimals = 1,
): string {
  if (value == null || !Number.isFinite(value)) return '—'
  const sign = value >= 0 ? '+' : ''
  return `${sign}${value.toFixed(decimals)}%`
}
