import { useMemo, useState } from 'react'
import { useAppStore } from '@/lib/store'
import type { Cue } from '@/lib/types'

type Territory = 'worldwide' | 'north_america' | 'europe' | 'single_country'
type TimePeriod = '1_year' | '3_years' | '5_years' | 'perpetuity'
type Medium = 'all_media' | 'film_tv' | 'online' | 'advertising'

const TERRITORY_OPTIONS: { value: Territory; label: string; multiplier: number }[] = [
  { value: 'worldwide', label: 'Worldwide', multiplier: 3.0 },
  { value: 'north_america', label: 'North America', multiplier: 1.5 },
  { value: 'europe', label: 'Europe', multiplier: 1.5 },
  { value: 'single_country', label: 'Single Country', multiplier: 1.0 },
]

const PERIOD_OPTIONS: { value: TimePeriod; label: string; multiplier: number }[] = [
  { value: '1_year', label: '1 Year', multiplier: 1.0 },
  { value: '3_years', label: '3 Years', multiplier: 1.5 },
  { value: '5_years', label: '5 Years', multiplier: 2.5 },
  { value: 'perpetuity', label: 'In Perpetuity', multiplier: 4.0 },
]

const MEDIUM_OPTIONS: { value: Medium; label: string; multiplier: number }[] = [
  { value: 'all_media', label: 'All Media', multiplier: 2.5 },
  { value: 'film_tv', label: 'Film / TV', multiplier: 2.0 },
  { value: 'online', label: 'Online / Streaming', multiplier: 1.2 },
  { value: 'advertising', label: 'Advertising', multiplier: 1.8 },
]

const BASE_FEE = 150

function calculateFee(territory: Territory, period: TimePeriod, medium: Medium): number {
  const t = TERRITORY_OPTIONS.find((o) => o.value === territory)!.multiplier
  const p = PERIOD_OPTIONS.find((o) => o.value === period)!.multiplier
  const m = MEDIUM_OPTIONS.find((o) => o.value === medium)!.multiplier
  return BASE_FEE * t * p * m
}

function formatCurrency(amount: number): string {
  return `$${amount.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`
}

const selectStyle: React.CSSProperties = {
  width: '100%',
  height: 36,
  padding: '0 10px',
  border: '1px solid #080808',
  borderRadius: 6,
  background: '#fff',
  fontFamily: 'var(--pl-font-mono)',
  fontSize: '0.75rem',
  color: '#080808',
  cursor: 'pointer',
  appearance: 'auto' as const,
}

export function LicensingPage() {
  const { project, fileName, setCurrentPage, reset } = useAppStore()

  const [territory, setTerritory] = useState<Territory>('worldwide')
  const [period, setPeriod] = useState<TimePeriod>('perpetuity')
  const [medium, setMedium] = useState<Medium>('all_media')

  const matchedCues = useMemo(
    () => (project?.cues ?? []).filter((c) => c.project_context?.status === 'matched'),
    [project]
  )

  const feePerTrack = calculateFee(territory, period, medium)
  const totalFee = feePerTrack * matchedCues.length

  const trackFileName = (cue: Cue): string => {
    const path = cue.project_context?.replacement?.track_path
    if (!path) return 'Unknown'
    return path.replace(/\\/g, '/').split('/').pop() ?? 'Unknown'
  }

  const handleDownload = (cue: Cue) => {
    alert(`Download placeholder: ${trackFileName(cue)}\n\nDownload API not yet implemented.`)
  }

  const handleDownloadAll = () => {
    alert(`Download All placeholder: ${matchedCues.length} tracks\n\nDownload API not yet implemented.`)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <button className="btn-pill" onClick={() => setCurrentPage('replacement')} style={{ height: 28, fontSize: '0.65rem' }}>
            ← Back
          </button>
          <button className="btn-pill" onClick={reset} style={{ height: 28, fontSize: '0.65rem' }}>
            New Import
          </button>
        </div>
        <div>
          <span style={{ fontSize: '1.2rem', fontWeight: 900, textTransform: 'uppercase' }}>Licensing & Download</span>
          <p className="mono" style={{ fontSize: '0.7rem', opacity: 0.5, marginTop: 4 }}>
            {fileName ?? 'Project'} · {matchedCues.length} matched {matchedCues.length === 1 ? 'track' : 'tracks'}
          </p>
        </div>
      </div>

      {/* Rights Configuration */}
      <div style={{ padding: 16, border: '1px solid #080808', borderRadius: 8 }}>
        <div className="label" style={{ marginBottom: 12 }}>Rights Configuration</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
          <div>
            <label className="label" style={{ display: 'block', fontSize: '0.55rem', marginBottom: 6, opacity: 0.6 }}>
              Territory
            </label>
            <select
              value={territory}
              onChange={(e) => setTerritory(e.target.value as Territory)}
              style={selectStyle}
            >
              {TERRITORY_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="label" style={{ display: 'block', fontSize: '0.55rem', marginBottom: 6, opacity: 0.6 }}>
              Time Period
            </label>
            <select
              value={period}
              onChange={(e) => setPeriod(e.target.value as TimePeriod)}
              style={selectStyle}
            >
              {PERIOD_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="label" style={{ display: 'block', fontSize: '0.55rem', marginBottom: 6, opacity: 0.6 }}>
              Medium
            </label>
            <select
              value={medium}
              onChange={(e) => setMedium(e.target.value as Medium)}
              style={selectStyle}
            >
              {MEDIUM_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Track Table */}
      <div style={{ padding: 16, border: '1px solid #080808', borderRadius: 8 }}>
        <div className="label" style={{ marginBottom: 12 }}>Tracks</div>

        {matchedCues.length === 0 ? (
          <p className="mono" style={{ fontSize: '0.7rem', opacity: 0.5 }}>No matched tracks to display.</p>
        ) : (
          <>
            {/* Column headers */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'auto 1fr auto auto auto',
              gap: 16,
              alignItems: 'center',
              padding: '0 8px 8px',
              borderBottom: '1px solid rgba(8,8,8,0.1)',
            }}>
              <span className="label" style={{ fontSize: '0.5rem' }}>Cue</span>
              <span className="label" style={{ fontSize: '0.5rem' }}>Replacement Track</span>
              <span className="label" style={{ fontSize: '0.5rem', textAlign: 'right' }}>Match</span>
              <span className="label" style={{ fontSize: '0.5rem', textAlign: 'right' }}>License Fee</span>
              <span className="label" style={{ fontSize: '0.5rem', textAlign: 'right' }}>Download</span>
            </div>

            {/* Track rows */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 8 }}>
              {matchedCues.map((cue) => {
                const cueIndex = project!.cues.findIndex((c) => c.id === cue.id)
                const score = cue.project_context?.replacement?.score ?? 0
                return (
                  <div
                    key={cue.id}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'auto 1fr auto auto auto',
                      gap: 16,
                      alignItems: 'center',
                      padding: '8px',
                      borderRadius: 4,
                      border: '1px solid rgba(8,8,8,0.1)',
                    }}
                  >
                    <div>
                      <span className="mono" style={{ fontSize: '0.75rem', fontWeight: 700 }}>Cue {cueIndex + 1}</span>
                      <div className="mono" style={{ fontSize: '0.6rem', opacity: 0.5 }}>{cue.start_tc} – {cue.end_tc}</div>
                    </div>
                    <div className="mono" style={{ fontSize: '0.7rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={trackFileName(cue)}>
                      {trackFileName(cue)}
                    </div>
                    <div className="mono" style={{ fontSize: '0.7rem', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                      {(score * 100).toFixed(1)}%
                    </div>
                    <div className="mono" style={{ fontSize: '0.7rem', textAlign: 'right', fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>
                      {formatCurrency(feePerTrack)}
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <button className="btn-pill" style={{ height: 24, fontSize: '0.55rem' }} onClick={() => handleDownload(cue)}>
                        Download
                      </button>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Summary row */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingTop: 12,
              marginTop: 12,
              borderTop: '1px solid rgba(8,8,8,0.1)',
            }}>
              <span className="mono" style={{ fontSize: '0.65rem', opacity: 0.5 }}>
                {matchedCues.length} {matchedCues.length === 1 ? 'track' : 'tracks'} · {TERRITORY_OPTIONS.find((o) => o.value === territory)!.label} · {PERIOD_OPTIONS.find((o) => o.value === period)!.label} · {MEDIUM_OPTIONS.find((o) => o.value === medium)!.label}
              </span>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span className="mono" style={{ fontSize: '1rem', fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>
                  Total: {formatCurrency(totalFee)}
                </span>
                <button className="btn-pill primary" onClick={handleDownloadAll} style={{ height: 36 }}>
                  Download All
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
