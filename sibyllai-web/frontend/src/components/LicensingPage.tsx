import { useMemo, useState } from 'react'
import { Button } from '@/components/ui/button'
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

  const selectClass =
    'bg-[hsla(0,0%,18%,0.6)] border border-[hsl(0_0%_100%/0.1)] rounded-lg px-3 py-2 text-sm text-foreground appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary/50 w-full'

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="glass px-6 py-4 rounded-2xl space-y-3">
        <div className="flex items-center justify-between">
          <Button variant="outline" size="sm" onClick={() => setCurrentPage('replacement')} className="glass-lighter border-primary/20">
            ← Back
          </Button>
          <Button variant="outline" size="sm" onClick={reset} className="glass-lighter border-primary/20">
            New Import
          </Button>
        </div>
        <div>
          <h2 className="text-2xl font-medium font-display">Licensing & Download</h2>
          <p className="text-sm text-foreground-muted">
            {fileName ?? 'Project'} · {matchedCues.length} matched {matchedCues.length === 1 ? 'track' : 'tracks'}
          </p>
        </div>
      </div>

      {/* Rights Configuration */}
      <div className="glass px-6 py-5 rounded-2xl space-y-3">
        <h3 className="text-lg font-medium font-display">Rights Configuration</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs text-foreground-muted uppercase tracking-wide mb-1.5">
              Territory
            </label>
            <select
              value={territory}
              onChange={(e) => setTerritory(e.target.value as Territory)}
              className={selectClass}
            >
              {TERRITORY_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-foreground-muted uppercase tracking-wide mb-1.5">
              Time Period
            </label>
            <select
              value={period}
              onChange={(e) => setPeriod(e.target.value as TimePeriod)}
              className={selectClass}
            >
              {PERIOD_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-foreground-muted uppercase tracking-wide mb-1.5">
              Medium
            </label>
            <select
              value={medium}
              onChange={(e) => setMedium(e.target.value as Medium)}
              className={selectClass}
            >
              {MEDIUM_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Track Table */}
      <div className="glass px-6 py-5 rounded-2xl space-y-4">
        <h3 className="text-lg font-medium font-display">Tracks</h3>

        {matchedCues.length === 0 ? (
          <p className="text-sm text-foreground-muted">No matched tracks to display.</p>
        ) : (
          <>
            {/* Column headers */}
            <div className="grid grid-cols-[auto_1fr_auto_auto_auto] gap-4 items-center text-xs text-foreground-muted uppercase tracking-wide px-3 pb-1 border-b border-[hsl(0_0%_100%/0.06)]">
              <span>Cue</span>
              <span>Replacement Track</span>
              <span className="text-right">Match</span>
              <span className="text-right">License Fee</span>
              <span className="text-right">Download</span>
            </div>

            {/* Track rows */}
            <div className="space-y-2">
              {matchedCues.map((cue) => {
                const cueIndex = project!.cues.findIndex((c) => c.id === cue.id)
                const score = cue.project_context?.replacement?.score ?? 0
                return (
                  <div
                    key={cue.id}
                    className="grid grid-cols-[auto_1fr_auto_auto_auto] gap-4 items-center px-3 py-3 rounded-lg border border-[hsl(0_0%_100%/0.06)] hover:border-primary/20 transition-colors"
                  >
                    <div className="text-sm">
                      <span className="font-medium">Cue {cueIndex + 1}</span>
                      <div className="text-xs text-foreground-muted">{cue.start_tc} – {cue.end_tc}</div>
                    </div>
                    <div className="text-sm truncate" title={trackFileName(cue)}>
                      {trackFileName(cue)}
                    </div>
                    <div className="text-sm text-right tabular-nums">
                      {(score * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-right font-medium tabular-nums">
                      {formatCurrency(feePerTrack)}
                    </div>
                    <div className="text-right">
                      <Button
                        variant="outline"
                        size="sm"
                        className="glass-lighter border-primary/20 text-xs"
                        onClick={() => handleDownload(cue)}
                      >
                        Download
                      </Button>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Summary row */}
            <div className="flex items-center justify-between pt-4 border-t border-[hsl(0_0%_100%/0.08)]">
              <div className="text-sm text-foreground-muted">
                {matchedCues.length} {matchedCues.length === 1 ? 'track' : 'tracks'} · {TERRITORY_OPTIONS.find((o) => o.value === territory)!.label} · {PERIOD_OPTIONS.find((o) => o.value === period)!.label} · {MEDIUM_OPTIONS.find((o) => o.value === medium)!.label}
              </div>
              <div className="flex items-center gap-4">
                <span className="text-lg font-medium tabular-nums">
                  Total: {formatCurrency(totalFee)}
                </span>
                <Button
                  className="btn-primary px-6"
                  onClick={handleDownloadAll}
                >
                  Download All
                </Button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
