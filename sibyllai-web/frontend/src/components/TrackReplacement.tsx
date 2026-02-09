import { useEffect, useMemo, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import { useAppStore } from '@/lib/store'
import { WaveformViewer } from '@/components/WaveformViewer'
import { CueCard } from '@/components/CueCard'
import type { LibraryMatch } from '@/lib/types'

export function TrackReplacement() {
  const { project, fileName, sessionId, activeCueId, setActiveCueId, updateCueContext } = useAppStore()

  const cueCount = project?.cues.length ?? 0
  const cueOptions = useMemo(
    () => (project?.cues ?? []).map((cue, index) => ({ id: cue.id, label: `Cue ${index + 1}` })),
    [project]
  )

  const folderInputRef = useRef<HTMLInputElement>(null)
  const [libraryFiles, setLibraryFiles] = useState<File[]>([])
  const [libraryFolderName, setLibraryFolderName] = useState<string | null>(null)
  const [buildJobId, setBuildJobId] = useState<string | null>(null)
  const [buildStatus, setBuildStatus] = useState<'idle' | 'running' | 'complete' | 'error'>('idle')
  const [buildMessage, setBuildMessage] = useState('')
  const [progressPercent, setProgressPercent] = useState(0)
  const [progressCurrent, setProgressCurrent] = useState(0)
  const [progressTotal, setProgressTotal] = useState(0)
  const [windowsIndexed, setWindowsIndexed] = useState(0)
  const [includeMoods, setIncludeMoods] = useState(true)
  const [isBuilding, setIsBuilding] = useState(false)
  const [isUploading, setIsUploading] = useState(false)

  const [selectedCueId, setSelectedCueId] = useState<string | null>(activeCueId)
  const [matchesByCue, setMatchesByCue] = useState<Record<string, LibraryMatch[]>>({})
  const [focusedMatchId, setFocusedMatchId] = useState<Record<string, string | null>>({})
  const [isMatching, setIsMatching] = useState(false)
  const [matchError, setMatchError] = useState('')
  const [indexAvailable, setIndexAvailable] = useState(false)
  const [indexInfo, setIndexInfo] = useState<{
    tracks: number
    windows: number
    builtAt?: string
    sourceName?: string
    sourceType?: string
  } | null>(null)

  useEffect(() => {
    let alive = true
    const loadInfo = async () => {
      try {
        const info = await api.getLibraryInfo()
        if (!alive) return
        setIndexAvailable(info.exists)
        const builtAt = typeof info.meta?.built_at === 'string' ? info.meta.built_at : undefined
        const sourceName = typeof info.meta?.source_name === 'string' ? info.meta.source_name : undefined
        const sourceType = typeof info.meta?.source_type === 'string' ? info.meta.source_type : undefined
        setIndexInfo(
          info.exists
            ? { tracks: info.tracks, windows: info.windows, builtAt, sourceName, sourceType }
            : null
        )
      } catch {
        if (!alive) return
        setIndexAvailable(false)
        setIndexInfo(null)
      }
    }
    loadInfo()
    return () => {
      alive = false
    }
  }, [])

  // Keep selected cue in sync with active cue
  useEffect(() => {
    if (activeCueId) {
      setSelectedCueId(activeCueId)
    } else if (!selectedCueId && cueOptions.length > 0) {
      setSelectedCueId(cueOptions[0].id)
      setActiveCueId(cueOptions[0].id)
    }
  }, [activeCueId, cueOptions, selectedCueId, setActiveCueId])

  useEffect(() => {
    let interval: number | undefined
    if (buildJobId && buildStatus === 'running') {
      interval = window.setInterval(async () => {
        try {
          const status = await api.getLibraryBuildStatus(buildJobId)
          if (typeof status.progress_percent === 'number') {
            setProgressPercent(status.progress_percent)
          }
          if (typeof status.current === 'number') {
            setProgressCurrent(status.current)
          }
          if (typeof status.total === 'number') {
            setProgressTotal(status.total)
          }
          if (typeof status.windows_indexed === 'number') {
            setWindowsIndexed(status.windows_indexed)
          }
          if (status.message) {
            setBuildMessage(status.message)
          }
          if (status.status === 'complete') {
            setBuildStatus('complete')
            setBuildMessage(
              `Indexed ${status.result?.tracks_indexed ?? 0} tracks, ${status.result?.windows_indexed ?? 0} windows`
            )
            setProgressPercent(100)
            setIsBuilding(false)
            setIndexAvailable(true)
            setIndexInfo({
              tracks: status.result?.tracks_indexed ?? 0,
              windows: status.result?.windows_indexed ?? 0,
              builtAt: new Date().toISOString(),
              sourceName: libraryFolderName ?? undefined,
              sourceType: 'upload',
            })
            window.clearInterval(interval)
          } else if (status.status === 'error') {
            setBuildStatus('error')
            setBuildMessage(status.error || 'Build failed')
            setIsBuilding(false)
            window.clearInterval(interval)
          }
        } catch (err) {
          setBuildStatus('error')
          setBuildMessage(err instanceof Error ? err.message : 'Build failed')
          setIsBuilding(false)
          window.clearInterval(interval)
        }
      }, 2000)
    }
    return () => {
      if (interval) window.clearInterval(interval)
    }
  }, [buildJobId, buildStatus, libraryFolderName])

  const handleBuildIndex = async () => {
    if (libraryFiles.length === 0) {
      setBuildStatus('error')
      setBuildMessage('Please select a library folder')
      return
    }
    setIsBuilding(true)
    setBuildStatus('running')
    setBuildMessage('Preparing upload...')
    setProgressPercent(0)
    setProgressCurrent(0)
    setProgressTotal(0)
    setWindowsIndexed(0)
    setMatchesByCue((prev) => ({ ...prev, [selectedCueId || '']: [] }))
    try {
      setIsUploading(true)
      const response = await api.buildLibraryIndexUpload(libraryFiles, includeMoods, true)
      setIsUploading(false)
      setBuildMessage('Building index...')
      setBuildJobId(response.job_id)
    } catch (err) {
      setBuildStatus('error')
      setBuildMessage(err instanceof Error ? err.message : 'Build failed')
      setIsBuilding(false)
      setIsUploading(false)
    }
  }

  const handleAddToLibrary = async () => {
    if (libraryFiles.length === 0) {
      setBuildStatus('error')
      setBuildMessage('Please select a library folder')
      return
    }
    setIsBuilding(true)
    setBuildStatus('running')
    setBuildMessage('Preparing upload...')
    setProgressPercent(0)
    setProgressCurrent(0)
    setProgressTotal(0)
    setWindowsIndexed(0)
    setMatchesByCue((prev) => ({ ...prev, [selectedCueId || '']: [] }))
    try {
      setIsUploading(true)
      const response = await api.buildLibraryIndexUpload(libraryFiles, includeMoods, false)
      setIsUploading(false)
      setBuildMessage('Adding to library...')
      setBuildJobId(response.job_id)
    } catch (err) {
      setBuildStatus('error')
      setBuildMessage(err instanceof Error ? err.message : 'Add failed')
      setIsBuilding(false)
      setIsUploading(false)
    }
  }

  const handleMatch = async () => {
    if (!sessionId || !selectedCueId) return
    setIsMatching(true)
    setMatchError('')
    try {
      const response = await api.matchLibrary({
        session_id: sessionId,
        cue_id: selectedCueId,
        top_n: 3,
        unique_tracks: true,
      })
      setMatchesByCue((prev) => ({ ...prev, [selectedCueId]: response.matches }))
      setFocusedMatchId((prev) => ({ ...prev, [selectedCueId]: response.matches[0]?.window_id ?? null }))
    } catch (err) {
      setMatchError(err instanceof Error ? err.message : 'Match failed')
    } finally {
      setIsMatching(false)
    }
  }

  const selectedCue = project?.cues.find((cue) => cue.id === selectedCueId) || null
  const matches = selectedCueId ? (matchesByCue[selectedCueId] || []) : []
  const focusedMatch = selectedCueId
    ? matches.find((m) => m.window_id === focusedMatchId[selectedCueId]) || null
    : null
  const selectedCueIndex = selectedCueId ? cueOptions.findIndex((c) => c.id === selectedCueId) : -1

  const applyMatch = async (match: LibraryMatch) => {
    if (!selectedCueId || !sessionId) return
    try {
      await api.updateCueReplacement(sessionId, selectedCueId, {
        track_path: match.track_path,
        start: match.start,
        end: match.end,
        score: match.score,
        window_size: match.window_size,
        status: 'matched',
      })
      updateCueContext(selectedCueId, {
        status: 'matched',
        replacement: {
          track_path: match.track_path,
          start: match.start,
          end: match.end,
          score: match.score,
          window_size: match.window_size,
        },
      })
      setFocusedMatchId((prev) => ({ ...prev, [selectedCueId]: match.window_id }))
    } catch (err) {
      setMatchError(err instanceof Error ? err.message : 'Failed to save match')
    }
  }

  const formatSeconds = (value: number | null | undefined) =>
    typeof value === 'number' ? `${value.toFixed(1)}s` : '—'

  const formatBpm = (bpm: number | null | undefined) => {
    if (typeof bpm !== 'number') return '—'
    // Round to 1 decimal if needed, otherwise show integer
    return bpm % 1 === 0 ? String(Math.round(bpm)) : bpm.toFixed(1)
  }

  return (
    <div className="space-y-6">
      <div className="glass p-8 rounded-2xl">
        <div className="flex items-start justify-between gap-6">
          <div className="space-y-2">
            <h2 className="text-3xl font-medium font-display">Track Replacement</h2>
            <p className="text-foreground-muted">
              Match temp cues to a music library using tempo, mood, instrumentation, and embeddings.
            </p>
            {project && (
              <p className="text-sm text-foreground-muted">
                Source: <span className="text-foreground">{fileName ?? 'Analysis'}</span> · {cueCount} cues
              </p>
            )}
          </div>
          <div className="flex items-center gap-3">
            <Button
              size="sm"
              variant="outline"
              className="glass-lighter border-primary/20"
              onClick={handleBuildIndex}
              disabled={isBuilding || libraryFiles.length === 0}
            >
              {isUploading ? 'Uploading…' : isBuilding ? 'Building Index…' : indexAvailable ? 'Rebuild Index' : 'Build Library Index'}
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="glass-lighter border-primary/20"
              onClick={handleAddToLibrary}
              disabled={isBuilding || libraryFiles.length === 0 || !indexAvailable}
            >
              {isUploading ? 'Uploading…' : isBuilding ? 'Adding…' : 'Add To Library'}
            </Button>
          </div>
        </div>
      </div>

      <div className="glass p-6 rounded-2xl space-y-4">
          <div>
            <h3 className="text-lg font-medium font-display mb-2">Library</h3>
            <p className="text-sm text-foreground-muted">
              Select a local folder to upload and build the demo index.
            </p>
          </div>
          <input
            ref={folderInputRef}
            type="file"
            multiple
            className="hidden"
            // @ts-expect-error webkitdirectory is non-standard but supported in Chromium
            webkitdirectory="true"
            directory=""
            onChange={(e) => {
              const files = Array.from(e.target.files || [])
              if (files.length === 0) return
              setLibraryFiles(files)
              const relPath = (files[0] as File & { webkitRelativePath?: string }).webkitRelativePath
              const folderName = relPath ? relPath.split('/')[0] : 'Selected Folder'
              setLibraryFolderName(folderName)
              setBuildStatus('idle')
              setBuildMessage('')
            }}
          />
          <div className="flex items-center gap-3">
            <Button
              size="sm"
              variant="outline"
              className="glass-lighter border-primary/20"
              onClick={() => folderInputRef.current?.click()}
              disabled={isBuilding}
            >
              Choose Folder
            </Button>
            {libraryFolderName && (
              <span className="text-xs text-foreground-muted">
                {libraryFolderName} · {libraryFiles.length} files
              </span>
            )}
          </div>
          <label className="flex items-center gap-2 text-xs text-foreground-muted">
            <input
              type="checkbox"
              checked={includeMoods}
              onChange={(e) => setIncludeMoods(e.target.checked)}
              className="accent-primary"
            />
            Include mood analysis (slower, but improves matching)
          </label>
          {buildStatus !== 'idle' && (
            <p className={`text-xs ${buildStatus === 'error' ? 'text-red-400' : 'text-foreground-muted'}`}>
              {buildMessage}
            </p>
          )}
          {!indexAvailable && buildStatus === 'idle' && (
            <p className="text-xs text-foreground-muted">
              No index found yet. Build one to enable matching.
            </p>
          )}
          {buildStatus === 'running' && (
            <div className="space-y-2">
              <div className="h-2 rounded-full bg-primary/10 overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${Math.min(100, Math.max(0, progressPercent))}%` }}
                />
              </div>
              <p className="text-xs text-foreground-muted">
                {progressCurrent}/{progressTotal || '–'} tracks · {windowsIndexed} windows
              </p>
            </div>
          )}
          {indexAvailable && indexInfo && buildStatus !== 'running' && (
            <div className="text-xs text-foreground-muted space-y-1">
              <p>Existing index: {indexInfo.tracks} tracks · {indexInfo.windows} windows</p>
              {indexInfo.sourceName && (
                <p>Source: {indexInfo.sourceName}{indexInfo.sourceType ? ` (${indexInfo.sourceType})` : ''}</p>
              )}
              {indexInfo.builtAt && <p>Last built: {new Date(indexInfo.builtAt).toLocaleString()}</p>}
            </div>
          )}
          {indexAvailable && libraryFiles.length > 0 && (
            <p className="text-xs text-foreground-muted">
              Use “Add To Library” to append new tracks without rebuilding the index.
            </p>
          )}
        </div>

      <div className="glass-glow rounded-2xl overflow-hidden p-1">
        <WaveformViewer />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="glass p-6 rounded-2xl space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium font-display">Cues</h3>
            <p className="text-sm text-foreground-muted">Click a cue card to set active cue.</p>
          </div>
          <div className="grid gap-4 max-h-[520px] overflow-y-auto pr-1">
            {project?.cues.map((cue, index) => (
              <CueCard key={cue.id} cue={cue} index={index} />
            ))}
          </div>
        </div>

        <div className="glass p-6 rounded-2xl space-y-3">
          <div>
            <div className="flex items-center justify-between gap-4">
              <div>
                <h3 className="text-lg font-medium font-display mb-1">Matches</h3>
                <p className="text-sm text-foreground-muted">
                  {selectedCue && selectedCueIndex >= 0
                    ? `Cue ${selectedCueIndex + 1} · ${selectedCue.start_tc}–${selectedCue.end_tc}`
                    : 'Select a cue to match'}
                </p>
              </div>
              <Button
                size="sm"
                className="btn-primary px-4"
                onClick={handleMatch}
                disabled={!indexAvailable || !sessionId || !selectedCueId || isMatching || buildStatus === 'running'}
              >
                {isMatching ? 'Matching…' : 'Find Matches'}
              </Button>
            </div>
          </div>
          {matchError && <p className="text-xs text-red-400">{matchError}</p>}
          {matches.length === 0 ? (
            <p className="text-sm text-foreground-muted">No matches yet.</p>
          ) : (
            <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
              {matches.map((match, idx) => {
                const fileName = match.track_path.replace(/\\/g, '/').split('/').pop()
                const isFocused = selectedCueId ? focusedMatchId[selectedCueId] === match.window_id : false
                const applied = selectedCue?.project_context?.replacement
                const isApplied = !!applied &&
                  applied.track_path === match.track_path &&
                  Math.abs(applied.start - match.start) < 0.01 &&
                  Math.abs(applied.end - match.end) < 0.01
                return (
                  <div
                    key={match.window_id}
                    className={`border rounded-lg px-3 py-2 text-xs ${isFocused ? 'border-primary/60 bg-primary/10' : 'border-primary/15'}`}
                    onClick={() => selectedCueId && setFocusedMatchId((prev) => ({ ...prev, [selectedCueId]: match.window_id }))}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">#{idx + 1} {fileName}</span>
                      <span className="text-foreground-muted">{match.score.toFixed(3)}</span>
                    </div>
                    <div className="text-foreground-muted">
                      {match.start.toFixed(1)}s – {match.end.toFixed(1)}s · window {match.window_size}s
                    </div>
                    {match.reasons && match.reasons.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {match.reasons.map((reason) => (
                          <span
                            key={`${match.window_id}-${reason}`}
                            className="text-[10px] px-2 py-0.5 rounded-full bg-primary/15 text-primary"
                          >
                            {reason}
                          </span>
                        ))}
                      </div>
                    )}
                    <div className="mt-2 flex items-center gap-2">
                      <Button size="sm" variant="outline" className="glass-lighter border-primary/20" onClick={() => applyMatch(match)}>
                        {isApplied ? 'Applied' : 'Use This'}
                      </Button>
                      {isApplied && (
                        <span className="text-[10px] uppercase tracking-wide text-primary">Matched</span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        <div className="glass p-6 rounded-2xl space-y-4">
          <div>
            <h3 className="text-lg font-medium font-display mb-2">Details</h3>
            <p className="text-sm text-foreground-muted">
              Compare the selected cue and the chosen match.
            </p>
          </div>
          <div className="space-y-4 text-sm">
            <div>
              <div className="text-foreground-muted uppercase tracking-wide text-xs mb-2">Cue</div>
              <div className="flex flex-wrap gap-x-4 gap-y-1">
                <span>Length: {selectedCue ? formatSeconds(selectedCue.end - selectedCue.start) : '—'}</span>
                <span>BPM: {formatBpm(selectedCue?.musical_profile.bpm)}</span>
                <span>Key: {selectedCue?.musical_profile.key ?? '—'}</span>
              </div>
            </div>
            <div>
              <div className="text-foreground-muted uppercase tracking-wide text-xs mb-2">Match</div>
              {focusedMatch ? (
                <div className="space-y-2">
                  <div className="flex flex-wrap gap-x-4 gap-y-1">
                    <span>BPM: {formatBpm(focusedMatch.bpm)}</span>
                    <span>Key: {focusedMatch.key ?? '—'}</span>
                    <span>Score: {focusedMatch.score.toFixed(3)}</span>
                  </div>
                  <div className="flex flex-wrap gap-x-4 gap-y-1">
                    <span>Range: {focusedMatch.start.toFixed(1)}s–{focusedMatch.end.toFixed(1)}s</span>
                    <span>Window: {focusedMatch.window_size}s</span>
                  </div>
                  {focusedMatch.genres && focusedMatch.genres.length > 0 && (
                    <div className="text-foreground-muted">
                      Genre: {focusedMatch.genres.join(', ')}
                    </div>
                  )}
                  {focusedMatch.instruments && focusedMatch.instruments.length > 0 && (
                    <div className="text-foreground-muted">
                      Instruments: {focusedMatch.instruments.join(', ')}
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-foreground-muted">Select a match to see details</div>
              )}
            </div>
            {selectedCue?.project_context?.replacement && (
              <div className="text-xs text-foreground-muted">
                Applied: {selectedCue.project_context.replacement.track_path.split('/').pop()}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
