import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { api } from '@/lib/api'
import { useAppStore } from '@/lib/store'
import { WaveformViewer } from '@/components/WaveformViewer'
import { CueCard } from '@/components/CueCard'
import { LibraryManager } from '@/components/LibraryManager'
import type { LibraryMatch } from '@/lib/types'

export function TrackReplacement() {
  const { project, fileName, sessionId, activeCueId, setActiveCueId, updateCueContext, setCurrentPage, reset, segments } = useAppStore()

  const cueCount = project?.cues.length ?? 0
  const cueOptions = useMemo(
    () => (project?.cues ?? []).map((cue, index) => ({ id: cue.id, label: `Cue ${index + 1}` })),
    [project]
  )

  // Tab state
  const [activeTab, setActiveTab] = useState<'match' | 'library'>('match')

  // Library build state (shared between Match and Library tabs)
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
  const [indexAvailable, setIndexAvailable] = useState(false)
  const [indexInfo, setIndexInfo] = useState<{
    tracks: number
    windows: number
  } | null>(null)

  // Match state
  const [selectedCueId, setSelectedCueId] = useState<string | null>(activeCueId)
  const [matchesByCue, setMatchesByCue] = useState<Record<string, LibraryMatch[]>>({})
  const [focusedMatchId, setFocusedMatchId] = useState<Record<string, string | null>>({})
  const [isMatching, setIsMatching] = useState(false)
  const [matchError, setMatchError] = useState('')

  // Audio playback for match previews
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playingMatchId, setPlayingMatchId] = useState<string | null>(null)
  const playEndTimerRef = useRef<number | null>(null)

  const stopMatchPlayback = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.src = ''
    }
    if (playEndTimerRef.current) {
      window.clearInterval(playEndTimerRef.current)
      playEndTimerRef.current = null
    }
    setPlayingMatchId(null)
  }, [])

  const playMatch = useCallback((match: LibraryMatch) => {
    if (playingMatchId === match.window_id) {
      stopMatchPlayback()
      return
    }
    stopMatchPlayback()
    const audio = audioRef.current || new Audio()
    audioRef.current = audio
    audio.src = api.getLibraryAudioUrl(match.track_id)
    audio.currentTime = match.start
    setPlayingMatchId(match.window_id)
    audio.play().catch(() => setPlayingMatchId(null))
    playEndTimerRef.current = window.setInterval(() => {
      if (audio.currentTime >= match.end || audio.paused) {
        stopMatchPlayback()
      }
    }, 100)
    audio.onended = () => stopMatchPlayback()
  }, [playingMatchId, stopMatchPlayback])

  // Cleanup audio on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current.src = ''
      }
      if (playEndTimerRef.current) {
        window.clearInterval(playEndTimerRef.current)
      }
    }
  }, [])

  // Load library info on mount
  useEffect(() => {
    let alive = true
    const loadInfo = async () => {
      try {
        const info = await api.getLibraryInfo()
        if (!alive) return
        setIndexAvailable(info.exists)
        setIndexInfo(info.exists ? { tracks: info.tracks, windows: info.windows } : null)
      } catch {
        if (!alive) return
        setIndexAvailable(false)
        setIndexInfo(null)
      }
    }
    loadInfo()
    return () => { alive = false }
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

  const handleBuildComplete = useCallback(async () => {
    try {
      const info = await api.getLibraryInfo()
      setIndexAvailable(info.exists)
      setIndexInfo(info.exists ? { tracks: info.tracks, windows: info.windows } : null)
    } catch {
      // ignore
    }
  }, [])

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

  const undoMatch = async () => {
    if (!selectedCueId || !sessionId) return
    try {
      await api.updateCueReplacement(sessionId, selectedCueId, {
        track_path: '',
        start: 0,
        end: 0,
        score: 0,
        window_size: 0,
        status: 'draft',
      })
      updateCueContext(selectedCueId, {
        status: 'draft',
        replacement: undefined,
      })
    } catch (err) {
      setMatchError(err instanceof Error ? err.message : 'Failed to undo match')
    }
  }

  const formatSeconds = (value: number | null | undefined) =>
    typeof value === 'number' ? `${value.toFixed(1)}s` : '—'

  const formatBpm = (bpm: number | null | undefined) => {
    if (typeof bpm !== 'number') return '—'
    return bpm % 1 === 0 ? String(Math.round(bpm)) : bpm.toFixed(1)
  }

  return (
    <div className="space-y-4">
      {/* Header with title and tabs */}
      <div className="glass px-6 py-4 rounded-2xl space-y-3">
        <div className="flex items-center justify-between">
          <Button variant="outline" size="sm" onClick={() => setCurrentPage('analysis')} className="glass-lighter border-primary/20">
            ← Back
          </Button>
          <Button variant="outline" size="sm" onClick={reset} className="glass-lighter border-primary/20">
            New Import
          </Button>
        </div>
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <h2 className="text-2xl font-medium font-display">Track Replacement</h2>
            {project && (
              <span className="text-sm text-foreground-muted">
                {fileName ?? 'Analysis'} · {cueCount} cues
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {/* Library summary */}
            {indexAvailable && indexInfo && (
              <span className="text-xs text-foreground-muted mr-3">
                {indexInfo.tracks} tracks · {indexInfo.windows} windows
              </span>
            )}
            {!indexAvailable && (
              <span className="text-xs text-foreground-muted mr-3">No library index</span>
            )}
            {/* Tab buttons */}
            <button
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'match'
                  ? 'bg-primary/20 text-primary'
                  : 'text-foreground-muted hover:text-foreground hover:bg-primary/5'
              }`}
              onClick={() => setActiveTab('match')}
            >
              Match
            </button>
            <button
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                activeTab === 'library'
                  ? 'bg-primary/20 text-primary'
                  : 'text-foreground-muted hover:text-foreground hover:bg-primary/5'
              }`}
              onClick={() => setActiveTab('library')}
            >
              Library
            </button>
          </div>
        </div>
      </div>

      {/* Match tab content */}
      {activeTab === 'match' && (
        <>
          <div className="glass-glow rounded-2xl overflow-hidden p-1">
            <WaveformViewer />
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-[45%_1fr_1fr] gap-6 items-start">
            <div className="glass p-6 rounded-2xl space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium font-display">Cues</h3>
                <p className="text-sm text-foreground-muted">Click a cue card to set active cue.</p>
              </div>
              <div className="grid gap-4">
                {[...(project?.cues ?? [])].sort((a, b) => a.start - b.start).map((cue) => {
                  const segIndex = segments.findIndex(([s, e]) =>
                    Math.abs(s - cue.start) < 0.5 && Math.abs(e - cue.end) < 0.5
                  )
                  return <CueCard key={cue.id} cue={cue} index={segIndex >= 0 ? segIndex : 0} />
                })}
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
                <div className="space-y-2">
                  {matches.map((match, idx) => {
                    const matchFileName = match.track_path.replace(/\\/g, '/').split('/').pop()
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
                          <span className="font-medium">#{idx + 1} {matchFileName}</span>
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
                          <Button
                            size="sm"
                            variant="outline"
                            className="glass-lighter border-primary/20"
                            onClick={(e) => { e.stopPropagation(); playMatch(match) }}
                          >
                            {playingMatchId === match.window_id ? 'Stop' : 'Play'}
                          </Button>
                          {isApplied ? (
                            <Button size="sm" variant="outline" className="glass-lighter border-white/20 text-foreground-muted" onClick={(e) => { e.stopPropagation(); undoMatch() }}>
                              Undo
                            </Button>
                          ) : (
                            <Button size="sm" variant="outline" className="glass-lighter border-primary/20" onClick={() => applyMatch(match)}>
                              Use This
                            </Button>
                          )}
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

            <div className="space-y-4">
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

              {/* Finalize button */}
              <Button
                className="btn-primary px-6 w-full"
                onClick={() => setCurrentPage('licensing')}
                disabled={!project?.cues.some((c) => c.project_context?.status === 'matched')}
              >
                Finalize Track Replacement →
              </Button>
            </div>
          </div>
        </>
      )}

      {/* Library tab content */}
      {activeTab === 'library' && (
        <LibraryManager
          libraryFiles={libraryFiles}
          setLibraryFiles={setLibraryFiles}
          libraryFolderName={libraryFolderName}
          setLibraryFolderName={setLibraryFolderName}
          includeMoods={includeMoods}
          setIncludeMoods={setIncludeMoods}
          buildJobId={buildJobId}
          setBuildJobId={setBuildJobId}
          buildStatus={buildStatus}
          setBuildStatus={setBuildStatus}
          buildMessage={buildMessage}
          setBuildMessage={setBuildMessage}
          progressPercent={progressPercent}
          setProgressPercent={setProgressPercent}
          progressCurrent={progressCurrent}
          setProgressCurrent={setProgressCurrent}
          progressTotal={progressTotal}
          setProgressTotal={setProgressTotal}
          windowsIndexed={windowsIndexed}
          setWindowsIndexed={setWindowsIndexed}
          isBuilding={isBuilding}
          setIsBuilding={setIsBuilding}
          isUploading={isUploading}
          setIsUploading={setIsUploading}
          indexAvailable={indexAvailable}
          setIndexAvailable={setIndexAvailable}
          onBuildComplete={handleBuildComplete}
        />
      )}
    </div>
  )
}
