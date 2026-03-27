import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '@/lib/api'
import { useAppStore } from '@/lib/store'
import { WaveformViewer } from '@/components/WaveformViewer'
import { LibraryManager } from '@/components/LibraryManager'
import type { LibraryMatch } from '@/lib/types'
import { secondsToTimecode } from '@/lib/timecode'

/* ═══════════════════════════════════════════════════════════════════════════
   DESIGN TOKENS
   ═══════════════════════════════════════════════════════════════════════════ */

const CANVAS = '#D6D6D4'
const BENTO = '#EBEBEB'
const LIST_BG = '#F3F3F1'
const ACCENT = '#FFD659'
const TEXT_SUB = '#4A4A4A'

const titleHeavy: React.CSSProperties = { fontWeight: 800, letterSpacing: '-0.04em' }
const labelBold: React.CSSProperties = {
  fontWeight: 700, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em', color: TEXT_SUB,
}

const pillBtn: React.CSSProperties = {
  display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 6,
  padding: '8px 20px', borderRadius: 999, fontWeight: 700, fontSize: 13,
  border: 'none', cursor: 'pointer', transition: 'all 0.2s',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
}

/* ═══════════════════════════════════════════════════════════════════════════
   ICONS
   ═══════════════════════════════════════════════════════════════════════════ */

function ArrowLeftIcon({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 12H5M12 19l-7-7 7-7" />
    </svg>
  )
}

function PlayIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M8 5.14v13.72a1 1 0 0 0 1.5.86l11.72-6.86a1 1 0 0 0 0-1.72L9.5 4.28A1 1 0 0 0 8 5.14z" />
    </svg>
  )
}

function StopIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </svg>
  )
}

function SearchIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.35-4.35" />
    </svg>
  )
}

function CheckIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}

function UndoIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 7v6h6" /><path d="M3 13a9 9 0 0 1 15.36-6.36L21 9" />
    </svg>
  )
}

function LibraryIcon({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" /><path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15z" />
    </svg>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════════════════ */

export function TrackReplacement() {
  const { project, fileName, sessionId, activeCueId, setActiveCueId, updateCueContext, setCurrentPage, reset, segments, userName, startTimecode, framerate } = useAppStore()

  const cueCount = project?.cues.length ?? 0
  const cueOptions = useMemo(
    () => (project?.cues ?? []).map((cue, index) => ({ id: cue.id, label: `Cue ${index + 1}` })),
    [project]
  )

  const [activeTab, setActiveTab] = useState<'match' | 'library'>('match')
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
  const [indexInfo, setIndexInfo] = useState<{ tracks: number; windows: number } | null>(null)
  const [segmentationMode, setSegmentationMode] = useState<'fixed' | 'structure'>('structure')
  const [minSectionLength, setMinSectionLength] = useState(8)

  const [selectedCueId, setSelectedCueId] = useState<string | null>(activeCueId)
  const [matchesByCue, setMatchesByCue] = useState<Record<string, LibraryMatch[]>>({})
  const [focusedMatchId, setFocusedMatchId] = useState<Record<string, string | null>>({})
  const [isMatching, setIsMatching] = useState(false)
  const [matchError, setMatchError] = useState('')

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playingMatchId, setPlayingMatchId] = useState<string | null>(null)
  const playEndTimerRef = useRef<number | null>(null)

  const initials = userName
    ? userName.split(/\s+/).map((w) => w[0]).join('').toUpperCase().slice(0, 2)
    : 'U'

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
    } catch { /* ignore */ }
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

  const cuesSorted = useMemo(
    () => [...(project?.cues ?? [])].sort((a, b) => a.start - b.start),
    [project]
  )

  const matchedCount = project?.cues.filter((c) => c.project_context?.status === 'matched').length ?? 0

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden',
      backgroundColor: CANVAS, color: '#000000',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    }}>
      {/* ── HEADER ── */}
      <header style={{
        height: 64, flexShrink: 0, backgroundColor: '#fff', borderBottom: '1px solid rgba(0,0,0,0.08)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 32px',
        position: 'relative', zIndex: 50,
      }}>
        {/* Left */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, width: '33%' }}>
          <button
            onClick={() => setCurrentPage('analysis')}
            style={{
              width: 40, height: 40, borderRadius: '50%', backgroundColor: '#fff',
              border: '1px solid rgba(0,0,0,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'pointer', transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#F9F9F9' }}
            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#fff' }}
          >
            <ArrowLeftIcon size={20} />
          </button>
          <div>
            <div style={{ ...titleHeavy, fontSize: 16 }}>Find Matches</div>
            <div style={{ fontSize: 11, color: TEXT_SUB, fontWeight: 500 }}>
              {fileName ?? 'Analysis'} · {cueCount} cues
            </div>
          </div>
        </div>

        {/* Center — tabs */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 4, backgroundColor: BENTO, borderRadius: 999, padding: 3 }}>
          <button
            onClick={() => setActiveTab('match')}
            style={{
              ...pillBtn, padding: '6px 18px', fontSize: 12,
              backgroundColor: activeTab === 'match' ? '#000' : 'transparent',
              color: activeTab === 'match' ? '#fff' : TEXT_SUB,
            }}
          >
            <SearchIcon size={14} />
            Match
          </button>
          <button
            onClick={() => setActiveTab('library')}
            style={{
              ...pillBtn, padding: '6px 18px', fontSize: 12,
              backgroundColor: activeTab === 'library' ? '#000' : 'transparent',
              color: activeTab === 'library' ? '#fff' : TEXT_SUB,
            }}
          >
            <LibraryIcon size={14} />
            Library
          </button>
        </div>

        {/* Right */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 12, width: '33%' }}>
          {indexAvailable && indexInfo && (
            <span style={{ fontSize: 11, color: TEXT_SUB, fontWeight: 600 }}>
              {indexInfo.tracks} tracks · {indexInfo.windows} windows
            </span>
          )}
          {!indexAvailable && (
            <span style={{ fontSize: 11, color: '#B44', fontWeight: 600 }}>No library index</span>
          )}
          <div style={{
            width: 40, height: 40, borderRadius: '50%', backgroundColor: '#000', color: '#fff',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 14,
            cursor: 'pointer',
          }}>
            {initials}
          </div>
        </div>
      </header>

      {/* ── MAIN CONTENT ── */}
      <main style={{ flex: 1, overflow: 'auto', padding: '24px 32px', display: 'flex', flexDirection: 'column', gap: 20 }}>

        {activeTab === 'match' && (
          <>
            {/* Waveform */}
            <div style={{ backgroundColor: BENTO, borderRadius: 28, padding: 20, boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
              <WaveformViewer />
            </div>

            {/* Three-column grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 20, alignItems: 'start' }}>

              {/* ── CUES column ── */}
              <div style={{ backgroundColor: BENTO, borderRadius: 28, padding: 24, boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                  <span style={{ ...titleHeavy, fontSize: 14 }}>Cues</span>
                  <span style={{ ...labelBold, fontSize: 9 }}>Click to select</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {cuesSorted.map((cue, idx) => {
                    const isSelected = cue.id === selectedCueId
                    const isMatched = cue.project_context?.status === 'matched'
                    const dur = cue.end - cue.start
                    const mm = String(Math.floor(dur / 60)).padStart(2, '0')
                    const ss = String(Math.floor(dur % 60)).padStart(2, '0')

                    return (
                      <div
                        key={cue.id}
                        onClick={() => { setSelectedCueId(cue.id); setActiveCueId(cue.id) }}
                        style={{
                          backgroundColor: isSelected ? '#fff' : LIST_BG,
                          borderRadius: 16, padding: '14px 16px', cursor: 'pointer',
                          border: isSelected ? '2px solid #000' : '2px solid transparent',
                          transition: 'all 0.15s ease',
                        }}
                        onMouseEnter={(e) => { if (!isSelected) e.currentTarget.style.backgroundColor = '#fff' }}
                        onMouseLeave={(e) => { if (!isSelected) e.currentTarget.style.backgroundColor = LIST_BG }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{
                              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                              width: 28, height: 28, borderRadius: '50%',
                              backgroundColor: isMatched ? '#22C55E' : '#000', color: '#fff',
                              fontSize: 11, fontWeight: 800,
                            }}>
                              {isMatched ? <CheckIcon size={14} /> : `#${idx + 1}`}
                            </span>
                            <span style={{ fontSize: 12, fontWeight: 600, fontFamily: 'SF Mono, monospace', color: TEXT_SUB }}>
                              {secondsToTimecode(cue.start, framerate, startTimecode)} – {secondsToTimecode(cue.end, framerate, startTimecode)}
                            </span>
                          </div>
                          <span style={{ fontSize: 11, fontWeight: 600, color: TEXT_SUB }}>{mm}:{ss}</span>
                        </div>

                        {/* Quick info row */}
                        <div style={{ display: 'flex', gap: 12, fontSize: 11, fontWeight: 600, color: TEXT_SUB }}>
                          <span>BPM {formatBpm(cue.musical_profile.bpm)}</span>
                          <span>{cue.musical_profile.key ?? '—'}</span>
                        </div>

                        {/* Tags */}
                        {cue.musical_profile.curated && (
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
                            {(cue.musical_profile.curated.genres ?? []).slice(0, 2).map((g) => (
                              <span key={g} style={{
                                display: 'inline-block', padding: '2px 8px', borderRadius: 999,
                                backgroundColor: '#000', color: '#fff', fontSize: 9, fontWeight: 700,
                                textTransform: 'uppercase', letterSpacing: '0.05em',
                              }}>{g}</span>
                            ))}
                            {(cue.musical_profile.curated.style ?? []).slice(0, 2).map((s) => (
                              <span key={s} style={{
                                display: 'inline-block', padding: '2px 8px', borderRadius: 999,
                                backgroundColor: 'transparent', color: '#000', fontSize: 9, fontWeight: 700,
                                textTransform: 'uppercase', letterSpacing: '0.05em',
                                border: '1px solid rgba(0,0,0,0.2)',
                              }}>{s}</span>
                            ))}
                          </div>
                        )}

                        {isMatched && (
                          <div style={{ marginTop: 6, fontSize: 10, fontWeight: 700, color: '#22C55E', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Matched
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* ── MATCHES column ── */}
              <div style={{ backgroundColor: BENTO, borderRadius: 28, padding: 24, boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                  <span style={{ ...titleHeavy, fontSize: 14 }}>Matches</span>
                  <button
                    onClick={handleMatch}
                    disabled={!indexAvailable || !sessionId || !selectedCueId || isMatching || buildStatus === 'running'}
                    style={{
                      ...pillBtn, padding: '6px 16px', fontSize: 12,
                      backgroundColor: ACCENT, color: '#000',
                      opacity: (!indexAvailable || !sessionId || !selectedCueId || isMatching) ? 0.4 : 1,
                    }}
                  >
                    <SearchIcon size={13} />
                    {isMatching ? 'Matching...' : 'Find'}
                  </button>
                </div>

                <div style={{ fontSize: 11, color: TEXT_SUB, fontWeight: 500, marginBottom: 16 }}>
                  {selectedCue && selectedCueIndex >= 0
                    ? `Cue ${selectedCueIndex + 1} · ${secondsToTimecode(selectedCue.start, framerate, startTimecode)}–${secondsToTimecode(selectedCue.end, framerate, startTimecode)}`
                    : 'Select a cue to match'}
                </div>

                {matchError && (
                  <div style={{ backgroundColor: '#FEE2E2', color: '#991B1B', borderRadius: 12, padding: '8px 12px', fontSize: 12, fontWeight: 600, marginBottom: 12 }}>
                    {matchError}
                  </div>
                )}

                {matches.length === 0 ? (
                  <div style={{ backgroundColor: LIST_BG, borderRadius: 16, padding: 32, textAlign: 'center' }}>
                    <div style={{ fontSize: 13, color: TEXT_SUB, fontWeight: 500 }}>
                      {indexAvailable ? 'No matches yet. Click "Find" to search.' : 'Build a library index first.'}
                    </div>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {matches.map((match, idx) => {
                      const matchFileName = match.track_path.replace(/\\/g, '/').split('/').pop()
                      const isFocused = selectedCueId ? focusedMatchId[selectedCueId] === match.window_id : false
                      const applied = selectedCue?.project_context?.replacement
                      const isApplied = !!applied &&
                        applied.track_path === match.track_path &&
                        Math.abs(applied.start - match.start) < 0.01 &&
                        Math.abs(applied.end - match.end) < 0.01
                      const similarityPct = Math.round(match.score * 100)

                      return (
                        <div
                          key={match.window_id}
                          onClick={() => selectedCueId && setFocusedMatchId((prev) => ({ ...prev, [selectedCueId]: match.window_id }))}
                          style={{
                            backgroundColor: isFocused ? '#fff' : LIST_BG,
                            borderRadius: 16, padding: '14px 16px', cursor: 'pointer',
                            border: isFocused ? '2px solid #000' : '2px solid transparent',
                            transition: 'all 0.15s ease',
                          }}
                          onMouseEnter={(e) => { if (!isFocused) e.currentTarget.style.backgroundColor = '#fff' }}
                          onMouseLeave={(e) => { if (!isFocused) e.currentTarget.style.backgroundColor = LIST_BG }}
                        >
                          {/* Top row */}
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <span style={{
                                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                                width: 24, height: 24, borderRadius: '50%', backgroundColor: '#000', color: '#fff',
                                fontSize: 10, fontWeight: 800,
                              }}>
                                {idx + 1}
                              </span>
                              <span style={{ fontSize: 13, fontWeight: 700 }}>{matchFileName}</span>
                            </div>
                            <span style={{ fontSize: 12, fontWeight: 700, color: similarityPct >= 90 ? '#22C55E' : TEXT_SUB }}>
                              {similarityPct}%
                            </span>
                          </div>

                          {/* Metadata */}
                          <div style={{ fontSize: 11, color: TEXT_SUB, fontWeight: 500, marginBottom: 8 }}>
                            {match.start.toFixed(1)}s – {match.end.toFixed(1)}s
                            {match.segment_type === 'structure' && match.section_index != null && match.total_sections != null
                              ? ` · section ${match.section_index + 1}/${match.total_sections}`
                              : ` · window ${match.window_size}s`}
                          </div>

                          {/* Similarity bar */}
                          <div style={{ height: 4, backgroundColor: '#D1D1D1', borderRadius: 999, overflow: 'hidden', marginBottom: 8 }}>
                            <div style={{ height: '100%', width: `${similarityPct}%`, backgroundColor: '#000', borderRadius: 999, transition: 'width 0.3s ease' }} />
                          </div>

                          {/* Reason tags */}
                          {match.reasons && match.reasons.length > 0 && (
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 10 }}>
                              {match.reasons.map((reason) => (
                                <span key={`${match.window_id}-${reason}`} style={{
                                  display: 'inline-block', padding: '2px 8px', borderRadius: 999,
                                  backgroundColor: 'transparent', border: '1px solid rgba(0,0,0,0.15)',
                                  fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em',
                                  color: TEXT_SUB,
                                }}>
                                  {reason}
                                </span>
                              ))}
                            </div>
                          )}

                          {/* Actions */}
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <button
                              onClick={(e) => { e.stopPropagation(); playMatch(match) }}
                              style={{
                                ...pillBtn, padding: '4px 12px', fontSize: 11,
                                backgroundColor: playingMatchId === match.window_id ? '#000' : '#fff',
                                color: playingMatchId === match.window_id ? '#fff' : '#000',
                                border: '1px solid rgba(0,0,0,0.15)',
                              }}
                            >
                              {playingMatchId === match.window_id ? <><StopIcon size={12} /> Stop</> : <><PlayIcon size={12} /> Play</>}
                            </button>
                            {isApplied ? (
                              <button
                                onClick={(e) => { e.stopPropagation(); undoMatch() }}
                                style={{ ...pillBtn, padding: '4px 12px', fontSize: 11, backgroundColor: '#fff', color: '#000', border: '1px solid rgba(0,0,0,0.15)' }}
                              >
                                <UndoIcon size={12} /> Undo
                              </button>
                            ) : (
                              <button
                                onClick={(e) => { e.stopPropagation(); applyMatch(match) }}
                                style={{ ...pillBtn, padding: '4px 12px', fontSize: 11, backgroundColor: '#000', color: '#fff' }}
                              >
                                <CheckIcon size={12} /> Use This
                              </button>
                            )}
                            {isApplied && (
                              <span style={{
                                display: 'inline-flex', alignItems: 'center', gap: 4,
                                padding: '2px 10px', borderRadius: 999, backgroundColor: '#22C55E', color: '#fff',
                                fontSize: 9, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em',
                              }}>
                                <CheckIcon size={10} /> Matched
                              </span>
                            )}
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>

              {/* ── DETAILS column ── */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <div style={{ backgroundColor: BENTO, borderRadius: 28, padding: 24, boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
                  <span style={{ ...titleHeavy, fontSize: 14, display: 'block', marginBottom: 8 }}>Comparison</span>
                  <p style={{ fontSize: 11, color: TEXT_SUB, fontWeight: 500, marginBottom: 16 }}>
                    Selected cue vs. chosen match.
                  </p>

                  {/* Cue details */}
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ ...labelBold, marginBottom: 8 }}>Cue</div>
                    <div style={{ backgroundColor: LIST_BG, borderRadius: 16, padding: '12px 14px' }}>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 16px', fontSize: 13 }}>
                        <span><strong>Length</strong> {selectedCue ? formatSeconds(selectedCue.end - selectedCue.start) : '—'}</span>
                        <span><strong>BPM</strong> {formatBpm(selectedCue?.musical_profile.bpm)}</span>
                        <span><strong>Key</strong> {selectedCue?.musical_profile.key ?? '—'}</span>
                      </div>
                    </div>
                  </div>

                  {/* Match details */}
                  <div>
                    <div style={{ ...labelBold, marginBottom: 8 }}>Match</div>
                    {focusedMatch ? (
                      <div style={{ backgroundColor: LIST_BG, borderRadius: 16, padding: '12px 14px' }}>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 16px', fontSize: 13, marginBottom: 6 }}>
                          <span><strong>BPM</strong> {formatBpm(focusedMatch.bpm)}</span>
                          <span><strong>Key</strong> {focusedMatch.key ?? '—'}</span>
                          <span><strong>Score</strong> {focusedMatch.score.toFixed(3)}</span>
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 16px', fontSize: 12, color: TEXT_SUB }}>
                          <span>Range: {focusedMatch.start.toFixed(1)}s–{focusedMatch.end.toFixed(1)}s</span>
                          <span>Window: {focusedMatch.window_size}s</span>
                        </div>
                        {focusedMatch.genres && focusedMatch.genres.length > 0 && (
                          <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                            {focusedMatch.genres.map((g) => (
                              <span key={g} style={{
                                display: 'inline-block', padding: '2px 8px', borderRadius: 999,
                                backgroundColor: '#000', color: '#fff', fontSize: 9, fontWeight: 700,
                                textTransform: 'uppercase', letterSpacing: '0.05em',
                              }}>{g}</span>
                            ))}
                          </div>
                        )}
                        {focusedMatch.instruments && focusedMatch.instruments.length > 0 && (
                          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                            {focusedMatch.instruments.map((i) => (
                              <span key={i} style={{
                                display: 'inline-block', padding: '2px 8px', borderRadius: 999,
                                border: '1px solid rgba(0,0,0,0.15)', fontSize: 9, fontWeight: 600,
                                textTransform: 'uppercase', letterSpacing: '0.05em', color: TEXT_SUB,
                              }}>{i}</span>
                            ))}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div style={{ backgroundColor: LIST_BG, borderRadius: 16, padding: '20px 14px', textAlign: 'center' }}>
                        <span style={{ fontSize: 12, color: TEXT_SUB }}>Select a match to compare</span>
                      </div>
                    )}
                  </div>

                  {selectedCue?.project_context?.replacement && (
                    <div style={{ marginTop: 12, padding: '8px 12px', backgroundColor: '#ECFDF5', borderRadius: 12, fontSize: 11, fontWeight: 600, color: '#065F46' }}>
                      Applied: {selectedCue.project_context.replacement.track_path.split('/').pop()}
                    </div>
                  )}
                </div>

                {/* Finalize button */}
                <button
                  onClick={() => setCurrentPage('rights')}
                  disabled={matchedCount === 0}
                  style={{
                    ...pillBtn, width: '100%', padding: '14px 24px', fontSize: 14,
                    backgroundColor: matchedCount > 0 ? '#000' : BENTO,
                    color: matchedCount > 0 ? '#fff' : TEXT_SUB,
                    opacity: matchedCount === 0 ? 0.5 : 1,
                  }}
                >
                  Set Rights ({matchedCount}/{cueCount} matched)
                </button>
              </div>
            </div>
          </>
        )}

        {/* Library tab */}
        {activeTab === 'library' && (
          <div style={{ backgroundColor: BENTO, borderRadius: 28, padding: 24, boxShadow: '0 1px 3px rgba(0,0,0,0.04)' }}>
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
              segmentationMode={segmentationMode}
              setSegmentationMode={setSegmentationMode}
              minSectionLength={minSectionLength}
              setMinSectionLength={setMinSectionLength}
            />
          </div>
        )}
      </main>

      {/* ── FOOTER ── */}
      <footer style={{
        height: 64, flexShrink: 0, backgroundColor: '#fff', borderTop: '1px solid rgba(0,0,0,0.08)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 32px',
        position: 'relative', zIndex: 50,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 13, fontWeight: 700, color: TEXT_SUB }}>
            {matchedCount} of {cueCount} cues matched
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button onClick={() => setCurrentPage('analysis')} style={{ ...pillBtn, backgroundColor: '#fff', color: '#000', border: '1px solid rgba(0,0,0,0.15)' }}>
            Back to Analysis
          </button>
          <button
            onClick={() => setCurrentPage('rights')}
            disabled={matchedCount === 0}
            style={{ ...pillBtn, backgroundColor: matchedCount > 0 ? ACCENT : BENTO, color: '#000', opacity: matchedCount === 0 ? 0.5 : 1 }}
          >
            Set Rights →
          </button>
        </div>
      </footer>
    </div>
  )
}
