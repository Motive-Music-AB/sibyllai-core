import { useState, useRef, useEffect, useCallback, type DragEvent, type ChangeEvent } from 'react'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions'
import { useAppStore } from '@/lib/store'
import { api } from '@/lib/api'
import { secondsToTimecode } from '@/lib/timecode'
import type { Cue } from '@/lib/types'
import './motive-pipeline.css'

function fmtDur(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/* =========================================================================
   SVG Icons — simple inline vectors for the brutalist UI
   ========================================================================= */
const PlayIcon = () => (
  <svg viewBox="0 0 24 24"><polygon points="8,5 19,12 8,19" /></svg>
)
const PauseIcon = () => (
  <svg viewBox="0 0 24 24"><rect x="6" y="5" width="4" height="14" /><rect x="14" y="5" width="4" height="14" /></svg>
)
const StopIcon = () => (
  <svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" /></svg>
)
const PrevIcon = () => (
  <svg viewBox="0 0 24 24"><rect x="4" y="5" width="3" height="14" /><polygon points="20,5 9,12 20,19" /></svg>
)
const NextIcon = () => (
  <svg viewBox="0 0 24 24"><rect x="17" y="5" width="3" height="14" /><polygon points="4,5 15,12 4,19" /></svg>
)

/* =========================================================================
   MAIN COMPONENT
   ========================================================================= */
export function MotivePipeline() {
  /* -- Store ------------------------------------------------------------ */
  const {
    uploadedFile, fileId, fileName, duration,
    segments, selectedSegments, setSelectedSegments,
    mixType, setMixType,
    musicThreshold, setMusicThreshold,
    minGap, setMinGap,
    minCueLength, setMinCueLength,
    silenceThreshold, setSilenceThreshold,
    isSegmenting, setIsSegmenting,
    isAnalyzing, setIsAnalyzing,
    analysisProgress, analysisStatus, setAnalysisProgress,
    project, sessionId,
    activeCueId, setActiveCueId, setPlayingCueId,
    setUploadedFile, setSegments, setProject,
    framerate, startTimecode,
    setCurrentPage, reset,
  } = useAppStore()

  /* -- Local state ------------------------------------------------------ */
  const [isPlaying, setIsPlaying] = useState(false)
  const [isReady, setIsReady] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)

  /* -- Refs ------------------------------------------------------------- */
  const waveformRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WaveSurfer | null>(null)
  const regionsRef = useRef<RegionsPlugin | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const lastTcRef = useRef(0)

  /* -- Derived ---------------------------------------------------------- */
  const activeCue: Cue | null = project?.cues.find(c => c.id === activeCueId) ?? null
  const cuesSorted = project ? [...project.cues].sort((a, b) => a.start - b.start) : []
  const itemCount = project ? cuesSorted.length : segments.length

  /* =====================================================================
     WAVESURFER
     ===================================================================== */
  useEffect(() => {
    if (!waveformRef.current || !uploadedFile || !fileId) return

    const ws = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#080808',
      progressColor: '#555',
      cursorColor: '#080808',
      cursorWidth: 1,
      barWidth: 4,
      barGap: 2,
      barRadius: 2,
      height: 'auto',
      normalize: true,
    })

    const regions = ws.registerPlugin(RegionsPlugin.create())
    wsRef.current = ws
    regionsRef.current = regions

    const url = URL.createObjectURL(uploadedFile)
    ws.load(url)

    ws.on('ready', () => {
      setIsReady(true)
      setCurrentTime(0)
      setTimeout(() => {
        if (wsRef.current !== ws) return
        try {
          const dur = ws.getDuration()
          const maxLen = Math.min(2000, Math.max(100, Math.ceil(dur * 50)))
          const peaks = ws.exportPeaks({ maxLength: maxLen })
          ws.setOptions({ peaks, duration: dur })
        } catch {}
      }, 0)
    })

    ws.on('play', () => setIsPlaying(true))
    ws.on('pause', () => setIsPlaying(false))
    ws.on('audioprocess', (t) => {
      const now = Date.now()
      if (now - lastTcRef.current < 50) return
      lastTcRef.current = now
      setCurrentTime(t)
    })
    ws.on('seeking', (t) => setCurrentTime(t))
    ws.on('click', () => setCurrentTime(ws.getCurrentTime()))

    return () => {
      URL.revokeObjectURL(url)
      ws.destroy()
      wsRef.current = null
      regionsRef.current = null
      setIsReady(false)
      setIsPlaying(false)
    }
  }, [uploadedFile, fileId])

  /* -- Sync regions ----------------------------------------------------- */
  useEffect(() => {
    const regions = regionsRef.current
    if (!regions || !isReady) return
    regions.clearRegions()

    const segs = project
      ? cuesSorted.map(c => [c.start, c.end] as [number, number])
      : segments

    segs.forEach(([start, end], i) => {
      const isSelected = selectedSegments.some(
        ([s, e]) => Math.abs(s - start) < 0.5 && Math.abs(e - end) < 0.5
      )
      const isCueActive = project && cuesSorted[i]?.id === activeCueId

      regions.addRegion({
        start,
        end,
        color: isCueActive
          ? 'rgba(8,8,8,0.12)'
          : isSelected
          ? 'rgba(8,8,8,0.06)'
          : 'rgba(0,0,0,0.02)',
        drag: false,
        resize: false,
      })
    })
  }, [segments, selectedSegments, isReady, activeCueId, project])

  /* =====================================================================
     FILE UPLOAD
     ===================================================================== */
  const handleFileSelect = useCallback(async (file: File) => {
    if (!file.type.startsWith('audio/') && !file.type.startsWith('video/')) {
      setError('Unsupported format')
      return
    }
    setError(null)
    setIsUploading(true)
    setUploadProgress(0)
    try {
      const res = await api.uploadFile(file, (p) => setUploadProgress(p))
      setUploadedFile(file, res.file_id, res.filename, '01:00:00:00', 24)
    } catch {
      setError('Upload failed — is the backend running on port 8003?')
    }
    setIsUploading(false)
  }, [setUploadedFile])

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelect(file)
  }, [handleFileSelect])

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleFileInput = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFileSelect(file)
  }, [handleFileSelect])

  /* =====================================================================
     TRANSPORT
     ===================================================================== */
  const handlePlay = useCallback(() => wsRef.current?.playPause(), [])

  const handleStop = useCallback(() => {
    const ws = wsRef.current
    if (!ws) return
    ws.pause()
    ws.setTime(0)
  }, [])

  const handlePrev = useCallback(() => {
    const ws = wsRef.current
    if (!ws) return
    const pos = ws.getCurrentTime()
    const starts = project ? cuesSorted.map(c => c.start) : segments.map(([s]) => s)
    const prev = [...starts].reverse().find(s => s < pos - 0.5)
    ws.setTime(prev ?? 0)
  }, [segments, project, cuesSorted])

  const handleNext = useCallback(() => {
    const ws = wsRef.current
    if (!ws) return
    const pos = ws.getCurrentTime()
    const starts = project ? cuesSorted.map(c => c.start) : segments.map(([s]) => s)
    const next = starts.find(s => s > pos + 0.5)
    if (next !== undefined) ws.setTime(next)
  }, [segments, project, cuesSorted])

  /* =====================================================================
     SEGMENT PREVIEW
     ===================================================================== */
  const handlePreview = useCallback(async () => {
    if (!fileId) return
    setIsSegmenting(true)
    setError(null)
    try {
      const res = await api.getSegmentPreview({
        file_id: fileId,
        music_thresh: musicThreshold,
        min_gap: minGap,
        min_cue_length: minCueLength,
        silence_thresh: silenceThreshold,
        mix_type: mixType,
      })
      setSegments(res.segments, res.duration)
    } catch {
      setError('Segment detection failed')
    }
    setIsSegmenting(false)
  }, [fileId, musicThreshold, minGap, minCueLength, silenceThreshold, mixType, setSegments, setIsSegmenting])

  /* =====================================================================
     ANALYSIS
     ===================================================================== */
  const handleAnalyze = useCallback(async () => {
    if (!fileId || selectedSegments.length === 0) return
    setIsAnalyzing(true)
    setAnalysisProgress(0, 'Starting...')
    setError(null)
    try {
      const res = await api.analyzeCues({
        file_id: fileId,
        segments: selectedSegments,
        fps: framerate,
      })
      const interval = setInterval(async () => {
        try {
          const st = await api.getAnalysisStatus(res.session_id)
          setAnalysisProgress(st.progress_percent, st.status)
          if (st.complete && st.project) {
            clearInterval(interval)
            setProject(res.session_id, st.project)
            setIsAnalyzing(false)
          }
          if (st.error) {
            clearInterval(interval)
            setIsAnalyzing(false)
            setError(st.error)
          }
        } catch {
          clearInterval(interval)
          setIsAnalyzing(false)
          setError('Lost backend connection')
        }
      }, 500)
    } catch {
      setIsAnalyzing(false)
      setError('Analysis start failed')
    }
  }, [fileId, selectedSegments, framerate, setIsAnalyzing, setAnalysisProgress, setProject])

  /* =====================================================================
     SEGMENT / CUE CLICK
     ===================================================================== */
  const handleSegClick = useCallback((idx: number) => {
    if (project && cuesSorted[idx]) {
      const cue = cuesSorted[idx]
      setActiveCueId(cue.id)
      setPlayingCueId(cue.id)
      wsRef.current?.setTime(cue.start)
    } else if (segments[idx]) {
      wsRef.current?.setTime(segments[idx][0])
    }
  }, [project, cuesSorted, segments, setActiveCueId, setPlayingCueId])

  const toggleSeg = useCallback((start: number, end: number) => {
    const isSelected = selectedSegments.some(
      ([s, e]) => Math.abs(s - start) < 0.5 && Math.abs(e - end) < 0.5
    )
    if (isSelected) {
      setSelectedSegments(selectedSegments.filter(
        ([s, e]) => !(Math.abs(s - start) < 0.5 && Math.abs(e - end) < 0.5)
      ))
    } else {
      setSelectedSegments([...selectedSegments, [start, end]])
    }
  }, [selectedSegments, setSelectedSegments])

  /* =====================================================================
     RENDER
     ===================================================================== */
  return (
    <div className="pipeline-root">
      <div className="noise-overlay" />

      {/* ==================== HEADER ==================== */}
      <header className="pipeline-header">
        <div className="brand">
          <div className="eye-icon" />
          MOTIVE
        </div>
        <div className="header-controls">
          <button className="btn-pill" onClick={() => reset()}>
            New Session
          </button>
          {project && (
            <button className="btn-pill" onClick={() => setCurrentPage('cuesynch')}>
              Export
            </button>
          )}
          {project && (
            <button className="btn-pill primary" onClick={() => setCurrentPage('replacement')}>
              Track Match
            </button>
          )}
        </div>
      </header>

      {/* ==================== WORKSPACE ==================== */}
      <div className="workspace">

        {/* ---------- LEFT SIDEBAR ---------- */}
        <aside className="sidebar">
          <div className="panel-header">
            <span className="label">Input Sources</span>
            <span className="mono" style={{ fontSize: '0.7rem' }}>[{itemCount || '—'}]</span>
          </div>

          <div className="track-list">
            {/* Uploaded file */}
            {fileId && fileName && (
              <div className="track-card active">
                <div className="track-card-inner">
                  <span className="track-card-name">{fileName}</span>
                  <span className="tag">{uploadedFile?.name.split('.').pop()?.toUpperCase() || 'WAV'}</span>
                </div>
                <div className="track-meta">
                  <span>{duration ? fmtDur(duration) : '--:--'}</span>
                  <span>48.0kHz</span>
                </div>
              </div>
            )}

            {/* Analyzed cues */}
            {cuesSorted.map((cue, i) => (
              <div
                key={cue.id}
                className={`track-card ${cue.id === activeCueId ? 'active' : ''}`}
                onClick={() => handleSegClick(i)}
              >
                <div className="track-card-inner">
                  <span className="track-card-name">
                    {cue.name || `Cue_${String(i + 1).padStart(2, '0')}`}
                  </span>
                  <span className="tag">{cue.musical_profile.key || '?'}</span>
                </div>
                <div className="track-meta">
                  <span>{fmtDur(cue.end - cue.start)}</span>
                  <span>{cue.musical_profile.bpm?.toFixed(0) || '?'} BPM</span>
                </div>
              </div>
            ))}

            {/* Non-analyzed segments */}
            {!project && segments.map(([start, end], i) => {
              const isSel = selectedSegments.some(
                ([s, e]) => Math.abs(s - start) < 0.5 && Math.abs(e - end) < 0.5
              )
              return (
                <div
                  key={i}
                  className={`track-card ${isSel ? 'active' : ''}`}
                  onClick={() => toggleSeg(start, end)}
                >
                  <div className="track-card-inner">
                    <span className="track-card-name">Seg_{String(i + 1).padStart(2, '0')}</span>
                    <span className="tag">{isSel ? '✓' : '○'}</span>
                  </div>
                  <div className="track-meta">
                    <span>{fmtDur(start)} — {fmtDur(end)}</span>
                    <span>{fmtDur(end - start)}</span>
                  </div>
                </div>
              )
            })}

            {/* Add source / upload */}
            <div
              className={`track-card add-source ${isDragOver ? 'drag-over' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="track-card-inner">
                <span className="track-card-name" style={{ opacity: 0.5 }}>
                  {isUploading ? `Uploading ${uploadProgress}%` : 'Add Source'}
                </span>
                <span className="tag">+</span>
              </div>
              {isUploading && (
                <div className="upload-bar">
                  <div className="upload-bar-fill" style={{ width: `${uploadProgress}%` }} />
                </div>
              )}
              <input
                ref={fileInputRef}
                type="file"
                hidden
                accept="audio/*,video/*"
                onChange={handleFileInput}
              />
            </div>
          </div>
        </aside>

        {/* ---------- CENTER CANVAS ---------- */}
        <main className="canvas-area">
          {/* Toolbar */}
          <div className="toolbar">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <button className="btn-icon" onClick={handlePrev} title="Prev"><PrevIcon /></button>
              <button className="btn-icon" onClick={handlePlay} title={isPlaying ? 'Pause' : 'Play'}>
                {isPlaying ? <PauseIcon /> : <PlayIcon />}
              </button>
              <button className="btn-icon" onClick={handleStop} title="Stop"><StopIcon /></button>
              <button className="btn-icon" onClick={handleNext} title="Next"><NextIcon /></button>
            </div>

            <div className="time-display">
              {secondsToTimecode(currentTime, framerate, startTimecode)}
            </div>

            <div style={{ flexGrow: 1 }} />

            {/* Mix type toggle */}
            <button
              className={`btn-pill ${mixType === 'clean_mx' ? 'primary' : ''}`}
              onClick={() => setMixType('clean_mx')}
              style={{ fontSize: '0.65rem' }}
            >
              Clean MX
            </button>
            <button
              className={`btn-pill ${mixType === 'full_mix' ? 'primary' : ''}`}
              onClick={() => setMixType('full_mix')}
              style={{ fontSize: '0.65rem' }}
            >
              Full Mix
            </button>
          </div>

          {/* Waveform area */}
          <div className="waveform-wrap">
            <div className="grid-lines" />
            {fileId && uploadedFile ? (
              <div className="waveform-inner">
                <div ref={waveformRef} />
              </div>
            ) : (
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span className="mono" style={{
                  background: 'var(--pl-bg)',
                  padding: '4px 12px',
                  border: 'var(--pl-border)',
                  fontSize: '0.75rem',
                }}>
                  NO SIGNAL — DROP AUDIO FILE TO BEGIN
                </span>
              </div>
            )}

            {/* Segment lane */}
            {(segments.length > 0 || cuesSorted.length > 0) && (
              <div className="segment-lane">
                {project ? (
                  cuesSorted.map((cue, i) => (
                    <button
                      key={cue.id}
                      className={`seg-chip ${cue.id === activeCueId ? 'active' : ''}`}
                      onClick={() => handleSegClick(i)}
                    >
                      #{i + 1} {cue.musical_profile.key || ''} {fmtDur(cue.start)}
                    </button>
                  ))
                ) : (
                  segments.map(([start, end], i) => {
                    const isSel = selectedSegments.some(
                      ([s, e]) => Math.abs(s - start) < 0.5 && Math.abs(e - end) < 0.5
                    )
                    return (
                      <button
                        key={i}
                        className={`seg-chip ${isSel ? 'selected' : ''}`}
                        onClick={() => toggleSeg(start, end)}
                      >
                        S{i + 1} {fmtDur(start)}
                      </button>
                    )
                  })
                )}
              </div>
            )}
          </div>
        </main>

        {/* ---------- RIGHT INSPECTOR ---------- */}
        <aside className="inspector">
          <div className="panel-header">
            <span className="label">Analysis Data</span>
          </div>

          {/* Data cells — show active cue info or placeholders */}
          <div className="data-grid">
            <div className="data-cell">
              <h4>BPM</h4>
              <div className="big-number">
                {activeCue?.musical_profile.bpm?.toFixed(1) || '—'}
              </div>
            </div>
            <div className="data-cell">
              <h4>Key</h4>
              <div className="big-number">
                {activeCue?.musical_profile.key || '—'}
              </div>
            </div>
            <div className="data-cell">
              <h4>Valence</h4>
              <div className="big-number">
                {activeCue ? `${(activeCue.musical_profile.valence * 100).toFixed(0)}` : '—'}
                {activeCue && <span className="unit">%</span>}
              </div>
            </div>
            <div className="data-cell">
              <h4>Arousal</h4>
              <div className="big-number">
                {activeCue ? `${(activeCue.musical_profile.arousal * 100).toFixed(0)}` : '—'}
                {activeCue && <span className="unit">%</span>}
              </div>
            </div>
          </div>

          {/* Instruments + genres as tags */}
          {activeCue && (
            <>
              <div className="panel-header" style={{ borderTop: 'var(--pl-border)' }}>
                <span className="label">Instruments</span>
              </div>
              <div className="tag-list">
                {activeCue.musical_profile.curated.instruments.map(inst => (
                  <span key={inst} className="tag-item">{inst}</span>
                ))}
                {activeCue.musical_profile.curated.instruments.length === 0 && (
                  <span className="mono" style={{ fontSize: '0.7rem', opacity: 0.5 }}>None detected</span>
                )}
              </div>

              <div className="panel-header" style={{ borderTop: 'var(--pl-border)' }}>
                <span className="label">Genre / Style</span>
              </div>
              <div className="tag-list">
                {[...activeCue.musical_profile.curated.genres, ...activeCue.musical_profile.curated.style].map(t => (
                  <span key={t} className="tag-item">{t}</span>
                ))}
              </div>

              {activeCue.musical_profile.curated.moods.length > 0 && (
                <>
                  <div className="panel-header" style={{ borderTop: 'var(--pl-border)' }}>
                    <span className="label">Moods</span>
                  </div>
                  <div className="tag-list">
                    {activeCue.musical_profile.curated.moods.map(m => (
                      <span key={m} className="tag-item">{m}</span>
                    ))}
                  </div>
                </>
              )}
            </>
          )}

          {/* Spectrogram placeholder (when no cue selected) */}
          {!activeCue && (
            <>
              <div className="panel-header" style={{ borderTop: 'var(--pl-border)' }}>
                <span className="label">Spectral Density</span>
              </div>
              <div className="spectrogram">
                <span className="spectrogram-label">
                  {project ? 'SELECT A CUE' : 'NO SIGNAL DETECTED'}
                </span>
              </div>
            </>
          )}

          {/* Controls */}
          {!project && (
            <>
              <div className="panel-header" style={{ borderTop: 'var(--pl-border)' }}>
                <span className="label">Detection Settings</span>
              </div>
              <div className="controls-section">
                <div className="control-row">
                  <span className="label">Threshold</span>
                  <input
                    type="range" min={0.01} max={1} step={0.01}
                    value={musicThreshold}
                    onChange={e => setMusicThreshold(parseFloat(e.target.value))}
                  />
                  <span className="data-val">{musicThreshold.toFixed(2)}</span>
                </div>
                <div className="control-row">
                  <span className="label">Min Gap</span>
                  <input
                    type="range" min={0.5} max={5} step={0.1}
                    value={minGap}
                    onChange={e => setMinGap(parseFloat(e.target.value))}
                  />
                  <span className="data-val">{minGap.toFixed(1)}s</span>
                </div>
                <div className="control-row">
                  <span className="label">Min Cue</span>
                  <input
                    type="range" min={1} max={10} step={0.5}
                    value={minCueLength}
                    onChange={e => setMinCueLength(parseFloat(e.target.value))}
                  />
                  <span className="data-val">{minCueLength.toFixed(1)}s</span>
                </div>
              </div>
            </>
          )}

          {/* Error */}
          {error && <div style={{ padding: '0 20px' }}><div className="error-msg">{error}</div></div>}

          {/* Progress */}
          {isAnalyzing && (
            <div style={{ padding: '12px 20px' }}>
              <div className="mono" style={{ fontSize: '0.65rem', marginBottom: 4 }}>
                {analysisStatus || 'Processing...'}
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${analysisProgress}%` }} />
              </div>
            </div>
          )}

          {/* Spacer */}
          <div style={{ flex: 1 }} />

          {/* Action buttons */}
          <div style={{ padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 8 }}>
            {fileId && !project && segments.length === 0 && !isAnalyzing && (
              <button className="btn-pill primary" onClick={handlePreview} disabled={isSegmenting} style={{ width: '100%', justifyContent: 'center' }}>
                {isSegmenting ? 'Detecting...' : 'Detect Segments'}
              </button>
            )}
            {segments.length > 0 && !project && !isAnalyzing && (
              <>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button className="btn-pill" onClick={() => setSelectedSegments([...segments])} style={{ flex: 1, justifyContent: 'center' }}>
                    Select All
                  </button>
                  <button className="btn-pill" onClick={() => setSelectedSegments([])} style={{ flex: 1, justifyContent: 'center' }}>
                    Clear
                  </button>
                </div>
                <button
                  className="btn-pill primary"
                  onClick={handleAnalyze}
                  disabled={selectedSegments.length === 0}
                  style={{ width: '100%', justifyContent: 'center' }}
                >
                  Analyze {selectedSegments.length} Segments
                </button>
              </>
            )}
            {isAnalyzing && (
              <button className="btn-pill" disabled style={{ width: '100%', justifyContent: 'center' }}>
                Analyzing {analysisProgress}%
              </button>
            )}
            {project && (
              <button className="btn-pill primary" onClick={() => setCurrentPage('cuesynch')} style={{ width: '100%', justifyContent: 'center' }}>
                Export Data
              </button>
            )}
            {!fileId && (
              <button className="btn-pill" disabled style={{ width: '100%', justifyContent: 'center' }}>
                Awaiting Input
              </button>
            )}
          </div>
        </aside>
      </div>

      {/* ==================== FOOTER ==================== */}
      <footer className="pipeline-footer">
        <div className="eye-icon" style={{ width: 16, height: 10 }} />
        MOTIVE AUDIO ANALYSIS
      </footer>
    </div>
  )
}
