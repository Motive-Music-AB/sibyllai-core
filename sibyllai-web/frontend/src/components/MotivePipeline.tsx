import { useState, useEffect, useRef, useCallback, type DragEvent, type ChangeEvent } from 'react'
import { useAppStore } from '@/lib/store'
import { api } from '@/lib/api'
import { secondsToTimecode } from '@/lib/timecode'
import { WaveformViewer } from '@/components/WaveformViewer'
import { CueCard } from '@/components/CueCard'
import { AddItemCombobox } from '@/components/AddItemCombobox'
import {
  YAMNET_INSTRUMENTS,
  YAMNET_GENRES,
  CLAP_STYLES,
  INSTRUMENT_THRESHOLD,
  GENRE_THRESHOLD,
  STYLE_THRESHOLD,
} from '@/lib/constants'
import type { CuratedAttributes } from '@/lib/types'
import './motive-pipeline.css'

/* -- Mix type defaults (from v1 ThresholdControls) ---------------------- */
const MIX_TYPE_DEFAULTS = {
  clean_mx: { musicThreshold: 0.5, silenceThreshold: 0.0005 },
  full_mix: { musicThreshold: 0.2, silenceThreshold: 0.01 },
}

function fmtDur(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/** Dynamic precision for small slider values */
function fmtPrecision(v: number): string {
  if (v < 0.001) return v.toFixed(6)
  if (v < 0.01) return v.toFixed(4)
  return v.toFixed(2)
}

/** Get items to display: detected items above threshold + curated items */
function getDisplayItems(
  detected: Record<string, number> | undefined,
  curated: string[] | undefined,
  threshold: number
): [string, number][] {
  const detectedMap = detected || {}
  const curatedSet = new Set(curated || [])

  const items = Object.entries(detectedMap)
    .filter(([name, score]) => score >= threshold || curatedSet.has(name))
    .sort((a, b) => b[1] - a[1])

  const detectedNames = new Set(items.map(([name]) => name))
  for (const name of curatedSet) {
    if (!detectedNames.has(name)) {
      items.push([name, 0])
    }
  }
  return items
}

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
    activeCueId, setActiveCueId,
    startTimecode,
    setUploadedFile, setSegments, setProject,
    updateCueInProject, resetToSegmentation,
    framerate,
    setCurrentPage, reset,
  } = useAppStore()

  /* -- Local state ------------------------------------------------------ */
  const [isDragOver, setIsDragOver] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [isEditingTags, setIsEditingTags] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [localCurated, setLocalCurated] = useState<CuratedAttributes | null>(null)

  /* -- Refs ------------------------------------------------------------- */
  const fileInputRef = useRef<HTMLInputElement>(null)

  /* -- Derived ---------------------------------------------------------- */
  const activeCue = project?.cues.find(c => c.id === activeCueId) ?? null
  const cuesSorted = project ? [...project.cues].sort((a, b) => a.start - b.start) : []
  const itemCount = project ? cuesSorted.length : segments.length
  const cueStatus = activeCue?.project_context?.status
  const isMatched = cueStatus === 'matched'

  /* -- Sync local curated state with active cue -- */
  useEffect(() => {
    if (activeCue) {
      setLocalCurated(activeCue.musical_profile.curated)
      setIsEditingTags(false)
    } else {
      setLocalCurated(null)
      setIsEditingTags(false)
    }
  }, [activeCue?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  /* =====================================================================
     TAG EDITING
     ===================================================================== */
  const toggleItem = (category: keyof CuratedAttributes, item: string) => {
    setLocalCurated((prev) => {
      if (!prev) return prev
      const current = (prev[category] as string[]) || []
      const updated = current.includes(item)
        ? current.filter((i) => i !== item)
        : [...current, item]
      return { ...prev, [category]: updated }
    })
  }

  const addItem = (category: keyof CuratedAttributes, item: string) => {
    setLocalCurated((prev) => {
      if (!prev) return prev
      const current = (prev[category] as string[]) || []
      if (current.includes(item)) return prev
      return { ...prev, [category]: [...current, item] }
    })
  }

  const handleSaveTags = async () => {
    if (!sessionId || !activeCue || !localCurated) return
    setIsSaving(true)
    try {
      await api.updateCue(sessionId, activeCue.id, {
        instruments: localCurated.instruments,
        genres: localCurated.genres,
        style: localCurated.style,
        moods: localCurated.moods,
      })
      updateCueInProject(activeCue.id, localCurated)
      setIsEditingTags(false)
    } catch {
      setError('Failed to save cue tags')
    } finally {
      setIsSaving(false)
    }
  }

  const handleCancelTags = () => {
    if (activeCue) {
      setLocalCurated(activeCue.musical_profile.curated)
    }
    setIsEditingTags(false)
  }

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
     MIX TYPE WITH AUTO-DEFAULTS
     ===================================================================== */
  const handleMixTypeChange = useCallback((type: 'clean_mx' | 'full_mix') => {
    setMixType(type)
    const defaults = MIX_TYPE_DEFAULTS[type]
    setMusicThreshold(defaults.musicThreshold)
    setSilenceThreshold(defaults.silenceThreshold)
  }, [setMixType, setMusicThreshold, setSilenceThreshold])

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
        mix_type: mixType,
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

  /* -- Detected items for tag editing -- */
  const detectedInstruments = activeCue && localCurated
    ? getDisplayItems(activeCue.musical_profile.detected?.instruments_yamnet, localCurated.instruments, INSTRUMENT_THRESHOLD)
    : []
  const detectedGenres = activeCue && localCurated
    ? getDisplayItems(activeCue.musical_profile.detected?.genres_yamnet, localCurated.genres, GENRE_THRESHOLD)
    : []
  const detectedStyles = activeCue && localCurated
    ? getDisplayItems(activeCue.musical_profile.detected?.clap_style, localCurated.style, STYLE_THRESHOLD)
    : []

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

            {/* Add source / upload */}
            <div
              className={`track-card add-source ${isDragOver ? 'drag-over' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="track-card-inner" style={{ justifyContent: 'center' }}>
                <span className="track-card-name" style={{ opacity: 0.5 }}>
                  {isUploading ? `Uploading ${uploadProgress}%` : '+ Add Source'}
                </span>
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
        <main
          className={`canvas-area ${isDragOver && !uploadedFile ? 'drag-over-canvas' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
        >
          {/* Mix type toggle toolbar */}
          <div className="toolbar">
            {/* Event counter */}
            {!project && segments.length > 0 && !isAnalyzing && (
              <span className="mono" style={{ fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.1em' }}>
                {segments.length} EVENT{segments.length !== 1 ? 'S' : ''} DETECTED
              </span>
            )}
            {/* Selection counter */}
            {!project && segments.length > 0 && !isAnalyzing && (
              <span className="mono" style={{ fontSize: '0.65rem', opacity: 0.6 }}>
                {selectedSegments.length} OF {segments.length} SELECTED
              </span>
            )}
            <div style={{ flexGrow: 1 }} />
            <button
              className={`btn-pill ${mixType === 'clean_mx' ? 'primary' : ''}`}
              onClick={() => handleMixTypeChange('clean_mx')}
              style={{ fontSize: '0.65rem' }}
            >
              Clean MX
            </button>
            <button
              className={`btn-pill ${mixType === 'full_mix' ? 'primary' : ''}`}
              onClick={() => handleMixTypeChange('full_mix')}
              style={{ fontSize: '0.65rem' }}
            >
              Full Mix
            </button>
          </div>

          {/* WaveformViewer — full featured: zoom, drag, split, etc. */}
          <div className="waveform-wrap">
            {fileId && uploadedFile ? (
              <WaveformViewer />
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
          </div>

          {/* ---- SEGMENT / CUE LIST ---- */}
          {!project && segments.length > 0 && (
            <div className="segment-list">
              <div className="segment-list-header">
                <span className="label">Segments</span>
                <span className="mono" style={{ fontSize: '0.6rem', opacity: 0.5 }}>
                  {selectedSegments.length}/{segments.length} selected
                </span>
              </div>
              <div className="segment-list-rows">
                {segments.map(([start, end], i) => {
                  const isSelected = selectedSegments.some(
                    ([s, e]) => Math.abs(s - start) < 0.01 && Math.abs(e - end) < 0.01
                  )
                  const tc0 = secondsToTimecode(start, framerate, startTimecode)
                  const tc1 = secondsToTimecode(end, framerate, startTimecode)
                  const dur = end - start
                  return (
                    <div
                      key={i}
                      className={`segment-row ${isSelected ? 'selected' : ''}`}
                      onClick={() => {
                        if (isSelected) {
                          setSelectedSegments(
                            selectedSegments.filter(
                              ([s, e]) => !(Math.abs(s - start) < 0.01 && Math.abs(e - end) < 0.01)
                            )
                          )
                        } else {
                          setSelectedSegments([...selectedSegments, [start, end]])
                        }
                      }}
                    >
                      <span className="segment-row-check">{isSelected ? '✓' : ''}</span>
                      <span className="segment-row-num">{i + 1}</span>
                      <span className="segment-row-tc">{tc0} — {tc1}</span>
                      <span className="segment-row-dur">{fmtDur(dur)}</span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {project && cuesSorted.length > 0 && (
            <div className="cue-cards-section">
              <div className="segment-list-header">
                <span className="label">Analysis Results</span>
                <span className="mono" style={{ fontSize: '0.6rem', opacity: 0.5 }}>
                  {cuesSorted.length} cues — click to view, click again to edit
                </span>
              </div>
              <div className="cue-cards-grid">
                {cuesSorted.map((cue) => {
                  const segIndex = segments.findIndex(([s, e]) =>
                    Math.abs(s - cue.start) < 0.5 && Math.abs(e - cue.end) < 0.5
                  )
                  return <CueCard key={cue.id} cue={cue} index={segIndex >= 0 ? segIndex : 0} />
                })}
              </div>
            </div>
          )}
        </main>

        {/* ---------- RIGHT INSPECTOR ---------- */}
        <aside className="inspector">
          <div className="panel-header">
            <span className="label">Analysis Data</span>
            {activeCue && isMatched && (
              <span className="tag" style={{ background: 'var(--pl-ink)', color: 'var(--pl-bg)' }}>MATCHED</span>
            )}
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

          {/* ---- TAG EDITING SECTION ---- */}
          {activeCue && localCurated && (
            <>
              {/* Instruments */}
              <div className="panel-header" style={{ borderTop: 'var(--pl-border)' }}>
                <span className="label">Instruments</span>
                {!isEditingTags && (
                  <button
                    className="btn-pill"
                    style={{ height: 24, fontSize: '0.6rem', padding: '0 10px' }}
                    onClick={() => setIsEditingTags(true)}
                  >
                    Edit
                  </button>
                )}
              </div>

              {!isEditingTags ? (
                /* -- Read-only tag display -- */
                <>
                  <div className="tag-list">
                    {localCurated.instruments.map(inst => (
                      <span key={inst} className="tag-item">{inst}</span>
                    ))}
                    {localCurated.instruments.length === 0 && (
                      <span className="mono" style={{ fontSize: '0.7rem', opacity: 0.5 }}>None detected</span>
                    )}
                  </div>

                  <div className="panel-header" style={{ borderTop: 'var(--pl-border)' }}>
                    <span className="label">Genre / Style</span>
                  </div>
                  <div className="tag-list">
                    {[...localCurated.genres, ...localCurated.style].map(t => (
                      <span key={t} className="tag-item">{t}</span>
                    ))}
                    {localCurated.genres.length === 0 && localCurated.style.length === 0 && (
                      <span className="mono" style={{ fontSize: '0.7rem', opacity: 0.5 }}>None detected</span>
                    )}
                  </div>

                  {localCurated.moods.length > 0 && (
                    <>
                      <div className="panel-header" style={{ borderTop: 'var(--pl-border)' }}>
                        <span className="label">Moods</span>
                      </div>
                      <div className="tag-list">
                        {localCurated.moods.map(m => (
                          <span key={m} className="tag-item">{m}</span>
                        ))}
                      </div>
                    </>
                  )}
                </>
              ) : (
                /* -- Editable tag section -- */
                <div className="tag-edit-section">
                  {/* Instruments edit */}
                  <div className="tag-edit-group">
                    <div className="tag-toggle-list">
                      {detectedInstruments.map(([name, score]) => {
                        const isSelected = localCurated.instruments?.includes(name)
                        return (
                          <button
                            key={name}
                            onClick={() => toggleItem('instruments', name)}
                            className={`tag-toggle ${isSelected ? 'selected' : ''}`}
                          >
                            {name} ({score.toFixed(3)})
                          </button>
                        )
                      })}
                    </div>
                    <AddItemCombobox
                      suggestions={YAMNET_INSTRUMENTS}
                      existingItems={localCurated.instruments}
                      onAdd={(item) => addItem('instruments', item)}
                      placeholder="Add instrument..."
                    />
                  </div>

                  {/* Genres edit */}
                  <div className="panel-header" style={{ borderTop: 'var(--pl-border)' }}>
                    <span className="label">Genres</span>
                  </div>
                  <div className="tag-edit-group">
                    <div className="tag-toggle-list">
                      {detectedGenres.map(([name, score]) => {
                        const isSelected = localCurated.genres?.includes(name)
                        return (
                          <button
                            key={name}
                            onClick={() => toggleItem('genres', name)}
                            className={`tag-toggle ${isSelected ? 'selected' : ''}`}
                          >
                            {name} ({score.toFixed(3)})
                          </button>
                        )
                      })}
                    </div>
                    <AddItemCombobox
                      suggestions={YAMNET_GENRES}
                      existingItems={localCurated.genres}
                      onAdd={(item) => addItem('genres', item)}
                      placeholder="Add genre..."
                    />
                  </div>

                  {/* Style edit */}
                  <div className="panel-header" style={{ borderTop: 'var(--pl-border)' }}>
                    <span className="label">Style</span>
                  </div>
                  <div className="tag-edit-group">
                    <div className="tag-toggle-list">
                      {detectedStyles.map(([name, score]) => {
                        const isSelected = localCurated.style?.includes(name)
                        return (
                          <button
                            key={name}
                            onClick={() => toggleItem('style', name)}
                            className={`tag-toggle ${isSelected ? 'selected' : ''}`}
                          >
                            {name} ({score.toFixed(3)})
                          </button>
                        )
                      })}
                    </div>
                    <AddItemCombobox
                      suggestions={CLAP_STYLES}
                      existingItems={localCurated.style}
                      onAdd={(item) => addItem('style', item)}
                      placeholder="Add style..."
                    />
                  </div>

                  {/* Save / Cancel */}
                  <div className="tag-edit-actions">
                    <button className="btn-pill" onClick={handleCancelTags} disabled={isSaving}>
                      Cancel
                    </button>
                    <button className="btn-pill primary" onClick={handleSaveTags} disabled={isSaving}>
                      {isSaving ? 'Saving...' : 'Save'}
                    </button>
                  </div>
                </div>
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

          {/* ---- DETECTION SETTINGS ---- */}
          {!project && (
            <>
              <div className="panel-header" style={{ borderTop: 'var(--pl-border)' }}>
                <span className="label">Detection Settings</span>
              </div>
              <div className="controls-section">
                {/* Primary threshold */}
                <div className="control-row">
                  <span className="label">Threshold</span>
                  <input
                    type="range" min={0.001} max={0.03} step={0.0001}
                    value={musicThreshold}
                    onChange={e => setMusicThreshold(parseFloat(e.target.value))}
                  />
                  <span className="data-val">{fmtPrecision(musicThreshold)}</span>
                </div>

                {/* Min Gap */}
                <div className="control-row">
                  <span className="label">Min Gap</span>
                  <input
                    type="range" min={0.1} max={15.0} step={0.1}
                    value={minGap}
                    onChange={e => setMinGap(parseFloat(e.target.value))}
                  />
                  <span className="data-val">{minGap.toFixed(1)}s</span>
                </div>

                {/* Min Cue */}
                <div className="control-row">
                  <span className="label">Min Cue</span>
                  <input
                    type="range" min={0.5} max={15.0} step={0.1}
                    value={minCueLength}
                    onChange={e => setMinCueLength(parseFloat(e.target.value))}
                  />
                  <span className="data-val">{minCueLength.toFixed(1)}s</span>
                </div>

                {/* Advanced Settings Toggle */}
                <button
                  className="btn-pill"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  style={{ width: '100%', justifyContent: 'center', fontSize: '0.6rem', height: 28 }}
                >
                  {showAdvanced ? 'Hide Advanced' : 'Advanced Settings'}
                </button>

                {/* Advanced: Silence Threshold */}
                <div
                  style={{
                    maxHeight: showAdvanced ? '200px' : '0px',
                    opacity: showAdvanced ? 1 : 0,
                    overflow: 'hidden',
                    transition: 'max-height 0.3s ease, opacity 0.3s ease',
                  }}
                >
                  <div className="control-row" style={{ paddingTop: 8 }}>
                    <span className="label">Silence</span>
                    <input
                      type="range" min={0.0001} max={0.1} step={0.0001}
                      value={silenceThreshold}
                      onChange={e => setSilenceThreshold(parseFloat(e.target.value))}
                    />
                    <span className="data-val">{fmtPrecision(silenceThreshold)}</span>
                  </div>
                </div>

                {/* Detect Segments button inside controls */}
                {fileId && segments.length === 0 && !isAnalyzing && (
                  <button
                    className="btn-pill primary"
                    onClick={handlePreview}
                    disabled={isSegmenting}
                    style={{ width: '100%', justifyContent: 'center', marginTop: 4 }}
                  >
                    {isSegmenting ? 'Detecting...' : 'Detect Segments'}
                  </button>
                )}

                {/* Re-detect button when segments exist */}
                {fileId && segments.length > 0 && !isAnalyzing && (
                  <button
                    className="btn-pill"
                    onClick={handlePreview}
                    disabled={isSegmenting}
                    style={{ width: '100%', justifyContent: 'center', marginTop: 4, fontSize: '0.6rem', height: 28 }}
                  >
                    {isSegmenting ? 'Detecting...' : 'Re-detect'}
                  </button>
                )}
              </div>
            </>
          )}

          {/* Error */}
          {error && <div style={{ padding: '0 20px' }}><div className="error-msg">{error}</div></div>}

          {/* Progress + DO NOT CLOSE warning */}
          {isAnalyzing && (
            <div style={{ padding: '12px 20px' }}>
              <div className="mono" style={{ fontSize: '0.65rem', marginBottom: 4 }}>
                {analysisStatus || 'Processing...'}
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${analysisProgress}%` }} />
              </div>
              <div className="mono" style={{ fontSize: '0.55rem', marginTop: 6, textAlign: 'center', letterSpacing: '0.1em', fontWeight: 700 }}>
                DO NOT CLOSE THIS TAB &middot; {Math.round(analysisProgress)}%
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
                Analyzing {Math.round(analysisProgress)}%
              </button>
            )}
            {project && (
              <>
                <button className="btn-pill" onClick={() => resetToSegmentation()} style={{ width: '100%', justifyContent: 'center', fontSize: '0.6rem', height: 28 }}>
                  Back to Segments
                </button>
                <button className="btn-pill primary" onClick={() => setCurrentPage('cuesynch')} style={{ width: '100%', justifyContent: 'center' }}>
                  Export Data
                </button>
              </>
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
