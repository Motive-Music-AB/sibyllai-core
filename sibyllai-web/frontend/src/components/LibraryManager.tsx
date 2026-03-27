import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '@/lib/api'
import type { LibrarySource, LibraryTrack } from '@/lib/types'

interface LibraryManagerProps {
  libraryFiles: File[]
  setLibraryFiles: (files: File[]) => void
  libraryFolderName: string | null
  setLibraryFolderName: (name: string | null) => void
  includeMoods: boolean
  setIncludeMoods: (v: boolean) => void
  buildJobId: string | null
  setBuildJobId: (id: string | null) => void
  buildStatus: 'idle' | 'running' | 'complete' | 'error'
  setBuildStatus: (s: 'idle' | 'running' | 'complete' | 'error') => void
  buildMessage: string
  setBuildMessage: (msg: string) => void
  progressPercent: number
  setProgressPercent: (v: number) => void
  progressCurrent: number
  setProgressCurrent: (v: number) => void
  progressTotal: number
  setProgressTotal: (v: number) => void
  windowsIndexed: number
  setWindowsIndexed: (v: number) => void
  isBuilding: boolean
  setIsBuilding: (v: boolean) => void
  isUploading: boolean
  setIsUploading: (v: boolean) => void
  indexAvailable: boolean
  setIndexAvailable: (v: boolean) => void
  onBuildComplete: () => void
  segmentationMode: 'fixed' | 'structure'
  setSegmentationMode: (mode: 'fixed' | 'structure') => void
  minSectionLength: number
  setMinSectionLength: (v: number) => void
}

export function LibraryManager({
  libraryFiles,
  setLibraryFiles,
  libraryFolderName,
  setLibraryFolderName,
  includeMoods,
  setIncludeMoods,
  buildJobId,
  setBuildJobId,
  buildStatus,
  setBuildStatus,
  buildMessage,
  setBuildMessage,
  progressPercent,
  setProgressPercent,
  progressCurrent,
  setProgressCurrent,
  progressTotal,
  setProgressTotal,
  windowsIndexed,
  setWindowsIndexed,
  isBuilding,
  setIsBuilding,
  isUploading,
  setIsUploading,
  indexAvailable,
  setIndexAvailable,
  onBuildComplete,
  segmentationMode,
  setSegmentationMode,
  minSectionLength,
  setMinSectionLength,
}: LibraryManagerProps) {
  const folderInputRef = useRef<HTMLInputElement>(null)

  const [sources, setSources] = useState<LibrarySource[]>([])
  const [expandedSource, setExpandedSource] = useState<string | null>(null)
  const [sourceTracks, setSourceTracks] = useState<Record<string, LibraryTrack[]>>({})
  const [loadingTracks, setLoadingTracks] = useState<string | null>(null)

  // Audio playback for track previews
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [playingTrackId, setPlayingTrackId] = useState<string | null>(null)

  const loadSources = useCallback(async () => {
    try {
      const data = await api.getLibrarySources()
      setSources(data)
    } catch {
      setSources([])
    }
  }, [])

  useEffect(() => {
    loadSources()
  }, [loadSources])

  // Poll build status
  useEffect(() => {
    let interval: number | undefined
    if (buildJobId && buildStatus === 'running') {
      interval = window.setInterval(async () => {
        try {
          const status = await api.getLibraryBuildStatus(buildJobId)
          if (typeof status.progress_percent === 'number') setProgressPercent(status.progress_percent)
          if (typeof status.current === 'number') setProgressCurrent(status.current)
          if (typeof status.total === 'number') setProgressTotal(status.total)
          if (typeof status.windows_indexed === 'number') setWindowsIndexed(status.windows_indexed)
          if (status.message) setBuildMessage(status.message)
          if (status.status === 'complete') {
            setBuildStatus('complete')
            setBuildMessage(
              `Indexed ${status.result?.tracks_indexed ?? 0} tracks, ${status.result?.windows_indexed ?? 0} windows`
            )
            setProgressPercent(100)
            setIsBuilding(false)
            setIndexAvailable(true)
            onBuildComplete()
            loadSources()
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
  }, [
    buildJobId, buildStatus, setBuildStatus, setBuildMessage, setProgressPercent,
    setProgressCurrent, setProgressTotal, setWindowsIndexed, setIsBuilding,
    setIndexAvailable, onBuildComplete, loadSources,
  ])

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
    try {
      setIsUploading(true)
      const response = await api.buildLibraryIndexUpload(libraryFiles, includeMoods, true, segmentationMode, minSectionLength)
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
    try {
      setIsUploading(true)
      const response = await api.buildLibraryIndexUpload(libraryFiles, includeMoods, false, segmentationMode, minSectionLength)
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

  const handleToggleSource = async (sourceName: string) => {
    if (expandedSource === sourceName) {
      setExpandedSource(null)
      return
    }
    setExpandedSource(sourceName)
    if (!sourceTracks[sourceName]) {
      setLoadingTracks(sourceName)
      try {
        const tracks = await api.getLibraryTracks(sourceName)
        setSourceTracks((prev) => ({ ...prev, [sourceName]: tracks }))
      } catch {
        setSourceTracks((prev) => ({ ...prev, [sourceName]: [] }))
      } finally {
        setLoadingTracks(null)
      }
    }
  }

  const handleDeleteSource = async (sourceName: string) => {
    try {
      await api.deleteLibrarySource(sourceName)
      setSources((prev) => prev.filter((s) => s.source_name !== sourceName))
      setSourceTracks((prev) => {
        const next = { ...prev }
        delete next[sourceName]
        return next
      })
      if (expandedSource === sourceName) setExpandedSource(null)
      const info = await api.getLibraryInfo()
      setIndexAvailable(info.exists && info.tracks > 0)
      onBuildComplete()
    } catch {
      // Silently fail
    }
  }

  const stopPlayback = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.src = ''
    }
    setPlayingTrackId(null)
  }, [])

  const playTrack = useCallback(
    (trackId: string) => {
      if (playingTrackId === trackId) {
        stopPlayback()
        return
      }
      stopPlayback()
      const audio = audioRef.current || new Audio()
      audioRef.current = audio
      audio.src = api.getLibraryAudioUrl(trackId)
      audio.currentTime = 0
      setPlayingTrackId(trackId)
      audio.play().catch(() => setPlayingTrackId(null))
      audio.onended = () => setPlayingTrackId(null)
    },
    [playingTrackId, stopPlayback]
  )

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current.src = ''
      }
    }
  }, [])

  const formatDuration = (seconds: number) => {
    const m = Math.floor(seconds / 60)
    const s = Math.floor(seconds % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  const formatSize = (bytes: number | null) => {
    if (!bytes) return '—'
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatTotalDuration = (seconds: number) => {
    const h = Math.floor(seconds / 3600)
    const m = Math.floor((seconds % 3600) / 60)
    if (h > 0) return `${h}h ${m}m`
    return `${m}m`
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <input
        ref={folderInputRef}
        type="file"
        multiple
        hidden
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

      {/* Add to library section */}
      <div style={{ padding: 16, border: '1px solid #080808', borderRadius: 8 }}>
        <div className="label" style={{ marginBottom: 12 }}>Add to Library</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <button
            className="btn-pill"
            style={{ height: 28, fontSize: '0.65rem' }}
            onClick={() => folderInputRef.current?.click()}
            disabled={isBuilding}
          >
            {libraryFolderName ? libraryFolderName : 'Choose Folder'}
          </button>
          <button
            className="btn-pill"
            style={{ height: 28, fontSize: '0.65rem' }}
            onClick={handleBuildIndex}
            disabled={isBuilding || libraryFiles.length === 0}
          >
            {isUploading ? 'Uploading...' : isBuilding ? 'Building...' : indexAvailable ? 'Replace All' : 'Build Index'}
          </button>
          <button
            className="btn-pill"
            style={{ height: 28, fontSize: '0.65rem' }}
            onClick={handleAddToLibrary}
            disabled={isBuilding || libraryFiles.length === 0 || !indexAvailable}
          >
            {isUploading ? 'Uploading...' : isBuilding ? 'Adding...' : 'Add to Library'}
          </button>
          <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.65rem', fontFamily: 'var(--pl-font-mono)', opacity: 0.6, marginLeft: 4, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={includeMoods}
              onChange={(e) => setIncludeMoods(e.target.checked)}
              style={{ accentColor: '#080808' }}
            />
            Moods
          </label>
        </div>

        {/* Segmentation mode toggle */}
        <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.65rem', fontFamily: 'var(--pl-font-mono)', cursor: 'pointer' }}>
              <input
                type="radio"
                name="seg-mode"
                checked={segmentationMode === 'structure'}
                onChange={() => setSegmentationMode('structure')}
                disabled={isBuilding}
                style={{ accentColor: '#080808' }}
              />
              Smart Sections
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.65rem', fontFamily: 'var(--pl-font-mono)', opacity: 0.6, cursor: 'pointer' }}>
              <input
                type="radio"
                name="seg-mode"
                checked={segmentationMode === 'fixed'}
                onChange={() => setSegmentationMode('fixed')}
                disabled={isBuilding}
                style={{ accentColor: '#080808' }}
              />
              Fixed Windows
            </label>
          </div>
          {segmentationMode === 'structure' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span className="mono" style={{ fontSize: '0.6rem', opacity: 0.5 }}>Min length</span>
              <input
                type="range"
                min={4}
                max={30}
                step={1}
                value={minSectionLength}
                onChange={(e) => setMinSectionLength(Number(e.target.value))}
                disabled={isBuilding}
                style={{ width: 80, accentColor: '#080808' }}
              />
              <span className="mono" style={{ fontSize: '0.6rem', opacity: 0.5 }}>{minSectionLength}s</span>
            </div>
          )}
        </div>

        {/* Build progress bar */}
        {buildStatus === 'running' && (
          <div style={{ marginTop: 12 }}>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, progressPercent))}%` }} />
            </div>
            <p className="mono" style={{ fontSize: '0.6rem', opacity: 0.6, marginTop: 4 }}>
              {buildMessage} · {progressCurrent}/{progressTotal || '–'} tracks · {windowsIndexed} windows
            </p>
          </div>
        )}
        {buildStatus === 'error' && (
          <p className="mono" style={{ fontSize: '0.65rem', color: '#c00', marginTop: 8 }}>{buildMessage}</p>
        )}
        {buildStatus === 'complete' && (
          <p className="mono" style={{ fontSize: '0.65rem', opacity: 0.6, marginTop: 8 }}>{buildMessage}</p>
        )}
      </div>

      {/* Sources table */}
      <div style={{ padding: 16, border: '1px solid #080808', borderRadius: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <span className="label">Indexed Sources</span>
          <span className="mono" style={{ fontSize: '0.6rem', opacity: 0.5 }}>
            {sources.length} source{sources.length !== 1 ? 's' : ''}
          </span>
        </div>

        {sources.length === 0 ? (
          <p className="mono" style={{ fontSize: '0.7rem', opacity: 0.5 }}>No sources indexed yet. Add a folder above to get started.</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {sources.map((src) => (
              <div key={src.source_name} style={{ border: '1px solid rgba(8,8,8,0.2)', borderRadius: 6, overflow: 'hidden' }}>
                {/* Source header row */}
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 12px',
                    cursor: 'pointer',
                    transition: 'background 0.1s',
                  }}
                  onClick={() => handleToggleSource(src.source_name)}
                  onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(8,8,8,0.03)')}
                  onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className="mono" style={{ fontSize: '0.6rem', opacity: 0.5 }}>
                      {expandedSource === src.source_name ? '▼' : '▶'}
                    </span>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700 }}>{src.source_name}</span>
                    <span className="mono" style={{ fontSize: '0.6rem', opacity: 0.4 }}>{src.source_type}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <span className="mono" style={{ fontSize: '0.6rem', opacity: 0.5 }}>{src.track_count} tracks</span>
                    <span className="mono" style={{ fontSize: '0.6rem', opacity: 0.5 }}>{src.total_windows} windows</span>
                    <span className="mono" style={{ fontSize: '0.6rem', opacity: 0.5 }}>{formatTotalDuration(src.total_duration)}</span>
                    <button
                      className="btn-pill"
                      style={{ height: 24, fontSize: '0.55rem', color: '#c00', borderColor: '#c00' }}
                      onClick={(e) => {
                        e.stopPropagation()
                        handleDeleteSource(src.source_name)
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </div>

                {/* Expanded track list */}
                {expandedSource === src.source_name && (
                  <div style={{ borderTop: '1px solid rgba(8,8,8,0.1)', maxHeight: 320, overflowY: 'auto' }}>
                    {loadingTracks === src.source_name ? (
                      <p className="mono" style={{ fontSize: '0.65rem', opacity: 0.5, padding: '8px 12px' }}>Loading tracks...</p>
                    ) : (sourceTracks[src.source_name] ?? []).length === 0 ? (
                      <p className="mono" style={{ fontSize: '0.65rem', opacity: 0.5, padding: '8px 12px' }}>No tracks found.</p>
                    ) : (
                      <table style={{ width: '100%', fontSize: '0.65rem', fontFamily: 'var(--pl-font-mono)', borderCollapse: 'collapse' }}>
                        <thead>
                          <tr style={{ textAlign: 'left', opacity: 0.5 }}>
                            <th style={{ padding: '6px 12px', fontWeight: 400 }}>Filename</th>
                            <th style={{ padding: '6px 8px', fontWeight: 400, width: 60 }}>Duration</th>
                            <th style={{ padding: '6px 8px', fontWeight: 400, width: 60 }}>Size</th>
                            <th style={{ padding: '6px 8px', fontWeight: 400, width: 50 }}>Windows</th>
                            <th style={{ padding: '6px 8px', fontWeight: 400, width: 50 }}></th>
                          </tr>
                        </thead>
                        <tbody>
                          {(sourceTracks[src.source_name] ?? []).map((track) => (
                            <tr key={track.id} style={{ borderTop: '1px solid rgba(8,8,8,0.06)' }}>
                              <td style={{ padding: '6px 12px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 300 }} title={track.path}>
                                {track.filename}
                              </td>
                              <td style={{ padding: '6px 8px', opacity: 0.6 }}>
                                {formatDuration(track.duration)}
                              </td>
                              <td style={{ padding: '6px 8px', opacity: 0.6 }}>
                                {formatSize(track.size)}
                              </td>
                              <td style={{ padding: '6px 8px', opacity: 0.6 }}>{track.window_count}</td>
                              <td style={{ padding: '6px 8px' }}>
                                <button
                                  className="btn-pill"
                                  style={{ height: 22, fontSize: '0.55rem', padding: '0 8px' }}
                                  onClick={() => playTrack(track.id)}
                                >
                                  {playingTrackId === track.id ? 'Stop' : 'Play'}
                                </button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
