import { useCallback, useEffect, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
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
      // Refresh index availability
      const info = await api.getLibraryInfo()
      setIndexAvailable(info.exists && info.tracks > 0)
      onBuildComplete()
    } catch {
      // Silently fail - could show error toast in future
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
    <div className="space-y-4">
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

      {/* Add to library section */}
      <div className="glass p-6 rounded-2xl space-y-4">
        <h3 className="text-lg font-medium font-display">Add to Library</h3>
        <div className="flex items-center gap-2 flex-wrap">
          <Button
            size="sm"
            variant="outline"
            className="glass-lighter border-primary/20"
            onClick={() => folderInputRef.current?.click()}
            disabled={isBuilding}
          >
            {libraryFolderName ? libraryFolderName : 'Choose Folder'}
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="glass-lighter border-primary/20"
            onClick={handleBuildIndex}
            disabled={isBuilding || libraryFiles.length === 0}
          >
            {isUploading ? 'Uploading…' : isBuilding ? 'Building…' : indexAvailable ? 'Rebuild Index' : 'Build Index'}
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="glass-lighter border-primary/20"
            onClick={handleAddToLibrary}
            disabled={isBuilding || libraryFiles.length === 0 || !indexAvailable}
          >
            {isUploading ? 'Uploading…' : isBuilding ? 'Adding…' : 'Add Folder'}
          </Button>
          <label className="flex items-center gap-1.5 text-xs text-foreground-muted ml-1">
            <input
              type="checkbox"
              checked={includeMoods}
              onChange={(e) => setIncludeMoods(e.target.checked)}
              className="accent-primary"
            />
            Moods
          </label>
        </div>
        {/* Build progress bar */}
        {buildStatus === 'running' && (
          <div className="space-y-1">
            <div className="h-1.5 rounded-full bg-primary/10 overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-300"
                style={{ width: `${Math.min(100, Math.max(0, progressPercent))}%` }}
              />
            </div>
            <p className="text-xs text-foreground-muted">
              {buildMessage} · {progressCurrent}/{progressTotal || '–'} tracks · {windowsIndexed} windows
            </p>
          </div>
        )}
        {buildStatus === 'error' && <p className="text-xs text-red-400">{buildMessage}</p>}
        {buildStatus === 'complete' && <p className="text-xs text-foreground-muted">{buildMessage}</p>}
      </div>

      {/* Sources table */}
      <div className="glass p-6 rounded-2xl space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium font-display">Indexed Sources</h3>
          <span className="text-xs text-foreground-muted">
            {sources.length} source{sources.length !== 1 ? 's' : ''}
          </span>
        </div>

        {sources.length === 0 ? (
          <p className="text-sm text-foreground-muted">No sources indexed yet. Add a folder above to get started.</p>
        ) : (
          <div className="space-y-2">
            {sources.map((src) => (
              <div key={src.source_name} className="border border-primary/15 rounded-lg overflow-hidden">
                <div
                  className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-primary/5 transition-colors"
                  onClick={() => handleToggleSource(src.source_name)}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-foreground-muted">
                      {expandedSource === src.source_name ? '▼' : '▶'}
                    </span>
                    <div>
                      <span className="text-sm font-medium">{src.source_name}</span>
                      <span className="text-xs text-foreground-muted ml-2">{src.source_type}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-xs text-foreground-muted">
                    <span>{src.track_count} tracks</span>
                    <span>{src.total_windows} windows</span>
                    <span>{formatTotalDuration(src.total_duration)}</span>
                    <Button
                      size="sm"
                      variant="outline"
                      className="glass-lighter border-red-400/30 text-red-400 hover:bg-red-400/10 h-7 px-2"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleDeleteSource(src.source_name)
                      }}
                    >
                      Delete
                    </Button>
                  </div>
                </div>

                {/* Expanded track list */}
                {expandedSource === src.source_name && (
                  <div className="border-t border-primary/10 px-4 py-2 max-h-80 overflow-y-auto">
                    {loadingTracks === src.source_name ? (
                      <p className="text-xs text-foreground-muted py-2">Loading tracks…</p>
                    ) : (sourceTracks[src.source_name] ?? []).length === 0 ? (
                      <p className="text-xs text-foreground-muted py-2">No tracks found.</p>
                    ) : (
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="text-foreground-muted text-left">
                            <th className="py-1 pr-4 font-normal">Filename</th>
                            <th className="py-1 pr-4 font-normal w-20">Duration</th>
                            <th className="py-1 pr-4 font-normal w-20">Size</th>
                            <th className="py-1 pr-4 font-normal w-16">Windows</th>
                            <th className="py-1 font-normal w-16"></th>
                          </tr>
                        </thead>
                        <tbody>
                          {(sourceTracks[src.source_name] ?? []).map((track) => (
                            <tr key={track.id} className="border-t border-primary/5">
                              <td className="py-1.5 pr-4 truncate max-w-[300px]" title={track.path}>
                                {track.filename}
                              </td>
                              <td className="py-1.5 pr-4 text-foreground-muted">
                                {formatDuration(track.duration)}
                              </td>
                              <td className="py-1.5 pr-4 text-foreground-muted">
                                {formatSize(track.size)}
                              </td>
                              <td className="py-1.5 pr-4 text-foreground-muted">{track.window_count}</td>
                              <td className="py-1.5">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="glass-lighter border-primary/20 h-6 px-2 text-[10px]"
                                  onClick={() => playTrack(track.id)}
                                >
                                  {playingTrackId === track.id ? 'Stop' : 'Play'}
                                </Button>
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
