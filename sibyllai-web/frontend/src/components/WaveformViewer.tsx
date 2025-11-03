import { useEffect, useRef, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/lib/store'
import { generateTicks, secondsToTimecode } from '@/lib/timecode'

// Define zoom levels for smoother progression
const ZOOM_LEVELS = [0, 10, 25, 50, 100, 200, 400, 800]

export function WaveformViewer() {
  const waveformRef = useRef<HTMLDivElement>(null)
  const wavesurferRef = useRef<WaveSurfer | null>(null)
  const regionsRef = useRef<RegionsPlugin | null>(null)
  const rulerRef = useRef<HTMLDivElement>(null)

  const [isPlaying, setIsPlaying] = useState(false)
  const [isReady, setIsReady] = useState(false)
  const [zoom, setZoom] = useState(0)
  const [playingCueId, setPlayingCueId] = useState<string | null>(null)
  const [currentTimecode, setCurrentTimecode] = useState<string>('')
  const [ticks, setTicks] = useState<Array<{ position: number; timecode: string; isMajor: boolean }>>([])
  const [waveformWidth, setWaveformWidth] = useState(0)
  const [waveformScrollLeft, setWaveformScrollLeft] = useState(0)
  const [rulerVersion, setRulerVersion] = useState(0) // Force re-render trigger
  const [draggingTimecodes, setDraggingTimecodes] = useState<{ start: string; end: string; startPos: number; endPos: number; activeHandle: 'start' | 'end' | 'both' } | null>(null)
  const dragStartRef = useRef<{ start: number; end: number } | null>(null)

  const {
    uploadedFile,
    segments,
    selectedSegments,
    setSelectedSegments,
    updateSegment,
    fileId,
    project,
    activeCueId,
    setActiveCueId,
    startTimecode,
    framerate,
  } = useAppStore()

  // Initialize WaveSurfer
  useEffect(() => {
    if (!waveformRef.current || !uploadedFile || !fileId) return

    // Create WaveSurfer instance
    const ws = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: 'rgb(200, 210, 220)',
      progressColor: 'rgb(59, 130, 246)',
      cursorColor: 'rgb(239, 68, 68)',
      barWidth: 2,
      barGap: 1,
      height: 180,
      normalize: true,
      scrollParent: true,
    })

    // Create regions plugin with drag-to-create enabled
    const regions = ws.registerPlugin(RegionsPlugin.create({
      dragSelection: {
        slop: 5,
      },
    }))

    wavesurferRef.current = ws
    regionsRef.current = regions

    // Load audio file
    const url = URL.createObjectURL(uploadedFile)
    ws.load(url)

    // Event handlers
    ws.on('ready', () => {
      setIsReady(true)
      // Waveform auto-fits by default (no zoom call needed)

      // Generate initial ticks
      const duration = ws.getDuration()
      const initialTicks = generateTicks(duration, 0, framerate, startTimecode)
      setTicks(initialTicks)

      // Initialize timecode display
      setCurrentTimecode(secondsToTimecode(0, framerate, startTimecode))

      // Set initial waveform width
      setTimeout(() => {
        const wrapper = ws.getWrapper()
        const initialWidth = wrapper?.scrollWidth || wrapper?.clientWidth || 0
        console.log('Initial waveformWidth set to:', initialWidth)
        setWaveformWidth(initialWidth)
      }, 0)
    })

    ws.on('play', () => setIsPlaying(true))
    ws.on('pause', () => setIsPlaying(false))

    // Update playhead timecode during playback
    ws.on('audioprocess', (currentTime) => {
      setCurrentTimecode(secondsToTimecode(currentTime, framerate, startTimecode))
    })

    // Update timecode when seeking
    ws.on('seeking', (currentTime) => {
      setCurrentTimecode(secondsToTimecode(currentTime, framerate, startTimecode))
    })

    // Update timecode on click
    ws.on('interaction', () => {
      const currentTime = ws.getCurrentTime()
      setCurrentTimecode(secondsToTimecode(currentTime, framerate, startTimecode))
    })

    // Cleanup
    return () => {
      ws.destroy()
      URL.revokeObjectURL(url)
    }
  }, [uploadedFile, fileId])

  // Track waveform scroll position for ruler alignment
  useEffect(() => {
    if (!isReady) return

    let animationFrameId: number

    const readScrollPosition = () => {
      const wrapper = wavesurferRef.current?.getWrapper() as HTMLElement | null
      const parentContainer = waveformRef.current?.parentElement ?? null

      const nextScrollLeft =
        wrapper?.scrollLeft ?? parentContainer?.scrollLeft ?? 0

      setWaveformScrollLeft((previous) => {
        if (Math.abs(previous - nextScrollLeft) < 0.5) {
          return previous
        }
        return nextScrollLeft
      })

      animationFrameId = requestAnimationFrame(readScrollPosition)
    }

    animationFrameId = requestAnimationFrame(readScrollPosition)

    return () => cancelAnimationFrame(animationFrameId)
  }, [isReady, zoom])

  // Sync width on WaveSurfer redraw events
  useEffect(() => {
    if (!wavesurferRef.current || !isReady || !waveformRef.current) return

    const handleRedraw = () => {
      // Always get actual width from WaveSurfer's wrapper after render
      const wrapper = wavesurferRef.current?.getWrapper()
      const calculatedWidth = wrapper?.scrollWidth || wrapper?.clientWidth || 0
      console.log('Redraw - setting waveformWidth to:', calculatedWidth)
      setWaveformWidth(calculatedWidth)
      setRulerVersion(v => v + 1)
    }

    // Set initial width
    handleRedraw()

    wavesurferRef.current.on('redraw', handleRedraw)
    return () => {
      wavesurferRef.current?.un('redraw', handleRedraw)
    }
  }, [zoom, isReady])

  // Helper to check if segment is selected
  const isSegmentSelected = (start: number, end: number) => {
    return selectedSegments.some(([s, e]) =>
      Math.abs(s - start) < 0.01 && Math.abs(e - end) < 0.01
    )
  }

  // Helper to find cue for a segment
  const findCueForSegment = (start: number, end: number) => {
    if (!project?.cues) return null
    return project.cues.find((cue) =>
      Math.abs(cue.start - start) < 0.01 && Math.abs(cue.end - end) < 0.01
    )
  }

  // Helper to set active cue (only used after analysis)
  const handleSegmentClick = (start: number, end: number) => {
    // Only handle clicks after analysis - clicking sets the active cue
    if (project) {
      const cue = findCueForSegment(start, end)
      if (cue) {
        setActiveCueId(cue.id)
      }
    }
    // Before analysis, do nothing - only the Select button should toggle selection
  }

  // Helper to toggle segment selection
  const toggleSegmentSelection = (start: number, end: number) => {
    const segment: [number, number] = [start, end]
    const isSelected = isSegmentSelected(start, end)

    if (isSelected) {
      // Remove from selection
      setSelectedSegments(selectedSegments.filter(([s, e]) =>
        !(Math.abs(s - start) < 0.01 && Math.abs(e - end) < 0.01)
      ))
    } else {
      // Add to selection
      setSelectedSegments([...selectedSegments, segment])
    }
  }

  // Play a specific cue
  const playActiveCue = () => {
    if (!wavesurferRef.current || !activeCueId || !project) return

    const activeCue = project.cues.find((cue) => cue.id === activeCueId)
    if (!activeCue) return

    const ws = wavesurferRef.current

    // If currently playing this cue, pause
    if (isPlaying && playingCueId === activeCueId) {
      ws.pause()
      setPlayingCueId(null)
      return
    }

    // Seek to cue start and play
    ws.seekTo(activeCue.start / ws.getDuration())
    ws.play()
    setPlayingCueId(activeCueId)

    // Stop at cue end
    const checkPosition = () => {
      if (ws.getCurrentTime() >= activeCue.end) {
        ws.pause()
        setPlayingCueId(null)
      }
    }

    ws.on('audioprocess', checkPosition)
    ws.once('pause', () => {
      ws.un('audioprocess', checkPosition)
    })
  }

  // Handle spacebar to play/pause waveform or active cue
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Only handle spacebar if not typing in an input
      if (
        e.code === 'Space' &&
        document.activeElement?.tagName !== 'INPUT' &&
        document.activeElement?.tagName !== 'TEXTAREA'
      ) {
        e.preventDefault()

        // If we have an active cue (after analysis), play that cue
        if (activeCueId && project) {
          playActiveCue()
        } else {
          // Otherwise, play/pause the main waveform
          handlePlayPause()
        }
      }
    }

    window.addEventListener('keydown', handleKeyPress)
    return () => window.removeEventListener('keydown', handleKeyPress)
  }, [activeCueId, project, isPlaying, playingCueId])

  // Update ticks when zoom, framerate, or startTimecode changes
  useEffect(() => {
    if (!wavesurferRef.current || !isReady || !waveformRef.current) return

    const duration = wavesurferRef.current.getDuration()
    const newTicks = generateTicks(duration, zoom, framerate, startTimecode)
    setTicks(newTicks)
  }, [zoom, isReady, framerate, startTimecode])

  // Update regions when segments, selection, or active cue changes
  useEffect(() => {
    if (!regionsRef.current || segments.length === 0) return

    // Clear existing regions
    regionsRef.current.clearRegions()

    // Add new regions with appropriate styling
    segments.forEach(([start, end], index) => {
      const duration = end - start
      const showLabel = duration > 5
      const selected = isSegmentSelected(start, end)
      const cue = findCueForSegment(start, end)
      const isActive = cue && activeCueId === cue.id

      // Determine color based on state
      let color: string
      if (isActive) {
        color = 'rgba(59, 130, 246, 0.5)' // Bright blue for active
      } else if (selected) {
        color = 'rgba(34, 197, 94, 0.3)' // Green for selected
      } else {
        color = 'rgba(148, 163, 184, 0.2)' // Light gray for unselected
      }

      // Enable dragging/resizing only before analysis
      const editable = !project

      const region = regionsRef.current?.addRegion({
        start,
        end,
        color,
        drag: editable,
        resize: editable,
        content: '',  // Remove number labels
      })

      // Add event handlers
      if (region) {
        // Click handler
        region.on('click', () => {
          handleSegmentClick(start, end)
        })

        // Update handler for drag/resize (only if editable)
        if (editable) {
          // Track initial positions when drag starts
          region.on('update-start', () => {
            dragStartRef.current = { start: region.start, end: region.end }
          })

          // Show timecode while dragging
          region.on('update', () => {
            const startTc = secondsToTimecode(region.start, framerate, startTimecode)
            const endTc = secondsToTimecode(region.end, framerate, startTimecode)

            // Calculate pixel positions relative to waveform container
            const duration = wavesurferRef.current?.getDuration() || 1

            // Get current width dynamically based on zoom
            let containerWidth: number
            if (zoom > 0) {
              containerWidth = duration * zoom
            } else {
              const wrapper = wavesurferRef.current?.getWrapper()
              containerWidth = wrapper?.scrollWidth || wrapper?.clientWidth || 1
            }

            const startPos = (region.start / duration) * containerWidth
            const endPos = (region.end / duration) * containerWidth

            // Detect which handle is being actively dragged
            let activeHandle: 'start' | 'end' | 'both' = 'both'
            if (dragStartRef.current) {
              const startChanged = Math.abs(region.start - dragStartRef.current.start) > 0.01
              const endChanged = Math.abs(region.end - dragStartRef.current.end) > 0.01

              if (startChanged && !endChanged) {
                activeHandle = 'start'
              } else if (endChanged && !startChanged) {
                activeHandle = 'end'
              }
            }

            setDraggingTimecodes({ start: startTc, end: endTc, startPos, endPos, activeHandle })
          })

          // Update segment when drag ends
          region.on('update-end', () => {
            const newStart = region.start
            const newEnd = region.end
            updateSegment(index, newStart, newEnd)
            // Clear dragging timecodes and ref
            setDraggingTimecodes(null)
            dragStartRef.current = null
          })
        }
      }
    })
  }, [segments, selectedSegments, activeCueId, project, framerate, startTimecode, updateSegment, zoom])

  const handlePlayPause = () => {
    if (wavesurferRef.current) {
      wavesurferRef.current.playPause()
    }
  }

  const handleZoomIn = () => {
    if (!wavesurferRef.current) return

    const ws = wavesurferRef.current
    const wrapper = ws.getWrapper()
    const duration = ws.getDuration()

    // Calculate the time position at the center of current view
    const scrollLeft = wrapper.scrollLeft
    const viewportWidth = wrapper.clientWidth
    const scrollWidth = wrapper.scrollWidth

    // Center of viewport in pixels
    const centerPx = scrollLeft + (viewportWidth / 2)

    // Center time position
    const centerTime = (centerPx / scrollWidth) * duration

    // Find current zoom level index and move to next level
    const currentIndex = ZOOM_LEVELS.findIndex(level => level >= zoom)
    const nextIndex = currentIndex < ZOOM_LEVELS.length - 1 ? currentIndex + 1 : ZOOM_LEVELS.length - 1
    const newZoom = ZOOM_LEVELS[nextIndex]

    // Apply zoom to WaveSurfer first, then update state
    ws.zoom(newZoom)
    setZoom(newZoom)

    // After zoom, recalculate scroll position to keep center time centered
    // Use double requestAnimationFrame to ensure DOM has fully updated
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const newWrapper = ws.getWrapper()
        const newScrollWidth = newWrapper.scrollWidth

        // Update ruler width and force re-render
        setWaveformWidth(newScrollWidth)
        setRulerVersion(v => v + 1)
        console.log('Zoom In - scrollWidth:', newScrollWidth, 'zoom:', newZoom)

        // Calculate where the center time is now in pixels
        const newCenterPx = (centerTime / duration) * newScrollWidth

        // Scroll to keep it centered
        const newScrollLeft = newCenterPx - (viewportWidth / 2)
        newWrapper.scrollLeft = Math.max(0, newScrollLeft)
      })
    })
  }

  const handleZoomOut = () => {
    if (!wavesurferRef.current) return

    const ws = wavesurferRef.current
    const wrapper = ws.getWrapper()
    const duration = ws.getDuration()

    // Calculate the time position at the center of current view
    const scrollLeft = wrapper.scrollLeft
    const viewportWidth = wrapper.clientWidth
    const scrollWidth = wrapper.scrollWidth

    // Center of viewport in pixels
    const centerPx = scrollLeft + (viewportWidth / 2)

    // Center time position
    const centerTime = (centerPx / scrollWidth) * duration

    // Find current zoom level index and move to previous level
    const currentIndex = ZOOM_LEVELS.findIndex(level => level >= zoom)
    const prevIndex = currentIndex > 0 ? currentIndex - 1 : 0
    const newZoom = ZOOM_LEVELS[prevIndex]

    // Apply zoom to WaveSurfer first, then update state
    ws.zoom(newZoom)
    setZoom(newZoom)

    // After zoom, recalculate scroll position to keep center time centered
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const newWrapper = ws.getWrapper()
        const newScrollWidth = newWrapper.scrollWidth

        // Update ruler width and force re-render
        setWaveformWidth(newScrollWidth)
        setRulerVersion(v => v + 1)
        console.log('Zoom Out - scrollWidth:', newScrollWidth, 'zoom:', newZoom)

        // Calculate where the center time is now in pixels
        const newCenterPx = (centerTime / duration) * newScrollWidth

        // Scroll to keep it centered
        const newScrollLeft = newCenterPx - (viewportWidth / 2)
        newWrapper.scrollLeft = Math.max(0, newScrollLeft)
      })
    })
  }

  const handleZoomReset = () => {
    setZoom(0)
    if (wavesurferRef.current) {
      wavesurferRef.current.zoom(0)

      // Update ruler width after reset
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          const wrapper = wavesurferRef.current?.getWrapper()
          const newScrollWidth = wrapper?.scrollWidth || wrapper?.clientWidth || 0
          setWaveformWidth(newScrollWidth)
          setRulerVersion(v => v + 1)
          console.log('Zoom Reset - scrollWidth:', newScrollWidth)
        })
      })
    }
  }

  if (!fileId) {
    return null
  }

  return (
    <Card className="w-full">
      <CardContent className="pt-6">
        <div className="space-y-4">
          {/* Select buttons row - above waveform, only show before analysis */}
          {!project && isReady && segments.length > 0 && (
            <div className="relative h-6 w-full">
              {segments.map(([start, end], index) => {
                const duration = wavesurferRef.current?.getDuration() || 1
                const leftPercent = (start / duration) * 100
                const widthPercent = ((end - start) / duration) * 100
                const selected = isSegmentSelected(start, end)

                return (
                  <button
                    key={index}
                    onClick={() => toggleSegmentSelection(start, end)}
                    className="absolute text-xs px-2 py-0.5 rounded transition-colors font-medium flex items-center justify-center"
                    style={{
                      left: `${leftPercent}%`,
                      width: `${widthPercent}%`,
                      minWidth: '60px',
                      backgroundColor: selected ? '#22c55e' : '#94a3b8',
                      color: 'white',
                      border: 'none',
                      cursor: 'pointer',
                    }}
                  >
                    {selected ? '✓' : 'Select'}
                  </button>
                )
              })}
            </div>
          )}

          {/* Timecode Ruler */}
          {isReady && (() => {
            // Always read fresh width directly from WaveSurfer DOM (rulerVersion just triggers re-render)
            const wrapper = wavesurferRef.current?.getWrapper()
            const actualWidth = wrapper?.scrollWidth || wrapper?.clientWidth || 0
            const duration = wavesurferRef.current?.getDuration() || 1

            return (
              <div
                ref={rulerRef}
                className="border rounded-t-lg overflow-hidden bg-gray-50"
              >
                <div
                  className="relative h-8"
                  style={{
                    width: `${actualWidth}px`,
                    minWidth: `${actualWidth}px`,
                    transform: `translateX(-${waveformScrollLeft}px)`,
                    willChange: 'transform',
                  }}
                >
                  {ticks.map((tick, index) => {
                      // Position tick proportionally - same method WaveSurfer uses internally
                      // This ensures perfect alignment with playhead at all zoom levels
                      const leftPx = (tick.position / duration) * actualWidth

                      // Debug logging (only first tick)
                      if (index === 0) {
                        console.log('Ruler render - actualWidth from DOM:', actualWidth, 'duration:', duration, 'zoom:', zoom, 'rulerVersion:', rulerVersion)
                      }

                      return (
                        <div
                          key={index}
                          className="absolute"
                          style={{ left: `${leftPx}px` }}
                        >
                          {/* Tick mark */}
                          <div
                            className="bg-gray-400"
                            style={{
                              width: '1px',
                              height: tick.isMajor ? '12px' : '6px',
                            }}
                          />
                          {/* Timecode label (only on major ticks) */}
                          {tick.isMajor && (
                            <div className="absolute top-3 -translate-x-1/2 text-xs text-gray-600 whitespace-nowrap">
                              {tick.timecode}
                            </div>
                          )}
                        </div>
                      )
                    })}
                </div>
              </div>
            )
          })()}

          <div className="relative overflow-x-auto border border-t-0 rounded-b-lg">
            <div ref={waveformRef} className="w-full min-w-full pt-10" />

            {/* Dragging timecode tooltips - positioned above handles */}
            {draggingTimecodes && (
              <>
                {draggingTimecodes.activeHandle === 'both' ? (
                  /* Combined tooltip when dragging whole region */
                  <div
                    className="absolute top-2 bg-blue-600 text-white px-3 py-2 rounded shadow-lg pointer-events-none text-xs font-mono whitespace-nowrap"
                    style={{
                      left: `${(draggingTimecodes.startPos + draggingTimecodes.endPos) / 2}px`,
                      transform: 'translateX(-50%)',
                      zIndex: 52
                    }}
                  >
                    <div>Start: {draggingTimecodes.start}</div>
                    <div>End: {draggingTimecodes.end}</div>
                  </div>
                ) : (
                  /* Separate tooltips when dragging individual handles */
                  <>
                    {/* Start handle tooltip */}
                    <div
                      className="absolute top-2 bg-blue-600 text-white px-2 py-1 rounded shadow-lg pointer-events-none text-xs font-mono whitespace-nowrap"
                      style={{
                        left: `${draggingTimecodes.startPos}px`,
                        transform: 'translateX(-50%)',
                        zIndex: draggingTimecodes.activeHandle === 'start' ? 52 : 50
                      }}
                    >
                      {draggingTimecodes.start}
                    </div>
                    {/* End handle tooltip */}
                    <div
                      className="absolute top-2 bg-blue-600 text-white px-2 py-1 rounded shadow-lg pointer-events-none text-xs font-mono whitespace-nowrap"
                      style={{
                        left: `${draggingTimecodes.endPos}px`,
                        transform: 'translateX(-50%)',
                        zIndex: draggingTimecodes.activeHandle === 'end' ? 52 : 50
                      }}
                    >
                      {draggingTimecodes.end}
                    </div>
                  </>
                )}
              </>
            )}
          </div>

          {isReady && (
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                {/* Playhead timecode display */}
                <div className="px-3 py-1.5 bg-gray-100 border rounded text-sm font-mono text-gray-700 min-w-[120px] text-center">
                  {currentTimecode || secondsToTimecode(0, framerate, startTimecode)}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handlePlayPause}
                  className="min-w-[70px]"
                >
                  {isPlaying ? 'Pause' : 'Play'}
                </Button>
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleZoomOut}
                  disabled={zoom === ZOOM_LEVELS[0]}
                >
                  Zoom Out
                </Button>
                <span className="text-sm text-muted-foreground min-w-16 text-center">
                  {zoom === 0 ? 'Fit' : `${zoom.toFixed(0)}x`}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleZoomIn}
                  disabled={zoom >= ZOOM_LEVELS[ZOOM_LEVELS.length - 1]}
                >
                  Zoom In
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleZoomReset}
                  disabled={zoom === 0}
                >
                  Reset
                </Button>
              </div>

              <div className="flex items-center gap-4 ml-auto">
                {segments.length > 0 && !project && (
                  <div className="text-xs text-muted-foreground">
                    Drag cue edges to adjust boundaries
                  </div>
                )}
                {project && (
                  <div className="text-sm text-muted-foreground">
                    Click cues to view details
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
