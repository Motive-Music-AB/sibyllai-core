import { useEffect, useLayoutEffect, useRef, useState, useCallback } from 'react'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/lib/store'
import { generateTicks, secondsToTimecode } from '@/lib/timecode'

/**
 * ZOOM LEVELS - pixels per second
 *
 * CRITICAL: WaveSurfer zoom works in pixels-per-second, NOT percentage.
 * - zoom=0 means "fit to container" (auto-calculated px/sec)
 * - zoom=N means N pixels per second
 *
 * THE BUG THAT KEPT BREAKING ZOOM:
 * When at zoom=0 (fit mode), the effective px/sec depends on audio duration
 * and container width. For example:
 *   - 20 second clip in 1000px container = 50 px/sec effective
 *   - If you naively go to zoom=10, you're ZOOMING OUT (10 < 50)!
 *
 * THE FIX: Always calculate effectiveZoom = viewportWidth / duration when
 * at zoom=0, then find a level that's actually > or < that value.
 *
 * DO NOT SIMPLIFY THIS LOGIC - it took weeks to debug!
 */
const ZOOM_LEVELS = [0, 10, 25, 50, 100]

export function WaveformViewer() {
  const waveformRef = useRef<HTMLDivElement>(null)
  const wavesurferRef = useRef<WaveSurfer | null>(null)
  const regionsRef = useRef<RegionsPlugin | null>(null)
  const rulerRef = useRef<HTMLDivElement>(null)
  const zoomRef = useRef(0) // Synchronous zoom tracking for redraw event handler

  const [isPlaying, setIsPlaying] = useState(false)
  const [isReady, setIsReady] = useState(false)
  const [zoom, setZoom] = useState(0)
  const [playingCueId, setPlayingCueId] = useState<string | null>(null)
  const [currentTimecode, setCurrentTimecode] = useState<string>('')
  const [ticks, setTicks] = useState<Array<{ position: number; timecode: string; isMajor: boolean }>>([])
  const [waveformWidth, setWaveformWidth] = useState(0)
  const [fallbackWidth, setFallbackWidth] = useState(0) // Fallback width calculated in useLayoutEffect to avoid DOM reads during render
  const [draggingTimecodes, setDraggingTimecodes] = useState<{ start: string; end: string; startPos: number; endPos: number; activeHandle: 'start' | 'end' | 'both' } | null>(null)
  const dragStartRef = useRef<{ start: number; end: number } | null>(null)
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; segmentIndex: number } | null>(null)
  const regionCreatedHandlerRef = useRef<((region: any) => void) | null>(null)

  const {
    uploadedFile,
    segments,
    selectedSegments,
    setSelectedSegments,
    updateSegment,
    splitSegment,
    addSegment,
    deleteSegment,
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
      waveColor: 'rgba(200, 130, 80, 0.6)',
      progressColor: 'rgb(232, 148, 58)',
      cursorColor: 'rgb(232, 148, 58)',
      barWidth: 2,
      barGap: 1,
      height: 180,
      normalize: true,
      backend: 'WebAudio',
    })

    // Create regions plugin with drag-to-create enabled
    const regions = ws.registerPlugin(RegionsPlugin.create())

    wavesurferRef.current = ws
    regionsRef.current = regions

    // Load audio file
    const url = URL.createObjectURL(uploadedFile)
    ws.load(url)

    // Event handlers
    ws.on('ready', () => {
      setIsReady(true)
      // Waveform auto-fits by default (no zoom call needed)
      // Initial ticks and width are set by handleRedraw (triggered by isReady state change)

      // Initialize timecode display
      setCurrentTimecode(secondsToTimecode(0, framerate, startTimecode))
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadedFile, fileId])

  // Ruler now scrolls naturally inside the waveform container - no sync needed!

  // Helper function to update width and ticks - can be called from anywhere
  const updateWidthAndTicks = useCallback(() => {
    const ws = wavesurferRef.current
    if (!ws) return

    const currentZoom = zoomRef.current
    const wrapper = ws.getWrapper()
    const duration = ws.getDuration()
    
    // Calculate width based on zoom level and duration
    // For zoom=0 (fit mode), use clientWidth (actual container width)
    // For zoom > 0, calculate as duration * zoom (pixels per second)
    let calculatedWidth: number
    if (currentZoom === 0) {
      calculatedWidth = wrapper?.clientWidth || 0
    } else {
      // Calculate expected width: duration (seconds) * zoom (pixels per second)
      // Don't read from DOM - it may still have the old value during zoom transitions
      calculatedWidth = duration * currentZoom
    }

    // Regenerate ticks with current zoom level from ref (synchronous, not React state)
    const newTicks = generateTicks(duration, currentZoom, framerate, startTimecode)

    console.log('Update width - width:', calculatedWidth, 'zoomRef:', currentZoom, 'zoomState:', zoom, 'ticks:', newTicks.length)

    // Update both width and ticks atomically
    setWaveformWidth(calculatedWidth)
    setTicks(newTicks)
  }, [zoom, framerate, startTimecode])

  // Sync width and ticks on WaveSurfer redraw events
  useEffect(() => {
    if (!wavesurferRef.current || !isReady || !waveformRef.current) return

    let previousZoom = zoomRef.current
    let isZoomChanging = false

    const handleRedraw = () => {
      const ws = wavesurferRef.current
      if (!ws) return

      const currentZoom = zoomRef.current
      const zoomChanged = currentZoom !== previousZoom

      // When zoom changes, skip the immediate redraw event
      // The zoom handlers will manually trigger updateWidthAndTicks after delay
      if (zoomChanged) {
        previousZoom = currentZoom
        isZoomChanging = true
        
        // Reset flag after a delay to prevent redraw events from overriding calculated width
        // Use longer delay for zoom out (DOM takes longer to update)
        setTimeout(() => {
          isZoomChanging = false
        }, 200)
        
        // Skip this redraw - zoom handlers will handle it
        return
      }

      // No zoom change - update width and ticks
      // Always use updateWidthAndTicks which calculates from duration for non-zero zoom
      if (!isZoomChanging) {
        updateWidthAndTicks()
      }
    }

    // Set initial width and ticks
    updateWidthAndTicks()

    wavesurferRef.current.on('redraw', handleRedraw)
    return () => {
      wavesurferRef.current?.un('redraw', handleRedraw)
    }
  }, [isReady, framerate, startTimecode, updateWidthAndTicks])

  // Calculate fallback width in useLayoutEffect to avoid DOM reads during render
  // This prevents React warning about state updates during render
  useLayoutEffect(() => {
    if (!isReady || waveformWidth > 0) {
      // If we have a valid width from handleRedraw, don't calculate fallback
      setFallbackWidth(0)
      return
    }

    // Only calculate fallback if waveformWidth is 0 or invalid
    const wrapper = wavesurferRef.current?.getWrapper()
    if (zoom === 0) {
      // In fit mode, use container width
      const container = waveformRef.current?.parentElement
      const width = container?.clientWidth || wrapper?.clientWidth || 0
      setFallbackWidth(width)
    } else {
      // In zoom mode, use scroll width
      const width = wrapper?.scrollWidth || wrapper?.clientWidth || 0
      setFallbackWidth(width)
    }
  }, [isReady, waveformWidth, zoom])

  // Helper to check if segment is selected
  const isSegmentSelected = useCallback((start: number, end: number) => {
    return selectedSegments.some(([s, e]) =>
      Math.abs(s - start) < 0.01 && Math.abs(e - end) < 0.01
    )
  }, [selectedSegments])

  // Helper to find cue for a segment
  const findCueForSegment = useCallback((start: number, end: number) => {
    if (!project?.cues) return null
    return project.cues.find((cue) =>
      Math.abs(cue.start - start) < 0.01 && Math.abs(cue.end - end) < 0.01
    )
  }, [project])

  // Helper to set active cue (only used after analysis)
  const handleSegmentClick = useCallback((start: number, end: number) => {
    // Only handle clicks after analysis - clicking sets the active cue
    if (project) {
      const cue = findCueForSegment(start, end)
      if (cue) {
        setActiveCueId(cue.id)
      }
    }
    // Before analysis, do nothing - only the Select button should toggle selection
  }, [project, findCueForSegment, setActiveCueId])

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCueId, project, isPlaying, playingCueId])

  // Close context menu on click outside
  useEffect(() => {
    const handleClickOutside = () => setContextMenu(null)
    if (contextMenu) {
      window.addEventListener('click', handleClickOutside)
      return () => window.removeEventListener('click', handleClickOutside)
    }
  }, [contextMenu])

  // Setup drag-to-create separately from region rendering
  useEffect(() => {
    if (!regionsRef.current) return

    const shouldEnableDragCreate = !project && segments.length > 0

    if (shouldEnableDragCreate && !regionCreatedHandlerRef.current) {
      // Enable drag selection for creating new regions
      regionsRef.current.enableDragSelection({
        color: 'rgba(148, 163, 184, 0.2)',
      })

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const handleRegionCreated = (region: any) => {
        // Batch the state update
        requestAnimationFrame(() => {
          addSegment(region.start, region.end)
        })
      }

      regionCreatedHandlerRef.current = handleRegionCreated
      regionsRef.current.on('region-created', handleRegionCreated)
    }

    // Cleanup when switching to analysis mode
    if (project && regionCreatedHandlerRef.current) {
      regionsRef.current?.off('region-created', regionCreatedHandlerRef.current)
      regionCreatedHandlerRef.current = null
    }
  }, [project, segments.length, addSegment])

  // Handle split cue at playhead
  const handleSplitAtPlayhead = (segmentIndex: number) => {
    if (!wavesurferRef.current) return
    const currentTime = wavesurferRef.current.getCurrentTime()
    splitSegment(segmentIndex, currentTime)
    setContextMenu(null)
  }

  // Handle delete cue
  const handleDeleteCue = (segmentIndex: number) => {
    deleteSegment(segmentIndex)
    setContextMenu(null)
  }

  // Note: Ticks are now generated in handleRedraw (above) to ensure they're synchronized
  // with WaveSurfer's width changes. This prevents race conditions where ticks are
  // regenerated with the old width before WaveSurfer finishes rendering.

  // Update regions when segments, selection, or active cue changes
  useEffect(() => {
    if (!regionsRef.current) return

    // Clear existing regions
    regionsRef.current.clearRegions()

    if (segments.length === 0) return

    // Add new regions with appropriate styling
    segments.forEach(([start, end], index) => {
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
        region.on('click', (e) => {
          // Check if it's a right-click
          if (e.button === 2) {
            e.preventDefault()
            e.stopPropagation()

            // Only show context menu before analysis
            if (!project) {
              setContextMenu({
                x: e.clientX,
                y: e.clientY,
                segmentIndex: index,
              })
            }
          } else {
            // Left click - normal behavior
            handleSegmentClick(start, end)
          }
        })

        // Prevent default context menu on region
        if (!project && region.element) {
          region.element.addEventListener('contextmenu', (e) => {
            e.preventDefault()
          })
        }

        // Update handler for drag/resize (only if editable)
        if (editable) {
          // Track initial positions when drag starts - store in a closure variable
          let dragStart: { start: number; end: number } | null = null

          // Listen to drag/resize events to capture start position
          const handleDragStart = () => {
            dragStart = { start: region.start, end: region.end }
            dragStartRef.current = dragStart
          }

          // WaveSurfer regions may not have update-start event, so we capture on first update
          let isFirstUpdate = true

          // Show timecode while dragging
          region.on('update', () => {
            // Capture initial position on first update (start of drag)
            if (isFirstUpdate) {
              handleDragStart()
              isFirstUpdate = false
            }

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
            if (dragStart) {
              const startChanged = Math.abs(region.start - dragStart.start) > 0.01
              const endChanged = Math.abs(region.end - dragStart.end) > 0.01

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
            // Clear dragging timecodes and reset state
            setDraggingTimecodes(null)
            dragStartRef.current = null
            dragStart = null
            isFirstUpdate = true
          })
        }
      }
    })
  }, [segments, selectedSegments, activeCueId, project, framerate, startTimecode, updateSegment, zoom, addSegment, isSegmentSelected, findCueForSegment, handleSegmentClick])

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

    // Calculate effective zoom level (pixels per second) when in fit mode
    const effectiveZoom = zoom === 0 ? viewportWidth / duration : zoom

    // Find the next zoom level that's actually higher than current effective zoom
    let newZoom = ZOOM_LEVELS[ZOOM_LEVELS.length - 1] // default to max
    for (const level of ZOOM_LEVELS) {
      if (level > effectiveZoom) {
        newZoom = level
        break
      }
    }

    console.log('Zoom In - effectiveZoom:', effectiveZoom, 'newZoom:', newZoom)

    // Update zoom ref synchronously BEFORE calling ws.zoom() (which fires redraw event)
    zoomRef.current = newZoom
    ws.zoom(newZoom)
    setZoom(newZoom)

    // Update width and ticks immediately (calculation-based, no DOM read needed)
    updateWidthAndTicks()

    // After zoom, recalculate scroll position to keep center time centered
    // Use requestAnimationFrame to ensure DOM has updated for scroll calculation
    requestAnimationFrame(() => {
      const newWrapper = ws.getWrapper()
      // Calculate expected width from zoom level
      const expectedWidth = newZoom === 0 ? newWrapper.clientWidth : duration * newZoom
      const newScrollWidth = newZoom === 0 ? newWrapper.clientWidth : expectedWidth

      // Calculate where the center time is now in pixels
      const newCenterPx = (centerTime / duration) * newScrollWidth

      // Scroll to keep it centered
      const newScrollLeft = newCenterPx - (viewportWidth / 2)
      newWrapper.scrollLeft = Math.max(0, newScrollLeft)
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

    // Calculate effective zoom level (pixels per second)
    const effectiveZoom = zoom === 0 ? viewportWidth / duration : zoom

    // Find the next zoom level that's lower than current effective zoom
    let newZoom = 0 // default to fit
    for (let i = ZOOM_LEVELS.length - 1; i >= 0; i--) {
      if (ZOOM_LEVELS[i] < effectiveZoom) {
        newZoom = ZOOM_LEVELS[i]
        break
      }
    }

    console.log('Zoom Out - effectiveZoom:', effectiveZoom, 'newZoom:', newZoom)

    // If already at fit/min zoom, don't do anything
    if (zoom === 0 || newZoom >= effectiveZoom) {
      console.log('Already at minimum zoom, skipping')
      return
    }

    // Update zoom ref synchronously BEFORE calling ws.zoom() (which fires redraw event)
    zoomRef.current = newZoom
    ws.zoom(newZoom)
    setZoom(newZoom)

    // Update width and ticks immediately (calculation-based, no DOM read needed)
    updateWidthAndTicks()

    // After zoom, recalculate scroll position to keep center time centered
    // Use requestAnimationFrame to ensure DOM has updated for scroll calculation
    requestAnimationFrame(() => {
      const newWrapper = ws.getWrapper()
      // Calculate expected width from zoom level
      const expectedWidth = newZoom === 0 ? newWrapper.clientWidth : duration * newZoom
      const newScrollWidth = newZoom === 0 ? newWrapper.clientWidth : expectedWidth

      // Calculate where the center time is now in pixels
      const newCenterPx = (centerTime / duration) * newScrollWidth

      // Scroll to keep it centered
      const newScrollLeft = newCenterPx - (viewportWidth / 2)
      newWrapper.scrollLeft = Math.max(0, newScrollLeft)
    })
  }

  const handleZoomReset = () => {
    console.log('Zoom Reset - current:', zoom, 'resetting to 0')

    if (!wavesurferRef.current) return

    // Update zoom ref synchronously BEFORE calling ws.zoom() (which fires redraw event)
    zoomRef.current = 0
    setZoom(0)
    wavesurferRef.current.zoom(0)

    // Update width and ticks immediately (for zoom=0, reads from DOM which is stable)
    updateWidthAndTicks()
  }

  if (!fileId) {
    return null
  }

  return (
    <Card className="w-full border-white/10 shadow-none">
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

          <div className="overflow-x-auto rounded-lg bg-[rgba(30,20,15,0.6)]">
            {/* Wrapper that contains both ruler and waveform - scrolls as one unit */}
            <div
              style={{
                display: zoom === 0 ? 'block' : 'inline-block',
                width: zoom === 0 ? '100%' : 'auto',
                minWidth: zoom === 0 ? '100%' : '0'
              }}
            >
              {/* Timecode Ruler - flows naturally in document, scrolls with waveform */}
              {isReady && (() => {
                // Always prefer waveformWidth from state (updated by handleRedraw) for consistency
                // Fallback to fallbackWidth state (calculated in useLayoutEffect) to avoid DOM reads during render
                const displayWidth = waveformWidth || fallbackWidth

                const duration = wavesurferRef.current?.getDuration() || 1

                return (
                  <div
                    ref={rulerRef}
                    className="relative h-8 bg-[rgba(40,28,22,0.8)] border-b border-white/10"
                    style={{
                      width: zoom === 0 ? '100%' : `${displayWidth}px`,
                      minWidth: zoom === 0 ? '100%' : `${displayWidth}px`,
                    }}
                  >
                    {ticks.map((tick, index) => {
                      const leftPx = (tick.position / duration) * displayWidth

                      if (index === 0) {
                        console.log('Ruler render - width:', displayWidth, 'zoom:', zoom, 'ticks:', ticks.length)
                      }

                      return (
                        <div
                          key={index}
                          className="absolute"
                          style={{ left: `${leftPx}px` }}
                        >
                          {/* Tick mark */}
                          <div
                            className="bg-foreground/40"
                            style={{
                              width: '1px',
                              height: tick.isMajor ? '12px' : '6px',
                            }}
                          />
                          {/* Timecode label (only on major ticks) */}
                          {tick.isMajor && (
                            <div className="absolute top-3 -translate-x-1/2 text-xs text-foreground/60 whitespace-nowrap">
                              {tick.timecode}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )
              })()}

              <div ref={waveformRef} className="w-full" />

              {/* Dragging timecode tooltips - positioned above handles */}
              {draggingTimecodes && (
                <>
                  {draggingTimecodes.activeHandle === 'both' ? (
                    /* Combined tooltip when dragging whole region */
                    <div
                      className="absolute top-2 bg-primary text-primary-foreground px-3 py-2 rounded shadow-lg pointer-events-none text-xs font-mono whitespace-nowrap"
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
                        className="absolute top-2 bg-primary text-primary-foreground px-2 py-1 rounded shadow-lg pointer-events-none text-xs font-mono whitespace-nowrap"
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
                        className="absolute top-2 bg-primary text-primary-foreground px-2 py-1 rounded shadow-lg pointer-events-none text-xs font-mono whitespace-nowrap"
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
          </div>

          {isReady && (
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                {/* Playhead timecode display */}
                <div className="px-3 py-1.5 bg-white/10 border border-white/10 rounded text-sm font-mono text-foreground min-w-[120px] text-center">
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
                    Drag edges to adjust • Drag empty area to add • Right-click to split/delete
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

          {/* Context Menu */}
          {contextMenu && (
            <div
              className="fixed bg-white border border-gray-300 rounded-md shadow-lg py-1 z-50"
              style={{
                left: `${contextMenu.x}px`,
                top: `${contextMenu.y}px`,
              }}
              onClick={(e) => e.stopPropagation()}
            >
              <button
                className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 flex items-center gap-2"
                onClick={() => handleSplitAtPlayhead(contextMenu.segmentIndex)}
              >
                <span>✂️</span>
                <span>Split at Playhead</span>
              </button>
              <button
                className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 flex items-center gap-2 text-red-600"
                onClick={() => handleDeleteCue(contextMenu.segmentIndex)}
              >
                <span>🗑️</span>
                <span>Delete Cue</span>
              </button>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
