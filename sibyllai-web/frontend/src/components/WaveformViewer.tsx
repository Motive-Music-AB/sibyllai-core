import { useEffect, useRef, useState, useCallback } from 'react'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin, { type Region } from 'wavesurfer.js/dist/plugins/regions'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/lib/store'
import { generateTicks, secondsToTimecode } from '@/lib/timecode'
import type { Cue } from '@/lib/types'

// Zoom levels are pixels-per-second; zoom=0 means "fit to container".
// Reduced max zoom to prevent slow renders. 200 px/sec gives 8+ pixels per frame at 24fps.
const ZOOM_LEVELS = [0, 10, 25, 50, 100, 200]

export function WaveformViewer() {
  const waveformRef = useRef<HTMLDivElement>(null)
  const wavesurferRef = useRef<WaveSurfer | null>(null)
  const regionsRef = useRef<RegionsPlugin | null>(null)
  const rulerRef = useRef<HTMLDivElement>(null)
  const zoomRef = useRef(0) // Synchronous zoom tracking for redraw event handler
  const pendingScrollRef = useRef<number | null>(null) // Track pending scroll position after zoom
  const lastDragUpdateRef = useRef(0) // Throttle drag updates to 30fps
  const peaksOptimizedRef = useRef(false) // Prevent repeated peak optimization per file
  const isZoomingRef = useRef(false) // Synchronous zoom guard to prevent race conditions

  // Refs for framerate/startTimecode - used in drag handlers without triggering region recreation
  const framerateRef = useRef(25)
  const startTimecodeRef = useRef('00:00:00:00')

  // Refs to hold latest callback values - avoids putting callbacks in useEffect dependency arrays
  // which would cause regions to be destroyed/recreated on every interaction
  const isSegmentSelectedRef = useRef<(start: number, end: number) => boolean>(() => false)
  const findCueForSegmentRef = useRef<(start: number, end: number) => Cue | null>((() => null))
  const handleSegmentClickRef = useRef<(start: number, end: number) => void>(() => {})
  const toggleSegmentSelectionRef = useRef<(start: number, end: number) => void>(() => {})

  const [isPlaying, setIsPlaying] = useState(false)
  const [isReady, setIsReady] = useState(false)
  const [zoom, setZoom] = useState(0)
  const [isZooming, setIsZooming] = useState(false) // For UI overlay only
  const [currentTimecode, setCurrentTimecode] = useState<string>('')
  // Combined state for ruler data - prevents double renders when updating width and ticks
  const [rulerData, setRulerData] = useState<{
    width: number
    ticks: Array<{ position: number; timecode: string; isMajor: boolean; leftPx: number }>
  }>({ width: 0, ticks: [] })
  // Derived values from rulerData for easier access
  const waveformWidth = rulerData.width
  const ticks = rulerData.ticks
  const [draggingTimecodes, setDraggingTimecodes] = useState<{ start: string; end: string; startPos: number; endPos: number; activeHandle: 'start' | 'end' | 'both'; mouseX?: number; mouseY?: number } | null>(null)
  const dragStartRef = useRef<{ start: number; end: number } | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState<{ index: number; x: number; y: number } | null>(null)
  const regionCreatedHandlerRef = useRef<((region: Region) => void) | null>(null)
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; segmentIndex: number; splitTime: number } | null>(null)

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
    setPlayingCueId,
    startTimecode,
    framerate,
  } = useAppStore()

  // Initialize WaveSurfer
  useEffect(() => {
    if (!waveformRef.current || !uploadedFile || !fileId) return

    // Reset per-file optimization state
    peaksOptimizedRef.current = false

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
      setCurrentTimecode(secondsToTimecode(0, framerate, startTimecode))

      // Performance: Replace full-resolution decoded data with downsampled peaks
      if (!peaksOptimizedRef.current) {
        peaksOptimizedRef.current = true
        const duration = ws.getDuration()
        // Cap at 2000 peaks total regardless of duration for consistent performance
        const MAX_PEAKS = 2000
        const maxLength = Math.min(MAX_PEAKS, Math.max(100, Math.ceil(duration * 50)))

        console.log(`[Waveform] Duration: ${duration.toFixed(1)}s, Downsampling to ${maxLength} peaks`)

        setTimeout(() => {
          if (wavesurferRef.current !== ws) return
          try {
            const peaks = ws.exportPeaks({ maxLength })
            console.log(`[Waveform] Exported ${peaks[0]?.length || 0} peaks`)
            ws.setOptions({ peaks, duration })
          } catch (err) {
            console.warn('Waveform peak optimization failed:', err)
          }
        }, 0)
      }
    })

    ws.on('play', () => setIsPlaying(true))
    ws.on('pause', () => setIsPlaying(false))

    ws.on('audioprocess', (currentTime) => {
      setCurrentTimecode(secondsToTimecode(currentTime, framerate, startTimecode))
    })

    ws.on('seeking', (currentTime) => {
      setCurrentTimecode(secondsToTimecode(currentTime, framerate, startTimecode))
    })

    ws.on('interaction', () => {
      const currentTime = ws.getCurrentTime()
      setCurrentTimecode(secondsToTimecode(currentTime, framerate, startTimecode))
    })

    // Cleanup
    return () => {
      ws.destroy()
      URL.revokeObjectURL(url)
      if (wavesurferRef.current === ws) {
        wavesurferRef.current = null
      }
      if (regionsRef.current === regions) {
        regionsRef.current = null
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [uploadedFile, fileId])

  // Keep framerate/startTimecode refs updated for drag handlers
  // These refs allow drag handlers to access current values without being in the regions useEffect deps
  useEffect(() => {
    framerateRef.current = framerate
    startTimecodeRef.current = startTimecode
  }, [framerate, startTimecode])

  // Ruler now scrolls naturally inside the waveform container - no sync needed!

  // Helper function to update width and ticks - can be called from anywhere
  // PERFORMANCE: Combined into single state update to prevent double renders
  const updateWidthAndTicks = useCallback(() => {
    const ws = wavesurferRef.current
    if (!ws) return

    const currentZoom = zoomRef.current
    const duration = ws.getDuration()

    // Calculate width based on zoom level and duration
    // For zoom=0 (fit mode), use clientWidth (actual container width)
    // For zoom > 0, calculate as duration * zoom (pixels per second)
    let calculatedWidth: number
    if (currentZoom === 0) {
      calculatedWidth = ws.getWidth() || 0
    } else {
      // Calculate expected width: duration (seconds) * zoom (pixels per second)
      // Don't read from DOM - it may still have the old value during zoom transitions
      calculatedWidth = duration * currentZoom
    }

    // Regenerate ticks with current zoom level from ref (synchronous, not React state)
    const newTicks = generateTicks(duration, currentZoom, framerate, startTimecode)

    // Pre-calculate pixel positions for each tick to avoid calculations in render loop
    const ticksWithPositions = newTicks.map(tick => ({
      ...tick,
      leftPx: (tick.position / duration) * calculatedWidth
    }))

    // Update both width and ticks atomically in single state update
    setRulerData({ width: calculatedWidth, ticks: ticksWithPositions })
  }, [framerate, startTimecode])

  // Set initial width and ticks when ready
  // Note: We no longer listen to redraw events - they fire too frequently and cause
  // performance issues. Width/ticks are only updated on zoom changes (handled by zoom functions).
  useEffect(() => {
    if (!wavesurferRef.current || !isReady || !waveformRef.current) return
    updateWidthAndTicks()
  }, [isReady, framerate, startTimecode, updateWidthAndTicks])

  // Helper to check if segment is selected
  const isSegmentSelected = useCallback((start: number, end: number) => {
    return selectedSegments.some(([s, e]) =>
      Math.abs(s - start) < 0.01 && Math.abs(e - end) < 0.01
    )
  }, [selectedSegments])

  // Helper to find cue for a segment
  const findCueForSegment = useCallback((start: number, end: number): Cue | null => {
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
      } else {
        // Clear active cue if clicking on unanalyzed segment
        setActiveCueId(null)
      }
    }
    // Before analysis, do nothing - only the Select button should toggle selection
  }, [project, findCueForSegment, setActiveCueId])

  // Keep refs updated with latest callback values
  // This allows the regions useEffect to use stable refs instead of callbacks in deps
  useEffect(() => {
    isSegmentSelectedRef.current = isSegmentSelected
    findCueForSegmentRef.current = findCueForSegment
    handleSegmentClickRef.current = handleSegmentClick
    toggleSegmentSelectionRef.current = toggleSegmentSelection
  })

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
  const playActiveCue = useCallback(() => {
    if (!wavesurferRef.current || !activeCueId || !project) return

    const activeCue = project.cues.find((cue) => cue.id === activeCueId)
    if (!activeCue) return

    const ws = wavesurferRef.current

    // If currently playing, pause
    if (isPlaying) {
      ws.pause()
      setPlayingCueId(null)
      return
    }

    // Seek to cue start and play
    ws.setTime(activeCue.start)
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
  }, [activeCueId, project, isPlaying, setPlayingCueId])

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
  }, [activeCueId, project, playActiveCue])

  // Close context menu on click outside
  useEffect(() => {
    const handleClickOutside = () => setContextMenu(null)
    if (contextMenu) {
      window.addEventListener('click', handleClickOutside)
      return () => window.removeEventListener('click', handleClickOutside)
    }
  }, [contextMenu])

  // Handle split cue at mouse position
  const handleSplitAtMouse = (segmentIndex: number, splitTime: number) => {
    splitSegment(segmentIndex, splitTime)
    setContextMenu(null)
  }

  // Handle delete cue - show confirmation dialog
  const handleDeleteCue = (segmentIndex: number, x: number, y: number) => {
    setDeleteConfirm({
      index: segmentIndex,
      x,
      y
    })
    setContextMenu(null)
  }

  // Note: Ticks are now generated in handleRedraw (above) to ensure they're synchronized
  // with WaveSurfer's width changes. This prevents race conditions where ticks are
  // regenerated with the old width before WaveSurfer finishes rendering.

  // Update regions when segments, selection, or active cue changes
  // PERFORMANCE: Uses refs for callbacks to avoid recreating all regions when callbacks change
  useEffect(() => {
    if (!regionsRef.current) return

    // Remove old region-created listener if it exists
    if (regionCreatedHandlerRef.current) {
      regionsRef.current.un('region-created', regionCreatedHandlerRef.current)
      regionCreatedHandlerRef.current = null
    }

    // Clear existing regions
    regionsRef.current.clearRegions()

    if (segments.length === 0) return

    // Safety check: Limit number of regions to prevent freezing
    const MAX_REGIONS = 20
    const segmentsToRender = segments.slice(0, MAX_REGIONS)

    if (segments.length > MAX_REGIONS) {
      console.warn(`Too many segments (${segments.length}). Only rendering first ${MAX_REGIONS}. Increase threshold to detect fewer cues.`)
    }

    // Add new regions synchronously - no staggered setTimeout
    // Staggering created competing timers that blocked scroll events
    segmentsToRender.forEach(([start, end]) => {
      if (!regionsRef.current) return

      const actualIndex = segments.findIndex(([s, e]) => s === start && e === end)
      // Use refs to get latest values without adding to dependency array
      const selected = isSegmentSelectedRef.current(start, end)
      const cue = findCueForSegmentRef.current(start, end)
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
        content: '',  // Empty content, we'll add button manually
      })

      // Add event handlers
      if (region && region.element) {
        // Add visual separation between adjacent cues (border on both sides)
        region.element.style.borderLeft = '2px solid rgba(100, 116, 139, 0.6)'
        region.element.style.borderRight = '2px solid rgba(100, 116, 139, 0.6)'
        region.element.style.boxSizing = 'border-box'

        // Create and add select button programmatically
        const button = document.createElement('button')
        button.className = 'region-select-btn'
        button.textContent = selected ? '✓' : 'Select'
        button.style.cssText = `
          position: absolute;
          top: 4px;
          right: 4px;
          padding: 2px 8px;
          font-size: 11px;
          font-weight: 500;
          border: none;
          border-radius: 3px;
          cursor: pointer;
          z-index: 10;
          background-color: ${selected ? 'rgba(34, 197, 94, 0.9)' : 'rgba(148, 163, 184, 0.7)'};
          color: white;
          transition: background-color 0.2s;
        `

        // Add hover effects
        button.addEventListener('mouseenter', () => {
          button.style.opacity = '0.8'
        })
        button.addEventListener('mouseleave', () => {
          button.style.opacity = '1'
        })

        // Add click handler for select button
        button.addEventListener('click', (e) => {
          e.stopPropagation() // Prevent region click
          toggleSegmentSelectionRef.current(start, end)
        })

        // Append button to region element
        region.element.appendChild(button)

        // Click handler - use ref to get latest callback
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
                segmentIndex: actualIndex,
              })
            }
          } else {
            // Left click - normal behavior, use ref for latest callback
            handleSegmentClickRef.current(start, end)
          }
        })

        // Right-click to show context menu (split/delete)
        if (!project && region.element) {
          region.element.addEventListener('contextmenu', (e) => {
            e.preventDefault()

            // Calculate the time position where the click occurred
            const ws = wavesurferRef.current
            if (!ws) return

            const rect = waveformRef.current?.getBoundingClientRect()
            if (!rect) return

            const clickX = e.clientX - rect.left
            const duration = ws.getDuration()
            const clickTime = (clickX / rect.width) * duration

            // Show context menu with split/delete options
            setContextMenu({
              segmentIndex: actualIndex,
              x: e.clientX,
              y: e.clientY,
              splitTime: clickTime
            })
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

          // Show timecode while dragging - THROTTLED to 30fps to prevent state update spam
          region.on('update', () => {
            // Throttle updates to max 30fps (every 33ms)
            const now = Date.now()
            if (now - lastDragUpdateRef.current < 33) return
            lastDragUpdateRef.current = now

            // Capture initial position on first update (start of drag)
            if (isFirstUpdate) {
              handleDragStart()
              isFirstUpdate = false
            }

            // Use refs to get current framerate/startTimecode without adding to useEffect deps
            const startTc = secondsToTimecode(region.start, framerateRef.current, startTimecodeRef.current)
            const endTc = secondsToTimecode(region.end, framerateRef.current, startTimecodeRef.current)

            // Calculate pixel positions relative to waveform container
            const duration = wavesurferRef.current?.getDuration() || 1

            // Get current width dynamically based on zoom (use ref to avoid dependency)
            let containerWidth: number
            const currentZoom = zoomRef.current
            if (currentZoom > 0) {
              containerWidth = duration * currentZoom
            } else {
              const wrapper = wavesurferRef.current?.getWrapper()
              containerWidth = wrapper?.scrollWidth || wrapper?.clientWidth || 1
            }

            const startPos = (region.start / duration) * containerWidth
            const endPos = (region.end / duration) * containerWidth

            // Detect which handle is being actively dragged
            let activeHandle: 'start' | 'end' | 'both' = 'both'
            if (dragStart) {
              const startDelta = Math.abs(region.start - dragStart.start)
              const endDelta = Math.abs(region.end - dragStart.end)
              const threshold = 0.001 // More sensitive threshold

              // Determine handle based on which changed more significantly
              if (startDelta > threshold && endDelta <= threshold) {
                activeHandle = 'start'
              } else if (endDelta > threshold && startDelta <= threshold) {
                activeHandle = 'end'
              } else if (startDelta > threshold && endDelta > threshold) {
                // Both changed - pick the one that changed more (likely floating point noise on the other)
                activeHandle = startDelta > endDelta ? 'start' : 'end'
              }
            }

            setDraggingTimecodes({ start: startTc, end: endTc, startPos, endPos, activeHandle })
          })

          // Update segment when drag ends
          region.on('update-end', () => {
            const newStart = region.start
            const newEnd = region.end
            updateSegment(actualIndex, newStart, newEnd)
            // Clear dragging timecodes and reset state
            setDraggingTimecodes(null)
            dragStartRef.current = null
            dragStart = null
            isFirstUpdate = true
          })
        }
      }
    })

    // Enable drag-to-create for user (only before analysis and after detection)
    if (!project && segments.length > 0 && regionsRef.current) {
      // Remove old region-created listener if it exists
      if (regionCreatedHandlerRef.current) {
        regionsRef.current.un('region-created', regionCreatedHandlerRef.current)
      }

      // Enable drag selection for creating new regions
      regionsRef.current.enableDragSelection({
        color: 'rgba(148, 163, 184, 0.2)',
      })

      // Listen for user-created regions (drag-to-create)
      const handleRegionCreated = (region: Region) => {
        // Only add if it's a new region (not already in segments)
        const exists = segments.some(([s, e]) =>
          Math.abs(s - region.start) < 0.1 && Math.abs(e - region.end) < 0.1
        )
        if (!exists) {
          // Use setTimeout(0) to defer state update to next tick
          setTimeout(() => {
            addSegment(region.start, region.end)
          }, 0)
        }
      }

      // Store reference and add listener
      regionCreatedHandlerRef.current = handleRegionCreated
      regionsRef.current.on('region-created', handleRegionCreated)
    }
  // PERFORMANCE: Callbacks and framerate/startTimecode removed from deps - we use refs
  // (isSegmentSelectedRef, findCueForSegmentRef, handleSegmentClickRef, framerateRef, startTimecodeRef)
  // to access latest values without triggering region recreation
  }, [segments, selectedSegments, activeCueId, project, updateSegment, addSegment])

  // Track mouse position during dragging
  useEffect(() => {
    if (!draggingTimecodes) return

    const handleMouseMove = (e: MouseEvent) => {
      if (draggingTimecodes) {
        setDraggingTimecodes(prev => prev ? {
          ...prev,
          mouseX: e.clientX,
          mouseY: e.clientY
        } : null)
      }
    }

    window.addEventListener('mousemove', handleMouseMove)
    return () => window.removeEventListener('mousemove', handleMouseMove)
  }, [draggingTimecodes])

  const handlePlayPause = () => {
    if (wavesurferRef.current) {
      wavesurferRef.current.playPause()
    }
  }

  const handleZoomIn = () => {
    // Use ref for synchronous guard to prevent race conditions when clicking rapidly
    if (!wavesurferRef.current || isZoomingRef.current) return

    isZoomingRef.current = true  // Synchronous - prevents overlapping zoom operations
    setIsZooming(true)  // For UI overlay only

    const ws = wavesurferRef.current
    const duration = ws.getDuration()
    const viewportWidth = ws.getWidth()
    const currentScrollLeft = ws.getScroll()

    // Calculate effective zoom level
    const effectiveZoom = zoom === 0 ? viewportWidth / duration : zoom

    // Find the next zoom level
    let newZoom = ZOOM_LEVELS[ZOOM_LEVELS.length - 1]
    for (const level of ZOOM_LEVELS) {
      if (level > effectiveZoom) {
        newZoom = level
        break
      }
    }

    // Calculate center time for scroll preservation
    const currentWidth = zoom === 0 ? viewportWidth : duration * zoom
    const viewportCenterPx = currentScrollLeft + (viewportWidth / 2)
    const centerTime = (viewportCenterPx / currentWidth) * duration

    zoomRef.current = newZoom
    ws.zoom(newZoom)
    setZoom(newZoom)

    // Calculate scroll position
    const expectedWidth = newZoom === 0 ? viewportWidth : duration * newZoom
    const centerPx = (centerTime / duration) * expectedWidth
    let targetScrollLeft = centerPx - (viewportWidth / 2)
    targetScrollLeft = Math.max(0, Math.min(targetScrollLeft, expectedWidth - viewportWidth))
    pendingScrollRef.current = targetScrollLeft

    updateWidthAndTicks()

    requestAnimationFrame(() => {
      if (pendingScrollRef.current !== null) {
        ws.setScroll(pendingScrollRef.current)
        pendingScrollRef.current = null
      }
      isZoomingRef.current = false
      setIsZooming(false)
    })
  }

  const handleZoomOut = () => {
    // Use ref for synchronous guard to prevent race conditions when clicking rapidly
    if (!wavesurferRef.current || isZoomingRef.current) return

    const ws = wavesurferRef.current
    const duration = ws.getDuration()
    const viewportWidth = ws.getWidth()
    const currentScrollLeft = ws.getScroll()

    // Calculate effective zoom level
    const effectiveZoom = zoom === 0 ? viewportWidth / duration : zoom

    // Find the next zoom level that's lower
    let newZoom = 0
    for (let i = ZOOM_LEVELS.length - 1; i >= 0; i--) {
      if (ZOOM_LEVELS[i] < effectiveZoom) {
        newZoom = ZOOM_LEVELS[i]
        break
      }
    }

    // If already at fit/min zoom, don't do anything
    if (zoom === 0 || newZoom >= effectiveZoom) {
      return
    }

    isZoomingRef.current = true  // Synchronous - prevents overlapping zoom operations
    setIsZooming(true)  // For UI overlay only

    // Defer zoom(0) until after layout so WaveSurfer uses the correct width.
    if (newZoom === 0) {
      zoomRef.current = 0
      setZoom(0)

      requestAnimationFrame(() => {
        const currentWs = wavesurferRef.current
        if (!currentWs) return
        currentWs.zoom(0)
        updateWidthAndTicks()
        currentWs.setScroll(0)
        isZoomingRef.current = false
        setIsZooming(false)
      })
      return
    }

    // Calculate center time for scroll preservation
    const currentWidth = zoom === 0 ? viewportWidth : duration * zoom
    const viewportCenterPx = currentScrollLeft + (viewportWidth / 2)
    const centerTime = (viewportCenterPx / currentWidth) * duration

    zoomRef.current = newZoom
    ws.zoom(newZoom)
    setZoom(newZoom)

    const expectedWidth = newZoom === 0 ? viewportWidth : duration * newZoom
    const centerPx = (centerTime / duration) * expectedWidth
    let targetScrollLeft = centerPx - (viewportWidth / 2)
    targetScrollLeft = Math.max(0, Math.min(targetScrollLeft, expectedWidth - viewportWidth))
    pendingScrollRef.current = targetScrollLeft

    updateWidthAndTicks()

    requestAnimationFrame(() => {
      if (pendingScrollRef.current !== null) {
        ws.setScroll(pendingScrollRef.current)
        pendingScrollRef.current = null
      }
      isZoomingRef.current = false
      setIsZooming(false)
    })
  }

  const handleZoomReset = () => {
    // Use ref for synchronous guard to prevent race conditions when clicking rapidly
    if (isZoomingRef.current) return

    isZoomingRef.current = true  // Synchronous - prevents overlapping zoom operations
    setIsZooming(true)  // For UI overlay only
    zoomRef.current = 0
    setZoom(0)

    requestAnimationFrame(() => {
      const ws = wavesurferRef.current
      if (!ws) return
      ws.zoom(0)
      updateWidthAndTicks()
      ws.setScroll(0)
      isZoomingRef.current = false
      setIsZooming(false)
    })
  }

  if (!fileId) {
    return null
  }

  return (
    <Card className="w-full border-white/10 shadow-none waveform-card">
      <CardContent className="pt-6">
        <div className="space-y-4">
          <div
            className="overflow-x-auto rounded-lg bg-[rgba(30,20,15,0.6)] relative waveform-scroll-container"
          >
            {/* Zoom loading overlay */}
            {isZooming && (
              <div className="absolute inset-0 bg-background/50 backdrop-blur-sm z-20 flex items-center justify-center">
                <div className="flex items-center gap-2 text-foreground-muted text-sm">
                  <div className="w-4 h-4 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                  Zooming...
                </div>
              </div>
            )}
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
                // Always prefer waveformWidth from state (updated by updateWidthAndTicks) for consistency
                // Fallback to direct DOM measurement if state not yet updated
                let displayWidth = waveformWidth
                if (!displayWidth || displayWidth === 0) {
                  const wrapper = wavesurferRef.current?.getWrapper()
                  if (zoom === 0) {
                    // In fit mode, use container width
                    const container = waveformRef.current?.parentElement
                    displayWidth = container?.clientWidth || wrapper?.clientWidth || 0
                  } else {
                    // In zoom mode, use scroll width
                    displayWidth = wrapper?.scrollWidth || wrapper?.clientWidth || 0
                  }
                }

                return (
                  <div
                    ref={rulerRef}
                    className="relative h-8 bg-[rgba(40,28,22,0.8)] border-b border-white/10"
                    style={{
                      width: zoom === 0 ? '100%' : `${displayWidth}px`,
                      minWidth: zoom === 0 ? '100%' : `${displayWidth}px`,
                    }}
                  >
                    {ticks.map((tick, index) => (
                      // PERFORMANCE: tick.leftPx is pre-calculated in updateWidthAndTicks()
                      <div
                        key={index}
                        className="absolute"
                        style={{ left: `${tick.leftPx}px` }}
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
                          <div
                            className={`absolute top-3 text-xs text-foreground/60 whitespace-nowrap ${
                              index === 0 ? '' : '-translate-x-1/2'
                            }`}
                          >
                            {tick.timecode}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )
              })()}

              <div ref={waveformRef} className="w-full" />

              {/* Dragging timecode tooltips - bottom-right corner near mouse cursor */}
              {draggingTimecodes && draggingTimecodes.mouseX !== undefined && draggingTimecodes.mouseY !== undefined && (
                <div
                  className="fixed bg-primary text-primary-foreground rounded-lg shadow-xl px-3 py-2 pointer-events-none text-sm font-mono whitespace-nowrap"
                  style={{
                    left: `${draggingTimecodes.mouseX}px`,
                    top: `${draggingTimecodes.mouseY}px`,
                    transform: 'translate(calc(-100% - 12px), calc(-100% - 12px))',
                    zIndex: 9999
                  }}
                >
                  {draggingTimecodes.activeHandle === 'start' && draggingTimecodes.start}
                  {draggingTimecodes.activeHandle === 'end' && draggingTimecodes.end}
                  {draggingTimecodes.activeHandle === 'both' && `${draggingTimecodes.start} - ${draggingTimecodes.end}`}
                </div>
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

                {/* Audio level meters - animated when playing */}
                <div className="flex items-end gap-[2px] h-6 px-2">
                  {[0, 1, 2, 3, 4].map((i) => (
                    <div
                      key={i}
                      className="w-[3px] bg-primary rounded-sm transition-all duration-75"
                      style={{
                        height: isPlaying ? undefined : '4px',
                        animation: isPlaying ? `soundbar 0.4s ease-in-out infinite` : 'none',
                        animationDelay: isPlaying ? `${i * 0.08}s` : '0s',
                      }}
                    />
                  ))}
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
                  disabled={zoom === ZOOM_LEVELS[0] || isZooming}
                >
                  Zoom Out
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleZoomReset}
                  disabled={zoom === 0 || isZooming}
                >
                  Fit
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleZoomIn}
                  disabled={zoom >= ZOOM_LEVELS[ZOOM_LEVELS.length - 1] || isZooming}
                >
                  Zoom In
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
            <>
              {/* Backdrop to close on click */}
              <div
                className="fixed inset-0 z-40"
                onClick={() => setContextMenu(null)}
              />

              <div
                className="fixed glass border border-white/10 rounded-lg shadow-2xl py-1 z-50"
                style={{
                  left: `${contextMenu.x}px`,
                  top: `${contextMenu.y}px`,
                }}
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  className="w-full px-4 py-2.5 text-left text-sm text-foreground hover:bg-white/10 flex items-center gap-3 transition-colors"
                  onClick={() => handleSplitAtMouse(contextMenu.segmentIndex, contextMenu.splitTime)}
                >
                  <span className="text-primary">✂</span>
                  <span>Split cue</span>
                </button>
                <button
                  className="w-full px-4 py-2.5 text-left text-sm text-red-400 hover:bg-white/10 flex items-center gap-3 transition-colors"
                  onClick={() => handleDeleteCue(contextMenu.segmentIndex, contextMenu.x, contextMenu.y)}
                >
                  <span>🗑</span>
                  <span>Delete cue</span>
                </button>
              </div>
            </>
          )}

          {/* Delete Confirmation Tooltip */}
          {deleteConfirm && (
            <>
              {/* Backdrop to close on click */}
              <div
                className="fixed inset-0 z-40"
                onClick={() => setDeleteConfirm(null)}
              />

              {/* Tooltip */}
              <div
                className="fixed glass border border-white/10 rounded-lg shadow-2xl p-4 z-50"
                style={{
                  left: `${deleteConfirm.x}px`,
                  top: `${deleteConfirm.y}px`,
                  transform: 'translate(-50%, -100%)',
                  marginTop: '-8px'
                }}
                onClick={(e) => e.stopPropagation()}
              >
                <p className="text-sm text-foreground mb-3">Delete this cue?</p>
                <div className="flex gap-2">
                  <button
                    className="px-3 py-1.5 text-sm bg-white/10 hover:bg-white/20 text-foreground rounded transition-colors"
                    onClick={() => setDeleteConfirm(null)}
                  >
                    Cancel
                  </button>
                  <button
                    className="px-3 py-1.5 text-sm bg-red-500/80 hover:bg-red-500 text-white rounded transition-colors"
                    onClick={() => {
                      deleteSegment(deleteConfirm.index)
                      setDeleteConfirm(null)
                    }}
                  >
                    Delete
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
