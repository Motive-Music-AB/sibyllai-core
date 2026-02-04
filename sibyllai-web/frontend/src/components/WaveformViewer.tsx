import { useEffect, useRef, useState, useCallback } from 'react'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/lib/store'
import { generateTicks, secondsToTimecode } from '@/lib/timecode'

// Define zoom levels for smoother progression
const ZOOM_LEVELS = [0, 5, 10, 20, 30, 50, 75, 100, 150]

export function WaveformViewer() {
  const waveformRef = useRef<HTMLDivElement>(null)
  const wavesurferRef = useRef<WaveSurfer | null>(null)
  const regionsRef = useRef<RegionsPlugin | null>(null)
  const rulerRef = useRef<HTMLDivElement>(null)
  const zoomRef = useRef(0) // Synchronous zoom tracking for redraw event handler
  const pendingScrollRef = useRef<number | null>(null) // Track pending scroll position after zoom

  const [isPlaying, setIsPlaying] = useState(false)
  const [isReady, setIsReady] = useState(false)
  const [zoom, setZoom] = useState(0)
  const [currentTimecode, setCurrentTimecode] = useState<string>('')
  const [ticks, setTicks] = useState<Array<{ position: number; timecode: string; isMajor: boolean }>>([])
  const [waveformWidth, setWaveformWidth] = useState(0)
  const [draggingTimecodes, setDraggingTimecodes] = useState<{ start: string; end: string; startPos: number; endPos: number; activeHandle: 'start' | 'end' | 'both'; mouseX?: number; mouseY?: number } | null>(null)
  const dragStartRef = useRef<{ start: number; end: number } | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState<{ index: number; x: number; y: number } | null>(null)
  const regionCreatedHandlerRef = useRef<((region: any) => void) | null>(null)
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
    playingCueId: _playingCueId,
    setPlayingCueId,
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
      } else {
        // Clear active cue if clicking on unanalyzed segment
        setActiveCueId(null)
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

    // Add new regions with appropriate styling
    // Use setTimeout with index to add regions asynchronously, preventing UI freeze
    segmentsToRender.forEach(([start, end], index) => {
      setTimeout(() => {
        if (!regionsRef.current) return

        const actualIndex = segments.findIndex(([s, e]) => s === start && e === end)
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
            toggleSegmentSelection(start, end)
          })

          // Append button to region element
          region.element.appendChild(button)

          // Click handler
          region.on('click', (e) => {
            // Check if it's a right-click
            if (e.button === 2) {
              e.preventDefault()
              e.stopPropagation()

              // Only show context menu before analysis
              if (!project) {
                // Calculate split time from click position
                const ws = wavesurferRef.current
                const rect = waveformRef.current?.getBoundingClientRect()
                let splitTime = region.start // fallback to region start
                if (ws && rect) {
                  const clickX = e.clientX - rect.left
                  const duration = ws.getDuration()
                  splitTime = (clickX / rect.width) * duration
                }
                setContextMenu({
                  x: e.clientX,
                  y: e.clientY,
                  segmentIndex: actualIndex,
                  splitTime,
                })
              }
          } else {
            // Left click - normal behavior
            handleSegmentClick(start, end)
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
            updateSegment(actualIndex, newStart, newEnd)
            // Clear dragging timecodes and reset state
            setDraggingTimecodes(null)
            dragStartRef.current = null
            dragStart = null
            isFirstUpdate = true
          })
        }
        }
      }, index * 10) // Stagger region creation by 10ms each to prevent UI freeze
    })

    // After all programmatic regions are added, enable drag-to-create for user
    // Only enable if before analysis (project is null) and detection has run
    if (!project && segments.length > 0) {
      setTimeout(() => {
        if (!regionsRef.current) return

        // Remove old region-created listener if it exists
        if (regionCreatedHandlerRef.current) {
          regionsRef.current.un('region-created', regionCreatedHandlerRef.current)
        }

        // Enable drag selection for creating new regions
        regionsRef.current.enableDragSelection({
          color: 'rgba(148, 163, 184, 0.2)',
        })

        // Listen for user-created regions (drag-to-create)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const handleRegionCreated = (region: any) => {
          // Only add if it's a new region (not already in segments)
          const exists = segments.some(([s, e]) =>
            Math.abs(s - region.start) < 0.1 && Math.abs(e - region.end) < 0.1
          )
          if (!exists) {
            setTimeout(() => {
              addSegment(region.start, region.end)
            }, 0)
          }
        }

        // Store reference and add listener
        regionCreatedHandlerRef.current = handleRegionCreated
        regionsRef.current.on('region-created', handleRegionCreated)
      }, (segmentsToRender.length + 1) * 10) // Wait for all regions to be added
    }
  }, [segments, selectedSegments, activeCueId, project, framerate, startTimecode, updateSegment, zoom, addSegment, isSegmentSelected, findCueForSegment, handleSegmentClick])

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
    if (!wavesurferRef.current) return

    const ws = wavesurferRef.current
    const wrapper = ws.getWrapper()
    const duration = ws.getDuration()
    const viewportWidth = wrapper.clientWidth
    const currentScrollLeft = wrapper.scrollLeft

    // Calculate the center point of the current viewport in time
    // This works whether audio is playing or not
    const currentWidth = zoom === 0 ? viewportWidth : duration * zoom
    const viewportCenterPx = currentScrollLeft + (viewportWidth / 2)
    const centerTime = (viewportCenterPx / currentWidth) * duration

    console.log('DEBUG ZOOM IN - centerTime:', centerTime, 'duration:', duration, 'currentScrollLeft:', currentScrollLeft)

    // Find current zoom level index - try exact match first, then find lowest level >= zoom
    let currentIndex = ZOOM_LEVELS.findIndex(level => level === zoom)
    if (currentIndex === -1) {
      // No exact match - find the lowest level that's >= current zoom
      currentIndex = ZOOM_LEVELS.findIndex(level => level >= zoom)
      // If still not found, default to last index
      if (currentIndex === -1) currentIndex = ZOOM_LEVELS.length - 1
    }

    // Move to next level (higher zoom)
    const nextIndex = currentIndex < ZOOM_LEVELS.length - 1 ? currentIndex + 1 : ZOOM_LEVELS.length - 1
    const newZoom = ZOOM_LEVELS[nextIndex]

    console.log('Zoom In - current:', zoom, 'currentIndex:', currentIndex, 'nextIndex:', nextIndex, 'newZoom:', newZoom)

    // Update zoom ref synchronously BEFORE calling ws.zoom() (which fires redraw event)
    zoomRef.current = newZoom
    console.log('DEBUG ZOOM IN - BEFORE ws.zoom(), wrapper dimensions:', {
      scrollWidth: wrapper.scrollWidth,
      clientWidth: wrapper.clientWidth,
      scrollLeft: wrapper.scrollLeft
    })
    ws.zoom(newZoom)
    console.log('DEBUG ZOOM IN - AFTER ws.zoom(), wrapper dimensions:', {
      scrollWidth: wrapper.scrollWidth,
      clientWidth: wrapper.clientWidth,
      scrollLeft: wrapper.scrollLeft
    })
    setZoom(newZoom)

    // Calculate scroll position immediately (before ws.zoom triggers redraw)
    // Calculate expected width from zoom level
    const expectedWidth = newZoom === 0 ? wrapper.clientWidth : duration * newZoom

    // Calculate where the center point is in pixels at the new zoom level
    const centerPx = (centerTime / duration) * expectedWidth

    // Calculate target scroll position (keep center centered)
    let targetScrollLeft = centerPx - (viewportWidth / 2)

    // Smart positioning: avoid showing too much empty space at edges
    const maxScrollLeft = expectedWidth - viewportWidth

    console.log('DEBUG ZOOM IN - scroll calc:', {
      expectedWidth,
      centerPx,
      viewportWidth,
      targetScrollLeft,
      maxScrollLeft
    })

    // If playhead near start, don't scroll past beginning
    if (targetScrollLeft < 0) {
      targetScrollLeft = 0
    }
    // If playhead near end, don't show empty space beyond waveform
    else if (targetScrollLeft > maxScrollLeft) {
      targetScrollLeft = Math.max(0, maxScrollLeft)
    }

    console.log('DEBUG ZOOM IN - storing pending scroll:', targetScrollLeft)
    // Store the target scroll position to be applied after redraw
    pendingScrollRef.current = targetScrollLeft

    // Update width and ticks immediately (calculation-based, no DOM read needed)
    updateWidthAndTicks()

    // Apply scroll after multiple animation frames to ensure DOM is fully updated
    // First frame: zoom is applied
    requestAnimationFrame(() => {
      // Second frame: redraw has happened
      requestAnimationFrame(() => {
        // Third frame: all layout calculations complete
        requestAnimationFrame(() => {
          if (pendingScrollRef.current !== null) {
            const newWrapper = ws.getWrapper()
            console.log('DEBUG ZOOM IN - applying scroll after RAF:', pendingScrollRef.current)
            console.log('DEBUG ZOOM IN - wrapper scrollWidth:', newWrapper.scrollWidth, 'clientWidth:', newWrapper.clientWidth)
            newWrapper.scrollLeft = pendingScrollRef.current
            console.log('DEBUG ZOOM IN - after setting, scrollLeft is now:', newWrapper.scrollLeft)
            pendingScrollRef.current = null
          }
        })
      })
    })
  }

  const handleZoomOut = () => {
    if (!wavesurferRef.current) return

    const ws = wavesurferRef.current
    const wrapper = ws.getWrapper()
    const duration = ws.getDuration()
    const viewportWidth = wrapper.clientWidth
    const currentScrollLeft = wrapper.scrollLeft

    // Calculate the center point of the current viewport in time
    // This works whether audio is playing or not
    const currentWidth = zoom === 0 ? viewportWidth : duration * zoom
    const viewportCenterPx = currentScrollLeft + (viewportWidth / 2)
    const centerTime = (viewportCenterPx / currentWidth) * duration

    console.log('DEBUG ZOOM OUT - centerTime:', centerTime, 'duration:', duration, 'currentScrollLeft:', currentScrollLeft)

    // Find current zoom level index - try exact match first, then find highest level <= zoom
    let currentIndex = ZOOM_LEVELS.findIndex(level => level === zoom)
    if (currentIndex === -1) {
      // No exact match - find the highest level that's <= current zoom
      for (let i = ZOOM_LEVELS.length - 1; i >= 0; i--) {
        if (ZOOM_LEVELS[i] <= zoom) {
          currentIndex = i
          break
        }
      }
      // If still not found, default to 0
      if (currentIndex === -1) currentIndex = 0
    }

    // Move to previous level (lower zoom)
    const prevIndex = currentIndex > 0 ? currentIndex - 1 : 0
    const newZoom = ZOOM_LEVELS[prevIndex]

    console.log('Zoom Out - current:', zoom, 'currentIndex:', currentIndex, 'prevIndex:', prevIndex, 'newZoom:', newZoom)

    // If already at fit/min zoom, don't do anything
    if (newZoom === zoom || currentIndex === 0) {
      console.log('Already at minimum zoom, skipping')
      return
    }

    // Update zoom ref synchronously BEFORE calling ws.zoom() (which fires redraw event)
    zoomRef.current = newZoom
    ws.zoom(newZoom)
    setZoom(newZoom)

    // Calculate scroll position immediately (before ws.zoom triggers redraw)
    // Calculate expected width from zoom level
    const expectedWidth = newZoom === 0 ? wrapper.clientWidth : duration * newZoom

    // Calculate where the center point is in pixels at the new zoom level
    const centerPx = (centerTime / duration) * expectedWidth

    // Calculate target scroll position (keep center centered)
    let targetScrollLeft = centerPx - (viewportWidth / 2)

    // Smart positioning: avoid showing too much empty space at edges
    const maxScrollLeft = expectedWidth - viewportWidth

    console.log('DEBUG ZOOM OUT - scroll calc:', {
      expectedWidth,
      centerPx,
      viewportWidth,
      targetScrollLeft,
      maxScrollLeft
    })

    // If center near start, don't scroll past beginning
    if (targetScrollLeft < 0) {
      targetScrollLeft = 0
    }
    // If center near end, don't show empty space beyond waveform
    else if (targetScrollLeft > maxScrollLeft) {
      targetScrollLeft = Math.max(0, maxScrollLeft)
    }

    console.log('DEBUG ZOOM OUT - storing pending scroll:', targetScrollLeft)
    // Store the target scroll position to be applied after redraw
    pendingScrollRef.current = targetScrollLeft

    // Update width and ticks immediately (calculation-based, no DOM read needed)
    updateWidthAndTicks()

    // Apply scroll after multiple animation frames to ensure DOM is fully updated
    // First frame: zoom is applied
    requestAnimationFrame(() => {
      // Second frame: redraw has happened
      requestAnimationFrame(() => {
        // Third frame: all layout calculations complete
        requestAnimationFrame(() => {
          if (pendingScrollRef.current !== null) {
            const newWrapper = ws.getWrapper()
            console.log('DEBUG ZOOM OUT - applying scroll after RAF:', pendingScrollRef.current)
            console.log('DEBUG ZOOM OUT - wrapper scrollWidth:', newWrapper.scrollWidth, 'clientWidth:', newWrapper.clientWidth)
            newWrapper.scrollLeft = pendingScrollRef.current
            console.log('DEBUG ZOOM OUT - after setting, scrollLeft is now:', newWrapper.scrollLeft)
            pendingScrollRef.current = null
          }
        })
      })
    })
  }

  const handleZoomReset = () => {
    console.log('Zoom Reset - current:', zoom, 'resetting to 0')

    // Update zoom ref synchronously BEFORE calling ws.zoom() (which fires redraw event)
    zoomRef.current = 0
    setZoom(0)
    if (wavesurferRef.current) {
      wavesurferRef.current.zoom(0)
    }

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

              {/* Dragging timecode tooltips - bottom-right corner near mouse cursor */}
              {draggingTimecodes && draggingTimecodes.mouseX !== undefined && draggingTimecodes.mouseY !== undefined && (
                <div
                  className="fixed bg-primary border border-primary/50 rounded-lg shadow-xl px-3 py-2 pointer-events-none text-sm font-mono whitespace-nowrap text-primary-foreground"
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
            <>
              {/* Backdrop to close on click */}
              <div
                className="fixed inset-0 z-40"
                onClick={() => setContextMenu(null)}
              />

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
                  onClick={() => handleSplitAtMouse(contextMenu.segmentIndex, contextMenu.splitTime)}
                >
                  <span>✂️</span>
                  <span>Split cue</span>
                </button>
                <button
                  className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 flex items-center gap-2 text-red-600"
                  onClick={() => handleDeleteCue(contextMenu.segmentIndex, contextMenu.x, contextMenu.y)}
                >
                  <span>🗑️</span>
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
                className="fixed bg-white border border-gray-300 rounded-lg shadow-xl p-4 z-50"
                style={{
                  left: `${deleteConfirm.x}px`,
                  top: `${deleteConfirm.y}px`,
                  transform: 'translate(-50%, -100%)',
                  marginTop: '-8px'
                }}
                onClick={(e) => e.stopPropagation()}
              >
                <p className="text-sm mb-3">Delete this cue?</p>
                <div className="flex gap-2">
                  <button
                    className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 rounded"
                    onClick={() => setDeleteConfirm(null)}
                  >
                    Cancel
                  </button>
                  <button
                    className="px-3 py-1.5 text-sm bg-red-600 hover:bg-red-700 text-white rounded"
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
