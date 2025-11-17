/**
 * Timecode utility functions for HH:MM:SS:FF format
 */

/**
 * Convert timecode string to total seconds
 */
export function timecodeToSeconds(timecode: string, framerate: number): number {
  const parts = timecode.split(':')
  if (parts.length !== 4) return 0

  const hours = parseInt(parts[0], 10)
  const minutes = parseInt(parts[1], 10)
  const seconds = parseInt(parts[2], 10)
  const frames = parseInt(parts[3], 10)

  return hours * 3600 + minutes * 60 + seconds + frames / framerate
}

/**
 * Convert seconds to timecode string (HH:MM:SS:FF)
 */
export function secondsToTimecode(seconds: number, framerate: number, startTimecode: string): string {
  // Add start timecode offset
  const startSeconds = timecodeToSeconds(startTimecode, framerate)
  const totalSeconds = startSeconds + seconds

  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const secs = Math.floor(totalSeconds % 60)
  const frames = Math.floor((totalSeconds % 1) * framerate)

  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}:${String(frames).padStart(2, '0')}`
}

/**
 * Calculate appropriate tick intervals based on zoom level and duration
 */
export function calculateIntervals(zoom: number, duration: number): { major: number } {
  if (zoom === 0) {
    // Fit mode - base intervals on duration
    if (duration > 1800) {
      // > 30 minutes: major every 60s
      return { major: 60 }
    } else if (duration > 300) {
      // > 5 minutes: major every 30s
      return { major: 30 }
    } else if (duration > 60) {
      // > 1 minute: major every 10s
      return { major: 10 }
    } else {
      // < 1 minute: major every 5s
      return { major: 5 }
    }
  } else if (zoom < 10) {
    // Very zoomed out (< 10 px/sec): major every 30s for readability
    return { major: 30 }
  } else if (zoom < 50) {
    // Zoomed out: major every 10s
    return { major: 10 }
  } else if (zoom < 150) {
    // Medium zoom: major every 5s
    return { major: 5 }
  } else if (zoom < 400) {
    // Zoomed in: major every 1s
    return { major: 1 }
  } else {
    // Very zoomed in: major every 0.5s
    return { major: 0.5 }
  }
}

/**
 * Generate tick positions for the ruler
 * All ticks show absolute timecode from file start (with startTimecode offset)
 */
export function generateTicks(
  duration: number,
  zoom: number,
  framerate: number,
  startTimecode: string
): Array<{ position: number; timecode: string; isMajor: boolean }> {
  const intervals = calculateIntervals(zoom, duration)
  const ticks: Array<{ position: number; timecode: string; isMajor: boolean }> = []

  // Generate ticks for the ENTIRE file duration, starting from time 0
  // Each tick shows its absolute time from the file start
  for (let absoluteTime = 0; absoluteTime <= duration; absoluteTime += intervals.major) {
    ticks.push({
      position: absoluteTime, // Absolute position in seconds from file start
      timecode: secondsToTimecode(absoluteTime, framerate, startTimecode), // Full HH:MM:SS:FF format with offset
      isMajor: true,
    })
  }

  // Always add the end tick if not already there
  const lastTick = ticks[ticks.length - 1]
  if (!lastTick || Math.abs(lastTick.position - duration) > 0.001) {
    ticks.push({
      position: duration,
      timecode: secondsToTimecode(duration, framerate, startTimecode),
      isMajor: true,
    })
  }

  return ticks
}
