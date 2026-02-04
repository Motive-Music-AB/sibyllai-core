/**
 * CSV Export Utilities
 * Export analysis results to CSV for use with CueSynch
 */

import type { SibylProject, Cue } from './types'

/**
 * Generate CSV content from project cues
 */
export function generateCueCSV(project: SibylProject): string {
  const headers = [
    'Timecode',
    'Cue',
    'Start',
    'End',
    'Duration',
    'BPM',
    'Key',
    'Instruments',
    'Genres',
    'Style',
  ]

  const rows = project.cues.map((cue, index) => {
    const duration = cue.end - cue.start
    const instruments = cue.musical_profile.curated.instruments.join(', ') || 'Unknown'
    const genres = cue.musical_profile.curated.genres.join(', ') || 'Unknown'
    const style = cue.musical_profile.curated.style.join(', ') || 'Unknown'

    return [
      cue.start_tc,
      `Cue ${index + 1}`,
      cue.start.toFixed(2),
      cue.end.toFixed(2),
      duration.toFixed(2),
      cue.musical_profile.bpm?.toFixed(1) || 'Unknown',
      cue.musical_profile.key || 'Unknown',
      `"${instruments}"`,
      `"${genres}"`,
      `"${style}"`,
    ].join(',')
  })

  return [headers.join(','), ...rows].join('\n')
}

/**
 * Download CSV content as a file
 */
export function downloadCSV(content: string, filename: string): void {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

/**
 * Export project cues to CSV file
 */
export function exportProjectToCSV(project: SibylProject, filename?: string): void {
  const csv = generateCueCSV(project)
  const name = filename || `${project.project.name}_cues.csv`
  downloadCSV(csv, name)
}
