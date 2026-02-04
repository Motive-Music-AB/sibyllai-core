import { create } from 'zustand'
import type { SibylProject, CuratedAttributes } from './types'

type Page = 'analysis' | 'cuesynch'

interface AppState {
  // Navigation state
  currentPage: Page

  // Upload state
  uploadedFile: File | null
  fileId: string | null
  fileName: string | null
  startTimecode: string
  framerate: number

  // Segmentation state (Phase 1)
  segments: [number, number][]
  selectedSegments: [number, number][]  // User-selected segments for analysis
  duration: number
  musicThreshold: number
  minGap: number
  minCueLength: number
  isSegmenting: boolean

  // Analysis state (Phase 2)
  isAnalyzing: boolean
  analysisProgress: number
  analysisStatus: string
  sessionId: string | null
  project: SibylProject | null

  // UI state
  activeCueId: string | null
  playingCueId: string | null
  showControls: boolean

  // Actions
  setCurrentPage: (page: Page) => void
  setUploadedFile: (file: File, fileId: string, fileName: string, startTimecode: string, framerate: number) => void
  setStartTimecode: (tc: string) => void
  setFramerate: (fps: number) => void
  setSegments: (segments: [number, number][], duration: number) => void
  setSelectedSegments: (segments: [number, number][]) => void
  updateSegment: (index: number, newStart: number, newEnd: number) => void
  splitSegment: (index: number, splitTime: number) => void
  addSegment: (start: number, end: number) => void
  deleteSegment: (index: number) => void
  setMusicThreshold: (threshold: number) => void
  setMinGap: (gap: number) => void
  setMinCueLength: (length: number) => void
  setIsSegmenting: (isSegmenting: boolean) => void
  setIsAnalyzing: (isAnalyzing: boolean) => void
  setAnalysisProgress: (progress: number, status: string) => void
  setProject: (sessionId: string, project: SibylProject) => void
  setActiveCueId: (cueId: string | null) => void
  setPlayingCueId: (cueId: string | null) => void
  setShowControls: (show: boolean) => void
  updateCueInProject: (cueId: string, curated: Partial<CuratedAttributes>) => void
  reset: () => void
  resetToSegmentation: () => void
}

const initialState = {
  currentPage: 'analysis' as Page,
  uploadedFile: null,
  fileId: null,
  fileName: null,
  startTimecode: '01:00:00:00',
  framerate: 24,
  segments: [],
  selectedSegments: [],
  duration: 0,
  musicThreshold: 0.01,
  minGap: 3.0,
  minCueLength: 3.0,
  isSegmenting: false,
  isAnalyzing: false,
  analysisProgress: 0,
  analysisStatus: '',
  sessionId: null,
  project: null,
  activeCueId: null,
  playingCueId: null,
  showControls: true,
}

export const useAppStore = create<AppState>((set) => ({
  ...initialState,

  setCurrentPage: (page) =>
    set({ currentPage: page }),

  setUploadedFile: (file, fileId, fileName, startTimecode, framerate) =>
    set({ uploadedFile: file, fileId, fileName, startTimecode, framerate }),

  setStartTimecode: (tc) =>
    set({ startTimecode: tc }),

  setFramerate: (fps) =>
    set({ framerate: fps }),

  setSegments: (segments, duration) =>
    set({ segments, duration, selectedSegments: [...segments] }),

  setSelectedSegments: (selectedSegments) =>
    set({ selectedSegments }),

  updateSegment: (index, newStart, newEnd) =>
    set((state) => {
      const newSegments = [...state.segments]
      const [oldStart, oldEnd] = newSegments[index]
      newSegments[index] = [newStart, newEnd]

      // Update selectedSegments if this segment was selected
      const newSelectedSegments: [number, number][] = state.selectedSegments.map(([s, e]) => {
        // If this selected segment matches the old coordinates, update to new coordinates
        if (Math.abs(s - oldStart) < 0.01 && Math.abs(e - oldEnd) < 0.01) {
          return [newStart, newEnd] as [number, number]
        }
        return [s, e] as [number, number]
      })

      return { segments: newSegments, selectedSegments: newSelectedSegments }
    }),

  splitSegment: (index, splitTime) =>
    set((state) => {
      const [start, end] = state.segments[index]

      // Validate split time is within segment bounds
      if (splitTime <= start || splitTime >= end) {
        return state
      }

      const newSegments = [...state.segments]
      // Replace original segment with two new segments
      newSegments.splice(index, 1, [start, splitTime], [splitTime, end])

      // Update selectedSegments if this segment was selected
      const wasSelected = state.selectedSegments.some(([s, e]) =>
        Math.abs(s - start) < 0.01 && Math.abs(e - end) < 0.01
      )

      let newSelectedSegments = state.selectedSegments.filter(([s, e]) =>
        !(Math.abs(s - start) < 0.01 && Math.abs(e - end) < 0.01)
      )

      // If original was selected, select both new segments
      if (wasSelected) {
        newSelectedSegments = [
          ...newSelectedSegments,
          [start, splitTime] as [number, number],
          [splitTime, end] as [number, number],
        ]
      }

      return { segments: newSegments, selectedSegments: newSelectedSegments }
    }),

  addSegment: (start, end) =>
    set((state) => {
      // Validate segment
      if (start >= end) {
        return state
      }

      const newSegment: [number, number] = [start, end]

      // Insert segment in chronological order
      const newSegments = [...state.segments, newSegment].sort((a, b) => a[0] - b[0])

      return { segments: newSegments }
    }),

  deleteSegment: (index) =>
    set((state) => {
      const [start, end] = state.segments[index]
      const newSegments = state.segments.filter((_, i) => i !== index)

      // Remove from selectedSegments if it was selected
      const newSelectedSegments = state.selectedSegments.filter(([s, e]) =>
        !(Math.abs(s - start) < 0.01 && Math.abs(e - end) < 0.01)
      )

      return { segments: newSegments, selectedSegments: newSelectedSegments }
    }),

  setMusicThreshold: (threshold) =>
    set({ musicThreshold: threshold }),

  setMinGap: (gap) =>
    set({ minGap: gap }),

  setMinCueLength: (length) =>
    set({ minCueLength: length }),

  setIsSegmenting: (isSegmenting) =>
    set({ isSegmenting }),

  setIsAnalyzing: (isAnalyzing) =>
    set({ isAnalyzing }),

  setAnalysisProgress: (progress, status) =>
    set({ analysisProgress: progress, analysisStatus: status }),

  setProject: (sessionId, project) =>
    set({ sessionId, project }),

  setActiveCueId: (cueId) =>
    set({ activeCueId: cueId }),

  setPlayingCueId: (cueId) =>
    set({ playingCueId: cueId }),

  setShowControls: (show) =>
    set({ showControls: show }),

  updateCueInProject: (cueId, curated) =>
    set((state) => {
      if (!state.project) return state
      const updatedCues = state.project.cues.map((cue) =>
        cue.id === cueId
          ? {
              ...cue,
              musical_profile: {
                ...cue.musical_profile,
                curated: { ...cue.musical_profile.curated, ...curated },
              },
            }
          : cue
      )
      return { project: { ...state.project, cues: updatedCues } }
    }),

  reset: () =>
    set(initialState),

  resetToSegmentation: () =>
    set({
      project: null,
      sessionId: null,
      isAnalyzing: false,
      analysisProgress: 0,
      analysisStatus: '',
      activeCueId: null,
      playingCueId: null,
      showControls: true,
    }),
}))
