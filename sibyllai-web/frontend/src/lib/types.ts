// API Request/Response Types

export interface UploadResponse {
  file_id: string
  filename: string
  size: number
}

export interface SegmentPreviewRequest {
  file_id: string
  music_thresh: number
  min_gap: number
  min_cue_length: number
  silence_thresh: number
  mix_type: 'clean_mx' | 'full_mix'
}

export interface SegmentPreviewResponse {
  segments: [number, number][]
  duration: number
  total_segments: number
}

export interface AnalyzeCuesRequest {
  file_id: string
  segments: [number, number][]
  fps?: number
  threshold?: number
}

export interface AnalysisResponse {
  session_id: string
  total_segments: number
  output_path: string
}

export interface AnalysisProgressUpdate {
  current: number
  total: number
  status: string
  progress_percent: number
}

export interface AnalysisStatusResponse {
  session_id: string
  complete: boolean
  current?: number
  total?: number
  progress_percent: number
  status: string
  project: SibylProject | null
  error: string | null
}

// .sibyl.json Format Types

export interface SibylProject {
  version: string
  project: ProjectMetadata
  cues: Cue[]
}

export interface ProjectMetadata {
  name: string
  mx_file: string
  created: string
  modified: string
  fps: number
}

export interface Cue {
  id: string
  name: string
  start: number
  end: number
  start_tc: string
  end_tc: string
  musical_profile: MusicalProfile
  project_context: ProjectContextCue
}

export interface ProjectContextCue {
  custom_tags: string[]
  notes: string
  color: string
  status: string
  replacement?: CueReplacement
}

export interface CueReplacement {
  track_path: string
  start: number
  end: number
  score: number
  window_size: number
}

export interface MusicalProfile {
  detected: DetectedAttributes
  curated: CuratedAttributes
  bpm: number | null
  key: string | null
  valence: number
  arousal: number
}

export interface DetectedAttributes {
  instruments_yamnet: Record<string, number>
  genres_yamnet: Record<string, number>  // YAMNet genre detection (pop, rock, jazz, etc.)
  moods: Record<string, number>
  clap_style: Record<string, number>  // CLAP film-scoring style (orchestral, hybrid, etc.)
  clap_genre?: Record<string, number>  // Legacy field for backwards compatibility
  clap_instrumentation: Record<string, number>
  clap_production: Record<string, number>
  clap_energy: Record<string, number>
  clap_era: Record<string, number>
  clap_function: Record<string, number>
}

export interface CuratedAttributes {
  instruments: string[]
  genres: string[]  // YAMNet genres (pop, rock, electronic, etc.)
  moods: string[]
  style: string[]  // CLAP film-scoring style (orchestral, hybrid, cinematic)
  genre?: string[]  // Legacy field for backwards compatibility
  instrumentation: string[]
  production: string[]
  energy: string[]
  era: string[]
  function: string[]
}

// Request type for updating cue curated attributes
export interface CueUpdateRequest {
  instruments?: string[]
  genres?: string[]
  style?: string[]
  moods?: string[]
}

// Response from cue update
export interface CueUpdateResponse {
  status: string
  cue_id: string
}

// Library index / matching types
export interface LibraryBuildRequest {
  folder: string
  window_sizes?: number[]
  hop_ratio?: number
  include_moods?: boolean
  reset?: boolean
  db_path?: string
}

export interface LibraryBuildResponse {
  job_id: string
  status: string
}

export interface LibraryBuildResult {
  tracks_indexed: number
  windows_indexed: number
  db_path: string
  vector_dim: number
}

export interface LibraryBuildStatus {
  status: 'running' | 'complete' | 'error'
  progress_percent?: number
  current?: number
  total?: number
  windows_indexed?: number
  message?: string
  result: LibraryBuildResult | null
  error: string | null
}

export interface LibraryMatchRequest {
  session_id: string
  cue_id: string
  top_n?: number
  db_path?: string
  unique_tracks?: boolean
}

export interface LibraryMatch {
  window_id: string
  track_id: string
  track_path: string
  start: number
  end: number
  duration: number
  window_size: number
  score: number
  reasons?: string[]
  // Match metadata for display
  bpm?: number | null
  key?: string | null
  instruments?: string[]
  genres?: string[]
}

export interface LibraryMatchResponse {
  cue_id: string
  cue_length: number
  matches: LibraryMatch[]
}

export interface CueReplacementRequest {
  track_path: string
  start: number
  end: number
  score: number
  window_size: number
  status?: string
}

export interface CueReplacementResponse {
  status: string
  cue_id: string
}

export interface LibraryInfo {
  exists: boolean
  db_path: string
  tracks: number
  windows: number
  meta: Record<string, unknown>
}
