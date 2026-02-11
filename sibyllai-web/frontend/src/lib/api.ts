import axios from 'axios'
import type {
  UploadResponse,
  SegmentPreviewRequest,
  SegmentPreviewResponse,
  AnalyzeCuesRequest,
  AnalysisResponse,
  AnalysisStatusResponse,
  SibylProject,
  CueUpdateRequest,
  CueUpdateResponse,
  LibraryBuildRequest,
  LibraryBuildResponse,
  LibraryBuildStatus,
  LibraryMatchRequest,
  LibraryMatchResponse,
  LibraryInfo,
  LibrarySource,
  LibraryTrack,
  CueReplacementRequest,
  CueReplacementResponse,
} from './types'

const API_BASE = '/api'

export const api = {
  /**
   * Upload an audio/video file for analysis
   */
  async uploadFile(file: File): Promise<UploadResponse> {
    const formData = new FormData()
    formData.append('file', file)

    const response = await axios.post<UploadResponse>(`${API_BASE}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    return response.data
  },

  /**
   * Phase 1: Fast segmentation preview using YAMNet
   */
  async getSegmentPreview(request: SegmentPreviewRequest): Promise<SegmentPreviewResponse> {
    const response = await axios.post<SegmentPreviewResponse>(
      `${API_BASE}/segment-preview`,
      request
    )
    return response.data
  },

  /**
   * Phase 2: Full analysis on user-confirmed segments
   * Returns immediately with session_id; analysis runs in background
   */
  async analyzeCues(request: AnalyzeCuesRequest): Promise<AnalysisResponse> {
    const response = await axios.post<AnalysisResponse>(`${API_BASE}/analyze-cues`, request)
    return response.data
  },

  /**
   * Get analysis progress status (poll this for updates)
   */
  async getAnalysisStatus(sessionId: string): Promise<AnalysisStatusResponse> {
    const response = await axios.get<AnalysisStatusResponse>(`${API_BASE}/analysis-status/${sessionId}`)
    return response.data
  },

  /**
   * Load a previously analyzed project
   */
  async getProject(sessionId: string): Promise<SibylProject> {
    const response = await axios.get<SibylProject>(`${API_BASE}/projects/${sessionId}`)
    return response.data
  },

  /**
   * Clean up uploaded files
   */
  async cleanup(fileId: string): Promise<void> {
    await axios.delete(`${API_BASE}/cleanup/${fileId}`)
  },

  /**
   * Update curated attributes for a cue
   */
  async updateCue(sessionId: string, cueId: string, update: CueUpdateRequest): Promise<CueUpdateResponse> {
    const response = await axios.put<CueUpdateResponse>(
      `${API_BASE}/projects/${sessionId}/cues/${cueId}`,
      update
    )
    return response.data
  },

  /**
   * Build library index from a folder of audio files
   */
  async buildLibraryIndex(request: LibraryBuildRequest): Promise<LibraryBuildResponse> {
    const response = await axios.post<LibraryBuildResponse>(`${API_BASE}/library/build`, request)
    return response.data
  },

  /**
   * Build library index by uploading a folder from the client
   */
  async buildLibraryIndexUpload(
    files: File[],
    includeMoods: boolean,
    reset: boolean
  ): Promise<LibraryBuildResponse> {
    const formData = new FormData()
    for (const file of files) {
      // Preserve relative path when available (webkitRelativePath)
      const relPath = (file as File & { webkitRelativePath?: string }).webkitRelativePath
      formData.append('files', file, relPath || file.name)
    }
    formData.append('include_moods', includeMoods ? 'true' : 'false')
    formData.append('reset', reset ? 'true' : 'false')
    formData.append('dedupe_simple', 'true')

    const response = await axios.post<LibraryBuildResponse>(`${API_BASE}/library/build-upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  /**
   * Get library build status
   */
  async getLibraryBuildStatus(jobId: string): Promise<LibraryBuildStatus> {
    const response = await axios.get<LibraryBuildStatus>(`${API_BASE}/library/status/${jobId}`)
    return response.data
  },

  /**
   * Match a cue to the library index
   */
  async matchLibrary(request: LibraryMatchRequest): Promise<LibraryMatchResponse> {
    const response = await axios.post<LibraryMatchResponse>(`${API_BASE}/library/match`, request)
    return response.data
  },

  /**
   * Persist replacement selection for a cue
   */
  async updateCueReplacement(
    sessionId: string,
    cueId: string,
    update: CueReplacementRequest
  ): Promise<CueReplacementResponse> {
    const response = await axios.put<CueReplacementResponse>(
      `${API_BASE}/projects/${sessionId}/cues/${cueId}/replacement`,
      update
    )
    return response.data
  },

  /**
   * Get the audio URL for a library track (for use with <audio> element)
   */
  getLibraryAudioUrl(trackId: string): string {
    return `${API_BASE}/library/audio/${encodeURIComponent(trackId)}`
  },

  /**
   * Get per-source stats for the library
   */
  async getLibrarySources(dbPath?: string): Promise<LibrarySource[]> {
    const response = await axios.get<LibrarySource[]>(`${API_BASE}/library/sources`, {
      params: dbPath ? { db_path: dbPath } : undefined,
    })
    return response.data
  },

  /**
   * Get tracks for a library source
   */
  async getLibraryTracks(sourceName: string, dbPath?: string): Promise<LibraryTrack[]> {
    const response = await axios.get<LibraryTrack[]>(
      `${API_BASE}/library/sources/${encodeURIComponent(sourceName)}/tracks`,
      { params: dbPath ? { db_path: dbPath } : undefined }
    )
    return response.data
  },

  /**
   * Delete all tracks and windows for a library source
   */
  async deleteLibrarySource(sourceName: string, dbPath?: string): Promise<{ tracks_deleted: number; windows_deleted: number }> {
    const response = await axios.delete<{ tracks_deleted: number; windows_deleted: number }>(
      `${API_BASE}/library/sources/${encodeURIComponent(sourceName)}`,
      { params: dbPath ? { db_path: dbPath } : undefined }
    )
    return response.data
  },

  /**
   * Get library index info (if it exists)
   */
  async getLibraryInfo(dbPath?: string): Promise<LibraryInfo> {
    const response = await axios.get<LibraryInfo>(`${API_BASE}/library/info`, {
      params: dbPath ? { db_path: dbPath } : undefined,
    })
    return response.data
  },
}

/**
 * Create WebSocket connection for progress updates
 */
export function createProgressWebSocket(
  sessionId: string,
  onProgress: (update: { current: number; total: number; status: string; progress_percent: number }) => void
): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/progress/${sessionId}`)

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data)
    onProgress(data)
  }

  return ws
}
