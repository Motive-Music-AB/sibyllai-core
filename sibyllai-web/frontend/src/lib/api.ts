import axios from 'axios'
import type {
  UploadResponse,
  SegmentPreviewRequest,
  SegmentPreviewResponse,
  AnalyzeCuesRequest,
  AnalysisResponse,
  SibylProject,
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
   */
  async analyzeCues(request: AnalyzeCuesRequest): Promise<AnalysisResponse> {
    const response = await axios.post<AnalysisResponse>(`${API_BASE}/analyze-cues`, request)
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
