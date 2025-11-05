import { useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { useAppStore } from '@/lib/store'
import { api, createProgressWebSocket } from '@/lib/api'

export function AnalyzeButton() {
  const {
    fileId,
    selectedSegments,
    isAnalyzing,
    analysisProgress,
    analysisStatus,
    setIsAnalyzing,
    setAnalysisProgress,
    setProject,
    setShowControls,
  } = useAppStore()

  const handleAnalyze = useCallback(async () => {
    if (!fileId || selectedSegments.length === 0) return

    setIsAnalyzing(true)
    setAnalysisProgress(0, 'Starting analysis...')

    try {
      // Start analysis on selected segments only
      const response = await api.analyzeCues({
        file_id: fileId,
        segments: selectedSegments,
        fps: 25,
        threshold: 0.5,
      })

      // Connect to WebSocket for progress updates
      const ws = createProgressWebSocket(response.session_id, (update) => {
        setAnalysisProgress(update.progress_percent, update.status)
      })

      // Store project when complete
      setProject(response.session_id, response.project)

      // Hide controls after successful analysis
      setShowControls(false)

      // Close WebSocket
      ws.close()
    } catch (err) {
      console.error('Analysis error:', err)
      setAnalysisProgress(0, 'Analysis failed')
    } finally {
      setIsAnalyzing(false)
    }
  }, [fileId, selectedSegments, setIsAnalyzing, setAnalysisProgress, setProject, setShowControls])

  if (!fileId) {
    return null
  }

  return (
    <Card className="w-full self-start">
      <CardHeader>
        <CardTitle>Full Musical Analysis</CardTitle>
        <CardDescription>
          {selectedSegments.length > 0
            ? `Run comprehensive analysis on ${selectedSegments.length} selected segment${selectedSegments.length !== 1 ? 's' : ''}`
            : 'Select segments from the waveform above to analyze'}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {!isAnalyzing && analysisProgress === 0 && (
          <Button
            onClick={handleAnalyze}
            className="w-full"
            size="lg"
            disabled={selectedSegments.length === 0}
          >
            {selectedSegments.length > 0
              ? `Analyze ${selectedSegments.length} Cue${selectedSegments.length !== 1 ? 's' : ''}`
              : 'Select segments to analyze'}
          </Button>
        )}

        {isAnalyzing && (
          <div className="space-y-3">
            <Progress value={analysisProgress} className="w-full" />
            <div className="text-sm text-center text-muted-foreground">
              {analysisStatus}
            </div>
            <div className="text-xs text-center text-muted-foreground">
              This may take several minutes for large files
            </div>
          </div>
        )}

        {!isAnalyzing && analysisProgress > 0 && (
          <div className="text-sm text-center text-green-600 font-medium">
            Analysis complete!
          </div>
        )}
      </CardContent>
    </Card>
  )
}
