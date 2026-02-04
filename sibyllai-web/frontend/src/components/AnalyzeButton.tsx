import { useCallback, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { useAppStore } from '@/lib/store'
import { api } from '@/lib/api'

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

  const pollingRef = useRef<NodeJS.Timeout | null>(null)

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
      }
    }
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!fileId || selectedSegments.length === 0) return

    setIsAnalyzing(true)
    setAnalysisProgress(0, 'Starting analysis...')

    try {
      // Start analysis (returns immediately with session_id)
      const response = await api.analyzeCues({
        file_id: fileId,
        segments: selectedSegments,
        fps: 25,
        threshold: 0.5,
      })

      const sessionId = response.session_id

      // Poll for progress updates
      const pollStatus = async () => {
        try {
          const status = await api.getAnalysisStatus(sessionId)

          // Update progress
          setAnalysisProgress(status.progress_percent, status.status)

          if (status.complete) {
            // Stop polling
            if (pollingRef.current) {
              clearInterval(pollingRef.current)
              pollingRef.current = null
            }

            if (status.error) {
              console.error('Analysis error:', status.error)
              setAnalysisProgress(0, `Analysis failed: ${status.error}`)
            } else if (status.project) {
              // Store project when complete
              setProject(sessionId, status.project)
              // Hide controls after successful analysis
              setShowControls(false)
            }

            setIsAnalyzing(false)
          }
        } catch (err) {
          console.error('Error polling status:', err)
        }
      }

      // Start polling every 500ms
      pollingRef.current = setInterval(pollStatus, 500)

      // Also poll immediately
      await pollStatus()

    } catch (err) {
      console.error('Analysis error:', err)
      setAnalysisProgress(0, 'Analysis failed')
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
