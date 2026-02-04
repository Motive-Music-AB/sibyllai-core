import { useCallback, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useAppStore } from '@/lib/store'
import { api } from '@/lib/api'

export function AnalyzeButton() {
  const {
    fileId,
    selectedSegments,
    isAnalyzing,
    analysisProgress,
    analysisStatus,
    framerate,
    setIsAnalyzing,
    setAnalysisProgress,
    setProject,
    setShowControls,
  } = useAppStore()

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null)

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
      const response = await api.analyzeCues({
        file_id: fileId,
        segments: selectedSegments,
        fps: framerate,
        threshold: 0.5,
      })

      const sessionId = response.session_id

      const pollStatus = async () => {
        try {
          const status = await api.getAnalysisStatus(sessionId)
          setAnalysisProgress(status.progress_percent, status.status)

          if (status.complete) {
            if (pollingRef.current) {
              clearInterval(pollingRef.current)
              pollingRef.current = null
            }

            if (status.error) {
              console.error('Analysis error:', status.error)
              setAnalysisProgress(0, `Analysis failed: ${status.error}`)
            } else if (status.project) {
              setProject(sessionId, status.project)
              setShowControls(false)
            }
            setIsAnalyzing(false)
          }
        } catch (err) {
          console.error('Error polling status:', err)
        }
      }

      pollingRef.current = setInterval(pollStatus, 500)
      await pollStatus()
    } catch (err) {
      console.error('Analysis error:', err)
      setAnalysisProgress(0, 'Analysis failed')
      setIsAnalyzing(false)
    }
  }, [fileId, selectedSegments, framerate, setIsAnalyzing, setAnalysisProgress, setProject, setShowControls])

  if (!fileId) return null

  return (
    <Card className="w-full glass border-white/5 p-4 animate-in fade-in slide-in-from-right-4 duration-700">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-2xl font-display font-medium">Full Musical Analysis</CardTitle>
          {selectedSegments.length > 0 && (
            <div className="text-sm font-bold text-primary">
              {selectedSegments.length} {selectedSegments.length === 1 ? 'segment' : 'segments'}
            </div>
          )}
        </div>
        <CardDescription className="text-foreground-muted">
          Uncover key signature, tempo, instrumentation, and structural analysis for the entire track.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">

        {!isAnalyzing && (
          <Button
            onClick={handleAnalyze}
            className="btn-primary w-full h-14 text-lg rounded-xl"
            disabled={selectedSegments.length === 0}
          >
            {selectedSegments.length > 0
              ? 'Initialize AI Analysis'
              : 'Queue Segments Above'}
          </Button>
        )}

        {isAnalyzing && (
          <div className="space-y-6 p-4 glass-lighter rounded-2xl animate-in fade-in zoom-in-95 duration-500">
            <div className="relative h-2 w-full bg-white/5 overflow-hidden rounded-full">
              <div
                className="absolute inset-y-0 left-0 bg-primary shadow-glow transition-all duration-500 ease-out"
                style={{ width: `${analysisProgress}%` }}
              />
            </div>
            <div className="space-y-2 text-center">
              <div className="text-sm font-bold font-mono tracking-widest text-primary uppercase">
                {analysisStatus}
              </div>
              <div className="text-[10px] text-foreground-muted font-bold uppercase tracking-widest">
                DO NOT CLOSE THIS TAB • {Math.round(analysisProgress)}% COMPLETE
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
