import { useCallback, useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { useAppStore } from '@/lib/store'
import { api } from '@/lib/api'

export function ThresholdControls() {
  const {
    fileId,
    musicThreshold,
    minGap,
    minCueLength,
    setMusicThreshold,
    setMinGap,
    setMinCueLength,
    setSegments,
    setIsSegmenting,
    isSegmenting,
    segments,
  } = useAppStore()

  const [localThreshold, setLocalThreshold] = useState(musicThreshold)
  const [localMinGap, setLocalMinGap] = useState(minGap)
  const [localMinCueLength, setLocalMinCueLength] = useState(minCueLength)
  const [showSettings, setShowSettings] = useState(false)

  // Debounced preview update
  useEffect(() => {
    const timer = setTimeout(() => {
      if (localThreshold !== musicThreshold) {
        setMusicThreshold(localThreshold)
      }
      if (localMinGap !== minGap) {
        setMinGap(localMinGap)
      }
      if (localMinCueLength !== minCueLength) {
        setMinCueLength(localMinCueLength)
      }
    }, 300) // 300ms debounce

    return () => clearTimeout(timer)
  }, [localThreshold, localMinGap, localMinCueLength, musicThreshold, minGap, minCueLength, setMusicThreshold, setMinGap, setMinCueLength])

  const handlePreview = useCallback(async () => {
    if (!fileId) return

    setIsSegmenting(true)

    try {
      const response = await api.getSegmentPreview({
        file_id: fileId,
        music_thresh: musicThreshold,
        min_gap: minGap,
        min_cue_length: minCueLength,
      })

      setSegments(response.segments, response.duration)
    } catch (err) {
      console.error('Segmentation error:', err)
    } finally {
      setIsSegmenting(false)
    }
  }, [fileId, musicThreshold, minGap, minCueLength, setSegments, setIsSegmenting])

  if (!fileId) {
    return null
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Cue Detection</CardTitle>
        <CardDescription>
          Adjust threshold to control music cue detection
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Button
          onClick={handlePreview}
          disabled={isSegmenting}
          className="w-full"
        >
          {isSegmenting ? 'Detecting cues...' : segments.length > 0 ? 'Update Preview' : 'Detect Cues'}
        </Button>

        <Button
          onClick={() => setShowSettings(!showSettings)}
          variant="outline"
          className="w-full"
        >
          {showSettings ? 'Hide Settings' : 'Settings'}
        </Button>

        <div
          className="overflow-hidden transition-all duration-300 ease-in-out"
          style={{
            maxHeight: showSettings ? '1000px' : '0px',
            opacity: showSettings ? 1 : 0
          }}
        >
          <div className="space-y-6 pt-2">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">
                  Detection Sensitivity
                </label>
                <span className="text-sm text-muted-foreground">
                  {localThreshold < 0.001 ? localThreshold.toFixed(6) : localThreshold < 0.01 ? localThreshold.toFixed(4) : localThreshold.toFixed(2)}
                </span>
              </div>
              <Slider
                value={[localThreshold]}
                onValueChange={([value]) => setLocalThreshold(value)}
                min={0.000001}
                max={0.9}
                step={0.000001}
                disabled={isSegmenting}
              />
              <p className="text-xs text-muted-foreground">
                Lower values = more sensitive (captures quieter/subtle music). Higher values = stricter (only clear/loud music)
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">
                  Minimum Silence Gap
                </label>
                <span className="text-sm text-muted-foreground">
                  {localMinGap.toFixed(1)}s
                </span>
              </div>
              <Slider
                value={[localMinGap]}
                onValueChange={([value]) => setLocalMinGap(value)}
                min={0.1}
                max={15.0}
                step={0.1}
                disabled={isSegmenting}
              />
              <p className="text-xs text-muted-foreground">
                Minimum silence duration required to split cues apart. Lower = splits more aggressively (more separate cues)
              </p>
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">
                  Minimum Cue Length
                </label>
                <span className="text-sm text-muted-foreground">
                  {localMinCueLength.toFixed(1)}s
                </span>
              </div>
              <Slider
                value={[localMinCueLength]}
                onValueChange={([value]) => setLocalMinCueLength(value)}
                min={0.5}
                max={15.0}
                step={0.1}
                disabled={isSegmenting}
              />
              <p className="text-xs text-muted-foreground">
                Minimum duration for a detected cue. Shorter cues will be filtered out
              </p>
            </div>
          </div>
        </div>

        {segments.length > 0 && (
          <div className="text-sm text-center text-muted-foreground">
            {segments.length} cue{segments.length !== 1 ? 's' : ''} detected
          </div>
        )}
      </CardContent>
    </Card>
  )
}
