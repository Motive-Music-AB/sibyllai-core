import { useCallback, useState } from 'react'
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
    silenceThreshold,
    setMusicThreshold,
    setMinGap,
    setMinCueLength,
    setSilenceThreshold,
    setSegments,
    setIsSegmenting,
    isSegmenting,
    segments,
  } = useAppStore()

  const [localThreshold, setLocalThreshold] = useState(musicThreshold)
  const [localMinGap, setLocalMinGap] = useState(minGap)
  const [localMinCueLength, setLocalMinCueLength] = useState(minCueLength)
  const [localSilenceThreshold, setLocalSilenceThreshold] = useState(silenceThreshold)
  const [showSettings, setShowSettings] = useState(false)

  const handlePreview = useCallback(async () => {
    if (!fileId) return
    setIsSegmenting(true)
    setMusicThreshold(localThreshold)
    setMinGap(localMinGap)
    setMinCueLength(localMinCueLength)
    setSilenceThreshold(localSilenceThreshold)

    try {
      const response = await api.getSegmentPreview({
        file_id: fileId,
        music_thresh: localThreshold,
        min_gap: localMinGap,
        min_cue_length: localMinCueLength,
        silence_thresh: localSilenceThreshold,
      })
      setSegments(response.segments, response.duration)
    } catch (err) {
      console.error('Segmentation error:', err)
    } finally {
      setIsSegmenting(false)
    }
  }, [fileId, localThreshold, localMinGap, localMinCueLength, localSilenceThreshold, setSegments, setIsSegmenting, setMusicThreshold, setMinGap, setMinCueLength, setSilenceThreshold])

  if (!fileId) return null

  return (
    <Card className="w-full glass border-white/5 p-4 animate-in fade-in slide-in-from-left-4 duration-700">
      <CardHeader className="pb-4">
        <CardTitle className="text-2xl font-display font-medium">Cue Detection</CardTitle>
        <CardDescription className="text-foreground-muted">
          Automatically identify precise in and out points for musical cues, stingers, and transitions with frame accuracy.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="flex flex-col gap-3">
          <Button
            onClick={handlePreview}
            disabled={isSegmenting}
            className="btn-primary h-14 text-lg rounded-xl"
          >
            {isSegmenting ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                Scanning Waveform...
              </div>
            ) : segments.length > 0 ? 'Update Preview' : 'Run Cue Detection'}
          </Button>

          <Button
            onClick={() => setShowSettings(!showSettings)}
            variant="ghost"
            className={`h-10 rounded-lg text-xs font-bold uppercase tracking-widest transition-colors ${showSettings ? 'bg-white/10 text-primary' : 'text-foreground-muted hover:bg-white/5 hover:text-foreground'}`}
          >
            {showSettings ? 'Hide Advanced Settings' : 'Advanced Settings'}
          </Button>
        </div>

        <div
          className="overflow-hidden transition-all duration-500 ease-in-out"
          style={{
            maxHeight: showSettings ? '1000px' : '0px',
            opacity: showSettings ? 1 : 0
          }}
        >
          <div className="space-y-8 pt-4 pb-2">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-sm font-bold tracking-tight text-foreground">
                    Silence Threshold
                  </label>
                  <p className="text-[10px] text-foreground-subtle font-medium uppercase tracking-wider">
                    Amplitude level to trim silence
                  </p>
                </div>
                <span className="glass-lighter px-3 py-1 rounded-lg font-mono font-bold text-xs text-primary border border-primary/20">
                  {localSilenceThreshold < 0.001 ? localSilenceThreshold.toFixed(4) : localSilenceThreshold.toFixed(3)}
                </span>
              </div>
              <Slider
                value={[localSilenceThreshold]}
                onValueChange={([value]) => setLocalSilenceThreshold(value)}
                min={0.001}
                max={0.1}
                step={0.001}
                disabled={isSegmenting}
                className="py-2"
              />
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-sm font-bold tracking-tight text-foreground">
                    In/Out Sensitivity
                  </label>
                  <p className="text-[10px] text-foreground-subtle font-medium uppercase tracking-wider">
                    Threshold for music start/end
                  </p>
                </div>
                <span className="glass-lighter px-3 py-1 rounded-lg font-mono font-bold text-xs text-primary border border-primary/20">
                  {localThreshold < 0.001 ? localThreshold.toFixed(6) : localThreshold < 0.01 ? localThreshold.toFixed(4) : localThreshold.toFixed(2)}
                </span>
              </div>
              <Slider
                value={[localThreshold]}
                onValueChange={([value]) => setLocalThreshold(value)}
                min={0.001}
                max={0.03}
                step={0.0001}
                disabled={isSegmenting}
                className="py-2"
              />
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-sm font-bold tracking-tight text-foreground">
                    Minimum Silence Gap
                  </label>
                  <p className="text-[10px] text-foreground-subtle font-medium uppercase tracking-wider">
                    Duration required to split cues
                  </p>
                </div>
                <span className="glass-lighter px-3 py-1 rounded-lg font-mono font-bold text-xs text-primary border border-primary/20">
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
                className="py-2"
              />
            </div>

            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <label className="text-sm font-bold tracking-tight text-foreground">
                    Minimum Cue Length
                  </label>
                  <p className="text-[10px] text-foreground-subtle font-medium uppercase tracking-wider">
                    Ignore cues shorter than this
                  </p>
                </div>
                <span className="glass-lighter px-3 py-1 rounded-lg font-mono font-bold text-xs text-primary border border-primary/20">
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
                className="py-2"
              />
            </div>
          </div>
        </div>

        {segments.length > 0 && (
          <div className="flex items-center justify-center gap-2 pt-2 animate-in fade-in duration-500">
            <div className="h-[1px] flex-1 bg-white/5" />
            <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-foreground-subtle">
              {segments.length} EVENT{segments.length !== 1 ? 'S' : ''} DETECTED
            </div>
            <div className="h-[1px] flex-1 bg-white/5" />
          </div>
        )}
      </CardContent>
    </Card>
  )
}
