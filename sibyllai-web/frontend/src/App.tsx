import { FileUpload } from '@/components/FileUpload'
import { WaveformViewer } from '@/components/WaveformViewer'
import { SelectionControls } from '@/components/SelectionControls'
import { ThresholdControls } from '@/components/ThresholdControls'
import { AnalyzeButton } from '@/components/AnalyzeButton'
import { CueCard } from '@/components/CueCard'
import { DebugConsole } from '@/components/DebugConsole'
import { CueSynch } from '@/components/CueSynch'
import { TrackReplacement } from '@/components/TrackReplacement'
import { LicensingPage } from '@/components/LicensingPage'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/lib/store'

function App() {
  const { fileId, project, resetToSegmentation, reset, currentPage, setCurrentPage } = useAppStore()

  return (
    <div className="min-h-screen relative overflow-hidden selection:bg-primary/30 selection:text-foreground">
      {/* Global Debug Console */}
      <DebugConsole />

      <div className="relative z-10 max-w-[1800px] mx-auto px-6 py-12 lg:px-12 space-y-12">
        {/* Header */}
        <header className="relative">
          <div className="flex items-center justify-between">
            <div className="flex-1 flex justify-start">
              {currentPage === 'cuesynch' && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage('analysis')}
                  className="glass-lighter hover:bg-primary/20 border-primary/20 transition-all duration-300"
                >
                  ← Back
                </Button>
              )}
              {currentPage === 'replacement' && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setCurrentPage('analysis')}
                  className="glass-lighter hover:bg-primary/20 border-primary/20 transition-all duration-300"
                >
                  ← Back
                </Button>
              )}
              {currentPage === 'analysis' && project && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={resetToSegmentation}
                  className="glass-lighter hover:bg-primary/20 border-primary/20 transition-all duration-300"
                >
                  ← Back
                </Button>
              )}
            </div>
            <div className="text-center group">
              <h1 className="text-[78px] font-semibold tracking-tight font-display -mb-2 bg-gradient-to-b from-white to-white/60 bg-clip-text text-transparent group-hover:from-primary group-hover:to-primary/40 transition-all duration-500 cursor-default" style={{ filter: 'drop-shadow(0 2px 8px rgba(0,0,0,0.5))' }}>
                Motive
              </h1>
              <p className="text-white/80 font-medium tracking-[0.25em] uppercase text-xs" style={{ textShadow: '0 1px 6px rgba(0,0,0,0.6)' }}>
                Music Analysis for Media Files
              </p>
            </div>
            <div className="flex-1 flex justify-end">
              {((currentPage === 'analysis' && fileId && !project) || currentPage === 'cuesynch' || currentPage === 'replacement' || currentPage === 'licensing') && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={reset}
                  className="glass-lighter hover:bg-primary/20 border-primary/20 transition-all duration-300"
                >
                  New Import
                </Button>
              )}
            </div>
          </div>
        </header>

        {/* Main Content Areas */}
        <main>
          {currentPage === 'analysis' ? (
            <div className="space-y-12">
              {/* Upload Section */}
              {!fileId && <FileUpload />}

              {/* Main Analysis View */}
              {fileId && (
                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
                  {/* Selection Controls - Above Waveform */}
                  <SelectionControls />

                  {/* Full-width Waveform */}
                  <WaveformViewer />

                  {/* Controls or Results */}
                  {!project ? (
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
                      <ThresholdControls />
                      <AnalyzeButton />
                    </div>
                  ) : (
                    <div className="space-y-8 animate-in fade-in duration-500">
                      {/* Results Header */}
                      <div className="flex items-center justify-between glass p-4 rounded-2xl">
                        <div>
                          <h2 className="text-lg font-medium font-display">Analysis Results</h2>
                          <p className="text-foreground-muted text-sm">
                            {project.cues.length} cues analyzed - Click cues above or cards below to navigate
                          </p>
                        </div>
                        <div className="flex items-center gap-3">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => setCurrentPage('replacement')}
                            className="glass-lighter border-primary/20"
                          >
                            Track Replacement
                          </Button>
                          <Button
                            size="sm"
                            onClick={() => setCurrentPage('cuesynch')}
                            className="btn-primary px-6"
                          >
                            Export to DAW or NLE
                          </Button>
                        </div>
                      </div>

                      {/* Cue Cards */}
                      <div className="grid gap-6">
                        {project.cues.map((cue, index) => (
                          <CueCard key={cue.id} cue={cue} index={index} />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : currentPage === 'cuesynch' ? (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
              <CueSynch />
            </div>
          ) : currentPage === 'licensing' ? (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
              <LicensingPage />
            </div>
          ) : (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
              <TrackReplacement />
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
