import { FileUpload } from '@/components/FileUpload'
import { WaveformViewer } from '@/components/WaveformViewer'
import { SelectionControls } from '@/components/SelectionControls'
import { ThresholdControls } from '@/components/ThresholdControls'
import { AnalyzeButton } from '@/components/AnalyzeButton'
import { CueCard } from '@/components/CueCard'
import { DebugConsole } from '@/components/DebugConsole'
import { CueSynch } from '@/components/CueSynch'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/lib/store'
import { exportProjectToCSV } from '@/lib/csv-export'

function App() {
  const { fileId, project, showControls, setShowControls, resetToSegmentation, reset, currentPage, setCurrentPage } = useAppStore()

  return (
    <div className="min-h-screen bg-background p-8">
      {/* Global Debug Console */}
      <DebugConsole />
      <div className="max-w-[1800px] mx-auto space-y-8">
        {/* Header */}
        <header className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex-1" />
            <div className="text-center">
              <h1 className="text-4xl font-bold">Motive</h1>
              <p className="text-muted-foreground">
                Music Analysis for MX & Media Files
              </p>
            </div>
            <div className="flex-1 flex justify-end">
              {currentPage === 'analysis' && fileId && !project && (
                <Button variant="outline" size="sm" onClick={reset}>
                  New Import
                </Button>
              )}
            </div>
          </div>
        </header>

        {/* Analysis Page */}
        {currentPage === 'analysis' && (
          <>
            {/* Upload Section */}
            {!fileId && <FileUpload />}

            {/* Main Analysis View - Always show waveform when file is loaded */}
            {fileId && (
          <div className="space-y-6">
            {/* Selection Controls - Above Waveform */}
            <SelectionControls />

            {/* Full-width Waveform at Top - Always Visible */}
            <WaveformViewer />

            {/* Collapsible Controls Section */}
            {!project && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ThresholdControls />
                <AnalyzeButton />
              </div>
            )}

            {project && (
              <div className="space-y-4">
                {/* Collapsible Controls Header */}
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold">Analysis Results</h2>
                    <p className="text-muted-foreground">
                      {project.cues.length} cues analyzed - Click cues above or cards below to navigate
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={resetToSegmentation}
                    >
                      Back
                    </Button>
                    <Button
                      size="sm"
                      onClick={() => setCurrentPage('cuesynch')}
                    >
                      Export to DAW or NLE
                    </Button>
                  </div>
                </div>

                {/* Cue Cards */}
                <div className="grid gap-4">
                  {project.cues.map((cue, index) => (
                    <CueCard key={cue.id} cue={cue} index={index} />
                  ))}
                </div>
              </div>
            )}
          </div>
            )}
          </>
        )}

        {/* CueSynch Page */}
        {currentPage === 'cuesynch' && <CueSynch />}
      </div>
    </div>
  )
}

export default App
