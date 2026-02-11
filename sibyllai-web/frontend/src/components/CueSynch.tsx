import { useState, useRef, useCallback, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/lib/store'
import { generateCueCSV } from '@/lib/csv-export'
import {
  parseCSVWithHeaders,
  parseCSVWithSelectedFields,
  detectTimeColumn,
  detectFrameRate,
  type CSVRow,
} from '@/lib/csv-parser'
import { generateWavWithMarkers, downloadBlob } from '@/lib/wav-generator'

interface CueData {
  headers: string[]
  rows: CSVRow[]
  frameRate: number
  detectedTimeColumn: string
  sourceName: string
}

export function CueSynch() {
  const { project, fileName, framerate, setCurrentPage, reset } = useAppStore()

  const [file, setFile] = useState<File | null>(null)
  const [frameRateSetting] = useState<string>('auto')
  const [sampleRate, setSampleRate] = useState<number>(48000)
  const [cueData, setCueData] = useState<CueData | null>(null)
  const [timeColumn, setTimeColumn] = useState<string>('')
  const [selectedFields, setSelectedFields] = useState<string[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string>('')
  const [useAnalysisData, setUseAnalysisData] = useState(true)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const analysisFrameRate = project?.project?.fps ?? framerate ?? 24

  const getDefaultSelectedFields = useCallback((headers: string[], timeField: string) => {
    // Select all non-time fields by default
    return headers.filter((header) => header !== timeField)
  }, [])

  // Convert project cues to CueData format
  const convertProjectToCueData = useCallback(() => {
    if (!project) return null

    const csvContent = generateCueCSV(project)
    const { headers, rows } = parseCSVWithHeaders(csvContent)
    const detectedTime = detectTimeColumn(headers)

    return {
      headers,
      rows,
      frameRate: analysisFrameRate,
      detectedTimeColumn: detectedTime,
      sourceName: fileName || project.project.name || 'analysis',
    }
  }, [project, analysisFrameRate, fileName])

  // Auto-load project data when available
  useEffect(() => {
    if (project && useAnalysisData) {
      const data = convertProjectToCueData()
      if (data) {
        setCueData(data)
        setTimeColumn(data.detectedTimeColumn)
        setSelectedFields(getDefaultSelectedFields(data.headers, data.detectedTimeColumn))
      }
    }
  }, [project, useAnalysisData, convertProjectToCueData, getDefaultSelectedFields])

  const analyzeCSV = useCallback(async (csvFile: File, frameSetting: string) => {
    setIsAnalyzing(true)
    setError('')

    try {
      const text = await csvFile.text()
      const { headers, rows } = parseCSVWithHeaders(text)

      if (rows.length === 0) {
        throw new Error('CSV file has no data rows')
      }

      const detectedTime = detectTimeColumn(headers)
      let frameRate = 30

      if (frameSetting === 'auto') {
        frameRate = detectFrameRate(rows, detectedTime)
      } else {
        frameRate = parseFloat(frameSetting)
      }

      setCueData({
        headers,
        rows,
        frameRate,
        detectedTimeColumn: detectedTime,
        sourceName: csvFile.name,
      })
      setTimeColumn(detectedTime)
      const nonTimeFields = headers.filter(h => h !== detectedTime)
      setSelectedFields(nonTimeFields)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze CSV')
      console.error(err)
    } finally {
      setIsAnalyzing(false)
    }
  }, [])

  const handleFileSelect = async (selectedFile: File) => {
    setFile(selectedFile)
    setUseAnalysisData(false)
    setError('')
    setCueData(null)
    setTimeColumn('')
    setSelectedFields([])
    await analyzeCSV(selectedFile, frameRateSetting)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile && droppedFile.name.endsWith('.csv')) {
      handleFileSelect(droppedFile)
    }
  }

  const handleGenerateWAV = () => {
    if (!cueData || !timeColumn || selectedFields.length === 0) {
      setError('Please select time column and at least one marker field')
      return
    }

    setIsGenerating(true)
    setError('')

    try {
      const markers = parseCSVWithSelectedFields(
        cueData.rows,
        cueData.frameRate,
        timeColumn,
        selectedFields
      )

      if (markers.length === 0) {
        throw new Error('No valid markers found')
      }

      const blob = generateWavWithMarkers(markers, sampleRate)
      const baseName = cueData.sourceName.replace(/\.(csv|wav|mp3|m4a)$/i, '')
      const rateLabel = sampleRate === 44100 ? '44k' : sampleRate === 48000 ? '48k' : '96k'
      const filename = `${baseName}_markers_${rateLabel}.wav`
      downloadBlob(blob, filename)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate WAV file')
      console.error(err)
    } finally {
      setIsGenerating(false)
    }
  }

  const toggleField = (field: string) => {
    setSelectedFields((prev) =>
      prev.includes(field) ? prev.filter((f) => f !== field) : [...prev, field]
    )
  }

  // Title case helper
  const toTitleCase = (str: string) => str.replace(/\b\w/g, char => char.toUpperCase())

  const getPreview = () => {
    if (!cueData?.rows || cueData.rows.length === 0) return ''
    const row = cueData.rows[0]
    const parts = selectedFields.map((field) => {
      const value = row[field]
      if (!value) return null
      // Add label prefixes with brackets for specific fields with capitalization
      const fieldLower = field.toLowerCase()
      const capitalizedValue = toTitleCase(value)
      if (fieldLower === 'bpm') return `[BPM ${value}]` // Keep BPM value as-is (it's a number)
      if (fieldLower === 'instruments') return `[Inst: ${capitalizedValue}]`
      if (fieldLower === 'genres') return `[Genre: ${capitalizedValue}]`
      if (fieldLower === 'style') return `[Style: ${capitalizedValue}]`
      return capitalizedValue
    }).filter(Boolean)
    return parts.join(' - ') || 'Marker'
  }

  const hasAnalysisData = !!project && project.cues.length > 0

  return (
    <div className="space-y-6">
      {/* Nav toolbar */}
      <div className="glass px-6 py-3 rounded-2xl flex items-center justify-between">
        <Button variant="outline" size="sm" onClick={() => setCurrentPage('analysis')} className="glass-lighter border-primary/20">
          ← Back
        </Button>
        <h2 className="text-lg font-medium font-display">Export to DAW or NLE</h2>
        <Button variant="outline" size="sm" onClick={reset} className="glass-lighter border-primary/20">
          New Import
        </Button>
      </div>

      {/* Data Source Info */}
      {hasAnalysisData && cueData && (
        <div className="glass rounded-2xl p-4">
          <p className="text-sm text-muted-foreground">
            Using {project.cues.length} cues from: <strong>{fileName}</strong>
          </p>
          <p className="text-sm text-muted-foreground">
            Frame rate: <strong>{analysisFrameRate} fps</strong>
          </p>
        </div>
      )}

      {/* File Upload - only show when no analysis data or explicitly choosing CSV */}
      {(!hasAnalysisData || !useAnalysisData) && (
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => fileInputRef.current?.click()}
          className={`glass rounded-2xl p-10 border-2 border-dashed transition-all cursor-pointer ${
            file
              ? 'border-green-500'
              : 'border-muted-foreground/30 hover:border-primary'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
            className="hidden"
          />
          <div className="text-center">
            {file ? (
              <>
                <div className="text-3xl mb-3">&#10003;</div>
                <p className="text-lg font-semibold text-green-500 mb-1.5">
                  {file.name}
                </p>
                <p className="text-muted-foreground text-sm">
                  Click to choose a different file
                </p>
              </>
            ) : (
              <>
                <div className="text-5xl mb-3">&#128193;</div>
                <p className="text-lg font-semibold mb-1.5">Drop CSV file here</p>
                <p className="text-muted-foreground text-sm">or click to browse</p>
              </>
            )}
          </div>
        </div>
      )}

      {/* Hidden file input for when using analysis data but want to switch */}
      {hasAnalysisData && useAnalysisData && (
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
          className="hidden"
        />
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-destructive/10 border border-destructive rounded-lg p-3">
          <p className="text-destructive text-sm">{error}</p>
        </div>
      )}

      {/* Analyzing State */}
      {isAnalyzing && (
        <div className="glass rounded-2xl p-5 text-center">
          <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-primary mb-3"></div>
          <p className="text-muted-foreground text-sm">Analyzing CSV...</p>
        </div>
      )}

      {/* Field Selection */}
      {cueData && cueData.headers && (
        <div className="glass rounded-2xl p-4">
          <h2 className="font-medium mb-4">Configure Markers</h2>

          {/* Sample Rate Selection */}
          <div className="mb-4">
            <p className="text-sm text-muted-foreground mb-2">
              Sample Rate
            </p>
            <div className="flex gap-2">
              {[44100, 48000, 96000].map((rate) => (
                <label
                  key={rate}
                  className={`inline-flex items-center px-4 py-2 rounded-lg cursor-pointer transition-colors text-sm ${
                    sampleRate === rate
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted hover:bg-muted/80'
                  }`}
                >
                  <input
                    type="radio"
                    name="sampleRate"
                    value={rate}
                    checked={sampleRate === rate}
                    onChange={() => setSampleRate(rate)}
                    className="hidden"
                  />
                  <span>{rate === 44100 ? '44.1' : rate / 1000} kHz</span>
                </label>
              ))}
            </div>
          </div>

          {/* Marker Fields Selection */}
          <div className="mb-4">
            <p className="text-sm text-muted-foreground mb-2">
              Select columns to include in marker names
            </p>
            <div className="flex flex-wrap gap-1.5">
              {cueData.headers
                .filter((h) => h !== timeColumn)
                .map((header) => (
                  <label
                    key={header}
                    className={`inline-flex items-center px-3 py-1.5 rounded-lg cursor-pointer transition-colors text-sm ${
                      selectedFields.includes(header)
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted hover:bg-muted/80'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedFields.includes(header)}
                      onChange={() => toggleField(header)}
                      className="hidden"
                    />
                    <span>{header}</span>
                  </label>
                ))}
            </div>
          </div>

          {/* Preview */}
          {selectedFields.length > 0 &&
            cueData.rows &&
            cueData.rows.length > 0 && (
              <div className="bg-muted rounded-lg p-3">
                <p className="text-sm text-muted-foreground mb-1">
                  Preview (first marker):
                </p>
                <p className="font-mono text-sm">{getPreview()}</p>
              </div>
            )}
        </div>
      )}

      {/* Generate Button */}
      {cueData && (
        <Button
          onClick={handleGenerateWAV}
          disabled={isGenerating || selectedFields.length === 0}
          className="w-full btn-primary h-14 text-lg rounded-xl shadow-lg"
          size="lg"
        >
          {isGenerating ? (
            <span className="flex items-center justify-center">
              <div className="inline-block animate-spin rounded-full h-5 w-5 border-b-2 border-primary-foreground mr-2.5"></div>
              Generating WAV...
            </span>
          ) : (
            'Generate & Download WAV'
          )}
        </Button>
      )}

      {/* Instructions */}
      {cueData && (
        <div className="glass rounded-2xl p-4">
          <h3 className="font-medium mb-3">Next Steps</h3>
          <ol className="space-y-2 text-sm text-muted-foreground">
            <li className="flex items-start">
              <span className="font-medium text-foreground mr-2">1.</span>
              <span>Download the generated WAV file</span>
            </li>
            <li className="flex items-start">
              <span className="font-medium text-foreground mr-2">2.</span>
              <span>Open Logic Pro with your project</span>
            </li>
            <li className="flex items-start">
              <span className="font-medium text-foreground mr-2">3.</span>
              <span>
                Import the audio file to the correct position (e.g.{' '}
                <strong>01:00:00:00</strong> or <strong>00:00:00:00</strong>)
              </span>
            </li>
            <li className="flex items-start">
              <span className="font-medium text-foreground mr-2">4.</span>
              <span>
                Go to{' '}
                <strong>Navigate &gt; Other &gt; Import Marker from Audio File</strong>
              </span>
            </li>
            <li className="flex items-start">
              <span className="font-medium text-foreground mr-2">5.</span>
              <span>Markers will appear at their correct timestamps!</span>
            </li>
          </ol>
          <div className="mt-3 bg-muted/50 rounded-lg p-3">
            <p className="text-sm text-muted-foreground">
              <strong>Note:</strong> Also works with other DAWs that support WAV marker metadata (e.g. Adobe Audition).
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
