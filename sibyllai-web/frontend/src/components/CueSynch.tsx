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
  const { project, fileName, framerate } = useAppStore()

  const [file, setFile] = useState<File | null>(null)
  const [frameRateSetting] = useState<string>('auto')
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

      const blob = generateWavWithMarkers(markers)
      const baseName = cueData.sourceName.replace(/\.(csv|wav|mp3|m4a)$/i, '')
      const filename = `${baseName}_marker_list.wav`
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

  const getPreview = () => {
    if (!cueData?.rows || cueData.rows.length === 0) return ''
    const row = cueData.rows[0]
    const parts = selectedFields.map((field) => row[field]).filter(Boolean)
    return parts.join(' - ') || 'Marker'
  }

  const hasAnalysisData = !!project && project.cues.length > 0

  return (
    <div className="space-y-6">
      {/* Data Source Info */}
      {hasAnalysisData && cueData && (
        <div className="bg-card rounded-lg p-4 border">
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
          className={`bg-card rounded-lg p-10 border-2 border-dashed transition-all cursor-pointer ${
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
        <div className="bg-card rounded-lg p-5 text-center">
          <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-primary mb-3"></div>
          <p className="text-muted-foreground text-sm">Analyzing CSV...</p>
        </div>
      )}

      {/* Field Selection */}
      {cueData && cueData.headers && (
        <div className="bg-card rounded-lg p-4 border">
          <h2 className="font-medium mb-4">Configure Markers</h2>

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
          className="w-full"
          size="lg"
        >
          {isGenerating ? (
            <span className="flex items-center justify-center">
              <div className="inline-block animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2.5"></div>
              Generating WAV...
            </span>
          ) : (
            'Generate & Download WAV'
          )}
        </Button>
      )}

      {/* Instructions */}
      {cueData && (
        <div className="bg-card rounded-lg p-4 border">
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
