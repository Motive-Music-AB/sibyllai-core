import { useState, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import {
  parseCSVWithHeaders,
  parseCSVWithSelectedFields,
  detectTimeColumn,
  detectFrameRate,
  type CSVRow,
} from '@/lib/csv-parser'
import { generateWavWithMarkers, downloadBlob } from '@/lib/wav-generator'

interface CSVData {
  headers: string[]
  rows: CSVRow[]
  frameRate: number
  detectedTimeColumn: string
}

export function CueSynch() {
  const [file, setFile] = useState<File | null>(null)
  const [frameRateSetting, setFrameRateSetting] = useState<string>('auto')
  const [csvData, setCSVData] = useState<CSVData | null>(null)
  const [timeColumn, setTimeColumn] = useState<string>('')
  const [selectedFields, setSelectedFields] = useState<string[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string>('')
  const fileInputRef = useRef<HTMLInputElement>(null)

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

      setCSVData({
        headers,
        rows,
        frameRate,
        detectedTimeColumn: detectedTime,
      })
      setTimeColumn(detectedTime)
      // Default to all non-time fields selected
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
    setError('')
    setCSVData(null)
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
    if (!csvData || !timeColumn || selectedFields.length === 0) {
      setError('Please select time column and at least one marker field')
      return
    }

    setIsGenerating(true)
    setError('')

    try {
      const markers = parseCSVWithSelectedFields(
        csvData.rows,
        csvData.frameRate,
        timeColumn,
        selectedFields
      )

      if (markers.length === 0) {
        throw new Error('No valid markers found in the CSV')
      }

      const blob = generateWavWithMarkers(markers)
      const filename = file
        ? file.name.replace('.csv', '_marker_list.wav')
        : 'marker_list.wav'
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
    if (!csvData?.rows || csvData.rows.length === 0) return ''
    const row = csvData.rows[0]
    const parts = selectedFields.map((field) => row[field]).filter(Boolean)
    return parts.join(' - ') || 'Marker'
  }

  const getTimecodePreview = () => {
    if (!csvData?.rows || csvData.rows.length === 0 || !timeColumn) return []
    return csvData.rows.slice(0, 3).map((row) => row[timeColumn]).filter(Boolean)
  }

  return (
    <div className="space-y-6">
      {/* Frame Rate Selector */}
      <div className="bg-card rounded-lg p-5 border">
        <label className="block mb-1.5 font-semibold text-sm">Frame Rate</label>
        <select
          value={frameRateSetting}
          onChange={(e) => setFrameRateSetting(e.target.value)}
          className="w-full p-2.5 bg-muted border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
        >
          <option value="auto">Auto Detect</option>
          <option value="23.976">23.976 fps (Film)</option>
          <option value="24">24 fps (Film)</option>
          <option value="25">25 fps (PAL)</option>
          <option value="29.97">29.97 fps (NTSC drop-frame)</option>
          <option value="30">30 fps (NTSC)</option>
          <option value="50">50 fps</option>
          <option value="60">60 fps</option>
        </select>
      </div>

      {/* File Upload */}
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
      {csvData && csvData.headers && (
        <div className="bg-card rounded-lg p-5 border">
          <h2 className="text-xl font-bold mb-5">Configure Markers</h2>

          {/* Time Column Selection */}
          <div className="mb-5">
            <label className="block mb-1.5 font-semibold text-sm">
              Time Column
            </label>
            <select
              value={timeColumn}
              onChange={(e) => setTimeColumn(e.target.value)}
              className="w-full p-2.5 bg-muted border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            >
              {csvData.headers.map((header) => (
                <option key={header} value={header}>
                  {header}
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground mt-1.5">
              Detected frame rate: {csvData.frameRate} fps
            </p>
            {getTimecodePreview().length > 0 && (
              <div className="mt-2.5 bg-muted rounded-lg p-3">
                <p className="text-xs text-muted-foreground mb-2.5">
                  Sample timecodes:
                </p>
                <div className="flex flex-wrap gap-2.5">
                  {getTimecodePreview().map((tc, idx) => (
                    <div key={idx} className="font-mono text-primary text-sm">
                      {tc}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Marker Fields Selection */}
          <div className="mb-5">
            <label className="block mb-1.5 font-semibold text-sm">
              Marker Fields
            </label>
            <p className="text-xs text-muted-foreground mb-2.5">
              Select columns to include in marker names
            </p>
            <div className="flex flex-wrap gap-1.5">
              {csvData.headers
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
            csvData.rows &&
            csvData.rows.length > 0 && (
              <div className="bg-muted rounded-lg p-3">
                <p className="text-xs text-muted-foreground mb-1.5">
                  Preview (first marker):
                </p>
                <p className="font-mono text-primary text-sm">{getPreview()}</p>
              </div>
            )}
        </div>
      )}

      {/* Generate Button */}
      {csvData && (
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
      {csvData && (
        <div className="bg-card rounded-lg p-5 border">
          <h3 className="text-lg font-bold mb-3">Next Steps</h3>
          <ol className="space-y-2.5 text-muted-foreground text-sm">
            <li className="flex items-start">
              <span className="font-bold text-primary mr-2.5">1.</span>
              <span>Download the generated WAV file</span>
            </li>
            <li className="flex items-start">
              <span className="font-bold text-primary mr-2.5">2.</span>
              <span>Open Logic Pro with your project</span>
            </li>
            <li className="flex items-start">
              <span className="font-bold text-primary mr-2.5">3.</span>
              <span>
                Import the audio file to the correct position (e.g.{' '}
                <strong>01:00:00:00</strong> or <strong>00:00:00:00</strong>)
              </span>
            </li>
            <li className="flex items-start">
              <span className="font-bold text-primary mr-2.5">4.</span>
              <span>
                Go to{' '}
                <strong>Navigate &gt; Other &gt; Import Marker from Audio File</strong>
              </span>
            </li>
            <li className="flex items-start">
              <span className="font-bold text-primary mr-2.5">5.</span>
              <span>Markers will appear at their correct timestamps!</span>
            </li>
          </ol>
          <div className="mt-4 bg-muted/50 rounded-lg p-3 border">
            <p className="text-xs text-muted-foreground">
              <strong>Note:</strong> This tool also works with other DAWs that
              support WAV marker metadata, including{' '}
              <strong>Adobe Audition</strong>.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
