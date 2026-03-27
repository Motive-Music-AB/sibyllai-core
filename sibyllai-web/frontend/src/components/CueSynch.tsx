import { useState, useRef, useCallback, useEffect } from 'react'
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
    return headers.filter((header) => header !== timeField)
  }, [])

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

  const toTitleCase = (str: string) => str.replace(/\b\w/g, char => char.toUpperCase())

  const getPreview = () => {
    if (!cueData?.rows || cueData.rows.length === 0) return ''
    const row = cueData.rows[0]
    const parts = selectedFields.map((field) => {
      const value = row[field]
      if (!value) return null
      const fieldLower = field.toLowerCase()
      const capitalizedValue = toTitleCase(value)
      if (fieldLower === 'bpm') return `[BPM ${value}]`
      if (fieldLower === 'instruments') return `[Inst: ${capitalizedValue}]`
      if (fieldLower === 'genres') return `[Genre: ${capitalizedValue}]`
      if (fieldLower === 'style') return `[Style: ${capitalizedValue}]`
      return capitalizedValue
    }).filter(Boolean)
    return parts.join(' - ') || 'Marker'
  }

  const hasAnalysisData = !!project && project.cues.length > 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Nav toolbar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <button className="btn-pill" onClick={() => setCurrentPage('pipeline')} style={{ height: 28, fontSize: '0.65rem' }}>
          ← Back
        </button>
        <span className="label">Export to DAW or NLE</span>
        <button className="btn-pill" onClick={reset} style={{ height: 28, fontSize: '0.65rem' }}>
          New Import
        </button>
      </div>

      {/* Data Source Info */}
      {hasAnalysisData && cueData && (
        <div style={{ padding: 12, border: '1px solid #080808', borderRadius: 8 }}>
          <p className="mono" style={{ fontSize: '0.75rem' }}>
            Using {project.cues.length} cues from: <strong>{fileName}</strong>
          </p>
          <p className="mono" style={{ fontSize: '0.75rem' }}>
            Frame rate: <strong>{analysisFrameRate} fps</strong>
          </p>
        </div>
      )}

      {/* File Upload */}
      {(!hasAnalysisData || !useAnalysisData) && (
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => fileInputRef.current?.click()}
          className="track-card add-source"
          style={{ padding: 40, textAlign: 'center' }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
            hidden
          />
          {file ? (
            <>
              <p style={{ fontSize: '0.85rem', fontWeight: 800, marginBottom: 4 }}>{file.name}</p>
              <p className="mono" style={{ fontSize: '0.7rem', opacity: 0.5 }}>Click to choose a different file</p>
            </>
          ) : (
            <>
              <p style={{ fontSize: '0.85rem', fontWeight: 800, marginBottom: 4 }}>Drop CSV file here</p>
              <p className="mono" style={{ fontSize: '0.7rem', opacity: 0.5 }}>or click to browse</p>
            </>
          )}
        </div>
      )}

      {hasAnalysisData && useAnalysisData && (
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
          hidden
        />
      )}

      {/* Error */}
      {error && <div className="error-msg">{error}</div>}

      {/* Analyzing State */}
      {isAnalyzing && (
        <div style={{ padding: 20, textAlign: 'center', border: '1px solid #080808', borderRadius: 8 }}>
          <p className="mono" style={{ fontSize: '0.75rem' }}>Analyzing CSV...</p>
        </div>
      )}

      {/* Field Selection */}
      {cueData && cueData.headers && (
        <div style={{ padding: 16, border: '1px solid #080808', borderRadius: 8 }}>
          <div className="label" style={{ marginBottom: 12 }}>Configure Markers</div>

          {/* Sample Rate Selection */}
          <div style={{ marginBottom: 16 }}>
            <p className="label" style={{ fontSize: '0.55rem', marginBottom: 6, opacity: 0.6 }}>Sample Rate</p>
            <div style={{ display: 'flex', gap: 8 }}>
              {[44100, 48000, 96000].map((rate) => (
                <label
                  key={rate}
                  className={`btn-pill ${sampleRate === rate ? 'primary' : ''}`}
                  style={{ height: 28, fontSize: '0.65rem', cursor: 'pointer' }}
                >
                  <input
                    type="radio"
                    name="sampleRate"
                    value={rate}
                    checked={sampleRate === rate}
                    onChange={() => setSampleRate(rate)}
                    hidden
                  />
                  {rate === 44100 ? '44.1' : rate / 1000} kHz
                </label>
              ))}
            </div>
          </div>

          {/* Marker Fields Selection */}
          <div style={{ marginBottom: 16 }}>
            <p className="label" style={{ fontSize: '0.55rem', marginBottom: 6, opacity: 0.6 }}>
              Columns to include in marker names
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {cueData.headers
                .filter((h) => h !== timeColumn)
                .map((header) => (
                  <label
                    key={header}
                    className={`tag-toggle ${selectedFields.includes(header) ? 'selected' : ''}`}
                    style={{ cursor: 'pointer' }}
                  >
                    <input
                      type="checkbox"
                      checked={selectedFields.includes(header)}
                      onChange={() => toggleField(header)}
                      hidden
                    />
                    {header}
                  </label>
                ))}
            </div>
          </div>

          {/* Preview */}
          {selectedFields.length > 0 && cueData.rows && cueData.rows.length > 0 && (
            <div style={{ padding: 10, background: '#E8E8E8', borderRadius: 6 }}>
              <p className="label" style={{ fontSize: '0.55rem', marginBottom: 4, opacity: 0.6 }}>Preview (first marker):</p>
              <p className="mono" style={{ fontSize: '0.75rem' }}>{getPreview()}</p>
            </div>
          )}
        </div>
      )}

      {/* Generate Button */}
      {cueData && (
        <button
          className="btn-pill primary"
          onClick={handleGenerateWAV}
          disabled={isGenerating || selectedFields.length === 0}
          style={{ width: '100%', justifyContent: 'center', height: 44, fontSize: '0.85rem' }}
        >
          {isGenerating ? 'Generating WAV...' : 'Generate & Download WAV'}
        </button>
      )}

      {/* Instructions */}
      {cueData && (
        <div style={{ padding: 16, border: '1px solid #080808', borderRadius: 8 }}>
          <div className="label" style={{ marginBottom: 10 }}>Next Steps</div>
          <ol style={{ paddingLeft: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 6 }}>
            {[
              'Download the generated WAV file',
              'Open Logic Pro with your project',
              'Import the audio file to the correct position (e.g. 01:00:00:00)',
              'Go to Navigate > Other > Import Marker from Audio File',
              'Markers will appear at their correct timestamps!',
            ].map((step, i) => (
              <li key={i} className="mono" style={{ fontSize: '0.7rem', display: 'flex', alignItems: 'flex-start' }}>
                <span style={{ fontWeight: 700, marginRight: 8, flexShrink: 0 }}>{i + 1}.</span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
          <div style={{ marginTop: 10, padding: 8, background: '#E8E8E8', borderRadius: 4 }}>
            <p className="mono" style={{ fontSize: '0.65rem', opacity: 0.7 }}>
              Also works with other DAWs that support WAV marker metadata (e.g. Adobe Audition).
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
