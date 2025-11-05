import { useCallback, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useAppStore } from '@/lib/store'
import { api } from '@/lib/api'

export function FileUpload() {
  const [isDragging, setIsDragging] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [startTimecode, setStartTimecode] = useState('01:00:00:00')
  const [framerate, setFramerate] = useState(24)

  const { setUploadedFile, fileId } = useAppStore()

  const handleFile = useCallback((file: File) => {
    // Validate file type
    const validTypes = ['audio/', 'video/']
    if (!validTypes.some(type => file.type.startsWith(type))) {
      setError('Please upload an audio or video file')
      return
    }

    setError(null)
    setSelectedFile(file)
  }, [])

  const handleUpload = useCallback(async () => {
    if (!selectedFile) return

    setIsUploading(true)
    setError(null)

    try {
      const result = await api.uploadFile(selectedFile)
      setUploadedFile(selectedFile, result.file_id, result.filename, startTimecode, framerate)
    } catch (err) {
      setError('Failed to upload file. Please try again.')
      console.error('Upload error:', err)
    } finally {
      setIsUploading(false)
    }
  }, [selectedFile, startTimecode, framerate, setUploadedFile])

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)

    const file = e.dataTransfer.files[0]
    if (file) {
      handleFile(file)
    }
  }, [handleFile])

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFile(file)
    }
  }, [handleFile])

  // Hide if already uploaded
  if (fileId) {
    return null
  }

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Upload Audio/Video File</CardTitle>
        <CardDescription>
          Upload your MX file or any audio/video file to begin music analysis
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!selectedFile ? (
          <div
            className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
              isDragging
                ? 'border-primary bg-primary/10'
                : 'border-muted-foreground/25 hover:border-primary/50'
            }`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <div className="space-y-4">
              <div className="text-lg font-medium">
                Drop your file here, or click to browse
              </div>
              <div className="text-sm text-muted-foreground">
                Supports: WAV, MP3, M4A, MP4, MOV
              </div>
              <input
                ref={(input) => {
                  if (input) {
                    (window as any).fileInput = input;
                  }
                }}
                id="file-input"
                type="file"
                accept="audio/*,video/*"
                className="hidden"
                onChange={handleFileSelect}
              />
              <Button
                variant="outline"
                className="cursor-pointer"
                type="button"
                onClick={() => document.getElementById('file-input')?.click()}
              >
                Choose File
              </Button>
            </div>

            {error && (
              <div className="mt-4 text-sm text-destructive">{error}</div>
            )}
          </div>
        ) : (
          <div className="space-y-6">
            {/* File info */}
            <div className="p-4 bg-muted rounded-lg">
              <div className="font-medium">{selectedFile.name}</div>
              <div className="text-sm text-muted-foreground">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </div>
            </div>

            {/* Timecode and framerate inputs */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="start-timecode">Start Timecode</Label>
                <Input
                  id="start-timecode"
                  type="text"
                  value={startTimecode}
                  onChange={(e) => setStartTimecode(e.target.value)}
                  placeholder="01:00:00:00"
                  disabled={isUploading}
                />
                <p className="text-xs text-muted-foreground">
                  Format: HH:MM:SS:FF
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="framerate">Framerate</Label>
                <select
                  id="framerate"
                  value={framerate}
                  onChange={(e) => setFramerate(Number(e.target.value))}
                  disabled={isUploading}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                >
                  <option value={23.976}>23.976 fps</option>
                  <option value={24}>24 fps</option>
                  <option value={25}>25 fps</option>
                  <option value={29.97}>29.97 fps</option>
                  <option value={30}>30 fps</option>
                  <option value={50}>50 fps</option>
                  <option value={59.94}>59.94 fps</option>
                  <option value={60}>60 fps</option>
                </select>
              </div>
            </div>

            {/* Upload button */}
            <div className="flex gap-2">
              <Button
                onClick={handleUpload}
                disabled={isUploading}
                className="flex-1"
              >
                {isUploading ? 'Uploading...' : 'Upload & Continue'}
              </Button>
              <Button
                variant="outline"
                onClick={() => setSelectedFile(null)}
                disabled={isUploading}
              >
                Cancel
              </Button>
            </div>

            {error && (
              <div className="text-sm text-destructive">{error}</div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
