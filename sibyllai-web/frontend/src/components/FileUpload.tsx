import { useCallback, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
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

  if (fileId) {
    return null
  }

  return (
    <Card className="w-full max-w-2xl mx-auto glass p-4 animate-in fade-in zoom-in-95 duration-700">
      <CardHeader>
        <CardTitle className="text-3xl font-display font-medium tracking-tight">Upload Media</CardTitle>
      </CardHeader>
      <CardContent className="pt-4">
        {!selectedFile ? (
          <div
            className={`group relative border-2 border-dashed rounded-2xl p-16 text-center transition-all duration-500 cursor-pointer overflow-hidden ${isDragging
                ? 'border-primary bg-primary/10 shadow-glow'
                : 'border-white/10 glass-lighter hover:border-primary/40 hover:bg-white/5'
              }`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => document.getElementById('file-input')?.click()}
          >
            {/* Background Glow */}
            <div className="absolute inset-0 bg-gradient-to-tr from-primary/5 to-transparent pointer-events-none" />

            <div className="relative z-10 space-y-6">
              <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center group-hover:scale-110 transition-transform duration-500">
                <svg className="w-8 h-8 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
              </div>
              <div className="space-y-2">
                <div className="text-xl font-medium font-display tracking-tight text-foreground">
                  Drop your file here, or click to browse
                </div>
                <div className="text-sm text-foreground-muted max-w-xs mx-auto">
                  High-fidelity audio or video files (WAV, MP3, M4A, MP4, MOV)
                </div>
              </div>
              <input
                id="file-input"
                type="file"
                accept="audio/*,video/*"
                className="hidden"
                onChange={handleFileSelect}
              />
              <Button
                variant="outline"
                className="glass-lighter border-primary/20 pointer-events-none group-hover:bg-primary/20 transition-colors"
                type="button"
              >
                Choose File
              </Button>
            </div>

            {error && (
              <div className="mt-6 text-sm text-destructive bg-destructive/10 p-3 rounded-lg animate-in shake duration-500">{error}</div>
            )}
          </div>
        ) : (
          <div className="space-y-8 animate-in fade-in slide-in-from-top-4 duration-500">
            {/* File info */}
            <div className="p-6 glass-lighter rounded-2xl border-l-4 border-l-primary">
              <div className="font-semibold text-base text-foreground truncate max-w-md">{selectedFile.name}</div>
              <div className="text-sm text-foreground-muted">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </div>
            </div>

            {/* Timecode and framerate inputs */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 p-6 glass-lighter rounded-2xl">
              <div className="space-y-3">
                <Label htmlFor="start-timecode" className="text-xs font-bold uppercase tracking-widest text-foreground-muted">Start Timecode</Label>
                <Input
                  id="start-timecode"
                  type="text"
                  value={startTimecode}
                  onChange={(e) => setStartTimecode(e.target.value)}
                  placeholder="01:00:00:00"
                  disabled={isUploading}
                  className="bg-white/5 border-white/10 h-12 text-lg font-mono focus:ring-primary/50"
                />
                <p className="text-[10px] text-foreground-subtle font-medium">
                  FORMAT: HH:MM:SS:FF
                </p>
              </div>

              <div className="space-y-3">
                <Label htmlFor="framerate" className="text-xs font-bold uppercase tracking-widest text-foreground-muted">Framerate</Label>
                <div className="relative">
                  <select
                    id="framerate"
                    value={framerate}
                    onChange={(e) => setFramerate(Number(e.target.value))}
                    disabled={isUploading}
                    className="flex h-12 w-full appearance-none rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-base font-medium text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50"
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
                  <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-foreground-muted">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>
              </div>
            </div>

            {/* Upload button */}
            <div className="flex gap-4 pt-2">
              <Button
                onClick={handleUpload}
                disabled={isUploading}
                className="btn-primary h-14 text-lg flex-1 rounded-xl"
              >
                {isUploading ? (
                  <div className="flex items-center gap-2">
                    <div className="w-5 h-5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                    Uploading Analysis...
                  </div>
                ) : 'Continue'}
              </Button>
              <Button
                variant="outline"
                onClick={() => setSelectedFile(null)}
                disabled={isUploading}
                className="glass-lighter h-14 px-8 rounded-xl"
              >
                Cancel
              </Button>
            </div>

            {error && (
              <div className="text-sm text-destructive bg-destructive/10 p-4 rounded-xl animate-in shake duration-500">{error}</div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
