import { useEffect, useState, useRef } from 'react'
import { Button } from '@/components/ui/button'

interface LogEntry {
  type: 'log' | 'error' | 'warn' | 'info'
  message: string
  timestamp: Date
}

export function DebugConsole() {
  const [isVisible, setIsVisible] = useState(false)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const logsEndRef = useRef<HTMLDivElement>(null)
  const pendingLogsRef = useRef<LogEntry[]>([])
  const maxLogs = 100 // Keep last 100 logs

  useEffect(() => {
    // Intercept console methods
    const originalLog = console.log
    const originalError = console.error
    const originalWarn = console.warn
    const originalInfo = console.info

    const addLog = (type: LogEntry['type'], args: any[]) => {
      const message = args.map(arg =>
        typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
      ).join(' ')

      const logEntry = { type, message, timestamp: new Date() }

      setLogs(prev => {
        const newLogs = [...prev, logEntry]
        return newLogs.slice(-maxLogs) // Keep only last maxLogs entries
      })

      // Add to pending logs for backend sync
      pendingLogsRef.current.push(logEntry)
    }

    console.log = (...args) => {
      originalLog(...args)
      addLog('log', args)
    }

    console.error = (...args) => {
      originalError(...args)
      addLog('error', args)
    }

    console.warn = (...args) => {
      originalWarn(...args)
      addLog('warn', args)
    }

    console.info = (...args) => {
      originalInfo(...args)
      addLog('info', args)
    }

    // Keyboard shortcut: Press 'D' to toggle debug console
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'd' && !e.ctrlKey && !e.metaKey && !e.altKey) {
        // Only toggle if not typing in an input
        if (
          document.activeElement?.tagName !== 'INPUT' &&
          document.activeElement?.tagName !== 'TEXTAREA'
        ) {
          setIsVisible(prev => !prev)
        }
      }
    }

    window.addEventListener('keydown', handleKeyPress)

    // Send pending logs to backend every 2 seconds
    const syncInterval = setInterval(async () => {
      if (pendingLogsRef.current.length > 0) {
        const logsToSend = [...pendingLogsRef.current]
        pendingLogsRef.current = []

        try {
          await fetch('http://localhost:8001/api/debug-logs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(
              logsToSend.map(log => ({
                type: log.type,
                message: log.message,
                timestamp: log.timestamp.toISOString()
              }))
            )
          })
        } catch (error) {
          // Silently fail - don't want to spam console with sync errors
        }
      }
    }, 2000)

    return () => {
      // Restore original console methods
      console.log = originalLog
      console.error = originalError
      console.warn = originalWarn
      console.info = originalInfo
      window.removeEventListener('keydown', handleKeyPress)
      clearInterval(syncInterval)
    }
  }, [])

  // Auto-scroll to bottom when new logs appear
  useEffect(() => {
    if (isVisible) {
      logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, isVisible])

  if (!isVisible) {
    return (
      <div className="fixed bottom-4 right-4 z-50">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsVisible(true)}
          className="shadow-lg bg-gray-900 text-white hover:bg-gray-800 border-gray-700"
        >
          Show Debug (D)
        </Button>
      </div>
    )
  }

  return (
    <div className="fixed bottom-4 right-4 z-50 w-[600px] max-h-[400px] flex flex-col bg-gray-900 text-gray-100 rounded-lg shadow-2xl border border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-700 bg-gray-800">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">Debug Console</span>
          <span className="text-xs text-gray-400">({logs.length} logs)</span>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setLogs([])}
            className="h-7 px-2 text-xs text-gray-300 hover:text-white hover:bg-gray-700"
          >
            Clear
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsVisible(false)}
            className="h-7 px-2 text-xs text-gray-300 hover:text-white hover:bg-gray-700"
          >
            Hide (D)
          </Button>
        </div>
      </div>

      {/* Log entries */}
      <div className="flex-1 overflow-y-auto px-4 py-2 font-mono text-xs">
        {logs.length === 0 ? (
          <div className="text-gray-500 text-center py-8">
            No logs yet. Console output will appear here.
          </div>
        ) : (
          logs.map((log, index) => {
            const time = log.timestamp.toLocaleTimeString('en-US', {
              hour12: false,
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
              fractionalSecondDigits: 3
            })

            let textColor = 'text-gray-300'
            let prefix = '→'

            switch (log.type) {
              case 'error':
                textColor = 'text-red-400'
                prefix = '✗'
                break
              case 'warn':
                textColor = 'text-yellow-400'
                prefix = '⚠'
                break
              case 'info':
                textColor = 'text-blue-400'
                prefix = 'ℹ'
                break
            }

            return (
              <div
                key={index}
                className={`py-1 border-b border-gray-800 last:border-0 ${textColor}`}
              >
                <div className="flex gap-2">
                  <span className="text-gray-500 shrink-0">{time}</span>
                  <span className="shrink-0">{prefix}</span>
                  <span className="break-all">{log.message}</span>
                </div>
              </div>
            )
          })
        )}
        <div ref={logsEndRef} />
      </div>

      {/* Footer hint */}
      <div className="px-4 py-1 border-t border-gray-700 bg-gray-800 text-xs text-gray-400">
        Press <kbd className="px-1 bg-gray-700 rounded">D</kbd> to toggle
      </div>
    </div>
  )
}
