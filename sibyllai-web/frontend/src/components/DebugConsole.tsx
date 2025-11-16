import { useEffect, useState, useRef } from 'react'
import { Button } from '@/components/ui/button'

interface LogEntry {
  type: 'log' | 'error' | 'warn' | 'info'
  message: string
  timestamp: Date
}

export function DebugConsole() {
  // DISABLED: This component was causing infinite render loops
  // The component has been disabled to prevent issues with state updates
  // If you need to re-enable it, ensure proper cleanup of event listeners
  // and avoid state updates during render cycles
  return null
}
