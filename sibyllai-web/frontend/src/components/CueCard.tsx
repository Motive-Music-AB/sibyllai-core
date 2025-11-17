import { useEffect, useRef } from 'react'
import { Card } from '@/components/ui/card'
import { useAppStore } from '@/lib/store'
import type { Cue } from '@/lib/types'

interface CueCardProps {
  cue: Cue
  index: number
}

export function CueCard({ cue, index }: CueCardProps) {
  const cardRef = useRef<HTMLDivElement>(null)
  const { activeCueId, setActiveCueId } = useAppStore()

  const isActive = activeCueId === cue.id

  // Scroll into view when this card becomes active
  useEffect(() => {
    if (isActive && cardRef.current) {
      cardRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [isActive])

  const handleClick = () => {
    setActiveCueId(cue.id)
  }

  return (
    <Card
      ref={cardRef}
      onClick={handleClick}
      className={`p-4 space-y-2 cursor-pointer transition-all ${
        isActive
          ? 'border-blue-500 border-2 shadow-lg bg-blue-50'
          : 'border hover:border-gray-400 hover:shadow-md'
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="font-medium">
          Cue {index + 1}: {cue.start_tc} - {cue.end_tc}
        </div>
        {isActive && (
          <div className="text-xs text-blue-600 font-medium">
            Press SPACE to play
          </div>
        )}
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div>
          <span className="text-muted-foreground">BPM:</span>{' '}
          {typeof cue.musical_profile.bpm === 'number'
            ? cue.musical_profile.bpm.toFixed(1)
            : cue.musical_profile.bpm || 'N/A'}
        </div>
        <div>
          <span className="text-muted-foreground">Key:</span>{' '}
          {cue.musical_profile.key || 'N/A'}
        </div>
        <div>
          <span className="text-muted-foreground">Genre:</span>{' '}
          {cue.musical_profile.curated.genre.join(', ') || 'N/A'}
        </div>
        <div>
          <span className="text-muted-foreground">Instruments:</span>{' '}
          {cue.musical_profile.curated.instrumentation.join(', ') || 'N/A'}
        </div>
      </div>
    </Card>
  )
}
