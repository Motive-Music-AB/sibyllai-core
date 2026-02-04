import { useEffect, useRef, useState } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/lib/store'
import { api } from '@/lib/api'
import { AddItemCombobox } from './AddItemCombobox'
import {
  YAMNET_INSTRUMENTS,
  YAMNET_GENRES,
  CLAP_STYLES,
  INSTRUMENT_THRESHOLD,
  GENRE_THRESHOLD,
  STYLE_THRESHOLD,
} from '@/lib/constants'
import type { Cue, CuratedAttributes } from '@/lib/types'

interface CueCardProps {
  cue: Cue
  index: number
}

// Helper to filter detected items above threshold
function getDetectedAboveThreshold(
  detected: Record<string, number> | undefined,
  threshold: number
) {
  if (!detected) return []
  return Object.entries(detected)
    .filter(([_, score]) => score >= threshold)
    .sort((a, b) => b[1] - a[1])
}

// Get items to display: detected items above threshold + curated items (even if below threshold)
// This ensures users can deselect items that were auto-curated but are below display threshold
function getDisplayItems(
  detected: Record<string, number> | undefined,
  curated: string[] | undefined,
  threshold: number
): [string, number][] {
  const detectedMap = detected || {}
  const curatedSet = new Set(curated || [])

  // Start with detected items above threshold
  const items = Object.entries(detectedMap)
    .filter(([name, score]) => score >= threshold || curatedSet.has(name))
    .sort((a, b) => b[1] - a[1])

  // Add any curated items not in detected (manually added)
  const detectedNames = new Set(items.map(([name]) => name))
  for (const name of curatedSet) {
    if (!detectedNames.has(name)) {
      items.push([name, 0]) // Score 0 for manually added items
    }
  }

  return items
}

export function CueCard({ cue, index }: CueCardProps) {
  const cardRef = useRef<HTMLDivElement>(null)
  const { activeCueId, setActiveCueId, playingCueId, sessionId, updateCueInProject } = useAppStore()

  const isActive = activeCueId === cue.id
  const isPlaying = playingCueId === cue.id
  const [isExpanded, setIsExpanded] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [localCurated, setLocalCurated] = useState<CuratedAttributes>(cue.musical_profile.curated)

  // Reset local state when cue changes
  useEffect(() => {
    setLocalCurated(cue.musical_profile.curated)
  }, [cue.musical_profile.curated])

  // Scroll into view when this card becomes active
  useEffect(() => {
    if (isActive && cardRef.current) {
      cardRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [isActive])

  const handleClick = () => {
    if (isActive) {
      // Second click on active cue - toggle expand
      setIsExpanded(!isExpanded)
    } else {
      // First click - just select
      setActiveCueId(cue.id)
    }
  }

  // Toggle item in/out of curated list
  const toggleItem = (category: keyof CuratedAttributes, item: string) => {
    setLocalCurated((prev) => {
      const current = (prev[category] as string[]) || []
      const updated = current.includes(item)
        ? current.filter((i) => i !== item)
        : [...current, item]
      return { ...prev, [category]: updated }
    })
  }

  // Add item to curated list
  const addItem = (category: keyof CuratedAttributes, item: string) => {
    setLocalCurated((prev) => {
      const current = (prev[category] as string[]) || []
      if (current.includes(item)) return prev
      return { ...prev, [category]: [...current, item] }
    })
  }

  // Save changes to backend
  const handleSave = async () => {
    if (!sessionId) return
    setIsSaving(true)
    try {
      await api.updateCue(sessionId, cue.id, {
        instruments: localCurated.instruments,
        genres: localCurated.genres,
        style: localCurated.style,
        moods: localCurated.moods,
      })
      updateCueInProject(cue.id, localCurated)
      setIsExpanded(false)
    } catch (error) {
      console.error('Failed to save cue:', error)
    } finally {
      setIsSaving(false)
    }
  }

  // Cancel and reset
  const handleCancel = () => {
    setLocalCurated(cue.musical_profile.curated)
    setIsExpanded(false)
  }

  // Get items to display for each category
  // Includes detected items above threshold + curated items (so they can be deselected)
  const detectedInstruments = getDisplayItems(
    cue.musical_profile.detected?.instruments_yamnet,
    localCurated.instruments,
    INSTRUMENT_THRESHOLD
  )
  const detectedGenres = getDisplayItems(
    cue.musical_profile.detected?.genres_yamnet,
    localCurated.genres,
    GENRE_THRESHOLD
  )
  const detectedStyles = getDisplayItems(
    cue.musical_profile.detected?.clap_style,
    localCurated.style,
    STYLE_THRESHOLD
  )

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
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 font-medium">
          <span>Cue {index + 1}: {cue.start_tc} - {cue.end_tc}</span>
          {isPlaying && (
            <span className="flex items-center gap-0.5">
              <span className="w-1 h-3 bg-blue-500 rounded-full animate-[soundbar_0.5s_ease-in-out_infinite_alternate]" style={{ animationDelay: '0ms' }} />
              <span className="w-1 h-3 bg-blue-500 rounded-full animate-[soundbar_0.5s_ease-in-out_infinite_alternate]" style={{ animationDelay: '150ms' }} />
              <span className="w-1 h-3 bg-blue-500 rounded-full animate-[soundbar_0.5s_ease-in-out_infinite_alternate]" style={{ animationDelay: '300ms' }} />
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isActive && !isExpanded && (
            <div className="text-xs text-blue-600 font-medium">
              Press SPACE to play
            </div>
          )}
          {isActive && !isExpanded && (
            <div className="text-xs text-muted-foreground">
              Click to edit
            </div>
          )}
        </div>
      </div>

      {/* Summary row (always visible) */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
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
          <span className="text-muted-foreground">Instruments:</span>{' '}
          {localCurated.instruments?.join(', ') || 'N/A'}
        </div>
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
        <div>
          <span className="text-muted-foreground">Genre:</span>{' '}
          {localCurated.genres?.join(', ') || localCurated.genre?.join(', ') || 'N/A'}
        </div>
        <div>
          <span className="text-muted-foreground">Style:</span>{' '}
          {localCurated.style?.join(', ') || 'N/A'}
        </div>
      </div>

      {/* Expanded edit section */}
      {isExpanded && (
        <div
          className="mt-4 pt-4 border-t space-y-4"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Instruments section */}
          <div>
            <div className="text-sm font-medium text-gray-700 mb-2">
              Instruments (select to keep)
            </div>
            <div className="flex flex-wrap gap-2 mb-2">
              {detectedInstruments.map(([name, score]) => {
                const isSelected = localCurated.instruments?.includes(name)
                return (
                  <button
                    key={name}
                    onClick={() => toggleItem('instruments', name)}
                    className={`px-2 py-1 text-xs rounded border transition-colors ${
                      isSelected
                        ? 'bg-blue-100 border-blue-400 text-blue-800'
                        : 'bg-gray-50 border-gray-200 text-gray-500 hover:bg-gray-100'
                    }`}
                  >
                    {name} ({score.toFixed(3)})
                  </button>
                )
              })}
            </div>
            <AddItemCombobox
              suggestions={YAMNET_INSTRUMENTS}
              existingItems={localCurated.instruments}
              onAdd={(item) => addItem('instruments', item)}
              placeholder="Add instrument..."
            />
          </div>

          {/* Genres section */}
          <div>
            <div className="text-sm font-medium text-gray-700 mb-2">
              Genres (select to keep)
            </div>
            <div className="flex flex-wrap gap-2 mb-2">
              {detectedGenres.map(([name, score]) => {
                const isSelected = localCurated.genres?.includes(name)
                return (
                  <button
                    key={name}
                    onClick={() => toggleItem('genres', name)}
                    className={`px-2 py-1 text-xs rounded border transition-colors ${
                      isSelected
                        ? 'bg-green-100 border-green-400 text-green-800'
                        : 'bg-gray-50 border-gray-200 text-gray-500 hover:bg-gray-100'
                    }`}
                  >
                    {name} ({score.toFixed(3)})
                  </button>
                )
              })}
            </div>
            <AddItemCombobox
              suggestions={YAMNET_GENRES}
              existingItems={localCurated.genres}
              onAdd={(item) => addItem('genres', item)}
              placeholder="Add genre..."
            />
          </div>

          {/* Style section */}
          <div>
            <div className="text-sm font-medium text-gray-700 mb-2">
              Style (select to keep)
            </div>
            <div className="flex flex-wrap gap-2 mb-2">
              {detectedStyles.map(([name, score]) => {
                const isSelected = localCurated.style?.includes(name)
                return (
                  <button
                    key={name}
                    onClick={() => toggleItem('style', name)}
                    className={`px-2 py-1 text-xs rounded border transition-colors ${
                      isSelected
                        ? 'bg-purple-100 border-purple-400 text-purple-800'
                        : 'bg-gray-50 border-gray-200 text-gray-500 hover:bg-gray-100'
                    }`}
                  >
                    {name} ({score.toFixed(3)})
                  </button>
                )
              })}
            </div>
            <AddItemCombobox
              suggestions={CLAP_STYLES}
              existingItems={localCurated.style}
              onAdd={(item) => addItem('style', item)}
              placeholder="Add style..."
            />
          </div>

          {/* Action buttons */}
          <div className="flex justify-end gap-2 pt-2">
            <Button
              size="sm"
              variant="outline"
              onClick={handleCancel}
              disabled={isSaving}
            >
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={handleSave}
              disabled={isSaving}
            >
              {isSaving ? 'Saving...' : 'Save'}
            </Button>
          </div>
        </div>
      )}
    </Card>
  )
}
