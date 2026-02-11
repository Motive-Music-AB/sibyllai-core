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

  // Title case helper
  const toTitleCase = (str: string) => str.replace(/\b\w/g, char => char.toUpperCase())

  // Limit items for display (with title case)
  const displayInstruments = (localCurated.instruments?.slice(0, 6) || []).map(toTitleCase)
  const displayGenres = (localCurated.genres?.slice(0, 2) || localCurated.genre?.slice(0, 2) || []).map(toTitleCase)
  const displayStyles = (localCurated.style?.slice(0, 2) || []).map(toTitleCase)
  const cueStatus = cue.project_context?.status
  const isMatched = cueStatus === 'matched'

  return (
    <Card
      ref={cardRef}
      onClick={handleClick}
      className={`p-4 cursor-pointer transition-all ${
        isActive
          ? 'border-primary border-2 shadow-lg'
          : 'border hover:border-primary/40'
      }`}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3 min-w-0">
          <span className="text-lg font-medium font-display shrink-0">Cue {index + 1}</span>
          <span className="text-sm text-foreground-muted font-mono shrink-0">{cue.start_tc} - {cue.end_tc}</span>
          {isMatched && (
            <span className="text-xs uppercase tracking-wide px-2 py-0.5 rounded-full bg-primary/20 text-primary shrink-0">
              Matched
            </span>
          )}
          {isPlaying && (
            <span className="flex items-center gap-0.5 ml-2 shrink-0">
              <span className="w-1 h-3 bg-primary rounded-full animate-[soundbar_0.5s_ease-in-out_infinite_alternate]" style={{ animationDelay: '0ms' }} />
              <span className="w-1 h-3 bg-primary rounded-full animate-[soundbar_0.5s_ease-in-out_infinite_alternate]" style={{ animationDelay: '150ms' }} />
              <span className="w-1 h-3 bg-primary rounded-full animate-[soundbar_0.5s_ease-in-out_infinite_alternate]" style={{ animationDelay: '300ms' }} />
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 text-xs shrink-0">
          {isActive && !isExpanded && (
            <span className="text-primary font-medium">SPACE to play</span>
          )}
        </div>
      </div>

      {/* Compact info grid */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-foreground-muted text-xs uppercase tracking-wide">BPM</span>
          <span className="font-medium">
            {typeof cue.musical_profile.bpm === 'number'
              ? Math.round(cue.musical_profile.bpm)
              : cue.musical_profile.bpm || '—'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-foreground-muted text-xs uppercase tracking-wide">Key</span>
          <span className="font-medium">{cue.musical_profile.key || '—'}</span>
        </div>
        {displayGenres.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-foreground-muted text-xs uppercase tracking-wide">Genre</span>
            <span className="font-medium">{displayGenres.join(', ')}</span>
          </div>
        )}
        {displayStyles.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-foreground-muted text-xs uppercase tracking-wide">Style</span>
            <span className="font-medium">{displayStyles.join(', ')}</span>
          </div>
        )}
        {displayInstruments.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-foreground-muted text-xs uppercase tracking-wide">Instruments</span>
            <span className="font-medium">{displayInstruments.join(', ')}</span>
          </div>
        )}
      </div>

      {/* Expanded edit section */}
      {isExpanded && (
        <div
          className="mt-4 p-4 glass rounded-xl space-y-6"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Instruments section */}
          <div>
            <div className="text-xs font-bold uppercase tracking-widest text-foreground-muted mb-2">
              Instruments
            </div>
            <div className="flex flex-wrap gap-2 mb-3">
              {detectedInstruments.map(([name, score]) => {
                const isSelected = localCurated.instruments?.includes(name)
                return (
                  <button
                    key={name}
                    onClick={() => toggleItem('instruments', name)}
                    className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${
                      isSelected
                        ? 'bg-primary/20 border-primary text-foreground'
                        : 'bg-white/5 border-white/10 text-foreground-muted hover:bg-white/10 hover:border-white/20'
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
            <div className="text-xs font-bold uppercase tracking-widest text-foreground-muted mb-2">
              Genres
            </div>
            <div className="flex flex-wrap gap-2 mb-3">
              {detectedGenres.map(([name, score]) => {
                const isSelected = localCurated.genres?.includes(name)
                return (
                  <button
                    key={name}
                    onClick={() => toggleItem('genres', name)}
                    className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${
                      isSelected
                        ? 'bg-primary/20 border-primary text-foreground'
                        : 'bg-white/5 border-white/10 text-foreground-muted hover:bg-white/10 hover:border-white/20'
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
            <div className="text-xs font-bold uppercase tracking-widest text-foreground-muted mb-2">
              Style
            </div>
            <div className="flex flex-wrap gap-2 mb-3">
              {detectedStyles.map(([name, score]) => {
                const isSelected = localCurated.style?.includes(name)
                return (
                  <button
                    key={name}
                    onClick={() => toggleItem('style', name)}
                    className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${
                      isSelected
                        ? 'bg-primary/20 border-primary text-foreground'
                        : 'bg-white/5 border-white/10 text-foreground-muted hover:bg-white/10 hover:border-white/20'
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
          <div className="flex justify-end gap-3 pt-4">
            <Button
              size="sm"
              variant="outline"
              onClick={handleCancel}
              disabled={isSaving}
              className="glass-lighter"
            >
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={handleSave}
              disabled={isSaving}
              className="btn-primary"
            >
              {isSaving ? 'Saving...' : 'Save'}
            </Button>
          </div>
        </div>
      )}
    </Card>
  )
}
