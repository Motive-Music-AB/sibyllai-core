import { useEffect, useRef, useState } from 'react'
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
import type { Cue, CuratedAttributes, CueSection } from '@/lib/types'

/** Map energy label to greyscale shade (brutalist palette). */
function energyToGrey(label: string): string {
  const l = label.toLowerCase()
  if (l.includes('high') || l.includes('climactic') || l.includes('intense')) return '#2a2a2a'
  if (l.includes('building') || l.includes('moderate') || l.includes('medium')) return '#707070'
  if (l.includes('low') || l.includes('gentle') || l.includes('calm') || l.includes('soft')) return '#c0c0c0'
  return '#909090' // default mid-grey for unknown
}

interface CueCardProps {
  cue: Cue
  index: number
}

// Get items to display: detected items above threshold + curated items (even if below threshold)
function getDisplayItems(
  detected: Record<string, number> | undefined,
  curated: string[] | undefined,
  threshold: number
): [string, number][] {
  const detectedMap = detected || {}
  const curatedSet = new Set(curated || [])

  const items = Object.entries(detectedMap)
    .filter(([name, score]) => score >= threshold || curatedSet.has(name))
    .sort((a, b) => b[1] - a[1])

  const detectedNames = new Set(items.map(([name]) => name))
  for (const name of curatedSet) {
    if (!detectedNames.has(name)) {
      items.push([name, 0])
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
      setIsExpanded(!isExpanded)
    } else {
      setActiveCueId(cue.id)
    }
  }

  const toggleItem = (category: keyof CuratedAttributes, item: string) => {
    setLocalCurated((prev) => {
      const current = (prev[category] as string[]) || []
      const updated = current.includes(item)
        ? current.filter((i) => i !== item)
        : [...current, item]
      return { ...prev, [category]: updated }
    })
  }

  const addItem = (category: keyof CuratedAttributes, item: string) => {
    setLocalCurated((prev) => {
      const current = (prev[category] as string[]) || []
      if (current.includes(item)) return prev
      return { ...prev, [category]: [...current, item] }
    })
  }

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

  const handleCancel = () => {
    setLocalCurated(cue.musical_profile.curated)
    setIsExpanded(false)
  }

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

  const toTitleCase = (str: string) => str.replace(/\b\w/g, char => char.toUpperCase())

  const displayInstruments = (localCurated.instruments?.slice(0, 6) || []).map(toTitleCase)
  const displayGenres = (localCurated.genres?.slice(0, 2) || localCurated.genre?.slice(0, 2) || []).map(toTitleCase)
  const displayStyles = (localCurated.style?.slice(0, 2) || []).map(toTitleCase)
  const cueStatus = cue.project_context?.status
  const isMatched = cueStatus === 'matched'

  const sections: CueSection[] = cue.musical_profile.sections || []
  const hasSections = sections.length > 1

  return (
    <div
      ref={cardRef}
      onClick={handleClick}
      className="track-card"
      style={isActive ? {
        border: '2px solid #080808',
        background: '#fff',
      } : {
        cursor: 'pointer',
      }}
    >
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
          <span style={{ fontSize: '0.9rem', fontWeight: 800, textTransform: 'uppercase' }}>Cue {index + 1}</span>
          <span className="mono" style={{ fontSize: '0.7rem', opacity: 0.6 }}>{cue.start_tc} - {cue.end_tc}</span>
          {isMatched && (
            <span className="tag" style={{ background: '#080808', color: '#F0F0F0' }}>MATCHED</span>
          )}
          {isPlaying && (
            <span style={{ display: 'flex', alignItems: 'center', gap: 1, marginLeft: 4 }}>
              <span style={{ width: 2, height: 10, background: '#080808', borderRadius: 1, animation: 'soundbar 0.5s ease-in-out infinite alternate', animationDelay: '0ms' }} />
              <span style={{ width: 2, height: 10, background: '#080808', borderRadius: 1, animation: 'soundbar 0.5s ease-in-out infinite alternate', animationDelay: '150ms' }} />
              <span style={{ width: 2, height: 10, background: '#080808', borderRadius: 1, animation: 'soundbar 0.5s ease-in-out infinite alternate', animationDelay: '300ms' }} />
            </span>
          )}
        </div>
      </div>

      {/* Compact info grid */}
      <div style={{ fontSize: '0.75rem' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '4px 16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span className="label" style={{ fontSize: '0.6rem' }}>BPM</span>
            <span className="mono">
              {typeof cue.musical_profile.bpm === 'number'
                ? Math.round(cue.musical_profile.bpm)
                : cue.musical_profile.bpm || '—'}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <span className="label" style={{ fontSize: '0.6rem' }}>Key</span>
            <span className="mono">{cue.musical_profile.key || '—'}</span>
          </div>
          {displayGenres.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span className="label" style={{ fontSize: '0.6rem' }}>Genre</span>
              <span className="mono">{displayGenres.join(', ')}</span>
            </div>
          )}
          {displayStyles.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span className="label" style={{ fontSize: '0.6rem' }}>Style</span>
              <span className="mono">{displayStyles.join(', ')}</span>
            </div>
          )}
        </div>
        {displayInstruments.length > 0 && (
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 4, marginTop: 4 }}>
            <span className="label" style={{ fontSize: '0.6rem', flexShrink: 0, marginTop: 1 }}>Inst</span>
            <span className="mono">{displayInstruments.join(', ')}</span>
          </div>
        )}
      </div>

      {/* Section structure bar */}
      {hasSections && (
        <div style={{ marginTop: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
            <span className="label" style={{ fontSize: '0.55rem' }}>STRUCTURE</span>
            <span className="mono" style={{ fontSize: '0.55rem', opacity: 0.4 }}>{sections.length} sections</span>
          </div>
          <div style={{ display: 'flex', height: 14, borderRadius: 3, overflow: 'hidden', border: '1px solid rgba(8,8,8,0.15)' }}>
            {sections.map((sec) => {
              const totalDuration = cue.end - cue.start
              const widthPercent = totalDuration > 0 ? (sec.duration / totalDuration) * 100 : 0
              const bg = energyToGrey(sec.energy_label)
              return (
                <div
                  key={sec.index}
                  title={`${sec.energy_label} · ${sec.duration.toFixed(1)}s${sec.bpm ? ` · ${Math.round(sec.bpm)} BPM` : ''}${sec.key ? ` · ${sec.key}` : ''}`}
                  style={{
                    width: `${widthPercent}%`,
                    minWidth: 2,
                    background: bg,
                    borderRight: sec.index < sections.length - 1 ? '1px solid rgba(255,255,255,0.4)' : 'none',
                    cursor: 'default',
                  }}
                />
              )
            })}
          </div>
        </div>
      )}

      {/* Expanded edit section */}
      {isExpanded && (
        <div
          style={{
            marginTop: 12,
            padding: 12,
            border: '1px solid #080808',
            borderRadius: 8,
            background: '#F8F8F8',
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Instruments section */}
          <div style={{ marginBottom: 16 }}>
            <div className="label" style={{ fontSize: '0.6rem', marginBottom: 6 }}>
              Instruments
            </div>
            <div className="tag-toggle-list">
              {detectedInstruments.map(([name, score]) => {
                const isSelected = localCurated.instruments?.includes(name)
                return (
                  <button
                    key={name}
                    onClick={() => toggleItem('instruments', name)}
                    className={`tag-toggle ${isSelected ? 'selected' : ''}`}
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
          <div style={{ marginBottom: 16 }}>
            <div className="label" style={{ fontSize: '0.6rem', marginBottom: 6 }}>
              Genres
            </div>
            <div className="tag-toggle-list">
              {detectedGenres.map(([name, score]) => {
                const isSelected = localCurated.genres?.includes(name)
                return (
                  <button
                    key={name}
                    onClick={() => toggleItem('genres', name)}
                    className={`tag-toggle ${isSelected ? 'selected' : ''}`}
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
          <div style={{ marginBottom: 16 }}>
            <div className="label" style={{ fontSize: '0.6rem', marginBottom: 6 }}>
              Style
            </div>
            <div className="tag-toggle-list">
              {detectedStyles.map(([name, score]) => {
                const isSelected = localCurated.style?.includes(name)
                return (
                  <button
                    key={name}
                    onClick={() => toggleItem('style', name)}
                    className={`tag-toggle ${isSelected ? 'selected' : ''}`}
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
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, paddingTop: 8, borderTop: '1px solid rgba(8,8,8,0.1)' }}>
            <button className="btn-pill" style={{ height: 28, fontSize: '0.65rem' }} onClick={handleCancel} disabled={isSaving}>
              Cancel
            </button>
            <button className="btn-pill primary" style={{ height: 28, fontSize: '0.65rem' }} onClick={handleSave} disabled={isSaving}>
              {isSaving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
