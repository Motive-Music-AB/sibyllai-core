import { useState, useEffect, useRef, useCallback, type DragEvent, type ChangeEvent } from 'react'
import { useAppStore } from '@/lib/store'
import { api } from '@/lib/api'
import { secondsToTimecode } from '@/lib/timecode'
import { WaveformViewer } from '@/components/WaveformViewer'
import type { Cue, CueSection, CuratedAttributes } from '@/lib/types'
import {
  YAMNET_INSTRUMENTS,
  YAMNET_GENRES,
  CLAP_STYLES,
  INSTRUMENT_THRESHOLD,
  GENRE_THRESHOLD,
  STYLE_THRESHOLD,
} from '@/lib/constants'
import { AddItemCombobox } from '@/components/AddItemCombobox'

/* ═══════════════════════════════════════════════════════════════════════════
   DESIGN TOKENS
   ═══════════════════════════════════════════════════════════════════════════ */

const CANVAS = '#D6D6D4'
const BENTO = '#EBEBEB'
const LIST_BG = '#F3F3F1'
const ACCENT = '#FFD659'
const TEXT_SUB = '#4A4A4A'

const titleHeavy: React.CSSProperties = { fontWeight: 800, letterSpacing: '-0.04em' }
const labelBold: React.CSSProperties = {
  fontWeight: 700, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em', color: TEXT_SUB,
}

/* ═══════════════════════════════════════════════════════════════════════════
   ICONS
   ═══════════════════════════════════════════════════════════════════════════ */

function PlayIcon({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M8 5.14v13.72a1 1 0 0 0 1.5.86l11.72-6.86a1 1 0 0 0 0-1.72L9.5 4.28A1 1 0 0 0 8 5.14z" />
    </svg>
  )
}

function ArrowLeftIcon({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 12H5M12 19l-7-7 7-7" />
    </svg>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   HELPERS
   ═══════════════════════════════════════════════════════════════════════════ */

const MIX_TYPE_DEFAULTS = {
  clean_mx: { musicThreshold: 0.5, silenceThreshold: 0.0005 },
  full_mix: { musicThreshold: 0.2, silenceThreshold: 0.01 },
}

function fmtDur(sec: number): string {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function fmtPrecision(v: number): string {
  if (v < 0.001) return v.toFixed(6)
  if (v < 0.01) return v.toFixed(4)
  return v.toFixed(2)
}

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
    if (!detectedNames.has(name)) items.push([name, 0])
  }
  return items
}

/* ═══════════════════════════════════════════════════════════════════════════
   ENERGY → COLOR MAPPING
   ═══════════════════════════════════════════════════════════════════════════ */

const ENERGY_COLORS: Record<string, { bg: string; fg: string; icon: string }> = {
  'high energy':       { bg: '#1A1A1A', fg: '#fff', icon: '⚡' },
  'climactic':         { bg: '#2D2D2D', fg: '#fff', icon: '🔺' },
  'aggressive':        { bg: '#3A1A1A', fg: '#fff', icon: '🔥' },
  'building tension':  { bg: '#4A4A4A', fg: '#fff', icon: '↗' },
  'low energy':        { bg: '#B8B8B8', fg: '#000', icon: '○' },
  'gentle':            { bg: '#D4D4D4', fg: '#000', icon: '~' },
}

function getEnergyStyle(label: string) {
  const lower = label.toLowerCase()
  for (const [key, val] of Object.entries(ENERGY_COLORS)) {
    if (lower.includes(key)) return val
  }
  return { bg: '#909090', fg: '#fff', icon: '•' }
}

/** Normalize CLAP energy scores to 0–1 range for bar display.
 *  Raw CLAP scores can be negative; we shift and scale to [0,1]. */
function normalizeEnergy(clap: Record<string, number>): Record<string, number> {
  const vals = Object.values(clap)
  const min = Math.min(...vals)
  const max = Math.max(...vals)
  const range = max - min || 1
  const out: Record<string, number> = {}
  for (const [k, v] of Object.entries(clap)) {
    out[k] = (v - min) / range
  }
  return out
}

/* ═══════════════════════════════════════════════════════════════════════════
   SECTION STRUCTURE VISUALIZATION
   ═══════════════════════════════════════════════════════════════════════════ */

function SectionStructure({
  sections,
  cueDuration,
  isActive,
  linkedSectionIndex,
  onSectionHover,
  onSectionSelect,
}: {
  sections: CueSection[]
  cueDuration: number
  isActive: boolean
  linkedSectionIndex?: number | null
  onSectionHover?: (idx: number | null) => void
  onSectionSelect?: (idx: number | null) => void
}) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null)
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null)

  if (sections.length <= 1) return null

  const effectiveIdx = hoveredIdx ?? linkedSectionIndex ?? selectedIdx ?? 0
  const focusedSec = sections[effectiveIdx]
  const setHover = (idx: number | null) => {
    setHoveredIdx(idx)
    onSectionHover?.(idx)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, paddingTop: 4 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={labelBold}>Structure</span>
        <span style={{ fontSize: 10, fontWeight: 600, color: TEXT_SUB }}>
          {sections.length} sections
        </span>
      </div>

      {/* Timeline bar — proportional width, energy-colored, with BPM/Key inside */}
      <div style={{
        display: 'flex', height: 48, borderRadius: 10, overflow: 'hidden',
        border: '1px solid rgba(0,0,0,0.1)',
      }}>
        {sections.map((sec) => {
          const pct = cueDuration > 0 ? (sec.duration / cueDuration) * 100 : 0
          const style = getEnergyStyle(sec.energy_label)
          const isHov = hoveredIdx === sec.index
          const wide = pct > 15
          return (
            <div
              key={sec.index}
              onMouseEnter={() => setHover(sec.index)}
              onMouseLeave={() => setHover(null)}
              onClick={() => {
                const next = selectedIdx === sec.index ? null : sec.index
                setSelectedIdx(next)
                onSectionSelect?.(next)
              }}
              style={{
                width: `${pct}%`, minWidth: 24, backgroundColor: style.bg,
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                borderRight: sec.index < sections.length - 1 ? '1px solid rgba(255,255,255,0.15)' : 'none',
                cursor: 'pointer',
                opacity: effectiveIdx !== null && !isHov ? 0.4 : 1,
                transition: 'opacity 0.15s ease, box-shadow 0.15s ease',
                padding: '0 4px', overflow: 'hidden',
                boxShadow: (linkedSectionIndex === sec.index || selectedIdx === sec.index) ? 'inset 0 0 0 2px rgba(255,255,255,0.7)' : 'none',
              }}
            >
              <span style={{ fontSize: 8, fontWeight: 800, color: style.fg, letterSpacing: '0.05em', textTransform: 'uppercase', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', lineHeight: 1.2 }}>
                {wide ? sec.energy_label : style.icon}
              </span>
              {wide && sec.bpm && (
                <span style={{ fontSize: 7, fontWeight: 600, color: style.fg, opacity: 0.7, whiteSpace: 'nowrap', lineHeight: 1.4 }}>
                  {Math.round(sec.bpm)} · {sec.key || '—'}
                </span>
              )}
            </div>
          )
        })}
      </div>

      {/* Compact summary row — BPM + Key per section */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
        {sections.map((sec) => {
          const style = getEnergyStyle(sec.energy_label)
          const isHov = hoveredIdx === sec.index
          return (
            <div
              key={sec.index}
              onMouseEnter={() => setHover(sec.index)}
              onMouseLeave={() => setHover(null)}
              onClick={() => {
                const next = selectedIdx === sec.index ? null : sec.index
                setSelectedIdx(next)
                onSectionSelect?.(next)
              }}
              style={{
                display: 'flex', alignItems: 'center', gap: 5, padding: '3px 8px',
                borderRadius: 8,
                backgroundColor: (isHov || linkedSectionIndex === sec.index || selectedIdx === sec.index) ? '#fff' : (isActive ? '#F9F9F7' : LIST_BG),
                border: (isHov || linkedSectionIndex === sec.index || selectedIdx === sec.index) ? '1px solid rgba(0,0,0,0.15)' : '1px solid transparent',
                transition: 'all 0.12s ease', cursor: 'pointer',
              }}
            >
              <span style={{
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                width: 16, height: 16, borderRadius: 4, backgroundColor: style.bg, color: style.fg,
                fontSize: 8, fontWeight: 800,
              }}>
                {sec.index + 1}
              </span>
              {sec.bpm && <span style={{ fontSize: 10, fontWeight: 700 }}>{Math.round(sec.bpm)}</span>}
              {sec.key && <span style={{ fontSize: 10, fontWeight: 600, color: TEXT_SUB }}>{sec.key}</span>}
            </div>
          )
        })}
      </div>

      {/* Focus detail — persists for selected/linked section, hover temporarily overrides */}
      <div style={{
        backgroundColor: isActive ? '#fff' : LIST_BG, borderRadius: 12, padding: '10px 14px',
        border: '1px solid rgba(0,0,0,0.1)', transition: 'all 0.15s ease',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              width: 18, height: 18, borderRadius: 5, backgroundColor: '#000', color: '#fff',
              fontSize: 9, fontWeight: 800,
            }}>
              {focusedSec.index + 1}
            </span>
            <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.03em' }}>
              {focusedSec.energy_label}
            </span>
          </div>
          <span style={{ fontSize: 10, fontWeight: 600, color: TEXT_SUB }}>
            {String(Math.floor(focusedSec.duration / 60)).padStart(2, '0')}:{String(Math.floor(focusedSec.duration % 60)).padStart(2, '0')}
          </span>
        </div>

        <div style={{ display: 'flex', gap: 10, marginBottom: 8 }}>
          <div style={{
            flex: 1, borderRadius: 8, border: '1px solid rgba(0,0,0,0.08)',
            padding: '4px 8px', backgroundColor: 'rgba(255,255,255,0.45)',
          }}>
            <span style={{ ...labelBold, fontSize: 8 }}>BPM</span>
            <div style={{ fontSize: 12, fontWeight: 800 }}>{focusedSec.bpm ? Math.round(focusedSec.bpm) : '—'}</div>
          </div>
          <div style={{
            flex: 1, borderRadius: 8, border: '1px solid rgba(0,0,0,0.08)',
            padding: '4px 8px', backgroundColor: 'rgba(255,255,255,0.45)',
          }}>
            <span style={{ ...labelBold, fontSize: 8 }}>Key</span>
            <div style={{ fontSize: 12, fontWeight: 800 }}>{focusedSec.key || '—'}</div>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {Object.entries(focusedSec.clap_energy)
            .sort((a, b) => b[1] - a[1])
            .map(([label, rawScore]) => {
              const norm = normalizeEnergy(focusedSec.clap_energy)[label]
              const isTop = label === focusedSec.energy_label
              return (
                <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{
                    width: 92, fontSize: 8, fontWeight: isTop ? 800 : 600,
                    textTransform: 'uppercase', letterSpacing: '0.05em',
                    color: isTop ? '#000' : TEXT_SUB, textAlign: 'right',
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {label}
                  </span>
                  <div style={{ flex: 1, height: 4, backgroundColor: 'rgba(0,0,0,0.08)', borderRadius: 999, overflow: 'hidden' }}>
                    <div style={{
                      height: '100%', borderRadius: 999,
                      width: `${(norm * 100).toFixed(0)}%`,
                      backgroundColor: isTop ? '#000' : 'rgba(0,0,0,0.25)',
                    }} />
                  </div>
                  <span style={{ width: 30, fontSize: 8, fontWeight: isTop ? 800 : 600, color: TEXT_SUB, fontFamily: 'monospace' }}>
                    {rawScore.toFixed(2)}
                  </span>
                </div>
              )
            })}
        </div>
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   ANALYSIS CUE CARD (new design)
   ═══════════════════════════════════════════════════════════════════════════ */

function AnalysisCueCard({
  cue, index, isActive, onClick, linkedSectionIndex, onSectionHover, onSectionSelect,
}: {
  cue: Cue; index: number; isActive: boolean; onClick: () => void
  linkedSectionIndex?: number | null
  onSectionHover?: (idx: number | null) => void
  onSectionSelect?: (idx: number | null) => void
}) {
  const [hovered, setHovered] = useState(false)
  const { sessionId, updateCueInProject } = useAppStore()
  const [isEditing, setIsEditing] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [localCurated, setLocalCurated] = useState<CuratedAttributes>(cue.musical_profile.curated)

  useEffect(() => {
    setLocalCurated(cue.musical_profile.curated)
    setIsEditing(false)
  }, [cue.id]) // eslint-disable-line react-hooks/exhaustive-deps

  const { framerate, startTimecode } = useAppStore()
  const tc0 = secondsToTimecode(cue.start, framerate, startTimecode)
  const tc1 = secondsToTimecode(cue.end, framerate, startTimecode)
  const dur = fmtDur(cue.end - cue.start)

  const elevated = isActive || hovered

  const toggleItem = (category: keyof CuratedAttributes, item: string) => {
    setLocalCurated((prev) => {
      const current = (prev[category] as string[]) || []
      const updated = current.includes(item) ? current.filter(i => i !== item) : [...current, item]
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
      setIsEditing(false)
    } catch { /* */ }
    setIsSaving(false)
  }

  const detectedInstruments = getDisplayItems(cue.musical_profile.detected?.instruments_yamnet, localCurated.instruments, INSTRUMENT_THRESHOLD)
  const detectedGenres = getDisplayItems(cue.musical_profile.detected?.genres_yamnet, localCurated.genres, GENRE_THRESHOLD)
  const detectedStyles = getDisplayItems(cue.musical_profile.detected?.clap_style, localCurated.style, STYLE_THRESHOLD)

  function TagPill({ label, isCurated }: { label: string; isCurated: boolean }) {
    return (
      <span style={{
        backgroundColor: isCurated ? '#000' : LIST_BG,
        color: isCurated ? '#fff' : '#000',
        padding: '4px 10px', borderRadius: 999, fontSize: 10, fontWeight: 700,
        boxShadow: isCurated ? '0 1px 2px rgba(0,0,0,0.1)' : 'none',
      }}>
        {label.toUpperCase()}
      </span>
    )
  }

  function TagToggle({ name, score, selected, onToggle }: { name: string; score: number; selected: boolean; onToggle: () => void }) {
    return (
      <button onClick={onToggle} style={{
        padding: '4px 10px', borderRadius: 999, fontSize: 10, fontWeight: 700, cursor: 'pointer',
        backgroundColor: selected ? '#000' : LIST_BG, color: selected ? '#fff' : '#000',
        border: 'none', transition: 'all 0.15s ease',
      }}>
        {name} ({score.toFixed(3)})
      </button>
    )
  }

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        backgroundColor: isActive ? '#fff' : BENTO,
        borderRadius: 28, padding: 24,
        borderLeft: isActive ? `4px solid ${ACCENT}` : '4px solid transparent',
        boxShadow: elevated ? '0 12px 24px -8px rgba(0,0,0,0.15), 0 4px 8px -4px rgba(0,0,0,0.1)' : 'none',
        transform: elevated ? 'translateY(-2px)' : 'translateY(0)',
        transition: 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)',
        border: isActive ? undefined : hovered ? '1px solid rgba(0,0,0,0.1)' : '1px solid transparent',
        cursor: 'pointer',
        display: 'flex', flexDirection: 'column', gap: 24,
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid rgba(0,0,0,0.05)', paddingBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ backgroundColor: '#000', color: '#fff', padding: '4px 10px', borderRadius: 999, fontSize: 11, fontWeight: 700, boxShadow: '0 1px 2px rgba(0,0,0,0.1)' }}>#{index + 1}</span>
          <span style={{ fontFamily: 'monospace', fontSize: 12, fontWeight: 500, color: isActive ? '#000' : TEXT_SUB }}>{tc0} — {tc1}</span>
        </div>
        <span style={{ fontSize: 12, color: TEXT_SUB, fontWeight: 700 }}>{dur}</span>
      </div>

      {/* BPM & Key */}
      <div style={{ display: 'flex', gap: 48 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={labelBold}>BPM</span>
          <span style={{ fontSize: 30, ...titleHeavy, lineHeight: 1 }}>{cue.musical_profile.bpm?.toFixed(0) || '—'}</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={labelBold}>Key</span>
          <span style={{ fontSize: 30, ...titleHeavy, lineHeight: 1 }}>{cue.musical_profile.key || '—'}</span>
        </div>
      </div>

      {/* Tags */}
      {!isEditing ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <span style={labelBold}>Instruments</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {localCurated.instruments.map(i => <TagPill key={i} label={i} isCurated={true} />)}
              {localCurated.instruments.length === 0 && <span style={{ fontSize: 11, color: TEXT_SUB }}>None detected</span>}
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <span style={labelBold}>Genre / Style</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {[...localCurated.genres, ...localCurated.style].map(t => <TagPill key={t} label={t} isCurated={localCurated.genres.includes(t)} />)}
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <span style={labelBold}>Mood</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {localCurated.moods.map(m => <TagPill key={m} label={m} isCurated={true} />)}
              {localCurated.moods.length === 0 && <span style={{ fontSize: 11, color: TEXT_SUB }}>None detected</span>}
            </div>
          </div>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }} onClick={(e) => e.stopPropagation()}>
          {/* Instruments edit */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <span style={labelBold}>Instruments</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {detectedInstruments.map(([name, score]) => (
                <TagToggle key={name} name={name} score={score} selected={localCurated.instruments.includes(name)} onToggle={() => toggleItem('instruments', name)} />
              ))}
            </div>
            <AddItemCombobox suggestions={YAMNET_INSTRUMENTS} existingItems={localCurated.instruments} onAdd={(item) => addItem('instruments', item)} placeholder="Add instrument..." />
          </div>
          {/* Genres edit */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <span style={labelBold}>Genres</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {detectedGenres.map(([name, score]) => (
                <TagToggle key={name} name={name} score={score} selected={localCurated.genres.includes(name)} onToggle={() => toggleItem('genres', name)} />
              ))}
            </div>
            <AddItemCombobox suggestions={YAMNET_GENRES} existingItems={localCurated.genres} onAdd={(item) => addItem('genres', item)} placeholder="Add genre..." />
          </div>
          {/* Style edit */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <span style={labelBold}>Style</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {detectedStyles.map(([name, score]) => (
                <TagToggle key={name} name={name} score={score} selected={localCurated.style.includes(name)} onToggle={() => toggleItem('style', name)} />
              ))}
            </div>
            <AddItemCombobox suggestions={CLAP_STYLES} existingItems={localCurated.style} onAdd={(item) => addItem('style', item)} placeholder="Add style..." />
          </div>
          {/* Save/Cancel */}
          <div style={{ display: 'flex', gap: 8, paddingTop: 8 }}>
            <button onClick={(e) => { e.stopPropagation(); setLocalCurated(cue.musical_profile.curated); setIsEditing(false) }} style={{ flex: 1, padding: '8px 16px', borderRadius: 999, fontWeight: 700, fontSize: 13, backgroundColor: '#fff', border: '1px solid rgba(0,0,0,0.1)', cursor: 'pointer' }}>Cancel</button>
            <button onClick={(e) => { e.stopPropagation(); handleSave() }} disabled={isSaving} style={{ flex: 1, padding: '8px 16px', borderRadius: 999, fontWeight: 700, fontSize: 13, backgroundColor: '#000', color: '#fff', border: 'none', cursor: 'pointer' }}>{isSaving ? 'Saving...' : 'Save'}</button>
          </div>
        </div>
      )}

      {/* Valence / Arousal bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, paddingTop: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ ...labelBold, width: 56 }}>Valence</span>
          <div style={{ flex: 1, height: 6, backgroundColor: isActive ? '#E5E5E5' : '#D1D1D1', borderRadius: 999, overflow: 'hidden' }}>
            <div style={{ height: '100%', backgroundColor: isActive ? '#000' : TEXT_SUB, borderRadius: 999, width: `${(Math.min(1, Math.max(0, cue.musical_profile.valence)) * 100).toFixed(0)}%` }} />
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ ...labelBold, width: 56 }}>Arousal</span>
          <div style={{ flex: 1, height: 6, backgroundColor: isActive ? '#E5E5E5' : '#D1D1D1', borderRadius: 999, overflow: 'hidden' }}>
            <div style={{ height: '100%', backgroundColor: isActive ? '#000' : TEXT_SUB, borderRadius: 999, width: `${(Math.min(1, Math.max(0, cue.musical_profile.arousal)) * 100).toFixed(0)}%` }} />
          </div>
        </div>
      </div>

      {/* Section Structure */}
      <SectionStructure
        sections={cue.musical_profile.sections || []}
        cueDuration={cue.end - cue.start}
        isActive={isActive}
        linkedSectionIndex={linkedSectionIndex}
        onSectionHover={onSectionHover}
        onSectionSelect={onSectionSelect}
      />

      {/* Edit Tags button */}
      {!isEditing && (
        <button
          onClick={(e) => { e.stopPropagation(); setIsEditing(true) }}
          style={{
            width: '100%', padding: '10px 0', borderRadius: 999, fontWeight: 700, fontSize: 13,
            backgroundColor: isActive ? '#fff' : 'transparent',
            border: '2px solid rgba(0,0,0,0.1)', cursor: 'pointer',
            transition: 'all 0.2s ease', marginTop: 8, color: '#000',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'rgba(0,0,0,0.3)'; e.currentTarget.style.backgroundColor = '#fff' }}
          onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'rgba(0,0,0,0.1)'; e.currentTarget.style.backgroundColor = isActive ? '#fff' : 'transparent' }}
        >
          Edit Tags
        </button>
      )}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════════════════ */

export function FileAnalysis() {
  const {
    uploadedFile, fileId, fileName, duration,
    segments, selectedSegments, setSelectedSegments,
    mixType, setMixType,
    musicThreshold, setMusicThreshold,
    minGap, setMinGap,
    minCueLength, setMinCueLength,
    silenceThreshold, setSilenceThreshold,
    isSegmenting, setIsSegmenting,
    isAnalyzing, setIsAnalyzing,
    analysisProgress, analysisStatus, setAnalysisProgress,
    project, sessionId,
    activeCueId, setActiveCueId,
    startTimecode, framerate,
    setUploadedFile, setSegments, setProject,
    resetToSegmentation, setCurrentPage, userName,
  } = useAppStore()

  const [isDragOver, setIsDragOver] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [linkedSectionIndex, setLinkedSectionIndex] = useState<number | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const cuesSorted = project ? [...project.cues].sort((a, b) => a.start - b.start) : []
  const initials = userName ? userName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2) : 'JD'

  // Determine file state
  const fileState = project ? 'analyzed' : isAnalyzing ? 'analyzing' : segments.length > 0 ? 'segmented' : fileId ? 'new' : 'empty'

  /* ── Handlers (ported from MotivePipeline) ── */

  const handleFileSelect = useCallback(async (file: File) => {
    if (!file.type.startsWith('audio/') && !file.type.startsWith('video/')) { setError('Unsupported format'); return }
    setError(null); setIsUploading(true); setUploadProgress(0)
    try {
      const res = await api.uploadFile(file, (p) => setUploadProgress(p))
      setUploadedFile(file, res.file_id, res.filename, '01:00:00:00', 24)
    } catch { setError('Upload failed — is the backend running on port 8003?') }
    setIsUploading(false)
  }, [setUploadedFile])

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault(); setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelect(file)
  }, [handleFileSelect])

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => { e.preventDefault(); setIsDragOver(true) }, [])

  const handleFileInput = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFileSelect(file)
  }, [handleFileSelect])

  const handleMixTypeChange = useCallback((type: 'clean_mx' | 'full_mix') => {
    setMixType(type)
    const defaults = MIX_TYPE_DEFAULTS[type]
    setMusicThreshold(defaults.musicThreshold)
    setSilenceThreshold(defaults.silenceThreshold)
  }, [setMixType, setMusicThreshold, setSilenceThreshold])

  const handlePreview = useCallback(async () => {
    if (!fileId) return
    setIsSegmenting(true); setError(null)
    try {
      const res = await api.getSegmentPreview({ file_id: fileId, music_thresh: musicThreshold, min_gap: minGap, min_cue_length: minCueLength, silence_thresh: silenceThreshold, mix_type: mixType })
      setSegments(res.segments, res.duration)
    } catch { setError('Segment detection failed') }
    setIsSegmenting(false)
  }, [fileId, musicThreshold, minGap, minCueLength, silenceThreshold, mixType, setSegments, setIsSegmenting])

  const handleAnalyze = useCallback(async () => {
    if (!fileId || selectedSegments.length === 0) return
    setIsAnalyzing(true); setAnalysisProgress(0, 'Starting...'); setError(null)
    try {
      const res = await api.analyzeCues({ file_id: fileId, segments: selectedSegments, fps: framerate, mix_type: mixType })
      const interval = setInterval(async () => {
        try {
          const st = await api.getAnalysisStatus(res.session_id)
          setAnalysisProgress(st.progress_percent, st.status)
          if (st.complete && st.project) { clearInterval(interval); setProject(res.session_id, st.project); setIsAnalyzing(false) }
          if (st.error) { clearInterval(interval); setIsAnalyzing(false); setError(st.error) }
        } catch { clearInterval(interval); setIsAnalyzing(false); setError('Lost backend connection') }
      }, 500)
    } catch { setIsAnalyzing(false); setError('Analysis start failed') }
  }, [fileId, selectedSegments, framerate, setIsAnalyzing, setAnalysisProgress, setProject])

  /* ════════════════════════════════════════════════════════════════════════
     RENDER
     ════════════════════════════════════════════════════════════════════════ */

  return (
    <div style={{
      height: '100vh', width: '100vw', overflow: 'hidden', display: 'flex', flexDirection: 'column',
      backgroundColor: CANVAS, color: '#000000',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
      WebkitFontSmoothing: 'antialiased',
    }}>

      {/* ── HEADER ── */}
      <header style={{
        height: 80, flexShrink: 0, padding: '0 32px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        position: 'relative', zIndex: 30, backgroundColor: CANVAS,
      }}>
        {/* Left */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, width: '33%' }}>
          <div style={{ width: 48, height: 48, backgroundColor: '#000', borderRadius: 14, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff', boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
            <PlayIcon size={24} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h1 style={{ fontSize: 24, ...titleHeavy, lineHeight: 1, marginTop: 4 }}>MOTIVE</h1>
            <span style={{ fontSize: 10, fontWeight: 700, color: TEXT_SUB, letterSpacing: '0.15em', textTransform: 'uppercase', marginTop: 2 }}>Studio Edition</span>
          </div>
        </div>

        {/* Center: file info */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12, width: '33%' }}>
          {fileName && (
            <>
              <span style={{ fontWeight: 700, fontSize: 16 }}>{fileName}</span>
              <span style={{ backgroundColor: '#fff', border: '1px solid rgba(0,0,0,0.1)', padding: '2px 8px', borderRadius: 4, fontSize: 10, fontWeight: 700, letterSpacing: '0.05em' }}>
                {uploadedFile?.name.split('.').pop()?.toUpperCase() || 'WAV'}
              </span>
              <span style={{ fontSize: 12, color: TEXT_SUB, fontWeight: 500 }}>{duration ? fmtDur(duration) : '--:--'}</span>
            </>
          )}
        </div>

        {/* Right */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 12, width: '33%' }}>
          <button onClick={() => setCurrentPage('workspace')} style={{
            width: 40, height: 40, borderRadius: '50%', backgroundColor: '#fff', border: '1px solid rgba(0,0,0,0.1)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', boxShadow: '0 1px 2px rgba(0,0,0,0.05)', transition: 'all 0.2s',
          }}
            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#F9F9F9' }}
            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#fff' }}
          >
            <ArrowLeftIcon size={20} />
          </button>
          {project && (
            <>
              <button onClick={() => setCurrentPage('replacement')} style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', padding: '10px 20px', borderRadius: 999, fontWeight: 700, fontSize: 14, backgroundColor: ACCENT, color: '#000', border: 'none', cursor: 'pointer', boxShadow: '0 1px 2px rgba(0,0,0,0.05)', transition: 'all 0.2s' }}>
                Find Matches
              </button>
              <button onClick={() => setCurrentPage('rights')} style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', padding: '10px 20px', borderRadius: 999, fontWeight: 700, fontSize: 14, backgroundColor: '#000', color: '#fff', border: 'none', cursor: 'pointer', boxShadow: '0 1px 2px rgba(0,0,0,0.05)', transition: 'all 0.2s' }}>
                Set Rights
              </button>
            </>
          )}
          <div style={{ width: 40, height: 40, borderRadius: '50%', backgroundColor: '#000', color: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 14, cursor: 'pointer', marginLeft: 8 }}>
            {initials}
          </div>
        </div>
      </header>

      {/* ── TOOLBAR ── */}
      <div style={{
        position: 'sticky', top: 0, zIndex: 20,
        backgroundColor: `${CANVAS}F2`, backdropFilter: 'blur(12px)',
        padding: '12px 32px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        boxShadow: '0 4px 20px -10px rgba(0,0,0,0.1)', marginBottom: 16,
      }}>
        {/* Mix type toggle */}
        <div style={{ backgroundColor: LIST_BG, borderRadius: 999, padding: 4, display: 'flex', alignItems: 'center', boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.05)' }}>
          <button onClick={() => handleMixTypeChange('clean_mx')} style={{
            backgroundColor: mixType === 'clean_mx' ? '#000' : 'transparent',
            color: mixType === 'clean_mx' ? '#fff' : TEXT_SUB,
            borderRadius: 999, padding: '6px 20px', fontSize: 11, fontWeight: 700, letterSpacing: '0.05em',
            border: 'none', cursor: 'pointer', transition: 'all 0.2s',
          }}>
            Clean MX
          </button>
          <button onClick={() => handleMixTypeChange('full_mix')} style={{
            backgroundColor: mixType === 'full_mix' ? '#000' : 'transparent',
            color: mixType === 'full_mix' ? '#fff' : TEXT_SUB,
            borderRadius: 999, padding: '6px 20px', fontSize: 11, fontWeight: 700, letterSpacing: '0.05em',
            border: 'none', cursor: 'pointer', transition: 'all 0.2s',
          }}>
            Full Mix
          </button>
        </div>

        {/* Center status */}
        <div style={{ fontFamily: 'monospace', fontSize: 10, textTransform: 'uppercase', color: TEXT_SUB, fontWeight: 700, letterSpacing: '0.15em', display: 'flex', alignItems: 'center', gap: 8 }}>
          {segments.length > 0 && <span>{segments.length} Events Detected</span>}
          {segments.length > 0 && <span style={{ width: 4, height: 4, borderRadius: '50%', backgroundColor: 'rgba(0,0,0,0.2)' }} />}
          {project && <span style={{ color: '#000' }}>{cuesSorted.length} of {segments.length} Analyzed</span>}
          {!project && segments.length > 0 && !isAnalyzing && <span style={{ color: '#000' }}>{selectedSegments.length} of {segments.length} Selected</span>}
        </div>

        {/* Right action */}
        <div>
          {fileId && segments.length > 0 && !project && !isAnalyzing && (
            <button onClick={handlePreview} disabled={isSegmenting} style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              padding: '8px 24px', borderRadius: 999, fontWeight: 700, fontSize: 12,
              backgroundColor: BENTO, color: '#000', border: '1px solid transparent', cursor: 'pointer',
              transition: 'all 0.2s',
            }}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#fff'; e.currentTarget.style.borderColor = 'rgba(0,0,0,0.1)' }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = BENTO; e.currentTarget.style.borderColor = 'transparent' }}
            >
              {isSegmenting ? 'Detecting...' : 'Re-detect'}
            </button>
          )}
        </div>
      </div>

      {/* ── SCROLLABLE MAIN CONTENT ── */}
      <main style={{ flex: 1, overflowY: 'auto', paddingBottom: 96 }}>
        <div style={{ maxWidth: 1400, margin: '0 auto', width: '100%', padding: '0 32px', display: 'flex', flexDirection: 'column', gap: 24 }}>

          {/* ── WAVEFORM ── */}
          <section style={{
            backgroundColor: BENTO, borderRadius: 28, padding: 24,
            display: 'flex', flexDirection: 'column', gap: 8,
            position: 'relative', height: fileId ? 320 : 200, flexShrink: 0,
            boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
          }}>
            {fileId && uploadedFile ? (
              <div style={{ flex: 1, borderRadius: 16, overflow: 'hidden', backgroundColor: '#fff', border: '1px solid rgba(0,0,0,0.05)', boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.03)' }}>
                <WaveformViewer
                  linkedSectionIndex={linkedSectionIndex}
                  onLinkedSectionHover={setLinkedSectionIndex}
                  onLinkedSectionSelect={setLinkedSectionIndex}
                />
              </div>
            ) : (
              <div
                onDragOver={handleDragOver}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                style={{
                  flex: 1, borderRadius: 16, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  backgroundColor: isDragOver ? '#fff' : LIST_BG,
                  border: isDragOver ? '2px dashed #000' : '2px dashed rgba(0,0,0,0.15)',
                  cursor: 'pointer', transition: 'all 0.2s',
                }}
              >
                <span style={{ fontFamily: 'monospace', fontSize: 14, fontWeight: 700, color: TEXT_SUB, letterSpacing: '0.05em' }}>
                  {isUploading ? `UPLOADING ${uploadProgress}%` : 'DROP AUDIO OR VIDEO FILE TO BEGIN'}
                </span>
                <input ref={fileInputRef} type="file" hidden accept="audio/*,video/*" onChange={handleFileInput} />
              </div>
            )}
          </section>

          {/* ── DETECTION SETTINGS (pre-analysis) ── */}
          {!project && !isAnalyzing && fileId && (
            <section style={{ backgroundColor: BENTO, borderRadius: 28, padding: 24, boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 16 }}>Detection Settings</h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <SliderRow label="Threshold" min={0.001} max={0.03} step={0.0001} value={musicThreshold} onChange={setMusicThreshold} format={fmtPrecision} />
                <SliderRow label="Min Gap" min={0.1} max={15} step={0.1} value={minGap} onChange={setMinGap} format={(v) => `${v.toFixed(1)}s`} />
                <SliderRow label="Min Cue" min={0.5} max={15} step={0.1} value={minCueLength} onChange={setMinCueLength} format={(v) => `${v.toFixed(1)}s`} />

                <button onClick={() => setShowAdvanced(!showAdvanced)} style={{
                  width: '100%', padding: '6px 0', borderRadius: 999, fontWeight: 700, fontSize: 11,
                  backgroundColor: LIST_BG, border: 'none', cursor: 'pointer', color: TEXT_SUB,
                }}>
                  {showAdvanced ? 'Hide Advanced' : 'Advanced Settings'}
                </button>

                {showAdvanced && (
                  <SliderRow label="Silence" min={0.0001} max={0.1} step={0.0001} value={silenceThreshold} onChange={setSilenceThreshold} format={fmtPrecision} />
                )}

                {segments.length === 0 && (
                  <button onClick={handlePreview} disabled={isSegmenting} style={{
                    width: '100%', padding: '12px 0', borderRadius: 999, fontWeight: 700, fontSize: 14,
                    backgroundColor: '#000', color: '#fff', border: 'none', cursor: 'pointer', marginTop: 8,
                  }}>
                    {isSegmenting ? 'Detecting...' : 'Detect Segments'}
                  </button>
                )}

                {segments.length > 0 && (
                  <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                    <button onClick={() => setSelectedSegments([...segments])} style={{ flex: 1, padding: '8px 0', borderRadius: 999, fontWeight: 700, fontSize: 13, backgroundColor: '#fff', border: '1px solid rgba(0,0,0,0.15)', cursor: 'pointer' }}>Select All</button>
                    <button onClick={() => setSelectedSegments([])} style={{ flex: 1, padding: '8px 0', borderRadius: 999, fontWeight: 700, fontSize: 13, backgroundColor: '#fff', border: '1px solid rgba(0,0,0,0.15)', cursor: 'pointer' }}>Clear</button>
                  </div>
                )}
              </div>
            </section>
          )}

          {/* ── SEGMENT LIST (pre-analysis) ── */}
          {!project && segments.length > 0 && !isAnalyzing && (
            <section>
              <h2 style={{ fontSize: 18, ...titleHeavy, marginBottom: 12 }}>Segments</h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {segments.map(([start, end], i) => {
                  const isSelected = selectedSegments.some(([s, e]) => Math.abs(s - start) < 0.01 && Math.abs(e - end) < 0.01)
                  const tc0 = secondsToTimecode(start, framerate, startTimecode)
                  const tc1 = secondsToTimecode(end, framerate, startTimecode)
                  return <SegmentRow key={i} index={i} tc0={tc0} tc1={tc1} dur={fmtDur(end - start)} selected={isSelected} onClick={() => {
                    if (isSelected) setSelectedSegments(selectedSegments.filter(([s, e]) => !(Math.abs(s - start) < 0.01 && Math.abs(e - end) < 0.01)))
                    else setSelectedSegments([...selectedSegments, [start, end]])
                  }} />
                })}
              </div>
            </section>
          )}

          {/* ── ANALYSIS PROGRESS ── */}
          {isAnalyzing && (
            <section style={{ backgroundColor: BENTO, borderRadius: 28, padding: 32, boxShadow: '0 1px 2px rgba(0,0,0,0.05)', textAlign: 'center' }}>
              <div style={{ fontFamily: 'monospace', fontSize: 12, marginBottom: 12, color: TEXT_SUB }}>{analysisStatus || 'Processing...'}</div>
              <div style={{ height: 8, backgroundColor: 'rgba(0,0,0,0.05)', borderRadius: 999, overflow: 'hidden', marginBottom: 12 }}>
                <div style={{ height: '100%', backgroundColor: '#000', borderRadius: 999, width: `${analysisProgress}%`, transition: 'width 0.3s ease' }} />
              </div>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', color: TEXT_SUB }}>
                DO NOT CLOSE THIS TAB &middot; {Math.round(analysisProgress)}%
              </div>
            </section>
          )}

          {/* ── ANALYSIS RESULTS (post-analysis) ── */}
          {project && cuesSorted.length > 0 && (
            <section>
              <h2 style={{ fontSize: 20, ...titleHeavy, marginBottom: 16 }}>Analysis Results</h2>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                {cuesSorted.map((cue, i) => (
                  <AnalysisCueCard
                    key={cue.id}
                    cue={cue}
                    index={i}
                    isActive={activeCueId === cue.id}
                    onClick={() => { setActiveCueId(cue.id); setLinkedSectionIndex(null) }}
                    linkedSectionIndex={activeCueId === cue.id ? linkedSectionIndex : null}
                    onSectionHover={activeCueId === cue.id ? setLinkedSectionIndex : undefined}
                    onSectionSelect={activeCueId === cue.id ? setLinkedSectionIndex : undefined}
                  />
                ))}
              </div>
            </section>
          )}

          {/* Error */}
          {error && (
            <div style={{ backgroundColor: '#FEF2F2', border: '1px solid #FECACA', borderRadius: 16, padding: 16, fontSize: 14, fontWeight: 500, color: '#991B1B' }}>
              {error}
            </div>
          )}
        </div>
      </main>

      {/* ── FOOTER ── */}
      <footer style={{
        height: 72, flexShrink: 0, backgroundColor: '#fff', borderTop: '1px solid rgba(0,0,0,0.1)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 32px',
        position: 'relative', zIndex: 50, boxShadow: '0 -4px 20px rgba(0,0,0,0.03)',
      }}>
        {/* Left: state badge + info */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <StateBadge state={fileState} />
          <div style={{ fontSize: 14, fontWeight: 700, color: TEXT_SUB, display: 'flex', alignItems: 'center', gap: 8 }}>
            {(segments.length > 0 || cuesSorted.length > 0) && <span>{cuesSorted.length || segments.length} segments</span>}
            {duration > 0 && (
              <>
                <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: 'rgba(0,0,0,0.2)' }} />
                <span>{fmtDur(duration)} total duration</span>
              </>
            )}
          </div>
        </div>

        {/* Right: actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {fileState === 'empty' && (
            <button onClick={() => fileInputRef.current?.click()} style={{ ...pillBtn, backgroundColor: '#000', color: '#fff' }}>
              Upload File
            </button>
          )}
          {fileState === 'new' && (
            <button onClick={handlePreview} disabled={isSegmenting} style={{ ...pillBtn, backgroundColor: '#000', color: '#fff' }}>
              {isSegmenting ? 'Detecting...' : 'Detect Segments'}
            </button>
          )}
          {fileState === 'segmented' && (
            <button onClick={handleAnalyze} disabled={selectedSegments.length === 0} style={{ ...pillBtn, backgroundColor: '#000', color: '#fff', opacity: selectedSegments.length === 0 ? 0.4 : 1 }}>
              Analyze {selectedSegments.length} Segments
            </button>
          )}
          {fileState === 'analyzing' && (
            <button disabled style={{ ...pillBtn, backgroundColor: BENTO, color: TEXT_SUB }}>
              Analyzing {Math.round(analysisProgress)}%...
            </button>
          )}
          {fileState === 'analyzed' && (
            <>
              <button onClick={() => setCurrentPage('workspace')} style={{ ...pillBtn, backgroundColor: '#fff', color: '#000', border: '1px solid rgba(0,0,0,0.2)' }}>Back to Workspace</button>
              <button onClick={() => setCurrentPage('replacement')} style={{ ...pillBtn, backgroundColor: ACCENT, color: '#000' }}>Find Matches</button>
            </>
          )}
        </div>
      </footer>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   SUB-COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */

const pillBtn: React.CSSProperties = {
  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
  padding: '10px 24px', borderRadius: 999, fontWeight: 700, fontSize: 14,
  border: 'none', cursor: 'pointer', transition: 'all 0.2s',
}

function StateBadge({ state }: { state: string }) {
  const styles: Record<string, React.CSSProperties> = {
    empty: { backgroundColor: BENTO, color: TEXT_SUB },
    new: { backgroundColor: BENTO, color: '#000' },
    segmented: { backgroundColor: '#EBEBEB', color: '#000' },
    analyzing: { backgroundColor: ACCENT, color: '#000' },
    analyzed: { backgroundColor: '#000', color: '#fff' },
    matched: { backgroundColor: '#000', color: '#fff' },
  }
  const labels: Record<string, string> = {
    empty: 'No File', new: 'New', segmented: 'Segmented',
    analyzing: 'Analyzing', analyzed: 'Analyzed', matched: 'Matched',
  }
  return (
    <div style={{ ...styles[state], padding: '6px 12px', borderRadius: 999, fontSize: 12, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6, boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
      {labels[state]}
    </div>
  )
}

function SegmentRow({ index, tc0, tc1, dur, selected, onClick }: {
  index: number; tc0: string; tc1: string; dur: string; selected: boolean; onClick: () => void
}) {
  const [hovered, setHovered] = useState(false)
  const elevated = hovered

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        backgroundColor: elevated ? '#fff' : LIST_BG,
        borderRadius: 16, padding: '12px 16px',
        borderLeft: `4px solid ${selected ? ACCENT : 'transparent'}`,
        border: elevated ? '1px solid rgba(0,0,0,0.1)' : '1px solid transparent',
        boxShadow: elevated ? '0 10px 25px -5px rgba(0,0,0,0.1)' : 'none',
        transform: elevated ? 'translateY(-4px)' : 'translateY(0)',
        transition: 'all 0.2s ease', cursor: 'pointer',
        display: 'flex', alignItems: 'center', gap: 12,
      }}
    >
      {/* Checkbox */}
      <div style={{
        width: 24, height: 24, borderRadius: '50%',
        backgroundColor: selected ? '#16a34a' : 'rgba(0,0,0,0.05)',
        border: selected ? 'none' : '2px solid rgba(0,0,0,0.15)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: '#fff', fontSize: 12, fontWeight: 700, flexShrink: 0,
      }}>
        {selected && '✓'}
      </div>
      <span style={{ fontWeight: 700, fontSize: 14, width: 28 }}>#{index + 1}</span>
      <span style={{ fontFamily: 'monospace', fontSize: 12, fontWeight: 500, color: TEXT_SUB, flex: 1 }}>{tc0} — {tc1}</span>
      <span style={{ fontFamily: 'monospace', fontSize: 12, fontWeight: 700 }}>{dur}</span>
    </div>
  )
}

function SliderRow({ label, min, max, step, value, onChange, format }: {
  label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void; format: (v: number) => string
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
      <span style={{ ...labelBold, width: 72 }}>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ flex: 1, accentColor: '#000' }} />
      <span style={{ fontFamily: 'monospace', fontSize: 12, fontWeight: 700, width: 64, textAlign: 'right' }}>{format(value)}</span>
    </div>
  )
}
