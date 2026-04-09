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
   DESIGN TOKENS — Dark "Studio Edition" theme
   ═══════════════════════════════════════════════════════════════════════════ */

const BG = '#0b0d12'
const BG_ELEVATED = '#121722'
const BG_CARD = '#161d2a'
const LINE = 'rgba(255, 255, 255, 0.08)'
const TEXT = '#edf2ff'
const MUTED = '#8b96b0'
const ACCENT = '#63b7ff'
const ACCENT_WARM = '#f2b84d'
const PILL_ON = 'rgba(99, 183, 255, 0.2)'

const SECTION_COLORS = [
  'rgba(99, 183, 255, 0.35)',
  'rgba(180, 150, 120, 0.32)',
  'rgba(243, 89, 107, 0.38)',
  'rgba(96, 198, 177, 0.32)',
]

const FONT_UI = 'Manrope, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
const FONT_MONO = '"IBM Plex Mono", monospace'

const titleHeavy: React.CSSProperties = { fontWeight: 800, letterSpacing: '-0.04em' }
const labelBold: React.CSSProperties = {
  fontWeight: 700, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.1em', color: MUTED,
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
  'high energy':       { bg: 'rgba(243, 89, 107, 0.5)', fg: TEXT, icon: '⚡' },
  'climactic':         { bg: 'rgba(243, 89, 107, 0.6)', fg: TEXT, icon: '🔺' },
  'aggressive':        { bg: 'rgba(243, 89, 107, 0.45)', fg: TEXT, icon: '🔥' },
  'building tension':  { bg: 'rgba(180, 150, 120, 0.45)', fg: TEXT, icon: '↗' },
  'low energy':        { bg: 'rgba(99, 183, 255, 0.3)', fg: TEXT, icon: '○' },
  'gentle':            { bg: 'rgba(96, 198, 177, 0.3)', fg: TEXT, icon: '~' },
}

function getEnergyStyle(label: string) {
  const lower = label.toLowerCase()
  for (const [key, val] of Object.entries(ENERGY_COLORS)) {
    if (lower.includes(key)) return val
  }
  return { bg: 'rgba(255,255,255,0.1)', fg: TEXT, icon: '•' }
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
        <span style={{ fontSize: 10, fontWeight: 600, color: MUTED }}>
          {sections.length} sections
        </span>
      </div>

      {/* Timeline bar — proportional width, energy-colored, with BPM/Key inside */}
      <div style={{
        display: 'flex', height: 48, borderRadius: 10, overflow: 'hidden',
        border: `1px solid ${LINE}`,
      }}>
        {sections.map((sec) => {
          const pct = cueDuration > 0 ? (sec.duration / cueDuration) * 100 : 0
          const style = getEnergyStyle(sec.energy_label)
          const isHov = hoveredIdx === sec.index
          const wide = pct > 15
          const secColor = SECTION_COLORS[sec.index % SECTION_COLORS.length]
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
                width: `${pct}%`, minWidth: 24, backgroundColor: secColor,
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                borderRight: sec.index < sections.length - 1 ? '1px solid rgba(255,255,255,0.12)' : 'none',
                cursor: 'pointer',
                opacity: effectiveIdx !== null && !isHov && effectiveIdx !== sec.index ? 0.38 : 1,
                transition: 'opacity 0.15s ease, box-shadow 0.15s ease',
                padding: '0 4px', overflow: 'hidden',
                boxShadow: (linkedSectionIndex === sec.index || selectedIdx === sec.index) ? 'inset 0 0 0 2px rgba(248,251,255,0.85)' : 'none',
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

      {/* Compact summary row — pill buttons per section */}
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(sections.length, 4)}, 1fr)`, gap: 8 }}>
        {sections.map((sec) => {
          const isPinned = linkedSectionIndex === sec.index || selectedIdx === sec.index
          const isHov = hoveredIdx === sec.index
          return (
            <button
              key={sec.index}
              onMouseEnter={() => setHover(sec.index)}
              onMouseLeave={() => setHover(null)}
              onClick={() => {
                const next = selectedIdx === sec.index ? null : sec.index
                setSelectedIdx(next)
                onSectionSelect?.(next)
              }}
              style={{
                display: 'flex', alignItems: 'center', gap: 5, padding: '8px 10px',
                borderRadius: 10,
                backgroundColor: 'rgba(255,255,255,0.04)',
                border: (isPinned || isHov) ? '1px solid rgba(248,251,255,0.45)' : `1px solid ${LINE}`,
                boxShadow: isPinned ? 'inset 0 0 0 1px rgba(248,251,255,0.2)' : 'none',
                transition: 'all 0.12s ease', cursor: 'pointer',
                color: TEXT, fontFamily: FONT_MONO, fontSize: 11, fontWeight: 600,
                background: 'none',
              }}
            >
              <strong style={{ fontFamily: FONT_MONO, fontSize: 11 }}>#{sec.index + 1}</strong>
              {sec.bpm && <span>{Math.round(sec.bpm)}</span>}
              {sec.key && <span>{sec.key}</span>}
            </button>
          )
        })}
      </div>

      {/* Focus detail panel */}
      <div style={{
        backgroundColor: BG_CARD, borderRadius: 14, padding: '14px 16px',
        border: `1px solid ${LINE}`, transition: 'all 0.15s ease',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              width: 18, height: 18, borderRadius: 5, backgroundColor: 'rgba(255,255,255,0.1)', color: TEXT,
              fontSize: 9, fontWeight: 800,
            }}>
              {focusedSec.index + 1}
            </span>
            <span style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.03em', color: TEXT }}>
              {focusedSec.energy_label}
            </span>
          </div>
          <span style={{ fontSize: 10, fontWeight: 600, color: MUTED, fontFamily: FONT_MONO }}>
            {String(Math.floor(focusedSec.duration / 60)).padStart(2, '0')}:{String(Math.floor(focusedSec.duration % 60)).padStart(2, '0')}
          </span>
        </div>

        <div style={{ display: 'flex', gap: 10, marginBottom: 8 }}>
          <div style={{
            flex: 1, borderRadius: 8, border: `1px solid ${LINE}`,
            padding: '4px 8px', backgroundColor: 'rgba(255,255,255,0.04)',
          }}>
            <span style={{ ...labelBold, fontSize: 8 }}>BPM</span>
            <div style={{ fontSize: 12, fontWeight: 800, color: TEXT }}>{focusedSec.bpm ? Math.round(focusedSec.bpm) : '—'}</div>
          </div>
          <div style={{
            flex: 1, borderRadius: 8, border: `1px solid ${LINE}`,
            padding: '4px 8px', backgroundColor: 'rgba(255,255,255,0.04)',
          }}>
            <span style={{ ...labelBold, fontSize: 8 }}>Key</span>
            <div style={{ fontSize: 12, fontWeight: 800, color: TEXT }}>{focusedSec.key || '—'}</div>
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
                    color: isTop ? TEXT : MUTED, textAlign: 'right',
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {label}
                  </span>
                  <div style={{ flex: 1, height: 8, backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: 999, overflow: 'hidden' }}>
                    <div style={{
                      height: '100%', borderRadius: 999,
                      width: `${(norm * 100).toFixed(0)}%`,
                      background: isTop ? 'linear-gradient(90deg, #7fc2ff, #f3596b)' : 'rgba(255,255,255,0.2)',
                    }} />
                  </div>
                  <span style={{ width: 30, fontSize: 8, fontWeight: isTop ? 800 : 600, color: MUTED, fontFamily: FONT_MONO }}>
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

  // Merge YAMNet individual instruments with CLAP ensemble-level instrumentation
  const yamnetInstr = cue.musical_profile.detected?.instruments_yamnet || {}
  const clapInstr = cue.musical_profile.detected?.clap_instrumentation || {}
  const mergedInstruments = { ...yamnetInstr, ...clapInstr }
  const detectedInstruments = getDisplayItems(mergedInstruments, localCurated.instruments, INSTRUMENT_THRESHOLD)
  const detectedGenres = getDisplayItems(cue.musical_profile.detected?.genres_yamnet, localCurated.genres, GENRE_THRESHOLD)
  const detectedStyles = getDisplayItems(cue.musical_profile.detected?.clap_style, localCurated.style, STYLE_THRESHOLD)

  function TagPill({ label, isCurated }: { label: string; isCurated: boolean }) {
    return (
      <span style={{
        backgroundColor: isCurated ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.04)',
        color: isCurated ? TEXT : MUTED,
        padding: '4px 10px', borderRadius: 999, fontSize: 10, fontWeight: 700,
        border: `1px solid ${LINE}`,
      }}>
        {label.toUpperCase()}
      </span>
    )
  }

  function TagToggle({ name, score, selected, onToggle }: { name: string; score: number; selected: boolean; onToggle: () => void }) {
    return (
      <button onClick={onToggle} style={{
        padding: '4px 10px', borderRadius: 999, fontSize: 10, fontWeight: 700, cursor: 'pointer',
        backgroundColor: selected ? PILL_ON : 'rgba(255,255,255,0.04)',
        color: selected ? '#b6deff' : MUTED,
        border: selected ? '1px solid rgba(99,183,255,0.3)' : `1px solid ${LINE}`,
        transition: 'all 0.15s ease', fontFamily: FONT_UI,
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
        backgroundColor: BG_CARD,
        borderRadius: 16, padding: 18,
        borderLeft: isActive ? `3px solid ${ACCENT}` : '3px solid transparent',
        boxShadow: elevated ? '0 12px 24px -8px rgba(0,0,0,0.4)' : 'none',
        transform: elevated ? 'translateY(-2px)' : 'translateY(0)',
        transition: 'all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1)',
        border: `1px solid ${LINE}`,
        cursor: 'pointer',
        display: 'flex', flexDirection: 'column', gap: 14,
        opacity: isActive ? 1 : 0.92,
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: `1px solid ${LINE}`, paddingBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ backgroundColor: 'rgba(255,255,255,0.1)', color: TEXT, padding: '4px 10px', borderRadius: 999, fontSize: 11, fontWeight: 800 }}>#{index + 1}</span>
          <span style={{ fontFamily: FONT_MONO, fontSize: 11, fontWeight: 500, color: MUTED }}>{tc0} — {tc1}</span>
        </div>
        <span style={{ fontSize: 11, color: MUTED, fontWeight: 700, fontFamily: FONT_MONO }}>{dur}</span>
      </div>

      {/* BPM & Key */}
      <div style={{ display: 'flex', gap: 32 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={labelBold}>BPM</span>
          <span style={{ fontSize: 28, ...titleHeavy, lineHeight: 1, color: TEXT }}>{cue.musical_profile.bpm?.toFixed(0) || '—'}</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span style={labelBold}>Key</span>
          <span style={{ fontSize: 28, ...titleHeavy, lineHeight: 1, color: TEXT }}>{cue.musical_profile.key || '—'}</span>
        </div>
      </div>

      {/* Tags */}
      {!isEditing ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <span style={labelBold}>Instruments</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {localCurated.instruments.map(i => <TagPill key={i} label={i} isCurated={true} />)}
              {localCurated.instruments.length === 0 && <span style={{ fontSize: 11, color: MUTED }}>None detected</span>}
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <span style={labelBold}>Genre</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {localCurated.genres.map(t => <TagPill key={t} label={t} isCurated={true} />)}
              {localCurated.genres.length === 0 && <span style={{ fontSize: 11, color: MUTED }}>None detected</span>}
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <span style={labelBold}>Style</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {localCurated.style.map(t => <TagPill key={t} label={t} isCurated={true} />)}
              {localCurated.style.length === 0 && <span style={{ fontSize: 11, color: MUTED }}>None detected</span>}
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <span style={labelBold}>Mood</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {localCurated.moods.map(m => <TagPill key={m} label={m} isCurated={true} />)}
              {localCurated.moods.length === 0 && <span style={{ fontSize: 11, color: MUTED }}>None detected</span>}
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
          {/* Moods edit */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <span style={labelBold}>Mood</span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {Object.entries(cue.musical_profile.detected?.moods || {})
                .sort((a, b) => b[1] - a[1])
                .map(([name, score]) => (
                  <TagToggle key={name} name={name} score={score} selected={localCurated.moods.includes(name)} onToggle={() => toggleItem('moods', name)} />
                ))}
            </div>
          </div>
          {/* Save/Cancel */}
          <div style={{ display: 'flex', gap: 8, paddingTop: 8 }}>
            <button onClick={(e) => { e.stopPropagation(); setLocalCurated(cue.musical_profile.curated); setIsEditing(false) }} style={{ flex: 1, padding: '8px 16px', borderRadius: 999, fontWeight: 700, fontSize: 13, backgroundColor: 'transparent', border: `1px solid ${LINE}`, cursor: 'pointer', color: TEXT, fontFamily: FONT_UI }}>Cancel</button>
            <button onClick={(e) => { e.stopPropagation(); handleSave() }} disabled={isSaving} style={{ flex: 1, padding: '8px 16px', borderRadius: 999, fontWeight: 700, fontSize: 13, backgroundColor: ACCENT_WARM, color: '#1a1408', border: 'none', cursor: 'pointer', fontFamily: FONT_UI }}>{isSaving ? 'Saving...' : 'Save'}</button>
          </div>
        </div>
      )}

      {/* Valence / Arousal bars — hidden for now, data still computed */}

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
            backgroundColor: 'transparent',
            border: `1px solid ${LINE}`, cursor: 'pointer',
            transition: 'all 0.2s ease', marginTop: 8, color: TEXT, fontFamily: FONT_UI,
          }}
          onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.25)'; e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.04)' }}
          onMouseLeave={(e) => { e.currentTarget.style.borderColor = LINE; e.currentTarget.style.backgroundColor = 'transparent' }}
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
  const [heroHoveredIdx, setHeroHoveredIdx] = useState<number | null>(null)
  const [heroPinnedIdx, setHeroPinnedIdx] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const cuesSorted = project ? [...project.cues].sort((a, b) => a.start - b.start) : []
  const initials = userName ? userName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2) : 'JD'
  const activeCue = project?.cues.find((c) => c.id === activeCueId) ?? cuesSorted[0] ?? null
  const heroSections = activeCue?.musical_profile.sections ?? []
  // Hover and clicks on waveform overlays update linkedSectionIndex.
  // The detail panel prefers the live hovered/linked value, falling back to the last pinned section.
  const heroEffectiveIdx = heroHoveredIdx ?? linkedSectionIndex ?? heroPinnedIdx
  const heroFocusedSec = heroSections[heroEffectiveIdx] ?? heroSections[0]

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
      background: `radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 183, 255, 0.12), transparent), ${BG}`,
      color: TEXT,
      fontFamily: FONT_UI,
      WebkitFontSmoothing: 'antialiased',
    }}>

      {/* ── HEADER ── */}
      <header style={{
        height: 72, flexShrink: 0, padding: '0 28px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        position: 'relative', zIndex: 30,
        borderBottom: `1px solid ${LINE}`,
        background: 'rgba(11, 13, 18, 0.85)',
        backdropFilter: 'blur(12px)',
      }}>
        {/* Left: brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            width: 44, height: 44, borderRadius: 12,
            background: 'linear-gradient(145deg, #2a3142, #1a2030)',
            border: `1px solid ${LINE}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center', color: TEXT,
          }}>
            <PlayIcon size={20} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <h1 style={{ fontSize: 20, ...titleHeavy, lineHeight: 1, margin: 0, color: TEXT }}>MOTIVE</h1>
            <span style={{ fontSize: 9, fontWeight: 700, color: MUTED, letterSpacing: '0.18em', textTransform: 'uppercase' }}>Studio Edition</span>
          </div>
        </div>

        {/* Center: file info */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, fontSize: 14, fontWeight: 600 }}>
          {fileName && (
            <>
              <span style={{ color: TEXT }}>{fileName}</span>
              <span style={{
                fontSize: 10, fontWeight: 700, letterSpacing: '0.08em',
                padding: '2px 8px', borderRadius: 4,
                border: `1px solid ${LINE}`, color: MUTED,
              }}>
                {uploadedFile?.name.split('.').pop()?.toUpperCase() || 'WAV'}
              </span>
              <span style={{ fontSize: 12, color: MUTED, fontFamily: FONT_MONO }}>{duration ? fmtDur(duration) : '--:--'}</span>
            </>
          )}
        </div>

        {/* Right: actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button onClick={() => setCurrentPage('workspace')} style={{
            width: 40, height: 40, borderRadius: '50%',
            border: `1px solid ${LINE}`, background: BG_ELEVATED,
            display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
            color: TEXT, transition: 'all 0.2s',
          }}
            onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)' }}
            onMouseLeave={(e) => { e.currentTarget.style.borderColor = LINE }}
          >
            <ArrowLeftIcon size={18} />
          </button>
          {project && (
            <>
              <button onClick={() => setCurrentPage('replacement')} style={{
                padding: '10px 18px', borderRadius: 999, border: 'none', fontWeight: 700, fontSize: 13,
                cursor: 'pointer', fontFamily: FONT_UI,
                background: ACCENT_WARM, color: '#1a1408',
              }}>
                Find Matches
              </button>
              <button onClick={() => setCurrentPage('rights')} style={{
                padding: '10px 18px', borderRadius: 999, fontWeight: 700, fontSize: 13,
                cursor: 'pointer', fontFamily: FONT_UI,
                background: '#2a3142', color: TEXT, border: `1px solid ${LINE}`,
              }}>
                Set Rights
              </button>
            </>
          )}
          <div style={{
            width: 40, height: 40, borderRadius: '50%',
            background: 'linear-gradient(135deg, #4a5568, #2d3748)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 13, fontWeight: 700, color: TEXT, cursor: 'pointer',
          }}>
            {initials}
          </div>
        </div>
      </header>

      {/* ── TOOLBAR ── */}
      <div style={{
        flexShrink: 0,
        padding: '12px 28px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        gap: 16,
        borderBottom: `1px solid ${LINE}`,
      }}>
        {/* Mix type toggle */}
        <div style={{ display: 'flex', background: BG_CARD, borderRadius: 999, padding: 4, border: `1px solid ${LINE}` }}>
          <button onClick={() => handleMixTypeChange('clean_mx')} style={{
            border: 'none',
            background: mixType === 'clean_mx' ? 'rgba(255,255,255,0.1)' : 'transparent',
            color: mixType === 'clean_mx' ? TEXT : MUTED,
            padding: '8px 18px', borderRadius: 999, fontSize: 11, fontWeight: 700,
            letterSpacing: '0.06em', cursor: 'pointer', fontFamily: FONT_UI,
          }}>
            Clean MX
          </button>
          <button onClick={() => handleMixTypeChange('full_mix')} style={{
            border: 'none',
            background: mixType === 'full_mix' ? 'rgba(255,255,255,0.1)' : 'transparent',
            color: mixType === 'full_mix' ? TEXT : MUTED,
            padding: '8px 18px', borderRadius: 999, fontSize: 11, fontWeight: 700,
            letterSpacing: '0.06em', cursor: 'pointer', fontFamily: FONT_UI,
          }}>
            Full Mix
          </button>
        </div>

        {/* Center status */}
        <div style={{ fontFamily: FONT_MONO, fontSize: 10, fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: MUTED, display: 'flex', alignItems: 'center', gap: 8 }}>
          {segments.length > 0 && <span>{segments.length} events detected</span>}
          {segments.length > 0 && <span style={{ width: 4, height: 4, borderRadius: '50%', backgroundColor: MUTED }} />}
          {project && <span><strong style={{ color: TEXT }}>{cuesSorted.length} of {segments.length}</strong> analyzed</span>}
          {!project && segments.length > 0 && !isAnalyzing && <span><strong style={{ color: TEXT }}>{selectedSegments.length} of {segments.length}</strong> selected</span>}
        </div>

        {/* Right action */}
        <div>
          {fileId && segments.length > 0 && !project && !isAnalyzing && (
            <button onClick={handlePreview} disabled={isSegmenting} style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              padding: '8px 24px', borderRadius: 999, fontWeight: 700, fontSize: 12,
              backgroundColor: '#2a3142', color: TEXT, border: `1px solid ${LINE}`, cursor: 'pointer',
              transition: 'all 0.2s', fontFamily: FONT_UI,
            }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.2)' }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = LINE }}
            >
              {isSegmenting ? 'Detecting...' : 'Re-detect'}
            </button>
          )}
        </div>
      </div>

      {/* ── SCROLLABLE MAIN CONTENT ── */}
      <main style={{ flex: 1, overflowY: 'auto', padding: '20px 28px 100px' }}>
        <div style={{ maxWidth: 1440, margin: '0 auto', width: '100%', display: 'flex', flexDirection: 'column', gap: 20 }}>

          {/* ── WAVEFORM + STRUCTURE HERO ── */}
          <section className="dark-studio" style={{
            border: `1px solid ${LINE}`,
            borderRadius: 18,
            background: 'linear-gradient(170deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01))',
            padding: 24,
            boxShadow: '0 20px 50px rgba(0,0,0,0.45)',
            flexShrink: 0,
          }}>
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: 16, marginBottom: 18 }}>
              <div>
                <h2 style={{ margin: 0, fontSize: 28, fontWeight: 800, letterSpacing: '-0.02em', color: TEXT }}>
                  {project && heroSections.length > 1 ? 'Cue Structure & Energy' : 'Waveform & Structure'}
                </h2>
                <p style={{ margin: '4px 0 0', color: MUTED, fontSize: 13 }}>
                  {project && heroSections.length > 1 ? 'Waveform-linked section energy overview' : 'Upload audio or video to begin analysis'}
                </p>
              </div>
              {project && heroSections.length > 1 && (
                <div style={{
                  padding: '8px 12px', borderRadius: 999, border: `1px solid ${LINE}`,
                  background: 'rgba(99, 183, 255, 0.14)', color: '#b6deff',
                  fontSize: 12, fontFamily: FONT_MONO,
                }}>
                  Energy Overlay: ON
                </div>
              )}
            </div>

            {/* Grid: waveform area | detail panel */}
            <div style={{
              display: (project && heroSections.length > 1 && heroFocusedSec) ? 'grid' : 'block',
              gridTemplateColumns: '1fr 360px',
              gap: 18,
            }}>
              {/* Left: waveform + pills */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
                {/* Waveform container */}
                {fileId && uploadedFile ? (
                  <div style={{
                    borderRadius: 16, overflow: 'hidden',
                    backgroundColor: '#0a0f1a', border: `1px solid ${LINE}`,
                    minHeight: 260,
                  }}>
                    <WaveformViewer
                      linkedSectionIndex={linkedSectionIndex}
                      onLinkedSectionHover={setLinkedSectionIndex}
                      onLinkedSectionSelect={(idx) => {
                        setLinkedSectionIndex(idx)
                        if (idx !== null) setHeroPinnedIdx(idx)
                      }}
                    />
                  </div>
                ) : (
                  <div
                    onDragOver={handleDragOver}
                    onDragLeave={() => setIsDragOver(false)}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    style={{
                      borderRadius: 16, display: 'flex', alignItems: 'center', justifyContent: 'center',
                      minHeight: 200,
                      backgroundColor: isDragOver ? BG_ELEVATED : BG_CARD,
                      border: isDragOver ? `2px dashed ${ACCENT}` : `2px dashed rgba(255,255,255,0.15)`,
                      cursor: 'pointer', transition: 'all 0.2s',
                    }}
                  >
                    <span style={{ fontFamily: FONT_MONO, fontSize: 14, fontWeight: 700, color: MUTED, letterSpacing: '0.05em' }}>
                      {isUploading ? `UPLOADING ${uploadProgress}%` : 'DROP AUDIO OR VIDEO FILE TO BEGIN'}
                    </span>
                    <input ref={fileInputRef} type="file" hidden accept="audio/*,video/*" onChange={handleFileInput} />
                  </div>
                )}

                {/* Section pills — below waveform */}
                {project && heroSections.length > 1 && (
                  <div style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 8,
                    marginTop: 10,
                  }}>
                    {heroSections.map((sec) => {
                      const isOn = heroEffectiveIdx === sec.index
                      return (
                        <button
                          key={sec.index}
                          onMouseEnter={() => { setHeroHoveredIdx(sec.index); setLinkedSectionIndex(sec.index) }}
                          onMouseLeave={() => { setHeroHoveredIdx(null); setLinkedSectionIndex(null) }}
                          onClick={() => { setHeroPinnedIdx(sec.index); setLinkedSectionIndex(sec.index) }}
                          style={{
                            flex: '1 1 0',
                            minWidth: 0,
                            borderRadius: 10,
                            border: isOn ? '1px solid rgba(248,251,255,0.45)' : `1px solid ${LINE}`,
                            boxShadow: isOn ? '0 0 0 1px rgba(248,251,255,0.24) inset' : 'none',
                            padding: '8px 10px', fontSize: 12, fontWeight: 600,
                            background: 'rgba(255,255,255,0.03)', cursor: 'pointer',
                            color: '#f3f7ff', fontFamily: FONT_UI,
                            transition: 'transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease',
                            transform: isOn ? 'translateY(-1px)' : 'none',
                            textAlign: 'center',
                          }}
                        >
                          <strong style={{ fontFamily: FONT_MONO, fontSize: 11, opacity: 0.95 }}>#{sec.index + 1}</strong>
                          {' '}{sec.bpm ? Math.round(sec.bpm) : '—'} BPM {sec.key || '—'}
                        </button>
                      )
                    })}
                  </div>
                )}
              </div>

              {/* Right: detail panel (only when sections exist) */}
              {project && heroSections.length > 1 && heroFocusedSec && (
                <aside style={{
                  background: BG_CARD, border: `1px solid ${LINE}`, borderRadius: 16, padding: 14,
                  display: 'flex', flexDirection: 'column',
                }}>
                  <h3 style={{ margin: 0, fontSize: 18, fontWeight: 800, color: TEXT }}>
                    Section #{heroFocusedSec.index + 1} — {heroFocusedSec.energy_label}
                  </h3>
                  <p style={{ margin: '4px 0 12px', fontSize: 12, color: MUTED, fontFamily: FONT_MONO }}>
                    {Math.floor(heroFocusedSec.start_relative / 60)}:{String(Math.floor(heroFocusedSec.start_relative % 60)).padStart(2, '0')}
                    {' – '}
                    {Math.floor(heroFocusedSec.end_relative / 60)}:{String(Math.floor(heroFocusedSec.end_relative % 60)).padStart(2, '0')}
                    {' · '}{heroFocusedSec.duration.toFixed(1)}s
                  </p>

                  <div style={{ display: 'flex', justifyContent: 'space-between', margin: '8px 0', fontSize: 13 }}>
                    <span style={{ color: MUTED }}>BPM</span>
                    <strong style={{ color: TEXT }}>{heroFocusedSec.bpm ? Math.round(heroFocusedSec.bpm) : '—'}</strong>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', margin: '8px 0', fontSize: 13 }}>
                    <span style={{ color: MUTED }}>Key</span>
                    <strong style={{ color: TEXT }}>{heroFocusedSec.key || '—'}</strong>
                  </div>

                  {/* Energy bars */}
                  <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {Object.entries(heroFocusedSec.clap_energy)
                      .sort((a, b) => b[1] - a[1])
                      .map(([label, rawScore]) => {
                        const norm = normalizeEnergy(heroFocusedSec.clap_energy)[label]
                        return (
                          <div key={label} className="bar-line" style={{ fontSize: 12, color: '#d9e4ff' }}>
                            <strong>{label}</strong> {rawScore.toFixed(2)}
                            <div style={{ marginTop: 3, height: 8, borderRadius: 999, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                              <div style={{ height: '100%', borderRadius: 999, background: 'linear-gradient(90deg, #7fc2ff 0%, #f3596b 100%)', width: `${Math.round(norm * 100)}%` }} />
                            </div>
                          </div>
                        )
                      })}
                  </div>

                  <p style={{
                    marginTop: 14, paddingTop: 10,
                    borderTop: '1px dashed rgba(255,255,255,0.2)',
                    fontSize: 12, color: '#a9b6d3', lineHeight: 1.45,
                  }}>
                    Hover any section band or pill to sync highlights with the waveform. Click to pin selection.
                  </p>
                </aside>
              )}
            </div>
          </section>

          {/* ── DETECTION SETTINGS (pre-analysis) ── */}
          {!project && !isAnalyzing && fileId && (
            <section style={{
              border: `1px solid ${LINE}`,
              borderRadius: 16,
              backgroundColor: BG_ELEVATED,
              padding: '20px 22px',
            }}>
              <h3 style={{ fontSize: 15, fontWeight: 800, marginBottom: 16, marginTop: 0, color: TEXT }}>Detection Settings</h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <SliderRow label="Threshold" min={0.001} max={0.03} step={0.0001} value={musicThreshold} onChange={setMusicThreshold} format={fmtPrecision} />
                <SliderRow label="Min Gap" min={0.1} max={15} step={0.1} value={minGap} onChange={setMinGap} format={(v) => `${v.toFixed(1)}s`} />
                <SliderRow label="Min Cue" min={0.5} max={15} step={0.1} value={minCueLength} onChange={setMinCueLength} format={(v) => `${v.toFixed(1)}s`} />

                <button onClick={() => setShowAdvanced(!showAdvanced)} style={{
                  width: '100%', padding: '6px 0', borderRadius: 999, fontWeight: 700, fontSize: 11,
                  backgroundColor: 'rgba(255,255,255,0.04)', border: `1px solid ${LINE}`, cursor: 'pointer', color: MUTED,
                  fontFamily: FONT_UI,
                }}>
                  {showAdvanced ? 'Hide Advanced' : 'Advanced Settings'}
                </button>

                {showAdvanced && (
                  <SliderRow label="Silence" min={0.0001} max={0.1} step={0.0001} value={silenceThreshold} onChange={setSilenceThreshold} format={fmtPrecision} />
                )}

                {segments.length === 0 && (
                  <button onClick={handlePreview} disabled={isSegmenting} style={{
                    width: '100%', padding: '12px 0', borderRadius: 999, fontWeight: 700, fontSize: 14,
                    backgroundColor: ACCENT_WARM, color: '#1a1408', border: 'none', cursor: 'pointer', marginTop: 8,
                    fontFamily: FONT_UI,
                  }}>
                    {isSegmenting ? 'Detecting...' : 'Detect Segments'}
                  </button>
                )}

                {segments.length > 0 && (
                  <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                    <button onClick={() => setSelectedSegments([...segments])} style={{ flex: 1, padding: '8px 0', borderRadius: 999, fontWeight: 700, fontSize: 13, backgroundColor: 'transparent', border: `1px solid ${LINE}`, cursor: 'pointer', color: TEXT, fontFamily: FONT_UI }}>Select All</button>
                    <button onClick={() => setSelectedSegments([])} style={{ flex: 1, padding: '8px 0', borderRadius: 999, fontWeight: 700, fontSize: 13, backgroundColor: 'transparent', border: `1px solid ${LINE}`, cursor: 'pointer', color: TEXT, fontFamily: FONT_UI }}>Clear</button>
                  </div>
                )}
              </div>
            </section>
          )}

          {/* ── SEGMENT LIST (pre-analysis) ── */}
          {!project && segments.length > 0 && !isAnalyzing && (
            <section>
              <h2 style={{ fontSize: 18, ...titleHeavy, marginBottom: 14, color: TEXT }}>Segments</h2>
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
            <section style={{
              border: `1px solid ${LINE}`,
              borderRadius: 16,
              backgroundColor: BG_ELEVATED,
              padding: 32, textAlign: 'center',
            }}>
              <div style={{ fontFamily: FONT_MONO, fontSize: 12, marginBottom: 12, color: MUTED }}>{analysisStatus || 'Processing...'}</div>
              <div style={{ height: 8, backgroundColor: 'rgba(255,255,255,0.08)', borderRadius: 999, overflow: 'hidden', marginBottom: 12 }}>
                <div style={{ height: '100%', background: 'linear-gradient(90deg, #7fc2ff, #f3596b)', borderRadius: 999, width: `${analysisProgress}%`, transition: 'width 0.3s ease' }} />
              </div>
              <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', color: MUTED }}>
                DO NOT CLOSE THIS TAB &middot; {Math.round(analysisProgress)}%
              </div>
            </section>
          )}

          {/* ── ANALYSIS RESULTS (post-analysis) ── */}
          {project && cuesSorted.length > 0 && (
            <section>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                <h2 style={{ fontSize: 18, ...titleHeavy, margin: 0, color: TEXT }}>Analysis Results</h2>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
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
            <div style={{ backgroundColor: 'rgba(153, 27, 27, 0.2)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 16, padding: 16, fontSize: 14, fontWeight: 500, color: '#fca5a5' }}>
              {error}
            </div>
          )}
        </div>
      </main>

      {/* ── FOOTER ── */}
      <footer style={{
        height: 68, flexShrink: 0,
        borderTop: `1px solid ${LINE}`,
        background: 'rgba(11, 13, 18, 0.95)',
        backdropFilter: 'blur(12px)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 28px',
        position: 'relative', zIndex: 50,
      }}>
        {/* Left: state badge + info */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <StateBadge state={fileState} />
          <div style={{ fontSize: 12, color: MUTED, display: 'flex', alignItems: 'center', gap: 10 }}>
            {(segments.length > 0 || cuesSorted.length > 0) && <span>{cuesSorted.length || segments.length} segments</span>}
            {duration > 0 && (
              <>
                <span style={{ width: 4, height: 4, borderRadius: '50%', backgroundColor: MUTED }} />
                <span>{fmtDur(duration)} total</span>
              </>
            )}
          </div>
        </div>

        {/* Right: actions */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {fileState === 'empty' && (
            <button onClick={() => fileInputRef.current?.click()} style={{ ...pillBtn, backgroundColor: ACCENT_WARM, color: '#1a1408' }}>
              Upload File
            </button>
          )}
          {fileState === 'new' && (
            <button onClick={handlePreview} disabled={isSegmenting} style={{ ...pillBtn, backgroundColor: ACCENT_WARM, color: '#1a1408' }}>
              {isSegmenting ? 'Detecting...' : 'Detect Segments'}
            </button>
          )}
          {fileState === 'segmented' && (
            <button onClick={handleAnalyze} disabled={selectedSegments.length === 0} style={{ ...pillBtn, backgroundColor: ACCENT_WARM, color: '#1a1408', opacity: selectedSegments.length === 0 ? 0.4 : 1 }}>
              Analyze {selectedSegments.length} Segments
            </button>
          )}
          {fileState === 'analyzing' && (
            <button disabled style={{ ...pillBtn, backgroundColor: '#2a3142', color: MUTED, border: `1px solid ${LINE}` }}>
              Analyzing {Math.round(analysisProgress)}%...
            </button>
          )}
          {fileState === 'analyzed' && (
            <>
              <button onClick={() => resetToSegmentation()} style={{ ...pillBtn, backgroundColor: 'transparent', color: TEXT, border: `1px solid ${LINE}` }}>Re-analyze</button>
              <button onClick={() => setCurrentPage('workspace')} style={{ ...pillBtn, backgroundColor: 'transparent', color: TEXT, border: `1px solid ${LINE}` }}>Back to Workspace</button>
              <button onClick={() => setCurrentPage('replacement')} style={{ ...pillBtn, backgroundColor: ACCENT_WARM, color: '#1a1408' }}>Find Matches</button>
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
  padding: '10px 20px', borderRadius: 999, fontWeight: 700, fontSize: 13,
  border: 'none', cursor: 'pointer', transition: 'all 0.2s',
  fontFamily: FONT_UI,
}

function StateBadge({ state }: { state: string }) {
  const styles: Record<string, React.CSSProperties> = {
    empty: { backgroundColor: '#2a3142', color: MUTED, border: `1px solid ${LINE}` },
    new: { backgroundColor: '#2a3142', color: TEXT, border: `1px solid ${LINE}` },
    segmented: { backgroundColor: 'rgba(99, 183, 255, 0.15)', color: '#9ecfff', border: '1px solid rgba(99, 183, 255, 0.25)' },
    analyzing: { backgroundColor: 'rgba(242, 184, 77, 0.2)', color: ACCENT_WARM, border: '1px solid rgba(242, 184, 77, 0.3)' },
    analyzed: { backgroundColor: 'rgba(99, 183, 255, 0.15)', color: '#9ecfff', border: '1px solid rgba(99, 183, 255, 0.25)' },
    matched: { backgroundColor: 'rgba(96, 198, 177, 0.15)', color: '#60c6b1', border: '1px solid rgba(96, 198, 177, 0.25)' },
  }
  const labels: Record<string, string> = {
    empty: 'No File', new: 'New', segmented: 'Segmented',
    analyzing: 'Analyzing', analyzed: 'Analyzed', matched: 'Matched',
  }
  return (
    <div style={{ ...styles[state], padding: '6px 12px', borderRadius: 999, fontSize: 11, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 6 }}>
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
        backgroundColor: elevated ? BG_CARD : BG_ELEVATED,
        borderRadius: 16, padding: '12px 16px',
        borderLeft: `3px solid ${selected ? ACCENT : 'transparent'}`,
        border: `1px solid ${elevated ? 'rgba(255,255,255,0.12)' : LINE}`,
        boxShadow: elevated ? '0 10px 25px -5px rgba(0,0,0,0.3)' : 'none',
        transform: elevated ? 'translateY(-2px)' : 'translateY(0)',
        transition: 'all 0.2s ease', cursor: 'pointer',
        display: 'flex', alignItems: 'center', gap: 12,
      }}
    >
      {/* Checkbox */}
      <div style={{
        width: 24, height: 24, borderRadius: '50%',
        backgroundColor: selected ? '#16a34a' : 'rgba(255,255,255,0.06)',
        border: selected ? 'none' : '2px solid rgba(255,255,255,0.15)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: '#fff', fontSize: 12, fontWeight: 700, flexShrink: 0,
      }}>
        {selected && '✓'}
      </div>
      <span style={{ fontWeight: 700, fontSize: 14, width: 28, color: TEXT }}>#{index + 1}</span>
      <span style={{ fontFamily: FONT_MONO, fontSize: 12, fontWeight: 500, color: MUTED, flex: 1 }}>{tc0} — {tc1}</span>
      <span style={{ fontFamily: FONT_MONO, fontSize: 12, fontWeight: 700, color: TEXT }}>{dur}</span>
    </div>
  )
}

function SliderRow({ label, min, max, step, value, onChange, format }: {
  label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void; format: (v: number) => string
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
      <label style={{ width: 88, fontSize: 11, fontWeight: 700, color: MUTED, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</label>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ flex: 1, accentColor: ACCENT }} />
      <span style={{ width: 64, textAlign: 'right', fontFamily: FONT_MONO, fontSize: 12, color: TEXT }}>{format(value)}</span>
    </div>
  )
}
