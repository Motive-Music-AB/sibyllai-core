import { useState } from 'react'
import { useAppStore } from '@/lib/store'

/* ═══════════════════════════════════════════════════════════════════════════
   ICONS
   ═══════════════════════════════════════════════════════════════════════════ */

function AudioBarsIcon({ size = 24 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M216,64V192a24,24,0,0,1-24,24H64a24,24,0,0,1-24-24V64A24,24,0,0,1,64,40H192A24,24,0,0,1,216,64ZM144,88a8,8,0,0,0-8-8H120a8,8,0,0,0-8,8v80a8,8,0,0,0,8,8h16a8,8,0,0,0,8-8ZM96,112a8,8,0,0,0-8-8H72a8,8,0,0,0-8,8v32a8,8,0,0,0,8,8H88a8,8,0,0,0,8-8Zm88-16a8,8,0,0,0-8-8H160a8,8,0,0,0-8,8v64a8,8,0,0,0,8,8h16a8,8,0,0,0,8-8Z" />
    </svg>
  )
}

function PlusIcon({ size = 16 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M224,128a8,8,0,0,1-8,8H136v80a8,8,0,0,1-16,0V136H40a8,8,0,0,1,0-16h80V40a8,8,0,0,1,16,0v80h80A8,8,0,0,1,224,128Z" />
    </svg>
  )
}

function ChevronRightIcon({ size = 20 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M181.66,133.66l-80,80a8,8,0,0,1-11.32-11.32L164.69,128,90.34,53.66a8,8,0,0,1,11.32-11.32l80,80A8,8,0,0,1,181.66,133.66Z" />
    </svg>
  )
}

function ChevronDownIcon({ size = 12 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M213.66,101.66l-80,80a8,8,0,0,1-11.32,0l-80-80A8,8,0,0,1,53.66,90.34L128,164.69l74.34-74.35a8,8,0,0,1,11.32,11.32Z" />
    </svg>
  )
}

function SearchIcon({ size = 18 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M229.66,218.34l-50.07-50.06a88.11,88.11,0,1,0-11.31,11.31l50.06,50.07a8,8,0,0,0,11.32-11.32ZM40,112a72,72,0,1,1,72,72A72.08,72.08,0,0,1,40,112Z" />
    </svg>
  )
}

function FilterIcon({ size = 14 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M200,128a8,8,0,0,1-8,8H64a8,8,0,0,1,0-16H192A8,8,0,0,1,200,128Zm24-56H32a8,8,0,0,0,0,16H224a8,8,0,0,0,0-16Zm-72,112H104a8,8,0,0,0,0,16h48a8,8,0,0,0,0-16Z" />
    </svg>
  )
}

function FolderIcon({ size = 16 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M216,40H40A16,16,0,0,0,24,56V200a16,16,0,0,0,16,16H216a16,16,0,0,0,16-16V56A16,16,0,0,0,216,40Zm0,16V72H40V56ZM40,200V88H216V200Z" />
    </svg>
  )
}

function ClockIcon({ size = 16 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M128,24A104,104,0,1,0,232,128,104.11,104.11,0,0,0,128,24Zm0,192a88,88,0,1,1,88-88A88.1,88.1,0,0,1,128,216Zm64-88a8,8,0,0,1-8,8H128a8,8,0,0,1-8-8V72a8,8,0,0,1,16,0v48h48A8,8,0,0,1,192,128Z" />
    </svg>
  )
}

function ArrowLeftIcon({ size = 16 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M224,128a8,8,0,0,1-8,8H59.31l58.35,58.34a8,8,0,0,1-11.32,11.32l-72-72a8,8,0,0,1,0-11.32l72-72a8,8,0,0,1,11.32,11.32L59.31,120H216A8,8,0,0,1,224,128Z" />
    </svg>
  )
}

function InfoIcon({ size = 24 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M228,128a100,100,0,1,1-100-100A100.11,100.11,0,0,1,228,128Zm-100-8a8,8,0,0,0-8,8v48a8,8,0,0,0,16,0V128A8,8,0,0,0,128,120Zm0-32a12,12,0,1,0,12,12A12,12,0,0,0,128,88Z" />
    </svg>
  )
}

function DraftThumbIcon({ size = 20 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M208,32H48A16,16,0,0,0,32,48V208a16,16,0,0,0,16,16H208a16,16,0,0,0,16-16V48A16,16,0,0,0,208,32Zm-32,80H136v40a8,8,0,0,1-16,0V112H80a8,8,0,0,1,0-16h40V56a8,8,0,0,1,16,0V96h40a8,8,0,0,1,0,16Z" />
    </svg>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   DATA
   ═══════════════════════════════════════════════════════════════════════════ */

type ProjectStatus = 'complete' | 'processing' | 'draft'

interface DemoProject {
  id: string
  name: string
  status: ProjectStatus
  analysisPercent?: number
  modifiedLabel: string
  createdLabel: string
  cues: number | null
  tracks: number | null
  duration: string
  thumbType: 'bars-yellow' | 'bars-white' | 'processing' | 'info' | 'draft'
  barHeights?: number[]
}

const DEMO_PROJECTS: DemoProject[] = [
  {
    id: 'p1',
    name: 'Cinematic Trailer Score Vol. 1',
    status: 'complete',
    modifiedLabel: 'Modified 2h ago',
    createdLabel: 'Created Oct 24, 2023',
    cues: 42,
    tracks: 8,
    duration: '01:14:30',
    thumbType: 'bars-yellow',
    barHeights: [40, 80, 50, 100, 30, 60],
  },
  {
    id: 'p2',
    name: 'Podcast Ep. 142 - Deep Dive',
    status: 'processing',
    analysisPercent: 64,
    modifiedLabel: '',
    createdLabel: 'Created Today, 09:15 AM',
    cues: null,
    tracks: 1,
    duration: '02:05:11',
    thumbType: 'processing',
    barHeights: [20, 60, 90, 40, 70, 30],
  },
  {
    id: 'p3',
    name: 'Corporate Promo Q4 - Draft',
    status: 'complete',
    modifiedLabel: 'Modified Yesterday',
    createdLabel: 'Created Oct 20, 2023',
    cues: 12,
    tracks: 3,
    duration: '00:08:45',
    thumbType: 'info',
    barHeights: [],
  },
  {
    id: 'p4',
    name: 'Untitled Session 04',
    status: 'draft',
    modifiedLabel: '',
    createdLabel: 'Created Oct 18, 2023',
    cues: 0,
    tracks: 0,
    duration: '--:--:--',
    thumbType: 'draft',
    barHeights: [],
  },
  {
    id: 'p5',
    name: 'Night City Ambience Batch',
    status: 'complete',
    modifiedLabel: 'Modified Oct 15, 2023',
    createdLabel: 'Created Oct 12, 2023',
    cues: 156,
    tracks: 24,
    duration: '04:32:15',
    thumbType: 'bars-white',
    barHeights: [60, 30, 90, 100, 70, 40],
  },
  {
    id: 'p6',
    name: 'Documentary Feature: Oceans',
    status: 'complete',
    modifiedLabel: 'Modified Sep 28, 2023',
    createdLabel: 'Created Sep 05, 2023',
    cues: 89,
    tracks: 12,
    duration: '01:45:00',
    thumbType: 'bars-white',
    barHeights: [20, 40, 80, 100, 50, 30],
  },
]

/* ═══════════════════════════════════════════════════════════════════════════
   STYLES
   ═══════════════════════════════════════════════════════════════════════════ */

const PAGE_BG = '#D6D6D4'
const CARD_BG = '#EBEBEB'
const TEXT_MAIN = '#000000'
const TEXT_SUB = '#4A4A4A'
const ACCENT_YELLOW = '#FFD659'
const RADIUS_LG = 28
const RADIUS_MD = 16
const FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'

const titleHeavy: React.CSSProperties = {
  fontWeight: 800,
  letterSpacing: '-0.04em',
  lineHeight: 1.1,
}

/* ═══════════════════════════════════════════════════════════════════════════
   SUB-COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */

/** Waveform thumbnail — small black box with colored bars */
function WaveformThumb({ bars, color }: { bars: number[]; color: string }) {
  return (
    <div
      style={{
        width: 80,
        height: 56,
        backgroundColor: '#000',
        borderRadius: 12,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 8,
        flexShrink: 0,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          justifyContent: 'center',
          gap: 2,
          height: 24,
          width: '100%',
          opacity: 0.8,
        }}
      >
        {bars.map((h, i) => (
          <div
            key={i}
            style={{
              width: 6,
              height: `${h}%`,
              backgroundColor: color,
              borderRadius: 99,
            }}
          />
        ))}
      </div>
    </div>
  )
}

/** Processing thumbnail with shimmer + animated wave bars */
function ProcessingThumb({ bars }: { bars: number[] }) {
  return (
    <div
      style={{
        width: 80,
        height: 56,
        backgroundColor: '#F3F3F1',
        borderRadius: 12,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 8,
        flexShrink: 0,
        border: '1px solid rgba(0,0,0,0.05)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Shimmer */}
      <div
        className="shimmer-slide"
        style={{
          position: 'absolute',
          inset: 0,
          background: `linear-gradient(to right, transparent, ${ACCENT_YELLOW}66, transparent)`,
          width: '200%',
        }}
      />
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          justifyContent: 'center',
          gap: 2,
          height: 24,
          width: '100%',
          opacity: 0.3,
        }}
      >
        {bars.map((h, i) => (
          <div
            key={i}
            className="wave-bar"
            style={{
              width: 6,
              height: `${h}%`,
              backgroundColor: '#000',
              borderRadius: 99,
              transformOrigin: 'bottom',
            }}
          />
        ))}
      </div>
    </div>
  )
}

/** Info icon thumbnail (for projects with no waveform but complete) */
function InfoThumb() {
  return (
    <div
      style={{
        width: 80,
        height: 56,
        backgroundColor: CARD_BG,
        borderRadius: 12,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        border: '1px solid rgba(0,0,0,0.05)',
        color: 'rgba(0,0,0,0.3)',
      }}
    >
      <InfoIcon />
    </div>
  )
}

/** Draft thumbnail — dashed border + plus icon */
function DraftThumb() {
  return (
    <div
      className="draft-thumb"
      style={{
        width: 80,
        height: 56,
        backgroundColor: 'transparent',
        borderRadius: 12,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        border: '2px dashed rgba(0,0,0,0.1)',
        color: 'rgba(0,0,0,0.2)',
        transition: 'border-color 0.2s',
      }}
    >
      <DraftThumbIcon />
    </div>
  )
}

/** Badge */
function Badge({ variant, children }: { variant: 'dark' | 'light' | 'accent'; children: React.ReactNode }) {
  const styles: Record<string, React.CSSProperties> = {
    dark: { background: '#000', color: '#fff' },
    light: { background: 'rgba(0,0,0,0.05)', color: TEXT_SUB, border: '1px solid rgba(0,0,0,0.1)' },
    accent: { background: ACCENT_YELLOW, color: '#000' },
  }
  return (
    <span
      style={{
        fontSize: 10,
        fontWeight: 800,
        padding: '4px 8px',
        borderRadius: 6,
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        ...styles[variant],
      }}
    >
      {children}
    </span>
  )
}

/** Pill button */
function PillButton({
  children,
  onClick,
  style,
}: {
  children: React.ReactNode
  onClick?: () => void
  style?: React.CSSProperties
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        padding: '8px 20px',
        borderRadius: 999,
        fontWeight: 700,
        fontSize: 14,
        border: '1px solid rgba(0,0,0,0.05)',
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        backgroundColor: CARD_BG,
        color: TEXT_MAIN,
        fontFamily: FONT_FAMILY,
        ...style,
      }}
    >
      {children}
    </button>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   PROJECT CARD
   ═══════════════════════════════════════════════════════════════════════════ */

function ProjectCard({ project, onOpen }: { project: DemoProject; onOpen: () => void }) {
  const [hovered, setHovered] = useState(false)
  const isProcessing = project.status === 'processing'
  const isDraft = project.status === 'draft'

  const cardBg = isProcessing ? '#FFFFFF' : '#F3F3F1'
  const cardBorder = isProcessing
    ? '2px solid rgba(0,0,0,0.1)'
    : '2px solid transparent'

  const cardStyle: React.CSSProperties = {
    backgroundColor: cardBg,
    borderRadius: RADIUS_MD,
    padding: 16,
    display: 'flex',
    alignItems: 'center',
    gap: 24,
    cursor: 'pointer',
    transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
    border: cardBorder,
    boxShadow: isProcessing ? '0 2px 8px rgba(0,0,0,0.06)' : 'none',
    opacity: isDraft ? 0.8 : 1,
    ...(hovered
      ? {
          transform: 'translateY(-4px)',
          backgroundColor: '#FFFFFF',
          borderColor: TEXT_MAIN,
          boxShadow: '0 10px 30px -10px rgba(0,0,0,0.1)',
          opacity: 1,
        }
      : {}),
  }

  // Render thumbnail
  let thumb: React.ReactNode
  if (project.thumbType === 'bars-yellow') {
    thumb = <WaveformThumb bars={project.barHeights || []} color={ACCENT_YELLOW} />
  } else if (project.thumbType === 'bars-white') {
    thumb = <WaveformThumb bars={project.barHeights || []} color="#FFFFFF" />
  } else if (project.thumbType === 'processing') {
    thumb = <ProcessingThumb bars={project.barHeights || []} />
  } else if (project.thumbType === 'info') {
    thumb = <InfoThumb />
  } else {
    thumb = <DraftThumb />
  }

  return (
    <div
      style={cardStyle}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={onOpen}
    >
      {/* Thumbnail */}
      {thumb}

      {/* Info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <h3
          style={{
            fontWeight: 700,
            fontSize: 18,
            lineHeight: 1.2,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            margin: 0,
            textDecoration: hovered && !isDraft ? 'underline' : 'none',
            textDecorationThickness: 2,
            textUnderlineOffset: 2,
            color: isDraft && !hovered ? 'rgba(0,0,0,0.6)' : TEXT_MAIN,
            transition: 'color 0.2s',
          }}
        >
          {project.name}
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 6 }}>
          {isProcessing ? (
            <>
              <span
                style={{
                  fontSize: 12,
                  color: ACCENT_YELLOW,
                  fontWeight: 700,
                  backgroundColor: '#000',
                  padding: '2px 8px',
                  borderRadius: 2,
                }}
              >
                Analyzing audio... {project.analysisPercent}%
              </span>
              <span style={{ width: 4, height: 4, borderRadius: '50%', backgroundColor: 'rgba(0,0,0,0.2)' }} />
              <span style={{ fontSize: 12, color: TEXT_SUB }}>{project.createdLabel}</span>
            </>
          ) : (
            <>
              {project.modifiedLabel && (
                <>
                  <span style={{ fontSize: 12, color: TEXT_SUB, fontWeight: 500 }}>{project.modifiedLabel}</span>
                  <span style={{ width: 4, height: 4, borderRadius: '50%', backgroundColor: 'rgba(0,0,0,0.2)' }} />
                </>
              )}
              <span style={{ fontSize: 12, color: TEXT_SUB }}>{project.createdLabel}</span>
            </>
          )}
        </div>
      </div>

      {/* Cues / Tracks column */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          padding: '0 32px',
          borderLeft: '1px solid rgba(0,0,0,0.1)',
          height: 40,
          flexShrink: 0,
          minWidth: 140,
          opacity: isProcessing ? 0.5 : isDraft && (project.cues === 0) ? 0.3 : 1,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
          <span style={{ fontWeight: 700, fontSize: 16 }}>{project.cues ?? '--'}</span>
          <span style={{ fontSize: 12, fontWeight: 700, color: TEXT_SUB, textTransform: 'uppercase' }}>Cues</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 2 }}>
          <span style={{ fontWeight: 700, fontSize: 14 }}>{project.tracks ?? '--'}</span>
          <span style={{ fontSize: 12, color: TEXT_SUB, textTransform: 'uppercase' }}>
            {project.tracks === 1 ? 'Track' : 'Tracks'}
          </span>
        </div>
      </div>

      {/* Duration column */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          padding: '0 32px',
          borderLeft: '1px solid rgba(0,0,0,0.1)',
          height: 40,
          flexShrink: 0,
          minWidth: 120,
          opacity: project.duration === '--:--:--' ? 0.3 : 1,
        }}
      >
        <span style={{ fontWeight: 700, fontSize: 16, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}>
          {project.duration}
        </span>
        <span style={{ fontSize: 10, fontWeight: 700, color: TEXT_SUB, textTransform: 'uppercase', marginTop: 2 }}>
          Total Duration
        </span>
      </div>

      {/* Status badge */}
      <div style={{ width: 112, display: 'flex', justifyContent: 'flex-end', flexShrink: 0 }}>
        {project.status === 'complete' && <Badge variant="dark">Complete</Badge>}
        {project.status === 'processing' && (
          <Badge variant="accent">
            <span
              style={{
                width: 6,
                height: 6,
                backgroundColor: '#000',
                borderRadius: '50%',
                animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
              }}
            />
            Processing
          </Badge>
        )}
        {project.status === 'draft' && <Badge variant="light">Draft</Badge>}
      </div>

      {/* Chevron */}
      <div
        style={{
          width: 32,
          display: 'flex',
          justifyContent: 'flex-end',
          flexShrink: 0,
          color: hovered ? '#000' : 'rgba(0,0,0,0.2)',
          transition: 'all 0.2s',
          transform: hovered ? 'translateX(4px)' : 'translateX(0)',
        }}
      >
        <ChevronRightIcon />
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   STAT CARDS
   ═══════════════════════════════════════════════════════════════════════════ */

function StatCard({
  icon,
  label,
  value,
  suffix,
  dark,
}: {
  icon: React.ReactNode
  label: string
  value: string
  suffix?: string
  dark?: boolean
}) {
  return (
    <div
      style={{
        flex: 1,
        backgroundColor: dark ? '#000' : CARD_BG,
        color: dark ? '#fff' : TEXT_MAIN,
        borderRadius: RADIUS_LG,
        padding: 24,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        height: 128,
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span
          style={{
            fontSize: 11,
            fontWeight: 700,
            color: dark ? 'rgba(255,255,255,0.5)' : TEXT_SUB,
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
          }}
        >
          {label}
        </span>
        <span style={{ color: dark ? 'rgba(255,255,255,0.3)' : 'rgba(74,74,74,0.5)' }}>{icon}</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span
          style={{
            fontSize: 48,
            ...titleHeavy,
            color: dark ? ACCENT_YELLOW : TEXT_MAIN,
          }}
        >
          {value}
        </span>
        {suffix && (
          <span style={{ fontSize: 14, fontWeight: 700, color: dark ? 'rgba(255,255,255,0.5)' : TEXT_SUB }}>
            {suffix}
          </span>
        )}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════════════════ */

export function ProjectsPage() {
  const { logout, setCurrentPage, userName } = useAppStore()
  const [searchQuery, setSearchQuery] = useState('')

  const handleOpenProject = () => {
    setCurrentPage('workspace')
  }

  const handleSignOut = () => {
    logout()
  }

  const initials = userName
    ? userName
        .split(/\s+/)
        .map((w) => w[0])
        .join('')
        .toUpperCase()
        .slice(0, 2)
    : 'U'

  return (
    <div
      style={{
        height: '100vh',
        width: '100vw',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        padding: 32,
        gap: 32,
        boxSizing: 'border-box',
        backgroundColor: PAGE_BG,
        color: TEXT_MAIN,
        fontFamily: FONT_FAMILY,
        WebkitFontSmoothing: 'antialiased',
      }}
    >
      {/* ── HEADER ── */}
      <header style={{ height: 56, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div
            style={{
              width: 48,
              height: 48,
              backgroundColor: '#000',
              color: '#fff',
              borderRadius: 14,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
            }}
          >
            <AudioBarsIcon />
          </div>
          <div>
            <h1 style={{ fontSize: 24, ...titleHeavy, margin: 0, marginTop: 4 }}>MOTIVE</h1>
            <div
              style={{
                fontSize: 10,
                fontWeight: 700,
                color: TEXT_SUB,
                textTransform: 'uppercase',
                letterSpacing: '0.15em',
                marginTop: 2,
              }}
            >
              Studio Edition
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <button
            style={{
              width: 40,
              height: 40,
              borderRadius: '50%',
              border: '2px solid rgba(0,0,0,0.1)',
              backgroundColor: 'transparent',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              transition: 'background-color 0.2s',
              color: TEXT_MAIN,
            }}
            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(0,0,0,0.05)' }}
            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent' }}
          >
            <SearchIcon />
          </button>
          <button
            onClick={handleOpenProject}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
              padding: '8px 20px',
              borderRadius: 999,
              fontWeight: 700,
              fontSize: 14,
              backgroundColor: ACCENT_YELLOW,
              color: TEXT_MAIN,
              border: 'none',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
              fontFamily: FONT_FAMILY,
            }}
          >
            <PlusIcon />
            New Project
          </button>
          <div style={{ width: 1, height: 24, backgroundColor: 'rgba(0,0,0,0.1)', margin: '0 8px' }} />
          <button
            onClick={handleSignOut}
            style={{
              width: 40,
              height: 40,
              borderRadius: '50%',
              backgroundColor: '#000',
              color: '#fff',
              border: 'none',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              fontWeight: 700,
              fontSize: 14,
              boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
              transition: 'all 0.2s',
              fontFamily: FONT_FAMILY,
            }}
          >
            {initials}
          </button>
        </div>
      </header>

      {/* ── STAT CARDS ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 24, flexShrink: 0 }}>
        <StatCard icon={<FolderIcon />} label="Total Projects" value="142" />
        <StatCard dark icon={<ClockIcon />} label="Hours Analyzed" value="3,402" suffix="hrs" />
        <StatCard icon={<ArrowLeftIcon />} label="Tracks Processed" value="12,845" />
      </div>

      {/* ── PROJECT LIST SECTION ── */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, gap: 16 }}>
        {/* Header + filters */}
        <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flexShrink: 0, padding: '0 8px' }}>
          <h2 style={{ fontSize: 24, ...titleHeavy, margin: 0 }}>All Projects</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {/* Inline search */}
            <div style={{ position: 'relative' }}>
              <div
                style={{
                  position: 'absolute',
                  left: 12,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: TEXT_SUB,
                  pointerEvents: 'none',
                }}
              >
                <SearchIcon size={14} />
              </div>
              <input
                type="text"
                placeholder="Search projects..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  backgroundColor: CARD_BG,
                  fontSize: 14,
                  fontWeight: 700,
                  paddingLeft: 36,
                  paddingRight: 16,
                  paddingTop: 8,
                  paddingBottom: 8,
                  borderRadius: 999,
                  border: '1px solid rgba(0,0,0,0.05)',
                  outline: 'none',
                  width: 256,
                  transition: 'all 0.2s',
                  color: TEXT_MAIN,
                  fontFamily: FONT_FAMILY,
                }}
                onFocus={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(0,0,0,0.3)'
                  e.currentTarget.style.backgroundColor = '#FFFFFF'
                }}
                onBlur={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(0,0,0,0.05)'
                  e.currentTarget.style.backgroundColor = CARD_BG
                }}
              />
            </div>
            <PillButton style={{ fontSize: 14 }}>
              <FilterIcon />
              Sort: Recent
            </PillButton>
            <PillButton style={{ fontSize: 14 }}>
              Status: All
              <ChevronDownIcon />
            </PillButton>
          </div>
        </div>

        {/* Scrollable list */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            paddingRight: 8,
            paddingBottom: 32,
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {DEMO_PROJECTS.map((project) => (
              <ProjectCard key={project.id} project={project} onOpen={handleOpenProject} />
            ))}

            {/* Load more */}
            <button
              style={{
                borderRadius: RADIUS_MD,
                padding: 16,
                marginTop: 8,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                border: '2px dashed rgba(0,0,0,0.1)',
                backgroundColor: 'transparent',
                cursor: 'pointer',
                transition: 'all 0.2s',
                width: '100%',
                fontFamily: FONT_FAMILY,
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'rgba(0,0,0,0.05)'
                e.currentTarget.style.borderColor = 'rgba(0,0,0,0.3)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent'
                e.currentTarget.style.borderColor = 'rgba(0,0,0,0.1)'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ color: 'rgba(0,0,0,0.5)' }}>
                  <PlusIcon size={20} />
                </span>
                <span style={{ fontWeight: 700, fontSize: 14, letterSpacing: '0.02em' }}>Load More Projects</span>
              </div>
            </button>
          </div>
        </div>
      </main>

      {/* Keyframe animations */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes pulse-height {
          0%, 100% { transform: scaleY(0.4); }
          50% { transform: scaleY(1); }
        }
        .wave-bar {
          transform-origin: bottom;
          animation: pulse-height 1.5s ease-in-out infinite;
        }
        .wave-bar:nth-child(2) { animation-delay: 0.2s; }
        .wave-bar:nth-child(3) { animation-delay: 0.4s; }
        .wave-bar:nth-child(4) { animation-delay: 0.1s; }
        .wave-bar:nth-child(5) { animation-delay: 0.5s; }
        @keyframes slide {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(50%); }
        }
        .shimmer-slide {
          animation: slide 1.5s ease-in-out infinite;
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.25); }
      `}</style>
    </div>
  )
}
