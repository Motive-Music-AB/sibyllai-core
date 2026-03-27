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

function ArrowLeftIcon({ size = 20 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M224,128a8,8,0,0,1-8,8H59.31l58.35,58.34a8,8,0,0,1-11.32,11.32l-72-72a8,8,0,0,1,0-11.32l72-72a8,8,0,0,1,11.32,11.32L59.31,120H216A8,8,0,0,1,224,128Z" />
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

function FilterIcon({ size = 14 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M200,128a8,8,0,0,1-8,8H64a8,8,0,0,1,0-16H192A8,8,0,0,1,200,128Zm24-56H32a8,8,0,0,0,0,16H224a8,8,0,0,0,0-16Zm-72,112H104a8,8,0,0,0,0,16h48a8,8,0,0,0,0-16Z" />
    </svg>
  )
}

function FilmFrameIcon({ size = 24 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M216,40H40A16,16,0,0,0,24,56V200a16,16,0,0,0,16,16H216a16,16,0,0,0,16-16V56A16,16,0,0,0,216,40ZM40,56H72V88H40ZM40,104H72v48H40Zm0,64H72v32H40Zm176,32H88V56H216V200Z" />
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

type FileStatus = 'analyzed' | 'processing' | 'queued' | 'new'
type FileType = 'audio' | 'video'

interface DemoFile {
  id: string
  name: string
  fileType: FileType
  format: string
  specs: string
  durationShort: string
  durationFull: string
  status: FileStatus
  processingPercent?: number
  tags?: string
  cues: number | null
  sections: number | null
  thumbVariant: 'bars-yellow' | 'video-analyzed' | 'processing' | 'video-queued' | 'draft'
  barHeights?: number[]
}

const DEMO_FILES: DemoFile[] = [
  {
    id: 'f1',
    name: 'MX_Reel_01_Main_Titles.wav',
    fileType: 'audio',
    format: 'WAV',
    specs: '48.0 kHz · 24-bit',
    durationShort: '04:32',
    durationFull: '00:04:32',
    status: 'analyzed',
    tags: 'Epic Orchestral, Brass Section, Building',
    cues: 8,
    sections: 3,
    thumbVariant: 'bars-yellow',
    barHeights: [60, 30, 80, 100, 50, 70],
  },
  {
    id: 'f2',
    name: 'Spot_A_Summer_Campaign_v3.mp4',
    fileType: 'video',
    format: 'MP4',
    specs: '1920x1080 · 24fps',
    durationShort: '00:30',
    durationFull: '00:00:30',
    status: 'analyzed',
    tags: 'Upbeat Pop, Energetic, Synth',
    cues: 2,
    sections: 1,
    thumbVariant: 'video-analyzed',
  },
  {
    id: 'f3',
    name: 'MX_Reel_02_Love_Theme.wav',
    fileType: 'audio',
    format: 'WAV',
    specs: '48.0 kHz · 24-bit',
    durationShort: '06:15',
    durationFull: '00:06:15',
    status: 'processing',
    processingPercent: 64,
    cues: null,
    sections: null,
    thumbVariant: 'processing',
    barHeights: [40, 80, 60, 30, 90, 50],
  },
  {
    id: 'f4',
    name: 'Spot_B_Holiday_Edit.mov',
    fileType: 'video',
    format: 'MOV',
    specs: '3840x2160 · 30fps',
    durationShort: '00:45',
    durationFull: '00:00:45',
    status: 'queued',
    cues: null,
    sections: null,
    thumbVariant: 'video-queued',
  },
  {
    id: 'f5',
    name: 'MX_Reel_03_Chase_Sequence.wav',
    fileType: 'audio',
    format: 'WAV',
    specs: '48.0 kHz · 24-bit',
    durationShort: '08:47',
    durationFull: '00:08:47',
    status: 'analyzed',
    tags: 'Action, Percussion, Intense',
    cues: 14,
    sections: 6,
    thumbVariant: 'bars-yellow',
    barHeights: [90, 40, 100, 70, 50, 30],
  },
  {
    id: 'f6',
    name: 'Untitled_Recording_04.wav',
    fileType: 'audio',
    format: 'WAV',
    specs: '44.1 kHz · 16-bit',
    durationShort: '02:10',
    durationFull: '00:02:10',
    status: 'new',
    cues: 0,
    sections: 0,
    thumbVariant: 'draft',
  },
]

/* ═══════════════════════════════════════════════════════════════════════════
   STYLES
   ═══════════════════════════════════════════════════════════════════════════ */

const PAGE_BG = '#D6D6D4'
const CARD_BG = '#EBEBEB'
const LIST_BG = '#F3F3F1'
const TEXT_MAIN = '#000000'
const TEXT_SUB = '#4A4A4A'
const ACCENT_YELLOW = '#FFD659'
const FONT_FAMILY = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'

const titleHeavy: React.CSSProperties = {
  fontWeight: 800,
  letterSpacing: '-0.04em',
  lineHeight: 1.1,
}

/* ═══════════════════════════════════════════════════════════════════════════
   THUMBNAILS
   ═══════════════════════════════════════════════════════════════════════════ */

function AudioThumbYellow({ bars }: { bars: number[] }) {
  return (
    <div
      style={{
        width: 80,
        height: 56,
        backgroundColor: '#000',
        borderRadius: 12,
        display: 'flex',
        alignItems: 'flex-end',
        justifyContent: 'center',
        gap: 2,
        paddingBottom: 8,
        flexShrink: 0,
      }}
    >
      {bars.map((h, i) => (
        <div
          key={i}
          style={{
            width: 6,
            height: `${h}%`,
            backgroundColor: ACCENT_YELLOW,
            borderRadius: 99,
            opacity: 0.8,
          }}
        />
      ))}
    </div>
  )
}

function VideoThumbAnalyzed() {
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
        flexShrink: 0,
        color: 'rgba(255,255,255,0.5)',
      }}
    >
      <FilmFrameIcon />
    </div>
  )
}

function ProcessingThumb({ bars }: { bars: number[] }) {
  return (
    <div
      style={{
        width: 80,
        height: 56,
        backgroundColor: LIST_BG,
        border: '1px solid rgba(0,0,0,0.05)',
        borderRadius: 12,
        display: 'flex',
        alignItems: 'flex-end',
        justifyContent: 'center',
        gap: 2,
        paddingBottom: 8,
        flexShrink: 0,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div
        className="shimmer-slide"
        style={{
          position: 'absolute',
          inset: 0,
          background: `linear-gradient(to right, transparent, ${ACCENT_YELLOW}66, transparent)`,
          width: '200%',
        }}
      />
      {bars.map((h, i) => (
        <div
          key={i}
          className="wave-bar"
          style={{
            width: 6,
            height: `${h}%`,
            backgroundColor: '#000',
            borderRadius: 99,
            opacity: 0.3,
            transformOrigin: 'bottom',
            position: 'relative',
            zIndex: 10,
          }}
        />
      ))}
    </div>
  )
}

function VideoThumbQueued() {
  return (
    <div
      style={{
        width: 80,
        height: 56,
        backgroundColor: CARD_BG,
        border: '2px dashed rgba(0,0,0,0.1)',
        borderRadius: 12,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        color: 'rgba(0,0,0,0.2)',
      }}
    >
      <FilmFrameIcon />
    </div>
  )
}

function DraftThumb() {
  return (
    <div
      style={{
        width: 80,
        height: 56,
        backgroundColor: 'transparent',
        border: '2px dashed rgba(0,0,0,0.1)',
        borderRadius: 12,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        color: 'rgba(0,0,0,0.2)',
        transition: 'border-color 0.2s',
      }}
    >
      <DraftThumbIcon />
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   BADGE COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */

function FormatBadge({ format }: { format: string }) {
  return (
    <span
      style={{
        fontSize: 10,
        fontWeight: 700,
        textTransform: 'uppercase',
        backgroundColor: 'rgba(0,0,0,0.08)',
        padding: '4px 6px',
        borderRadius: 4,
        color: TEXT_MAIN,
        lineHeight: 1,
      }}
    >
      {format}
    </span>
  )
}

function StatusBadge({ status, percent }: { status: FileStatus; percent?: number }) {
  if (status === 'analyzed') {
    return (
      <span
        style={{
          fontSize: 10,
          fontWeight: 800,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          padding: '4px 8px',
          borderRadius: 6,
          backgroundColor: '#000',
          color: '#fff',
        }}
      >
        Analyzed
      </span>
    )
  }
  if (status === 'processing') {
    return (
      <span
        style={{
          fontSize: 10,
          fontWeight: 800,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          padding: '4px 8px',
          borderRadius: 6,
          backgroundColor: ACCENT_YELLOW,
          color: '#000',
          display: 'inline-flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            backgroundColor: '#000',
            borderRadius: '50%',
            animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
          }}
        />
        Processing {percent}%
      </span>
    )
  }
  if (status === 'queued') {
    return (
      <span
        style={{
          fontSize: 10,
          fontWeight: 800,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          padding: '4px 8px',
          borderRadius: 6,
          backgroundColor: 'rgba(0,0,0,0.05)',
          color: TEXT_SUB,
          border: '1px solid rgba(0,0,0,0.1)',
        }}
      >
        Queued
      </span>
    )
  }
  // new
  return (
    <span
      style={{
        fontSize: 10,
        fontWeight: 800,
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        padding: '4px 8px',
        borderRadius: 6,
        backgroundColor: 'transparent',
        color: TEXT_SUB,
        border: '1px dashed rgba(0,0,0,0.2)',
      }}
    >
      New
    </span>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   FILE CARD
   ═══════════════════════════════════════════════════════════════════════════ */

function FileCard({ file, onOpen }: { file: DemoFile; onOpen: () => void }) {
  const [hovered, setHovered] = useState(false)
  const isProcessing = file.status === 'processing'
  const isNew = file.status === 'new'

  // Thumbnail
  let thumb: React.ReactNode
  if (file.thumbVariant === 'bars-yellow') {
    thumb = <AudioThumbYellow bars={file.barHeights || []} />
  } else if (file.thumbVariant === 'video-analyzed') {
    thumb = <VideoThumbAnalyzed />
  } else if (file.thumbVariant === 'processing') {
    thumb = <ProcessingThumb bars={file.barHeights || []} />
  } else if (file.thumbVariant === 'video-queued') {
    thumb = <VideoThumbQueued />
  } else {
    thumb = <DraftThumb />
  }

  const cardBg = isProcessing ? '#FFFFFF' : LIST_BG
  const cardBorder = isProcessing
    ? '2px solid rgba(0,0,0,0.05)'
    : '2px solid transparent'
  const cardShadow = isProcessing ? '0 2px 8px rgba(0,0,0,0.04)' : 'none'

  return (
    <div
      style={{
        backgroundColor: cardBg,
        borderRadius: 16,
        padding: 16,
        border: cardBorder,
        boxShadow: cardShadow,
        display: 'flex',
        alignItems: 'center',
        gap: 20,
        cursor: 'pointer',
        transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
        opacity: 1,
        ...(hovered
          ? {
              transform: 'translateY(-4px)',
              backgroundColor: '#FFFFFF',
              borderColor: '#000',
              boxShadow: '0 10px 30px -10px rgba(0,0,0,0.15)',
            }
          : {}),
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={onOpen}
    >
      {/* Thumbnail */}
      {thumb}

      {/* File info */}
      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        <h3
          style={{
            fontWeight: 700,
            fontSize: 18,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            margin: 0,
            paddingBottom: 2,
            textDecoration: hovered && !isNew ? 'underline' : 'none',
            textDecorationThickness: 2,
            textUnderlineOffset: 2,
            color: isNew && !hovered ? 'rgba(0,0,0,0.6)' : TEXT_MAIN,
            transition: 'color 0.2s',
          }}
        >
          {file.name}
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
          <FormatBadge format={file.format} />
          <Dot />
          <span style={{ fontSize: 12, color: TEXT_SUB, fontWeight: 500 }}>{file.specs}</span>
          <Dot />
          <span style={{ fontSize: 12, color: TEXT_SUB, fontWeight: 500 }}>{file.durationShort}</span>
        </div>
        {file.tags && (
          <div
            style={{
              fontSize: 11,
              color: TEXT_SUB,
              marginTop: 6,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {file.tags}
          </div>
        )}
      </div>

      {/* Cues / Sections */}
      <div
        style={{
          borderLeft: '1px solid rgba(0,0,0,0.1)',
          padding: '0 32px',
          minWidth: 100,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          flexShrink: 0,
          opacity: file.cues === null ? 0.5 : file.cues === 0 ? 0.3 : 1,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
          <span style={{ fontWeight: 700, fontSize: 16 }}>{file.cues ?? '--'}</span>
          <span style={{ fontSize: 12, fontWeight: 700, color: TEXT_SUB, textTransform: 'uppercase' }}>Cues</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 2 }}>
          <span style={{ fontWeight: 700, fontSize: 14 }}>{file.sections ?? '--'}</span>
          <span style={{ fontSize: 12, fontWeight: 500, color: TEXT_SUB, textTransform: 'uppercase' }}>
            {file.sections === 1 ? 'Section' : 'Sections'}
          </span>
        </div>
      </div>

      {/* Duration */}
      <div
        style={{
          borderLeft: '1px solid rgba(0,0,0,0.1)',
          padding: '0 32px',
          minWidth: 120,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          flexShrink: 0,
        }}
      >
        <span
          style={{
            fontWeight: 700,
            fontSize: 16,
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            letterSpacing: '-0.02em',
          }}
        >
          {file.durationFull}
        </span>
        <span style={{ fontSize: 10, fontWeight: 700, color: TEXT_SUB, textTransform: 'uppercase', marginTop: 4 }}>
          Duration
        </span>
      </div>

      {/* Status badge */}
      <div style={{ width: 112, display: 'flex', justifyContent: 'flex-end', flexShrink: 0, paddingRight: 8 }}>
        <StatusBadge status={file.status} percent={file.processingPercent} />
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

/** Small dot separator */
function Dot() {
  return <span style={{ width: 3, height: 3, borderRadius: '50%', backgroundColor: 'rgba(0,0,0,0.2)', flexShrink: 0 }} />
}

/* ═══════════════════════════════════════════════════════════════════════════
   PILL BUTTON
   ═══════════════════════════════════════════════════════════════════════════ */

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
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════════════════ */

export function ProjectWorkspace() {
  const { setCurrentPage, logout, userName } = useAppStore()
  const [searchQuery, setSearchQuery] = useState('')

  const handleBack = () => {
    setCurrentPage('projects')
  }

  const handleOpenFile = () => {
    setCurrentPage('analysis')
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
      <header
        style={{
          height: 56,
          flexShrink: 0,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
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
          {/* Back button */}
          <button
            onClick={handleBack}
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
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(0,0,0,0.05)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent'
            }}
          >
            <ArrowLeftIcon />
          </button>
          {/* Search button */}
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
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(0,0,0,0.05)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent'
            }}
          >
            <SearchIcon />
          </button>
          {/* Add Files */}
          <button
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
            Add Files
          </button>
          <div style={{ width: 1, height: 24, backgroundColor: 'rgba(0,0,0,0.1)', margin: '0 8px' }} />
          {/* Avatar */}
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

      {/* ── PROJECT TITLE AREA ── */}
      <div
        style={{
          flexShrink: 0,
          display: 'flex',
          alignItems: 'flex-end',
          justifyContent: 'space-between',
          padding: '0 8px',
        }}
      >
        <div>
          <h1 style={{ fontSize: 30, ...titleHeavy, margin: 0 }}>Cinematic Trailer Score Vol. 1</h1>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 12 }}>
            <span
              style={{
                fontSize: 10,
                fontWeight: 800,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                padding: '4px 10px',
                borderRadius: 6,
                backgroundColor: '#000',
                color: '#fff',
              }}
            >
              12 of 14 Analyzed
            </span>
            <span style={{ fontSize: 12, color: TEXT_SUB, fontWeight: 500 }}>14 Files</span>
            <Dot />
            <span style={{ fontSize: 12, color: TEXT_SUB, fontWeight: 500 }}>01:14:30 total</span>
            <Dot />
            <span style={{ fontSize: 12, color: TEXT_SUB, fontWeight: 500 }}>Last modified 2h ago</span>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
          <button
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
              padding: '8px 20px',
              borderRadius: 999,
              fontWeight: 700,
              fontSize: 14,
              backgroundColor: '#000',
              color: '#fff',
              border: 'none',
              cursor: 'pointer',
              transition: 'all 0.2s',
              fontFamily: FONT_FAMILY,
            }}
          >
            Analyze All
          </button>
          <button
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
              padding: '8px 20px',
              borderRadius: 999,
              fontWeight: 700,
              fontSize: 14,
              backgroundColor: '#fff',
              color: TEXT_MAIN,
              border: '1px solid #000',
              cursor: 'pointer',
              transition: 'all 0.2s',
              fontFamily: FONT_FAMILY,
            }}
          >
            Export
          </button>
          <button
            onClick={() => setCurrentPage('rights')}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
              padding: '8px 20px',
              borderRadius: 999,
              fontWeight: 700,
              fontSize: 14,
              backgroundColor: '#fff',
              color: TEXT_MAIN,
              border: '1px solid rgba(0,0,0,0.15)',
              cursor: 'pointer',
              transition: 'all 0.2s',
              fontFamily: FONT_FAMILY,
            }}
          >
            Set Rights
          </button>
          <button
            onClick={() => setCurrentPage('delivery')}
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
              transition: 'all 0.2s',
              fontFamily: FONT_FAMILY,
            }}
          >
            Review Delivery
          </button>
        </div>
      </div>

      {/* ── FILES SECTION ── */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, gap: 16, marginTop: 8 }}>
        {/* Filter bar */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexShrink: 0,
            padding: '0 8px 8px',
          }}
        >
          <h2 style={{ fontSize: 24, ...titleHeavy, margin: 0 }}>Files</h2>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {/* Search */}
            <div style={{ position: 'relative' }}>
              <div
                style={{
                  position: 'absolute',
                  left: 14,
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: 'rgba(0,0,0,0.4)',
                  pointerEvents: 'none',
                }}
              >
                <SearchIcon size={14} />
              </div>
              <input
                type="text"
                placeholder="Search files..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  width: 256,
                  backgroundColor: CARD_BG,
                  fontSize: 14,
                  fontWeight: 700,
                  paddingLeft: 36,
                  paddingRight: 16,
                  paddingTop: 8,
                  paddingBottom: 8,
                  borderRadius: 999,
                  border: '1px solid transparent',
                  outline: 'none',
                  transition: 'all 0.2s',
                  color: TEXT_MAIN,
                  fontFamily: FONT_FAMILY,
                }}
                onFocus={(e) => {
                  e.currentTarget.style.borderColor = 'rgba(0,0,0,0.3)'
                  e.currentTarget.style.backgroundColor = '#FFFFFF'
                }}
                onBlur={(e) => {
                  e.currentTarget.style.borderColor = 'transparent'
                  e.currentTarget.style.backgroundColor = CARD_BG
                }}
              />
            </div>
            <PillButton>
              <FilterIcon />
              Sort: Timeline Order
            </PillButton>
            <PillButton>
              Type: All
              <ChevronDownIcon />
            </PillButton>
          </div>
        </div>

        {/* Scrollable file list */}
        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            paddingRight: 8,
            paddingBottom: 32,
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {DEMO_FILES.map((file) => (
              <FileCard key={file.id} file={file} onOpen={handleOpenFile} />
            ))}

            {/* Add more files button */}
            <button
              style={{
                width: '100%',
                marginTop: 8,
                padding: 24,
                borderRadius: 16,
                border: '2px dashed rgba(0,0,0,0.1)',
                backgroundColor: 'transparent',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 12,
                cursor: 'pointer',
                transition: 'all 0.2s',
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
              <span style={{ color: 'rgba(0,0,0,0.5)' }}>
                <PlusIcon size={24} />
              </span>
              <span style={{ fontWeight: 700, fontSize: 14, color: 'rgba(0,0,0,0.8)' }}>Add More Files</span>
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
        .wave-bar:nth-child(6) { animation-delay: 0.3s; }
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
