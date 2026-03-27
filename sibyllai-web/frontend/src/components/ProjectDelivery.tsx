import { useState } from 'react'
import { useAppStore } from '@/lib/store'

/* ═══════════════════════════════════════════════════════════════════════════
   DESIGN TOKENS
   ═══════════════════════════════════════════════════════════════════════════ */

const CANVAS = '#D6D6D4'
const BENTO = '#EBEBEB'
const LIST_BG = '#F3F3F1'
const ACCENT = '#FFD659'
const TEXT_SUB = '#4A4A4A'

/* ═══════════════════════════════════════════════════════════════════════════
   ICONS
   ═══════════════════════════════════════════════════════════════════════════ */

function ArrowLeftIcon({ size = 20 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M224,128a8,8,0,0,1-8,8H59.31l58.35,58.34a8,8,0,0,1-11.32,11.32l-72-72a8,8,0,0,1,0-11.32l72-72a8,8,0,0,1,11.32,11.32L59.31,120H216A8,8,0,0,1,224,128Z" />
    </svg>
  )
}

function PlayIcon({ size = 16 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M228.23,134.69l-136,80A16,16,0,0,1,68,201.27V41.3A16,16,0,0,1,92.23,27.88l136,80a16,16,0,0,1,0,26.82Z" />
    </svg>
  )
}

function PauseIcon({ size = 16 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M216,48V208a16,16,0,0,1-16,16H160a16,16,0,0,1-16-16V48a16,16,0,0,1,16-16h40A16,16,0,0,1,216,48ZM96,32H56A16,16,0,0,0,40,48V208a16,16,0,0,0,16,16H96a16,16,0,0,0,16-16V48A16,16,0,0,0,96,32Z" />
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

function ClockIcon({ size = 16 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M128,24A104,104,0,1,0,232,128,104.11,104.11,0,0,0,128,24Zm0,192a88,88,0,1,1,88-88A88.1,88.1,0,0,1,128,216Zm64-88a8,8,0,0,1-8,8H128a8,8,0,0,1-8-8V72a8,8,0,0,1,16,0v48h48A8,8,0,0,1,192,128Z" />
    </svg>
  )
}

function DollarIcon({ size = 16 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M128,24A104,104,0,1,0,232,128,104.11,104.11,0,0,0,128,24Zm0,192a88,88,0,1,1,88-88A88.1,88.1,0,0,1,128,216Zm16-136a8,8,0,0,1-8,8H120v24h16a24,24,0,0,1,0,48H128v16a8,8,0,0,1-16,0V160h-8a8,8,0,0,1,0-16h16V120H104a24,24,0,0,1,0-48h8V56a8,8,0,0,1,16,0v16h16A8,8,0,0,1,144,80Z" />
    </svg>
  )
}

function SearchIcon({ size = 14 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M229.66,218.34l-50.07-50.06a88.11,88.11,0,1,0-11.31,11.31l50.06,50.07a8,8,0,0,0,11.32-11.32ZM40,112a72,72,0,1,1,72,72A72.08,72.08,0,0,1,40,112Z" />
    </svg>
  )
}

function DownloadIcon({ size = 16 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M224,152v56a16,16,0,0,1-16,16H48a16,16,0,0,1-16-16V152a8,8,0,0,1,16,0v56H208V152a8,8,0,0,1,16,0Zm-101.66,5.66a8,8,0,0,0,11.32,0l40-40a8,8,0,0,0-11.32-11.32L136,132.69V32a8,8,0,0,0-16,0v100.69L93.66,106.34a8,8,0,0,0-11.32,11.32Z" />
    </svg>
  )
}

function LockIcon({ size = 16 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M208,88H176V56a48,48,0,0,0-96,0V88H48a16,16,0,0,0-16,16V208a16,16,0,0,0,16,16H208a16,16,0,0,0,16-16V104A16,16,0,0,0,208,88ZM96,56a32,32,0,0,1,64,0V88H96Zm112,152H48V104H208V208Zm-48-52a12,12,0,0,1-12,12H100a12,12,0,0,1,0-24h48A12,12,0,0,1,160,156Z" />
    </svg>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   DEMO DATA
   ═══════════════════════════════════════════════════════════════════════════ */

interface DeliveryTrack {
  id: string
  name: string
  catalog: string
  matchedTo: string
  duration: string
  license: number
  waveformHeights: number[]
  bpm: number
  key: string
  genre: string
  similarity: number
  originalTemp: string
  originalDetails: string
  videoFile?: string
  videoTimecode?: string
}

const DEMO_TRACKS: DeliveryTrack[] = [
  {
    id: '1', name: 'Nightfall Pulse', catalog: 'TRK-8921',
    matchedTo: 'MX_Reel_01 → Cue #3', duration: '03:24', license: 180,
    waveformHeights: [20,40,60,80,100,70,90,50,30,60,90,100,80,40,20,50,70,90,100,60,40,20,30,50,80,100,90,60,40,20],
    bpm: 118, key: 'Cm', genre: 'Cinematic, Dark', similarity: 92,
    originalTemp: 'Temp_Action_04.wav', originalDetails: 'Action • 120 BPM • Dm',
    videoFile: 'MX_Reel_01.mp4', videoTimecode: '01:14:22:00 - 01:17:46:00',
  },
  {
    id: '2', name: 'Deep Space Drone', catalog: 'AMB-4412',
    matchedTo: 'MX_Reel_01 → Cue #1', duration: '01:45', license: 120,
    waveformHeights: [30,35,40,45,50,45,40,45,50,55,60,65,60,55,50,45,40,35,30,25,30,35,40,45,50,55,60,55,50,45],
    bpm: 72, key: 'Am', genre: 'Ambient, Atmospheric', similarity: 87,
    originalTemp: 'Temp_Ambient_01.wav', originalDetails: 'Ambient • 70 BPM • Am',
  },
  {
    id: '3', name: 'Percussive Tension', catalog: 'ACT-1109',
    matchedTo: 'MX_Reel_02 → Cue #2', duration: '02:15', license: 150,
    waveformHeights: [80,20,30,90,20,40,100,30,50,80,20,40,90,30,20,100,40,30,80,20,50,90,30,20,100,40,30,80,20,50],
    bpm: 135, key: 'Em', genre: 'Action, Percussive', similarity: 89,
    originalTemp: 'Temp_Tension_02.wav', originalDetails: 'Tension • 132 BPM • Em',
  },
  {
    id: '4', name: 'Melancholy Strings', catalog: 'ORC-5521',
    matchedTo: 'MX_Reel_03 → Cue #1', duration: '04:10', license: 220,
    waveformHeights: [40,50,60,70,80,90,100,90,80,70,60,50,40,30,40,50,60,70,80,90,100,90,80,70,60,50,40,30,20,10],
    bpm: 68, key: 'Gm', genre: 'Orchestral, Emotional', similarity: 95,
    originalTemp: 'Temp_Strings_01.wav', originalDetails: 'Orchestral • 66 BPM • Gm',
  },
  {
    id: '5', name: 'Cyberpunk Chase', catalog: 'SYN-9902',
    matchedTo: 'MX_Reel_04 → Cue #5', duration: '02:45', license: 180,
    waveformHeights: [100,80,100,70,90,60,100,80,90,70,100,60,80,100,70,90,80,100,60,80,100,70,90,80,100,60,80,100,70,90],
    bpm: 140, key: 'F#m', genre: 'Electronic, Intense', similarity: 84,
    originalTemp: 'Temp_Chase_05.wav', originalDetails: 'Electronic • 142 BPM • F#m',
  },
  {
    id: '6', name: 'Ambient Underbed', catalog: 'AMB-1022',
    matchedTo: 'MX_Reel_05 → Cue #1', duration: '01:20', license: 100,
    waveformHeights: [20,25,20,30,25,20,25,30,25,20,25,30,25,20,25,20,25,30,25,20,25,30,25,20,25,20,25,30,25,20],
    bpm: 60, key: 'C', genre: 'Ambient, Underscore', similarity: 91,
    originalTemp: 'Temp_Ambient_03.wav', originalDetails: 'Ambient • 58 BPM • C',
  },
]

const TOTAL_TRACKS = DEMO_TRACKS.length
const TOTAL_DURATION = '42:30'
const TOTAL_COST = DEMO_TRACKS.reduce((sum, t) => sum + t.license, 0)

/* ═══════════════════════════════════════════════════════════════════════════
   MINI WAVEFORM
   ═══════════════════════════════════════════════════════════════════════════ */

function MiniWaveform({ heights }: { heights: number[] }) {
  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', height: 32, opacity: 0.3, gap: 2 }}>
      {heights.map((h, i) => (
        <div key={i} style={{ width: 2, height: `${h}%`, backgroundColor: '#000', borderRadius: 1 }} />
      ))}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   TRACK ROW
   ═══════════════════════════════════════════════════════════════════════════ */

function TrackRow({
  track,
  isPlaying,
  onPlay,
  onSelect,
}: {
  track: DeliveryTrack
  isPlaying: boolean
  onPlay: () => void
  onSelect: () => void
}) {
  const [hovered, setHovered] = useState(false)
  const elevated = hovered || isPlaying

  return (
    <div
      onClick={onSelect}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        backgroundColor: elevated ? '#FFFFFF' : LIST_BG,
        borderRadius: 16,
        border: elevated ? '1px solid #000000' : '1px solid transparent',
        boxShadow: elevated ? '0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1)' : 'none',
        transform: elevated ? 'translateY(-4px)' : 'translateY(0)',
        transition: 'all 0.2s ease',
        display: 'flex',
        alignItems: 'center',
        padding: 12,
        gap: 20,
        cursor: 'pointer',
        position: 'relative',
      }}
    >
      {/* Play/Pause button */}
      <button
        onClick={(e) => { e.stopPropagation(); onPlay() }}
        style={{
          width: 40, height: 40, borderRadius: '50%', backgroundColor: '#000', color: '#fff',
          display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
          border: 'none', cursor: 'pointer', position: 'relative', zIndex: 10,
          transform: hovered ? 'scale(1.05)' : 'scale(1)', transition: 'transform 0.2s ease',
        }}
      >
        {isPlaying && (
          <div style={{
            position: 'absolute', inset: 0, borderRadius: '50%',
            border: '2px solid #000', animation: 'ping 1s cubic-bezier(0,0,0.2,1) infinite', opacity: 0.3,
          }} />
        )}
        {isPlaying ? <PauseIcon size={16} /> : <PlayIcon size={16} />}
      </button>

      {/* Track info */}
      <div style={{ display: 'flex', flexDirection: 'column', width: 220, flexShrink: 0 }}>
        <div style={{ fontWeight: 700, fontSize: 16, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{track.name}</div>
        <div style={{ fontSize: 11, color: TEXT_SUB, marginTop: 2 }}>Catalog: {track.catalog}</div>
        <div style={{ fontSize: 11, color: TEXT_SUB, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>Matched to: {track.matchedTo}</div>
      </div>

      {/* Mini waveform */}
      <MiniWaveform heights={track.waveformHeights} />

      {/* Duration & License */}
      <div style={{ display: 'flex', gap: 24, flexShrink: 0, justifyContent: 'flex-end', width: 140 }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
          <span style={{ fontFamily: 'monospace', fontWeight: 700, fontSize: 14 }}>{track.duration}</span>
          <span style={{ fontSize: 10, color: TEXT_SUB, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700 }}>Duration</span>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', width: 50 }}>
          <span style={{ fontWeight: 700, fontSize: 14 }}>${track.license}</span>
          <span style={{ fontSize: 10, color: TEXT_SUB, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700 }}>License</span>
        </div>
      </div>

      {/* A/B button */}
      <button
        onClick={(e) => e.stopPropagation()}
        style={{
          flexShrink: 0, width: 48, height: 32, borderRadius: 999, backgroundColor: BENTO,
          border: '1px solid rgba(0,0,0,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 12, fontWeight: 700, cursor: 'pointer', position: 'relative', zIndex: 10,
          transition: 'all 0.2s ease',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#000'; e.currentTarget.style.color = '#fff' }}
        onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = BENTO; e.currentTarget.style.color = '#000' }}
      >
        A/B
      </button>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════════════════ */

export function ProjectDelivery() {
  const { setCurrentPage, userName } = useAppStore()
  const [playingId, setPlayingId] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string>(DEMO_TRACKS[0].id)
  const [searchQuery, setSearchQuery] = useState('')

  const selectedTrack = DEMO_TRACKS.find(t => t.id === selectedId) ?? DEMO_TRACKS[0]
  const initials = userName ? userName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2) : 'JD'

  const filteredTracks = searchQuery
    ? DEMO_TRACKS.filter(t =>
        t.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        t.catalog.toLowerCase().includes(searchQuery.toLowerCase()) ||
        t.matchedTo.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : DEMO_TRACKS

  return (
    <div style={{
      height: '100vh', width: '100vw', overflow: 'hidden', display: 'flex', flexDirection: 'column',
      backgroundColor: CANVAS, color: '#000000',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
      WebkitFontSmoothing: 'antialiased',
    }}>

      {/* ── HEADER ── */}
      <header style={{ height: 80, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 32px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{
            width: 48, height: 48, backgroundColor: '#000', borderRadius: 14,
            display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff',
          }}>
            <PlayIcon size={24} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ fontSize: 24, fontWeight: 800, letterSpacing: '-0.04em', lineHeight: 1 }}>MOTIVE</div>
            <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.15em', fontWeight: 700, color: TEXT_SUB, marginTop: 4 }}>Studio Edition</div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button
            onClick={() => setCurrentPage('rights')}
            style={{
              width: 40, height: 40, borderRadius: '50%', backgroundColor: '#fff',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1px solid rgba(0,0,0,0.1)', cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#000'; e.currentTarget.style.color = '#fff' }}
            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#fff'; e.currentTarget.style.color = '#000' }}
          >
            <ArrowLeftIcon size={20} />
          </button>

          <button
            onClick={() => {}}
            style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              padding: '8px 20px', borderRadius: 999, fontWeight: 700, fontSize: 14,
              backgroundColor: '#fff', border: '1px solid #EBEBEB', cursor: 'pointer',
              transition: 'all 0.2s ease', color: '#000',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'rgba(0,0,0,0.2)' }}
            onMouseLeave={(e) => { e.currentTarget.style.borderColor = '#EBEBEB' }}
          >
            Export PDF
          </button>

          <button
            onClick={() => {}}
            style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              padding: '8px 20px', borderRadius: 999, fontWeight: 700, fontSize: 14,
              backgroundColor: '#fff', border: '1px solid #EBEBEB', cursor: 'pointer',
              transition: 'all 0.2s ease', color: '#000',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'rgba(0,0,0,0.2)' }}
            onMouseLeave={(e) => { e.currentTarget.style.borderColor = '#EBEBEB' }}
          >
            Export CSV
          </button>

          <div style={{ width: 1, height: 24, backgroundColor: 'rgba(0,0,0,0.1)', margin: '0 8px' }} />

          <div style={{
            width: 40, height: 40, borderRadius: '50%', backgroundColor: '#000', color: '#fff',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 14, fontWeight: 700, cursor: 'pointer',
            transition: 'transform 0.2s ease',
          }}
            onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.05)' }}
            onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1)' }}
          >
            {initials}
          </div>
        </div>
      </header>

      {/* ── CONTENT AREA ── */}
      <div style={{ flex: 1, minHeight: 0, padding: '0 32px', display: 'flex', flexDirection: 'column', gap: 24 }}>

        {/* Title + Stats */}
        <div style={{ flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div>
            <h1 style={{ fontSize: 30, fontWeight: 800, letterSpacing: '-0.04em', lineHeight: 1, marginBottom: 4 }}>Cinematic Trailer Score Vol. 1</h1>
            <div style={{ fontSize: 12, fontWeight: 500, color: TEXT_SUB }}>Delivery Overview</div>
          </div>

          {/* Stat Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 24, height: 140 }}>
            {/* Matched Tracks */}
            <div style={{ backgroundColor: BENTO, borderRadius: 28, padding: 24, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', position: 'relative', overflow: 'hidden' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <span style={{ fontSize: 11, textTransform: 'uppercase', fontWeight: 700, color: TEXT_SUB, letterSpacing: '0.05em' }}>Matched Tracks</span>
                <PlusIcon size={16} />
              </div>
              <div>
                <div style={{ fontSize: 48, fontWeight: 800, letterSpacing: '-0.04em', lineHeight: 1, marginBottom: 4 }}>{TOTAL_TRACKS * 3}</div>
                <div style={{ fontSize: 12, fontWeight: 500, color: TEXT_SUB }}>across 6 files</div>
              </div>
            </div>

            {/* Total Duration — dark */}
            <div style={{ backgroundColor: '#000', borderRadius: 28, padding: 24, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', position: 'relative', overflow: 'hidden' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <span style={{ fontSize: 11, textTransform: 'uppercase', fontWeight: 700, color: 'rgba(255,255,255,0.5)', letterSpacing: '0.05em' }}>Total Duration</span>
                <span style={{ color: 'rgba(255,255,255,0.5)' }}><ClockIcon size={16} /></span>
              </div>
              <div>
                <div style={{ fontSize: 48, fontWeight: 800, letterSpacing: '-0.04em', lineHeight: 1, color: ACCENT, marginBottom: 4 }}>{TOTAL_DURATION}</div>
                <div style={{ fontSize: 12, fontWeight: 500, color: 'rgba(255,255,255,0.5)' }}>minutes of licensed music</div>
              </div>
            </div>

            {/* Estimated Cost — yellow */}
            <div style={{ backgroundColor: ACCENT, borderRadius: 28, padding: 24, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', position: 'relative', overflow: 'hidden' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <span style={{ fontSize: 11, textTransform: 'uppercase', fontWeight: 700, color: 'rgba(0,0,0,0.6)', letterSpacing: '0.05em' }}>Estimated Cost</span>
                <span style={{ color: 'rgba(0,0,0,0.6)' }}><DollarIcon size={16} /></span>
              </div>
              <div>
                <div style={{ fontSize: 48, fontWeight: 800, letterSpacing: '-0.04em', lineHeight: 1, marginBottom: 4 }}>${TOTAL_COST.toLocaleString()}</div>
                <div style={{ fontSize: 12, fontWeight: 500, color: 'rgba(0,0,0,0.6)' }}>based on standard licensing</div>
              </div>
            </div>
          </div>
        </div>

        {/* ── MAIN SPLIT: Track List + Sidebar ── */}
        <div style={{ flex: 1, minHeight: 0, display: 'flex', gap: 24, paddingBottom: 24 }}>

          {/* LEFT: Track list */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            {/* List header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 16, flexShrink: 0 }}>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                <h2 style={{ fontSize: 24, fontWeight: 800, letterSpacing: '-0.04em' }}>All Matched Tracks</h2>
                <span style={{ fontSize: 14, fontWeight: 500, color: TEXT_SUB }}>({filteredTracks.length} tracks)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{
                  backgroundColor: '#fff', borderRadius: 999, display: 'flex', alignItems: 'center',
                  padding: '6px 12px', border: '1px solid rgba(0,0,0,0.1)', boxShadow: '0 1px 2px rgba(0,0,0,0.05)', width: 200,
                }}>
                  <SearchIcon size={14} />
                  <input
                    type="text"
                    placeholder="Search tracks..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    style={{
                      background: 'transparent', border: 'none', outline: 'none', fontSize: 14,
                      marginLeft: 8, width: '100%', color: '#000',
                    }}
                  />
                </div>
                <button style={{
                  display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                  padding: '6px 20px', borderRadius: 999, fontWeight: 700, fontSize: 12,
                  backgroundColor: BENTO, border: 'none', cursor: 'pointer', color: '#000',
                }}>
                  Sort: By File
                </button>
              </div>
            </div>

            {/* Scrollable track list */}
            <div style={{ flex: 1, overflowY: 'auto', paddingRight: 12, display: 'flex', flexDirection: 'column', gap: 12 }}>
              {filteredTracks.map((track) => (
                <TrackRow
                  key={track.id}
                  track={track}
                  isPlaying={playingId === track.id}
                  onPlay={() => setPlayingId(playingId === track.id ? null : track.id)}
                  onSelect={() => setSelectedId(track.id)}
                />
              ))}
            </div>
          </div>

          {/* RIGHT: Sidebar */}
          <div style={{ width: 320, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 20, overflowY: 'auto', paddingRight: 8, paddingBottom: 8 }}>

            {/* Video Preview */}
            <div style={{ backgroundColor: BENTO, borderRadius: 28, padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div style={{
                width: '100%', aspectRatio: '16/9', backgroundColor: '#000', borderRadius: 14,
                position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden',
              }}>
                <button style={{
                  width: 56, height: 56, backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: '50%',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff',
                  border: 'none', cursor: 'pointer', transition: 'all 0.2s ease',
                  backdropFilter: 'blur(12px)',
                }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.3)'; e.currentTarget.style.transform = 'scale(1.1)' }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.2)'; e.currentTarget.style.transform = 'scale(1)' }}
                >
                  <PlayIcon size={24} />
                </button>
                <div style={{
                  position: 'absolute', bottom: 8, left: 8,
                  backgroundColor: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
                  color: '#fff', fontSize: 10, fontWeight: 700, padding: '4px 8px', borderRadius: 4,
                }}>
                  SYNC ON
                </div>
              </div>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {selectedTrack.videoFile ?? 'No video'}
                </div>
                <div style={{ fontSize: 12, fontFamily: 'monospace', color: TEXT_SUB, marginTop: 2 }}>
                  {selectedTrack.videoTimecode ?? '—'}
                </div>
              </div>
            </div>

            {/* Current Match Details */}
            <div style={{ backgroundColor: BENTO, borderRadius: 28, padding: 24 }}>
              <div style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: TEXT_SUB, marginBottom: 4 }}>Current Match</div>
              <div style={{ fontSize: 20, fontWeight: 800, letterSpacing: '-0.04em', lineHeight: 1, marginBottom: 4 }}>{selectedTrack.name}</div>
              <div style={{ fontSize: 12, fontWeight: 500, color: TEXT_SUB, marginBottom: 20 }}>Catalog ID: {selectedTrack.catalog}</div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
                <div>
                  <div style={{ fontSize: 10, color: TEXT_SUB, textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.05em', marginBottom: 2 }}>BPM</div>
                  <div style={{ fontWeight: 700, fontSize: 16 }}>{selectedTrack.bpm}</div>
                </div>
                <div>
                  <div style={{ fontSize: 10, color: TEXT_SUB, textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.05em', marginBottom: 2 }}>Key</div>
                  <div style={{ fontWeight: 700, fontSize: 16 }}>{selectedTrack.key}</div>
                </div>
                <div style={{ gridColumn: 'span 2' }}>
                  <div style={{ fontSize: 10, color: TEXT_SUB, textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.05em', marginBottom: 2 }}>Genre</div>
                  <div style={{ fontWeight: 700, fontSize: 16 }}>{selectedTrack.genre}</div>
                </div>
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
                  <span style={{ fontSize: 12, fontWeight: 700 }}>Similarity Score</span>
                  <span style={{ fontSize: 14, fontWeight: 800, letterSpacing: '-0.04em' }}>{selectedTrack.similarity}% Match</span>
                </div>
                <div style={{ height: 10, borderRadius: 999, backgroundColor: 'rgba(0,0,0,0.05)', overflow: 'hidden' }}>
                  <div style={{ height: '100%', backgroundColor: '#000', borderRadius: 999, width: `${selectedTrack.similarity}%` }} />
                </div>
              </div>
            </div>

            {/* A/B Compare Card — yellow */}
            <div style={{ backgroundColor: ACCENT, borderRadius: 28, padding: 24, display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'rgba(0,0,0,0.6)', marginBottom: 4 }}>Original Temp</div>
                <div style={{ fontSize: 14, fontWeight: 700, lineHeight: 1.3 }}>{selectedTrack.originalTemp}</div>
                <div style={{ fontSize: 12, color: 'rgba(0,0,0,0.6)', fontWeight: 500, marginTop: 2 }}>{selectedTrack.originalDetails}</div>
              </div>

              {/* VS divider */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, margin: '8px 0', opacity: 0.5 }}>
                <div style={{ height: 1, backgroundColor: '#000', flex: 1 }} />
                <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.15em' }}>VS</div>
                <div style={{ height: 1, backgroundColor: '#000', flex: 1 }} />
              </div>

              <div style={{ marginBottom: 16 }}>
                <div style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'rgba(0,0,0,0.6)', marginBottom: 4 }}>Replacement</div>
                <div style={{ fontSize: 14, fontWeight: 700, lineHeight: 1.3 }}>{selectedTrack.name}</div>
                <div style={{ fontSize: 12, color: 'rgba(0,0,0,0.6)', fontWeight: 500, marginTop: 2 }}>{selectedTrack.genre} • {selectedTrack.bpm} BPM • {selectedTrack.key}</div>
              </div>

              {/* Comparison bars */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 'auto' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 10, fontWeight: 700, width: 48, color: 'rgba(0,0,0,0.6)', textTransform: 'uppercase' }}>Temp</span>
                  <div style={{ flex: 1, height: 8, backgroundColor: 'rgba(0,0,0,0.1)', borderRadius: 999, overflow: 'hidden' }}>
                    <div style={{ height: '100%', backgroundColor: 'rgba(0,0,0,0.4)', borderRadius: 999, width: '85%' }} />
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 10, fontWeight: 700, width: 48, color: 'rgba(0,0,0,0.6)', textTransform: 'uppercase' }}>Match</span>
                  <div style={{ flex: 1, height: 8, backgroundColor: 'rgba(0,0,0,0.1)', borderRadius: 999, overflow: 'hidden' }}>
                    <div style={{ height: '100%', backgroundColor: '#000', borderRadius: 999, width: `${selectedTrack.similarity}%` }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── BOTTOM BAR ── */}
      <div style={{
        height: 64, flexShrink: 0, backgroundColor: '#fff', borderTop: '1px solid rgba(0,0,0,0.1)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 32px', zIndex: 20,
      }}>
        <div style={{ fontSize: 14, fontWeight: 500, color: TEXT_SUB }}>
          {TOTAL_TRACKS * 3} tracks selected
          <span style={{ margin: '0 8px' }}>•</span>
          {TOTAL_DURATION} total
          <span style={{ margin: '0 8px' }}>•</span>
          <span style={{ fontWeight: 700, color: '#000' }}>${TOTAL_COST.toLocaleString()} estimated</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
            padding: '8px 20px', borderRadius: 999, fontWeight: 700, fontSize: 14,
            backgroundColor: ACCENT, border: '1px solid transparent', cursor: 'pointer', color: '#000',
            transition: 'all 0.2s ease',
          }}
            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#F0C84A' }}
            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = ACCENT }}
          >
            <DownloadIcon size={16} />
            Download Package
          </button>
          <button
            onClick={() => setCurrentPage('workspace')}
            style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
            padding: '8px 20px', borderRadius: 999, fontWeight: 700, fontSize: 14,
            backgroundColor: '#000', border: '1px solid transparent', cursor: 'pointer', color: '#fff',
            transition: 'all 0.2s ease', marginLeft: 8,
          }}
            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(0,0,0,0.8)' }}
            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#000' }}
          >
            <LockIcon size={16} />
            Approve &amp; Lock
          </button>
        </div>
      </div>

      {/* Ping animation keyframes */}
      <style>{`
        @keyframes ping {
          75%, 100% { transform: scale(2); opacity: 0; }
        }
      `}</style>
    </div>
  )
}
