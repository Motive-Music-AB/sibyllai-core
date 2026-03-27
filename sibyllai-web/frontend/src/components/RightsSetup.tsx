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

function PlayIcon({ size = 24 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M228.23,134.69l-144,88A8,8,0,0,1,72,216V40a8,8,0,0,1,12.23-6.69l144,88A8,8,0,0,1,228.23,134.69Z" />
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

function ArrowRightIcon({ size = 16 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M221.66,133.66l-72,72a8,8,0,0,1-11.32-11.32L196.69,136H40a8,8,0,0,1,0-16H196.69L138.34,61.66a8,8,0,0,1,11.32-11.32l72,72A8,8,0,0,1,221.66,133.66Z" />
    </svg>
  )
}

function CheckCircleIcon({ size = 14 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256" style={{ color: '#16a34a' }}>
      <path d="M128,24A104,104,0,1,0,232,128,104.11,104.11,0,0,0,128,24Zm45.66,85.66-56,56a8,8,0,0,1-11.32,0l-24-24a8,8,0,0,1,11.32-11.32L112,148.69l50.34-50.35a8,8,0,0,1,11.32,11.32Z" />
    </svg>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   SELECTOR PILL
   ═══════════════════════════════════════════════════════════════════════════ */

function SelectorPill({
  label,
  selected,
  onClick,
  badge,
  icon,
}: {
  label: string
  selected: boolean
  onClick: () => void
  badge?: string
  icon?: React.ReactNode
}) {
  const [hovered, setHovered] = useState(false)

  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        padding: '8px 20px',
        borderRadius: 999,
        fontWeight: 700,
        fontSize: 14,
        cursor: 'pointer',
        border: '1px solid transparent',
        transition: 'all 0.2s ease',
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        backgroundColor: selected ? '#000' : hovered ? '#E2E2E0' : LIST_BG,
        color: selected ? '#fff' : '#000',
      }}
    >
      {label}
      {icon}
      {badge && (
        <span style={{
          fontSize: 10, fontWeight: 700,
          backgroundColor: '#FFF7ED', color: '#C2410C',
          padding: '2px 6px', borderRadius: 4, marginLeft: 4,
        }}>
          {badge}
        </span>
      )}
    </button>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   DEMO DATA
   ═══════════════════════════════════════════════════════════════════════════ */

const DEMO_TRACKS = [
  { name: 'Neon Nights - Instrumental', catalog: 'CAT-8921-A', cost: 1250 },
  { name: 'Ambient Pulse (Bed)', catalog: 'CAT-1104-B', cost: 800 },
  { name: 'Rising Tension Stinger', catalog: 'CAT-5592-C', cost: 350 },
]

const TERRITORIES = ['Worldwide', 'North America', 'Europe', 'UK', 'Single Country']
const MEDIUMS = ['All Media', 'TV + Web', 'TV Broadcast', 'Web/Digital', 'Theatrical', 'Social Only']
const PERIODS = ['1 Year', '3 Years', '5 Years', 'In Perpetuity']
const EXCLUSIVITIES = ['Non-Exclusive', 'Exclusive']

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN COMPONENT
   ═══════════════════════════════════════════════════════════════════════════ */

export function RightsSetup() {
  const { setCurrentPage, userName } = useAppStore()
  const initials = userName ? userName.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2) : 'JD'

  // Rights state
  const [territory, setTerritory] = useState('Worldwide')
  const [medium, setMedium] = useState('TV + Web')
  const [period, setPeriod] = useState('1 Year')
  const [exclusivity, setExclusivity] = useState('Non-Exclusive')

  // Client contact state
  const [clientName, setClientName] = useState('Sarah Jenkins')
  const [clientEmail, setClientEmail] = useState('s.jenkins@atlascreative.co')
  const [clientCompany, setClientCompany] = useState('Atlas Creative')
  const [clientRole, setClientRole] = useState('Producer')

  const totalCost = DEMO_TRACKS.reduce((sum, t) => sum + t.cost, 0)

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
        position: 'relative', zIndex: 10,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div style={{
            width: 48, height: 48, backgroundColor: '#000', borderRadius: 14,
            display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff',
            boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
          }}>
            <PlayIcon size={24} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <h1 style={{ fontSize: 24, fontWeight: 800, letterSpacing: '-0.04em', lineHeight: 1 }}>MOTIVE</h1>
            <div style={{ fontWeight: 700, fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.05em', color: TEXT_SUB, marginTop: 4 }}>Studio Edition</div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <button
            onClick={() => setCurrentPage('analysis')}
            style={{
              width: 40, height: 40, borderRadius: '50%', backgroundColor: '#fff',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              border: '1px solid rgba(0,0,0,0.1)', cursor: 'pointer',
              boxShadow: '0 1px 2px rgba(0,0,0,0.05)', transition: 'all 0.2s ease',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#F9F9F9' }}
            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#fff' }}
          >
            <ArrowLeftIcon size={20} />
          </button>
          <div style={{
            width: 40, height: 40, borderRadius: '50%', backgroundColor: '#000', color: '#fff',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontWeight: 700, fontSize: 14, cursor: 'pointer',
            boxShadow: '0 1px 2px rgba(0,0,0,0.05)', transition: 'all 0.2s ease',
          }}
            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#222' }}
            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#000' }}
          >
            {initials}
          </div>
        </div>
      </header>

      {/* ── MAIN CONTENT ── */}
      <main style={{ flex: 1, overflowY: 'auto', padding: '0 32px 100px' }}>
        <div style={{ maxWidth: 1440, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24, paddingTop: 8 }}>

          {/* Two-column layout */}
          <div style={{ display: 'grid', gridTemplateColumns: '7fr 5fr', gap: 32, alignItems: 'start' }}>

            {/* ── LEFT COLUMN ── */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>

              {/* Configuring Rights For */}
              <section>
                <h2 style={{ fontSize: 18, fontWeight: 800, letterSpacing: '-0.04em', marginBottom: 16 }}>Configuring Rights For</h2>

                <FileCard />
              </section>

              {/* Live Cost Preview */}
              <section>
                <h2 style={{ fontSize: 18, fontWeight: 800, letterSpacing: '-0.04em', marginBottom: 16 }}>Live Cost Preview</h2>

                <div style={{ backgroundColor: BENTO, borderRadius: 28, overflow: 'hidden' }}>
                  <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid rgba(0,0,0,0.05)' }}>
                        <th style={{ ...labelStyle, padding: '16px 24px', backgroundColor: 'rgba(255,255,255,0.4)' }}>Track Name</th>
                        <th style={{ ...labelStyle, padding: '16px 24px', backgroundColor: 'rgba(255,255,255,0.4)' }}>Catalog ID</th>
                        <th style={{ ...labelStyle, padding: '16px 24px', backgroundColor: 'rgba(255,255,255,0.4)', textAlign: 'right' }}>Cost</th>
                      </tr>
                    </thead>
                    <tbody style={{ fontSize: 14 }}>
                      {DEMO_TRACKS.map((track, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid rgba(0,0,0,0.05)', transition: 'background-color 0.2s' }}
                          onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.2)' }}
                          onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'transparent' }}
                        >
                          <td style={{ padding: '16px 24px', fontWeight: 600 }}>{track.name}</td>
                          <td style={{ padding: '16px 24px', color: TEXT_SUB, fontWeight: 500 }}>{track.catalog}</td>
                          <td style={{ padding: '16px 24px', textAlign: 'right', fontWeight: 700 }}>${track.cost.toLocaleString()}.00</td>
                        </tr>
                      ))}
                    </tbody>
                    <tfoot>
                      <tr style={{ backgroundColor: 'rgba(255,255,255,0.6)' }}>
                        <td colSpan={2} style={{ padding: '20px 24px', fontWeight: 700, fontSize: 16, textAlign: 'right' }}>Total Estimated License Cost:</td>
                        <td style={{ padding: '20px 24px', textAlign: 'right', fontWeight: 700, fontSize: 18 }}>${totalCost.toLocaleString()}.00</td>
                      </tr>
                    </tfoot>
                  </table>
                </div>
              </section>
            </div>

            {/* ── RIGHT COLUMN: License Scope ── */}
            <div style={{
              backgroundColor: BENTO, borderRadius: 28, padding: 32,
              display: 'flex', flexDirection: 'column', gap: 32,
              boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
            }}>
              <div>
                <h2 style={{ fontSize: 24, fontWeight: 800, letterSpacing: '-0.04em', marginBottom: 4 }}>License Scope</h2>
                <p style={{ color: TEXT_SUB, fontSize: 14, fontWeight: 500 }}>These rights apply to all selected files</p>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                {/* Territory */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <label style={labelStyle}>Territory</label>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {TERRITORIES.map((t) => (
                      <SelectorPill
                        key={t}
                        label={t}
                        selected={territory === t}
                        onClick={() => setTerritory(t)}
                        icon={t === 'Single Country' ? <ArrowRightIcon size={12} /> : undefined}
                      />
                    ))}
                  </div>
                </div>

                {/* Medium */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <label style={labelStyle}>Medium</label>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {MEDIUMS.map((m) => (
                      <SelectorPill key={m} label={m} selected={medium === m} onClick={() => setMedium(m)} />
                    ))}
                  </div>
                </div>

                {/* Time Period */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <label style={labelStyle}>Time Period</label>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {PERIODS.map((p) => (
                      <SelectorPill key={p} label={p} selected={period === p} onClick={() => setPeriod(p)} />
                    ))}
                  </div>
                </div>

                {/* Exclusivity */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <label style={labelStyle}>Exclusivity</label>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {EXCLUSIVITIES.map((ex) => (
                      <SelectorPill
                        key={ex}
                        label={ex}
                        selected={exclusivity === ex}
                        onClick={() => setExclusivity(ex)}
                        badge={ex === 'Exclusive' ? 'High Cost' : undefined}
                      />
                    ))}
                  </div>
                </div>
              </div>

              {/* Action buttons */}
              <div style={{
                display: 'flex', gap: 16, marginTop: 8, paddingTop: 24,
                borderTop: '1px solid rgba(0,0,0,0.05)',
              }}>
                <button style={{
                  flex: 1, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                  padding: '10px 24px', borderRadius: 999, fontWeight: 700, fontSize: 14,
                  backgroundColor: '#fff', color: '#000', border: '1px solid #000', cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#F9F9F9' }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#fff' }}
                >
                  Apply to Similar
                </button>
                <button style={{
                  flex: 1, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                  padding: '10px 24px', borderRadius: 999, fontWeight: 700, fontSize: 14,
                  backgroundColor: '#000', color: '#fff', border: 'none', cursor: 'pointer',
                  transition: 'all 0.2s ease',
                }}
                  onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#222' }}
                  onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#000' }}
                >
                  Save Rights
                </button>
              </div>
            </div>
          </div>

          {/* ── PROJECT CLIENT CONTACT ── */}
          <section style={{ marginTop: 16, marginBottom: 32 }}>
            <div style={{ backgroundColor: BENTO, borderRadius: 28, padding: 32, boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 24 }}>
                <div>
                  <h2 style={{ fontSize: 18, fontWeight: 800, letterSpacing: '-0.04em', marginBottom: 4 }}>Project Client Contact</h2>
                  <p style={{ color: TEXT_SUB, fontSize: 14, fontWeight: 500 }}>Client will receive delivery package and can approve licenses.</p>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 24 }}>
                <ContactField label="Name" value={clientName} onChange={setClientName} placeholder="Contact Name" />
                <ContactField label="Email" value={clientEmail} onChange={setClientEmail} placeholder="Email Address" type="email" />
                <ContactField label="Company" value={clientCompany} onChange={setClientCompany} placeholder="Company Name" />
                <ContactField label="Role" value={clientRole} onChange={setClientRole} placeholder="Job Title" />
              </div>
            </div>
          </section>
        </div>
      </main>

      {/* ── FOOTER BAR ── */}
      <footer style={{
        position: 'fixed', bottom: 0, left: 0, width: '100%', height: 80,
        backgroundColor: '#fff', borderTop: '1px solid rgba(0,0,0,0.1)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 32px', zIndex: 50,
        boxShadow: '0 -4px 20px rgba(0,0,0,0.02)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 14, fontWeight: 700 }}>
          <span>Rights configured for 1 of 6 files</span>
          <span style={{ width: 6, height: 6, borderRadius: '50%', backgroundColor: 'rgba(0,0,0,0.2)' }} />
          <span style={{ color: TEXT_SUB }}>4 files remaining</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <button style={{
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
            padding: '10px 24px', borderRadius: 999, fontWeight: 700, fontSize: 14,
            backgroundColor: ACCENT, color: '#000', border: 'none', cursor: 'pointer',
            transition: 'all 0.2s ease',
          }}
            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#F0C642' }}
            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = ACCENT }}
          >
            Save & Configure Next
            <ArrowRightIcon size={16} />
          </button>
          <button
            onClick={() => setCurrentPage('delivery')}
            style={{
              display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              padding: '10px 24px', borderRadius: 999, fontWeight: 700, fontSize: 14,
              backgroundColor: '#000', color: '#fff', border: 'none', cursor: 'pointer',
              transition: 'all 0.2s ease',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = '#222' }}
            onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = '#000' }}
          >
            Review Delivery
          </button>
        </div>
      </footer>

      {/* Waveform bar animation */}
      <style>{`
        @keyframes wfPulse {
          0%, 100% { height: 8px; }
          50% { height: 24px; }
        }
      `}</style>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════
   SUB-COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════ */

const labelStyle: React.CSSProperties = {
  fontWeight: 700, fontSize: 10, textTransform: 'uppercase',
  letterSpacing: '0.05em', color: TEXT_SUB,
}

function FileCard() {
  const [hovered, setHovered] = useState(false)

  const bars = [
    { h: 12, delay: '0.1s' },
    { h: 24, delay: '0.2s' },
    { h: 16, delay: '0.3s' },
    { h: 8, delay: '0.4s' },
    { h: 18, delay: '0.5s' },
  ]

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        backgroundColor: hovered ? '#fff' : LIST_BG,
        borderRadius: 16, padding: 16,
        border: hovered ? '1px solid #000' : '1px solid transparent',
        boxShadow: hovered ? '0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.1)' : 'none',
        transform: hovered ? 'translateY(-4px)' : 'translateY(0)',
        transition: 'all 0.2s ease',
        display: 'flex', alignItems: 'center', gap: 16, cursor: 'pointer',
      }}
    >
      {/* Waveform thumbnail */}
      <div style={{
        width: 64, height: 48, backgroundColor: '#000', borderRadius: 8, flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 2,
        position: 'relative', overflow: 'hidden',
      }}>
        {bars.map((bar, i) => (
          <div key={i} style={{
            width: 3, borderRadius: 2, backgroundColor: ACCENT,
            animation: `wfPulse 1.5s infinite ease-in-out`,
            animationDelay: bar.delay,
            height: bar.h,
          }} />
        ))}
      </div>

      {/* File info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
          <h3 style={{ fontWeight: 700, fontSize: 16, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            Commercial_Edit_v3_Final_Audio.wav
          </h3>
          <span style={{
            padding: '2px 8px', backgroundColor: '#fff', border: '1px solid rgba(0,0,0,0.1)',
            borderRadius: 4, fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
            letterSpacing: '0.05em', color: TEXT_SUB, flexShrink: 0,
          }}>
            WAV
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: 14, color: TEXT_SUB }}>
          <span style={{ fontWeight: 500 }}>1:45</span>
          <span style={{ width: 4, height: 4, backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '50%' }} />
          <span style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#000', fontWeight: 500 }}>
            <CheckCircleIcon size={14} />
            3 matched tracks
          </span>
        </div>
      </div>
    </div>
  )
}

function ContactField({
  label, value, onChange, placeholder, type = 'text',
}: {
  label: string
  value: string
  onChange: (v: string) => void
  placeholder: string
  type?: string
}) {
  const [focused, setFocused] = useState(false)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <label style={{ ...labelStyle, paddingLeft: 4 }}>{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        placeholder={placeholder}
        style={{
          backgroundColor: '#fff',
          border: focused ? '1px solid #000' : '1px solid transparent',
          borderRadius: 12, padding: '12px 16px',
          fontSize: 14, fontWeight: 500, outline: 'none',
          transition: 'border-color 0.2s ease',
          boxShadow: '0 1px 2px rgba(0,0,0,0.05)',
          width: '100%', color: '#000',
        }}
      />
    </div>
  )
}
