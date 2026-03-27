import { useState, type FormEvent } from 'react'
import { useAppStore } from '@/lib/store'

/* ─── Icons (inline SVGs matching the design) ─── */

function ArrowRightIcon({ size = 18 }: { size?: number }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={size} height={size} fill="currentColor" viewBox="0 0 256 256">
      <path d="M221.66,133.66l-72,72a8,8,0,0,1-11.32-11.32L196.69,136H40a8,8,0,0,1,0-16H196.69L138.34,61.66a8,8,0,0,1,11.32-11.32l72,72A8,8,0,0,1,221.66,133.66Z" />
    </svg>
  )
}

function GoogleIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={20} height={20} fill="currentColor" viewBox="0 0 256 256">
      <path d="M228,128a100,100,0,1,1-29.29-70.71,12,12,0,0,1-17,17A76,76,0,1,0,204,128H128a12,12,0,0,1,0-24h88A12,12,0,0,1,228,128Z" />
    </svg>
  )
}

function AppleIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width={20} height={20} fill="currentColor" viewBox="0 0 256 256">
      <path d="M148.16,63.1A35.85,35.85,0,0,0,158.42,38a12,12,0,0,0-15.54-15A36.19,36.19,0,0,0,131.6,48.24,35.85,35.85,0,0,0,121.34,73.3,12,12,0,0,0,136.88,88.3,36.19,36.19,0,0,0,148.16,63.1Zm58.15,53.21a52.12,52.12,0,0,1,16.48,37,51.52,51.52,0,0,1-21.75,41.49C191.86,201.29,176.4,228,148.37,228c-17.51,0-34.62-11.23-51.27-11.23S63.49,228,46,228c-30,0-46.76-29.58-55.85-45.19C-16.14,139-2.29,74.55,39,74.55c16.35,0,32,10.6,47.41,10.6s35.39-11.66,54-11.66c18,0,36,8,46.13,22.18A12,12,0,0,1,181.76,114,52.28,52.28,0,0,1,206.31,116.31Z" />
    </svg>
  )
}

/* ─── Main Component ─── */

export function LoginScreen() {
  const { setLoggedIn } = useAppStore()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    setLoggedIn(email || 'User')
  }

  return (
    <div
      className="min-h-screen w-full flex flex-col items-center justify-center p-6 box-border"
      style={{
        backgroundColor: '#D6D6D4',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
        WebkitFontSmoothing: 'antialiased',
      }}
    >
      <div className="w-full flex flex-col items-center" style={{ maxWidth: 460 }}>

        {/* ── Brand ── */}
        <div className="flex flex-col items-center w-full" style={{ marginBottom: 40 }}>
          <div
            className="flex items-center justify-center shadow-sm"
            style={{
              width: 64,
              height: 64,
              backgroundColor: '#FFD659',
              borderRadius: 20,
              marginBottom: 24,
            }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width={32} height={32} fill="currentColor" viewBox="0 0 256 256">
              <path d="M216.5,114.36,68.52,21.84A16,16,0,0,0,44,35.48V220.52a16,16,0,0,0,24.52,13.64l148-92.52A16,16,0,0,0,216.5,114.36Z" opacity="0.2" />
              <path d="M224,128a8,8,0,0,1-3.76,6.81l-148,92.52A8,8,0,0,1,60,220.52V35.48a8,8,0,0,1,12.24-6.81l148,92.52A8,8,0,0,1,224,128Z" />
            </svg>
          </div>
          <h1
            className="text-center"
            style={{
              fontSize: 40,
              fontWeight: 800,
              letterSpacing: '-0.04em',
              lineHeight: 1.1,
              marginBottom: 12,
              color: '#000',
            }}
          >
            MOTIVE
          </h1>
          <p
            className="text-center"
            style={{
              fontSize: 15,
              fontWeight: 700,
              color: '#4A4A4A',
            }}
          >
            Sign in to your workspace
          </p>
        </div>

        {/* ── Card ── */}
        <div
          className="w-full flex flex-col shadow-sm"
          style={{
            backgroundColor: '#EBEBEB',
            borderRadius: 28,
            padding: 32,
          }}
        >
          <form className="flex flex-col" style={{ gap: 24 }} onSubmit={handleSubmit}>
            {/* Email */}
            <div className="flex flex-col" style={{ gap: 10 }}>
              <label
                style={{
                  fontSize: 13,
                  fontWeight: 700,
                  letterSpacing: '-0.01em',
                  lineHeight: 1.3,
                  marginLeft: 4,
                  color: '#000',
                }}
              >
                Email Address
              </label>
              <input
                type="email"
                placeholder="hello@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoFocus
                style={{
                  background: '#FFFFFF',
                  border: '2px solid transparent',
                  borderRadius: 16,
                  padding: '18px 20px',
                  fontWeight: 600,
                  fontSize: 16,
                  outline: 'none',
                  transition: 'all 0.2s',
                  color: '#000',
                }}
                onFocus={(e) => { e.currentTarget.style.borderColor = '#000' }}
                onBlur={(e) => { e.currentTarget.style.borderColor = 'transparent' }}
              />
            </div>

            {/* Password */}
            <div className="flex flex-col" style={{ gap: 10 }}>
              <div className="flex justify-between items-end" style={{ marginLeft: 4 }}>
                <label
                  style={{
                    fontSize: 13,
                    fontWeight: 700,
                    letterSpacing: '-0.01em',
                    lineHeight: 1.3,
                    color: '#000',
                  }}
                >
                  Password
                </label>
                <a
                  href="#"
                  onClick={(e) => e.preventDefault()}
                  className="hover:underline"
                  style={{
                    fontSize: 12,
                    fontWeight: 700,
                    color: '#4A4A4A',
                    textDecorationOffset: '2px',
                    transition: 'color 0.2s',
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = '#000' }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = '#4A4A4A' }}
                >
                  Forgot password?
                </a>
              </div>
              <input
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                style={{
                  background: '#FFFFFF',
                  border: '2px solid transparent',
                  borderRadius: 16,
                  padding: '18px 20px',
                  fontWeight: 600,
                  fontSize: 16,
                  outline: 'none',
                  transition: 'all 0.2s',
                  color: '#000',
                }}
                onFocus={(e) => { e.currentTarget.style.borderColor = '#000' }}
                onBlur={(e) => { e.currentTarget.style.borderColor = 'transparent' }}
              />
            </div>

            {/* Submit */}
            <button
              type="submit"
              className="w-full"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                padding: '16px 32px',
                borderRadius: 999,
                fontWeight: 700,
                fontSize: 15,
                backgroundColor: '#000',
                color: '#fff',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                marginTop: 8,
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)'
                e.currentTarget.style.boxShadow = '0 8px 20px rgba(0,0,0,0.15)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)'
                e.currentTarget.style.boxShadow = 'none'
              }}
            >
              Sign In
              <ArrowRightIcon />
            </button>
          </form>

          {/* Divider */}
          <div className="flex items-center" style={{ gap: 16, margin: '28px 0' }}>
            <div className="flex-1" style={{ height: 2, backgroundColor: 'rgba(0,0,0,0.05)', borderRadius: 999 }} />
            <div
              style={{
                fontSize: 10,
                fontWeight: 700,
                color: '#4A4A4A',
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
              }}
            >
              Or continue with
            </div>
            <div className="flex-1" style={{ height: 2, backgroundColor: 'rgba(0,0,0,0.05)', borderRadius: 999 }} />
          </div>

          {/* Social buttons */}
          <div className="flex flex-col" style={{ gap: 12 }}>
            <SocialButton icon={<GoogleIcon />} label="Google" />
            <SocialButton icon={<AppleIcon />} label="Apple" />
          </div>
        </div>

        {/* Footer link */}
        <div
          style={{
            marginTop: 32,
            fontSize: 14,
            fontWeight: 700,
            color: '#4A4A4A',
          }}
        >
          Don't have an account?{' '}
          <a
            href="#"
            onClick={(e) => e.preventDefault()}
            className="hover:underline"
            style={{
              color: '#000',
              textDecorationOffset: '4px',
              marginLeft: 4,
            }}
          >
            Sign up for free
          </a>
        </div>
      </div>
    </div>
  )
}

/* ─── Social Button ─── */

function SocialButton({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <button
      type="button"
      className="flex items-center justify-center cursor-pointer"
      style={{
        background: '#FFFFFF',
        border: '2px solid transparent',
        padding: '16px 20px',
        borderRadius: 16,
        fontWeight: 700,
        fontSize: 14,
        gap: 12,
        transition: 'all 0.2s',
        color: '#000',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = '#000'
        e.currentTarget.style.transform = 'translateY(-1px)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = 'transparent'
        e.currentTarget.style.transform = 'translateY(0)'
      }}
    >
      {icon}
      {label}
    </button>
  )
}
