import { useEffect } from 'react'
import { MotivePipeline } from '@/components/MotivePipeline'
import { CueSynch } from '@/components/CueSynch'
import { TrackReplacement } from '@/components/TrackReplacement'
import { LicensingPage } from '@/components/LicensingPage'
import { LoginScreen } from '@/components/LoginScreen'
import { ProjectsPage } from '@/components/ProjectsPage'
import { ProjectWorkspace } from '@/components/ProjectWorkspace'
import { ProjectDelivery } from '@/components/ProjectDelivery'
import { RightsSetup } from '@/components/RightsSetup'
import { FileAnalysis } from '@/components/FileAnalysis'
import { useAppStore } from '@/lib/store'
import '@/components/motive-pipeline.css'

const PAGE_SET = new Set(['login', 'projects', 'workspace', 'rights', 'delivery', 'analysis', 'cuesynch', 'replacement', 'licensing', 'pipeline'])

/** Brutalist shell wrapper for sub-pages (CueSynch, TrackReplacement, Licensing) */
function BrutalistLayout({ children }: { children: React.ReactNode }) {
  const { reset, setCurrentPage } = useAppStore()

  return (
    <div className="pipeline-root">
      <div className="noise-overlay" />

      <header className="pipeline-header">
        <div className="brand">
          <div className="eye-icon" />
          MOTIVE
        </div>
        <div className="header-controls">
          <button className="btn-pill" onClick={() => reset()}>
            New Session
          </button>
          <button className="btn-pill primary" onClick={() => setCurrentPage('pipeline')}>
            Pipeline
          </button>
        </div>
      </header>

      <div style={{ flex: 1, overflow: 'auto', padding: '24px 40px' }}>
        {children}
      </div>

      <footer className="pipeline-footer">
        <div className="eye-icon" style={{ width: 16, height: 10 }} />
        MOTIVE AUDIO ANALYSIS
      </footer>
    </div>
  )
}

function App() {
  const { currentPage } = useAppStore()
  const setCurrentPage = useAppStore((s) => s.setCurrentPage)

  // Restore session on mount + listen for browser back/forward
  useEffect(() => {
    // Restore auth from sessionStorage
    try {
      const saved = sessionStorage.getItem('motive_auth')
      if (saved) {
        const { isLoggedIn: wasLoggedIn, userName } = JSON.parse(saved)
        if (wasLoggedIn) {
          useAppStore.setState({ isLoggedIn: true, userName, currentPage: 'projects' })
        }
      }
    } catch { /* ignore */ }

    // Resolve initial page from hash
    const hashPage = window.location.hash.replace('#/', '')
    if (hashPage && PAGE_SET.has(hashPage)) {
      useAppStore.setState({ currentPage: hashPage as typeof currentPage })
    }

    // Browser back/forward
    const onPopState = (e: PopStateEvent) => {
      const page = e.state?.page
      if (page && PAGE_SET.has(page)) {
        useAppStore.setState({ currentPage: page })
      }
    }

    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (currentPage === 'login') {
    return <LoginScreen />
  }

  if (currentPage === 'projects') {
    return <ProjectsPage />
  }

  if (currentPage === 'workspace') {
    return <ProjectWorkspace />
  }

  if (currentPage === 'rights') {
    return <RightsSetup />
  }

  if (currentPage === 'analysis') {
    return <FileAnalysis />
  }

  if (currentPage === 'delivery') {
    return <ProjectDelivery />
  }

  // Sub-pages wrapped in consistent brutalist shell
  if (currentPage === 'cuesynch') {
    return (
      <BrutalistLayout>
        <CueSynch />
      </BrutalistLayout>
    )
  }

  if (currentPage === 'replacement') {
    return <TrackReplacement />
  }

  if (currentPage === 'licensing') {
    return (
      <BrutalistLayout>
        <LicensingPage />
      </BrutalistLayout>
    )
  }

  // Default: Pipeline view is the primary UI
  return <MotivePipeline />
}

export default App
