import { MotivePipeline } from '@/components/MotivePipeline'
import { CueSynch } from '@/components/CueSynch'
import { TrackReplacement } from '@/components/TrackReplacement'
import { LicensingPage } from '@/components/LicensingPage'
import { useAppStore } from '@/lib/store'

function App() {
  const { currentPage } = useAppStore()

  // Sub-pages render inside the old layout shell for now
  if (currentPage === 'cuesynch') {
    return (
      <div className="min-h-screen relative overflow-hidden selection:bg-primary/30 selection:text-foreground">
        <div className="relative z-10 max-w-[1800px] mx-auto px-6 py-12 lg:px-12">
          <CueSynch />
        </div>
      </div>
    )
  }

  if (currentPage === 'replacement') {
    return (
      <div className="min-h-screen relative overflow-hidden selection:bg-primary/30 selection:text-foreground">
        <div className="relative z-10 max-w-[1800px] mx-auto px-6 py-12 lg:px-12">
          <TrackReplacement />
        </div>
      </div>
    )
  }

  if (currentPage === 'licensing') {
    return (
      <div className="min-h-screen relative overflow-hidden selection:bg-primary/30 selection:text-foreground">
        <div className="relative z-10 max-w-[1800px] mx-auto px-6 py-12 lg:px-12">
          <LicensingPage />
        </div>
      </div>
    )
  }

  // Default: Pipeline view is the primary UI
  return <MotivePipeline />
}

export default App
