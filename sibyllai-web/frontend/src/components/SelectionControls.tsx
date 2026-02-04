import { Button } from '@/components/ui/button'
import { useAppStore } from '@/lib/store'

export function SelectionControls() {
  const { segments, selectedSegments, setSelectedSegments, project } = useAppStore()

  // Don't show after analysis
  if (project || segments.length === 0) return null

  return (
    <div className="w-full flex items-center justify-between glass px-6 py-4 rounded-2xl border-white/5 animate-in slide-in-from-top-4 duration-500">
      <div className="flex items-center gap-3">
        <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
        <div className="text-sm font-medium text-foreground tracking-tight">
          Select segments to include in Musical Analysis
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="text-xs font-bold uppercase tracking-widest text-foreground-muted bg-white/5 px-3 py-1.5 rounded-lg border border-white/5">
          {selectedSegments.length} <span className="text-[10px] opacity-60">OF</span> {segments.length} <span className="text-[10px] opacity-60">SELECTED</span>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSelectedSegments(segments)}
            className="h-9 px-4 rounded-xl hover:bg-white/10 text-xs font-bold uppercase tracking-widest"
          >
            Select All
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSelectedSegments([])}
            disabled={selectedSegments.length === 0}
            className="h-9 px-4 rounded-xl hover:bg-white/10 text-xs font-bold uppercase tracking-widest disabled:opacity-30"
          >
            Clear
          </Button>
        </div>
      </div>
    </div>
  )
}
