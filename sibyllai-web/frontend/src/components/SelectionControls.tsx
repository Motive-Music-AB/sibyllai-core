import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/lib/store'

export function SelectionControls() {
  const { segments, selectedSegments, setSelectedSegments, project } = useAppStore()

  // Don't show after analysis
  if (project || segments.length === 0) return null

  return (
    <Card className="w-full">
      <CardContent className="py-3">
        <div className="flex items-center justify-between">
          <div className="text-sm text-muted-foreground">
            Click "Select" above cues to choose which ones to analyze
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelectedSegments(segments)}
            >
              Select All
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelectedSegments([])}
              disabled={selectedSegments.length === 0}
            >
              Clear All
            </Button>
            <div className="text-sm text-muted-foreground">
              {selectedSegments.length} / {segments.length} selected
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
