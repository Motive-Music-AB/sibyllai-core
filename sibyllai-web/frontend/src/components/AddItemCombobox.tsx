import { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'

interface AddItemComboboxProps {
  suggestions: string[]
  onAdd: (item: string) => void
  placeholder?: string
  existingItems?: string[]  // Items already selected (to filter from suggestions)
}

export function AddItemCombobox({
  suggestions,
  onAdd,
  placeholder = 'Type to add...',
  existingItems = [],
}: AddItemComboboxProps) {
  const [value, setValue] = useState('')
  const [showSuggestions, setShowSuggestions] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  // Filter suggestions based on input and exclude existing items
  const filteredSuggestions = suggestions
    .filter((s) => !existingItems.includes(s))
    .filter((s) => s.toLowerCase().includes(value.toLowerCase()))
    .slice(0, 10)  // Limit to 10 suggestions

  const handleAdd = () => {
    const trimmed = value.trim()
    if (trimmed && !existingItems.includes(trimmed)) {
      onAdd(trimmed)
      setValue('')
      setShowSuggestions(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAdd()
    } else if (e.key === 'Escape') {
      setShowSuggestions(false)
    }
  }

  const handleSuggestionClick = (suggestion: string) => {
    onAdd(suggestion)
    setValue('')
    setShowSuggestions(false)
    inputRef.current?.focus()
  }

  return (
    <div className="relative">
      <div className="flex gap-2">
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => {
            setValue(e.target.value)
            setShowSuggestions(true)
          }}
          onFocus={() => setShowSuggestions(true)}
          onBlur={() => {
            // Delay hiding to allow click on suggestion
            setTimeout(() => setShowSuggestions(false), 150)
          }}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="flex-1 px-3 py-2 text-sm rounded-lg bg-white/5 border border-white/10 text-foreground placeholder:text-foreground-muted focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary"
        />
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={handleAdd}
          disabled={!value.trim()}
          className="px-4 py-2 h-auto text-xs glass-lighter"
        >
          Add
        </Button>
      </div>

      {/* Suggestions dropdown */}
      {showSuggestions && value && filteredSuggestions.length > 0 && (
        <div className="absolute z-50 w-full mt-1 glass rounded-lg shadow-lg max-h-48 overflow-y-auto border border-white/10">
          {filteredSuggestions.map((suggestion) => (
            <div
              key={suggestion}
              className="px-3 py-2 text-sm cursor-pointer text-foreground hover:bg-primary/20 first:rounded-t-lg last:rounded-b-lg"
              onMouseDown={(e) => {
                e.preventDefault()  // Prevent blur
                handleSuggestionClick(suggestion)
              }}
            >
              {suggestion}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
