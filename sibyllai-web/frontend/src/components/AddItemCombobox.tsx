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
      <div className="flex gap-1">
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
          className="flex-1 px-2 py-1 text-sm border rounded bg-white focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={handleAdd}
          disabled={!value.trim()}
          className="px-2 py-1 h-auto text-xs"
        >
          Add
        </Button>
      </div>

      {/* Suggestions dropdown */}
      {showSuggestions && value && filteredSuggestions.length > 0 && (
        <div className="absolute z-50 w-full mt-1 bg-white border rounded shadow-lg max-h-48 overflow-y-auto">
          {filteredSuggestions.map((suggestion) => (
            <div
              key={suggestion}
              className="px-2 py-1 text-sm cursor-pointer hover:bg-blue-50"
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
