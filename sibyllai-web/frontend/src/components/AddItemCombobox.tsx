import { useState, useRef } from 'react'

interface AddItemComboboxProps {
  suggestions: string[]
  onAdd: (item: string) => void
  placeholder?: string
  existingItems?: string[]
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

  const filteredSuggestions = suggestions
    .filter((s) => !existingItems.includes(s))
    .filter((s) => s.toLowerCase().includes(value.toLowerCase()))
    .slice(0, 10)

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
    <div style={{ position: 'relative' }}>
      <div style={{ display: 'flex', gap: 6 }}>
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
            setTimeout(() => setShowSuggestions(false), 150)
          }}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          style={{
            flex: 1,
            padding: '6px 10px',
            fontSize: '0.7rem',
            fontFamily: 'var(--pl-font-mono)',
            border: '1px solid #080808',
            borderRadius: 6,
            background: '#fff',
            color: '#080808',
            outline: 'none',
          }}
        />
        <button
          className="btn-pill"
          style={{ height: 28, fontSize: '0.6rem', padding: '0 10px' }}
          onClick={handleAdd}
          disabled={!value.trim()}
        >
          Add
        </button>
      </div>

      {/* Suggestions dropdown */}
      {showSuggestions && value && filteredSuggestions.length > 0 && (
        <div style={{
          position: 'absolute',
          zIndex: 50,
          width: '100%',
          marginTop: 4,
          background: '#fff',
          border: '1px solid #080808',
          borderRadius: 6,
          boxShadow: '4px 4px 0 rgba(0,0,0,0.1)',
          maxHeight: 192,
          overflowY: 'auto',
        }}>
          {filteredSuggestions.map((suggestion) => (
            <div
              key={suggestion}
              style={{
                padding: '6px 10px',
                fontSize: '0.7rem',
                fontFamily: 'var(--pl-font-mono)',
                cursor: 'pointer',
              }}
              onMouseEnter={(e) => (e.currentTarget.style.background = '#F0F0F0')}
              onMouseLeave={(e) => (e.currentTarget.style.background = '#fff')}
              onMouseDown={(e) => {
                e.preventDefault()
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
