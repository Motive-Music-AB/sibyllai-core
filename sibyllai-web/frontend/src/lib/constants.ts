/**
 * Predefined lists for adding items to curated fields.
 * Based on YAMNet class map and CLAP tag categories.
 */

// YAMNet instrument classes (common musical instruments + vocals)
export const YAMNET_INSTRUMENTS = [
  // Keyboards
  "Piano", "Electric piano", "Keyboard (musical)", "Organ", "Hammond organ", "Harpsichord", "Synthesizer",
  // Guitars
  "Guitar", "Electric guitar", "Acoustic guitar", "Bass guitar", "Steel guitar, slide guitar",
  // Strings
  "Violin, fiddle", "Cello", "Double bass", "Viola", "Harp",
  "String section", "Bowed string instrument", "Plucked string instrument",
  // Drums & Percussion
  "Drum", "Drum kit", "Snare drum", "Bass drum", "Drum roll", "Drum machine",
  "Percussion", "Mallet percussion", "Timpani", "Cymbal", "Hi-hat", "Tambourine",
  "Gong", "Marimba, xylophone", "Vibraphone", "Steelpan",
  // Brass
  "Brass instrument", "Trumpet", "Trombone", "French horn", "Tuba",
  // Woodwinds
  "Flute", "Clarinet", "Saxophone", "Oboe", "Bassoon", "Harmonica", "Accordion",
  // Vocals
  "Singing", "Humming", "Whistling", "Choir", "Vocal music",
  "Child singing", "Male singing", "Female singing",
  "Rapping", "Beatboxing", "Chant", "A capella",
  "Speech", "Child speech, kid speaking",
  // Other
  "Banjo", "Ukulele", "Mandolin", "Sitar", "Bagpipes"
]

// YAMNet genre/music style classes
export const YAMNET_GENRES = [
  // Pop/Rock
  "Pop music", "Rock music", "Rock and roll", "Heavy metal", "Punk rock",
  "Grunge", "Progressive rock", "Psychedelic rock", "Indie rock",
  // Electronic
  "Electronic music", "Techno", "House music", "Trance music", "Dubstep",
  "Drum and bass", "Electronica", "Disco", "Dance music",
  // Jazz/Blues
  "Jazz", "Blues", "Soul music", "Rhythm and blues", "Funk", "Swing music",
  // Classical
  "Classical music", "Opera", "Choir",
  // World/Traditional
  "Country", "Folk music", "Bluegrass", "Reggae", "Ska",
  "Latin music", "Salsa music", "Flamenco", "Middle Eastern music",
  "Traditional music", "Gospel music", "Christian music",
  // Other
  "Hip hop music", "Soundtrack music", "Theme music", "Video game music",
  "Ambient music", "New-age music", "Music for children",
  // Mood-based (YAMNet classes)
  "Happy music", "Sad music", "Tender music", "Exciting music", "Angry music", "Scary music"
]

// CLAP film-scoring style tags (from clap.py CLAP_TAG_CATEGORIES)
export const CLAP_STYLES = [
  // Genre
  "orchestral", "electronic", "hybrid orchestral", "rock", "jazz",
  "classical", "minimalist", "ambient", "cinematic percussion",
  // Instrumentation
  "brass section", "string ensemble", "woodwinds", "choir vocals",
  "piano keys", "guitar bass", "synthesizers", "full orchestra",
  // Production
  "polished production", "raw production", "vintage sound",
  "modern production", "acoustic recording", "heavily processed",
  // Energy
  "high energy", "low energy", "building tension", "climactic", "gentle", "aggressive",
  // Era
  "retro", "futuristic", "timeless", "modern",
  // Function
  "action sequence", "dramatic underscore", "theme music", "transitional"
]

// Confidence thresholds for showing detected items
// Different detectors produce different score ranges:
// - YAMNet instruments: typically 0.001-0.01 (averaged frame probabilities)
// - YAMNet genres: typically 0.005-0.05
// - CLAP style: typically 0.1-0.4 (cosine similarity)
export const CONFIDENCE_THRESHOLD = 0.01  // Default 1% (kept for backwards compatibility)
export const INSTRUMENT_THRESHOLD = 0.0005  // 0.05% - YAMNet instrument scores are very low
export const GENRE_THRESHOLD = 0.005  // 0.5% - YAMNet genres slightly higher
export const STYLE_THRESHOLD = 0.01  // 1% - CLAP scores are normalized higher
