# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SibyllAI Core is a music analysis tool for film composers, designed to analyze temp music (MX) tracks and extract comprehensive musical characteristics. The primary use case is **pre-scoring analysis**: composers receive temp MX from editors and use SibyllAI to understand what musical elements (genre, instrumentation, mood, energy) the director responded to, informing their original compositions.

**Current Status:** MVP complete — web UI with two-phase analysis, library matching/track replacement, and brutalist design system.

**Track Replacement:** Reverse lookup capability — match detected musical profiles against composer's personal music library to suggest similar unused tracks/demos. MVP implemented with SQLite-backed library index and cosine similarity matching.

## Tech Stack

- Python 3.11+
- PyTorch 2.2.2 (pinned for compatibility; has known RCE vulnerability via torch.load - only use trusted model files)
- TensorFlow & Keras (for YAMNet)
- YAMNet (music cue segmentation + instrument detection from 521 audio event classes)
- CLAP (LAION-CLAP for genre, production style, energy, era, function, instrumentation — 37 categorized tags)
- Music2Emo (mood/emotion analysis: valence, arousal, mood tags)
- Essentia (rhythm/BPM extraction)
- Librosa (audio processing, resampling)

## Common Commands

### Installation
```bash
pip install -e .[dev]
```

### Running the Web App (Development)

**⚠️ CRITICAL: BACKEND MUST ALWAYS RUN ON PORT 8003 ⚠️**

The Vite proxy configuration is hardcoded to forward `/api` requests to `localhost:8003`. Using any other port will cause upload/analysis to freeze with no error messages.

Port 8001 is reserved for kazen — **never kill kazen or use port 8001**.

```bash
# Backend (from project root) - ALWAYS PORT 8003
cd sibyllai-web/backend && python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8003

# Frontend (from project root) - ALWAYS PORT 5174
cd sibyllai-web/frontend && npm run dev
```

If you need to change the port, update BOTH:
1. `sibyllai-web/frontend/vite.config.ts` (proxy target)
2. The uvicorn command port

### Running the CLI
```bash
# Basic usage
python -m sibyllai_core.cli <audio_or_video_file>

# With custom parameters
python -m sibyllai_core.cli <input_file> --fps 25 --thr 0.5
```

**Parameters:**
- `src`: Audio or video file (input) - required positional argument
- `--fps`: Time-code FPS (default: 25)
- `--thr`: Mood probability threshold (default: 0.5)

### Linting
```bash
ruff check src/
```

### Running Tests
No formal test suite exists yet. Manual testing is required. When adding features:
- Manually test affected code paths
- Reproduce bugs before and after fixes
- Provide minimal examples (script, CLI command, or usage note)

## High-Level Architecture

### Pipeline Flow

The system follows a sequential processing pipeline:

1. **CLI Entry** (`src/sibyllai_core/cli.py`):
   - Entry point via `main()` function
   - Parses arguments and calls `analyse()` from pipeline.py

2. **Audio Extraction** (`pipeline.py:_extract_audio()`):
   - Extracts audio from video/audio files using ffmpeg
   - Converts to mono 44.1 kHz WAV format for analysis
   - Creates temporary working directory

3. **Music Cue Detection** (`detectors/yamnet_segmenter.py`):
   - Two modes: YAMNet classification (Full Mix) or RMS amplitude (Clean MX)
   - Returns list of (start, end) time ranges in seconds
   - Filters segments shorter than 3 seconds (minimum cue length)

4. **Feature Analysis** (per detected cue):
   - **BPM Analysis**: Uses Essentia RhythmExtractor2013 (requires 44100 Hz)
   - **Key Detection**: Chord/key detection
   - **Instrument Detection**: YAMNet — top 15 from 521 audio event classes (includes vocals: Singing, Humming, Choir, etc.)
   - **Genre Detection**: YAMNet — Pop, Rock, Jazz, Electronic, Classical, Hip hop, Country, etc.
   - **Style/Tags**: CLAP analysis across 37 categorized tags (genre, production, energy, era, function, instrumentation)
   - **Mood Analysis**: Music2Emo model for valence, arousal, and mood tags
   - **Structure Detection**: Spectral novelty change-point detection for internal section boundaries (intro, verse, chorus, bridge, build-up, climax)

5. **Per-Section Lightweight Analysis** (for cues >16s):
   - Each detected section gets energy label (CLAP), BPM, and key
   - Section data stored with absolute and relative timestamps in `musical_profile.sections[]`

6. **Output Generation**:
   - Creates incremental run folders: `outputs/run_001/`, `outputs/run_002/`, etc.
   - Saves project file: `project.sibyl.json` with comprehensive cue data
   - Structured format separates universal musical attributes from project-specific context

### Key Components

**Detectors Module** (`src/sibyllai_core/detectors/`):
- Each detector provides standalone analysis functionality
- Exports consolidated in `__init__.py`: `music_probability`, `tag_chunk`, `global_moods`, `detect_sections`, `detect_sections_with_fallback`, `Section`
- Key files:
  - `yamnet_segmenter.py` — `segment_music_regions()`, `extract_instruments()`, `extract_genres()`
  - `clap.py` — `tag_chunk()` (37 categorized audio-text tags)
  - `m2e_wrapper.py` — `global_moods()` (valence, arousal, mood tags)
  - `chord_detector.py` — Key detection
  - `structure.py` — `detect_sections()`, `detect_sections_with_fallback()` (spectral novelty + agglomerative fallback)
  - `ast.py` — `music_probability()` (legacy audio spectrogram transformer)
- Detectors are designed to be independent and resilient (failures are caught and logged)

**Music2Emo Integration** (`src/sibyllai_core/thirdparty/music2emo/`):
- Integrated third-party package for music emotion recognition
- Main interface: `Music2emo` class in `music2emo.py`
- Wrapper function: `global_moods()` in `detectors/m2e_wrapper.py`
- Uses pre-trained models for valence/arousal/mood prediction
- Complex dependencies (PyTorch Lightning, Hydra configs, MERT encoder)

**Cue Data Model**:

Each detected music cue contains two main sections:

1. **`musical_profile`** - Universal musical attributes (used for future library matching):
   - **`detected`**: Comprehensive AI analysis with confidence scores
     - YAMNet instruments (all detected with confidence)
     - YAMNet genres (all detected with confidence)
     - Music2Emo moods (all detected with confidence)
     - CLAP tags organized by category: genre, production, energy, era, function, instrumentation
   - **`curated`**: User-approved subset of detected attributes
     - Top instruments, genres, moods, style, energy, era, etc.
     - User can promote/demote/remove via UI
   - **Musical fundamentals**: BPM, key, valence (0-1), arousal (0-1)
   - **`sections`**: Internal structure boundaries (for cues >16s)
     - Each section: index, start/end (absolute + relative), duration
     - Lightweight analysis: energy_label, clap_energy, bpm, key

2. **`project_context`** - Project-specific metadata (NOT used for matching):
   - Custom tags (e.g., "Hero's theme", "Director's favorite")
   - Composer notes
   - Visual color coding
   - Status (draft/review/locked)
   - Director feedback

This separation enables reverse-lookup: "Find tracks in my library matching this cue's musical_profile" (implemented as Track Replacement MVP).

**Web UI Components:**
- `MotivePipeline.tsx` — Primary view: upload, waveform, threshold tuning, cue cards
- `WaveformViewer.tsx` — WaveSurfer.js waveform with zoom, segments, playback, segment label lane
- `CueCard.tsx` — Per-cue analysis results (instruments, genres, style, mood, BPM, key)
- `TrackReplacement.tsx` — Library upload, cue matching, match details
- `CueSynch.tsx` — CSV cue marker import/synchronization
- `LibraryManager.tsx` — Library index management
- `LicensingPage.tsx` — Licensing cost calculator
- `LoginScreen.tsx` — Authentication
- `ProjectsPage.tsx` — Project listing and selection
- `AddItemCombobox.tsx` — Tag editing combobox for curated attributes

## Dependency Management

**Critical: All dependencies must be declared in `pyproject.toml` ONLY.**
- Do NOT modify `requirements.txt` (legacy/compatibility only)
- Specify minimum versions unless strict pinning is necessary
- Remove unused dependencies promptly
- `torch==2.2.2` is pinned due to compatibility constraints (see security notice in README)

## Code Style

- Follow PEP8
- Use 4 spaces (no tabs)
- Add type hints to all new functions
- Run `ruff` before committing
- Add docstrings to all public functions/classes/modules
- Add comments for non-obvious logic

## UI Design Guidelines

### Tooltip Implementation

When implementing tooltips in the React frontend:

**Positioning:**
- Use `position: fixed` with viewport coordinates (`e.clientX`, `e.clientY`)
- NEVER use `position: absolute` with container-relative coordinates - causes positioning mismatches
- Track mouse position with `window.addEventListener('mousemove')` and update state
- Clean up event listeners in useEffect return function

**Styling:**
- Match application design language: white background, gray border, shadow-xl
- Use TailwindCSS classes: `bg-white border border-gray-300 rounded-lg shadow-xl`
- Position tooltip above cursor: `transform: translate(-50%, calc(-100% - 8px))`
- Add high z-index for visibility: `z-index: 9999`
- Use `pointer-events-none` for non-interactive tooltips

**Example:**
```tsx
// State
const [tooltipData, setTooltipData] = useState<{ x: number; y: number; text: string } | null>(null)

// Mouse tracking
useEffect(() => {
  if (!tooltipData) return
  const handleMouseMove = (e: MouseEvent) => {
    setTooltipData(prev => prev ? { ...prev, x: e.clientX, y: e.clientY } : null)
  }
  window.addEventListener('mousemove', handleMouseMove)
  return () => window.removeEventListener('mousemove', handleMouseMove)
}, [tooltipData])

// Render
{tooltipData && (
  <div
    className="fixed bg-white border border-gray-300 rounded-lg shadow-xl px-3 py-2 pointer-events-none"
    style={{
      left: `${tooltipData.x}px`,
      top: `${tooltipData.y}px`,
      transform: 'translate(-50%, calc(-100% - 8px))',
      zIndex: 9999
    }}
  >
    {tooltipData.text}
  </div>
)}
```

## Error Handling Philosophy

The pipeline is designed to be resilient:
- Each detector analysis is wrapped in try-except
- Failures are logged as warnings but don't halt the pipeline
- Fallback values ("Unknown", 0.0) are used when analysis fails
- Segments with failed processing are skipped, not fatal

## Known Issues

- No formal automated testing structure exists
- DeepFilterNet/ directory is included but not integrated (ignore it)
- Some detectors may fail silently on edge cases
- Music2Emo analysis is the most fragile detector (run last in pipeline)

## Critical: WaveformViewer Zoom Implementation

**⚠️ DO NOT SIMPLIFY THE ZOOM LOGIC - IT TOOK WEEKS TO DEBUG ⚠️**

WaveSurfer.js zoom works in **pixels-per-second**, NOT percentage:
- `ws.zoom(0)` = "fit to container" (auto-calculated px/sec)
- `ws.zoom(N)` = N pixels per second

**The Bug That Kept Breaking Zoom:**

When at zoom=0 (fit mode), the effective px/sec depends on audio duration and container width:
- 20 second clip in 1000px container = **50 px/sec effective**
- If you naively jump to `zoom=10`, you're actually **ZOOMING OUT** (10 < 50)!

**The Fix (in `WaveformViewer.tsx`):**

Always calculate `effectiveZoom = viewportWidth / duration` when at zoom=0, then find a zoom level that's actually greater (for zoom in) or less (for zoom out) than that value.

```typescript
// Calculate effective zoom level (pixels per second) when in fit mode
const effectiveZoom = zoom === 0 ? viewportWidth / duration : zoom

// Find the next zoom level that's actually higher than current effective zoom
for (const level of ZOOM_LEVELS) {
  if (level > effectiveZoom) {
    newZoom = level
    break
  }
}
```

**Never:**
- Remove the effectiveZoom calculation
- Assume zoom=0 means "zoom level 0 in the array"
- Simplify to just incrementing/decrementing array indices

## Output Structure

```
outputs/
  run_001/
    project.sibyl.json         # Complete project file with all cues
  run_002/
    project.sibyl.json
  ...
```

Each run creates an incremental folder to avoid overwriting previous analyses.

**Project File Format (.sibyl.json)**:
```json
{
  "version": "1.0",
  "project": {
    "name": "Feature Film Final",
    "mx_file": "/path/to/mx.wav",
    "created": "2025-01-15T10:30:00Z",
    "fps": 24
  },
  "cues": [
    {
      "id": "cue_001",
      "name": "",
      "start": 323.5,
      "end": 465.2,
      "start_tc": "00:05:23:12",
      "end_tc": "00:07:45:05",
      "musical_profile": { /* detected + curated + fundamentals */ },
      "project_context": { /* custom_tags, notes, color, status */ }
    }
  ]
}
```

## Changelog

### February 2026 - Segment Label Lane & UI Cleanup

**Segment Label Lane (WaveformViewer):**
- Added a dedicated horizontal lane between the timecode ruler and the waveform
- Pre-analysis: renders selection checkboxes per segment (green ✓ when selected)
- Post-analysis: renders cue number labels (#1, #2, etc.) for analyzed segments
- Lane scrolls and resizes with zoom automatically (same width calc as ruler)
- Narrow segments (<16px) hide their label to avoid visual clutter
- Container is pointer-events-none so clicks in gaps pass through to the waveform
- Replaced old imperative DOM checkbox buttons that were appended inside WaveSurfer regions

**Removed Zoom Overlay:**
- Removed the orange "Zooming..." spinner overlay that appeared during zoom transitions
- The synchronous zoom guard (isZoomingRef) still prevents race conditions from rapid clicks

### November 2025 - MVP Backend Complete

**Major Performance Improvement:**
- Removed Demucs source separation (10-100x speed improvement)
- Significant speed improvement from removing Demucs
- Significant speed improvement from removing Demucs

**Enhanced CLAP Analysis:**
- Expanded from 5 to 37 categorized tags
- Tags organized into: genre, production, energy, era, function, **instrumentation**
- Added ensemble-level instrumentation detection (brass section, string ensemble, woodwinds, etc.)
- Pre-computed text embeddings for performance optimization
- CLAP provides 6-13% confidence for orchestral film music (10-13x better than YAMNet for ensemble detection)

**YAMNet Instrument Detection:**
- Added extract_instruments() function
- Detects top 15 instruments per segment from 521 audio classes
- Returns confidence scores for each detected instrument
- Note: YAMNet shows low confidence (<1%) for orchestral ensembles; CLAP instrumentation preferred for film music

**New .sibyl Project Format:**
- Structured JSON format with versioning
- Separates musical_profile (universal, searchable) from project_context (project-specific notes)
- Detected (all scores) vs curated (top items) for UI flexibility
- Curated lists: 8 instruments, 2 genres, 3 instrumentation tags, 2 moods
- Includes timecode conversion (HH:MM:SS:FF)
- Backwards-compatible CSV output maintained

**Web UI Updates:**
- UI displays CLAP instrumentation instead of YAMNet instruments
- UI displays CLAP genre instead of Music2Emo moods
- Both changes provide more accurate, film-music-appropriate tags
- Smart playback control: First spacebar plays from cue start, subsequent plays from current playhead position
- Click waveform to set playhead position for precise scrubbing

**Simplified Chord Analysis:**
- Removed complex chord detection (primary chord, confidence, complexity)
- Kept only key detection for cleaner, more reliable results

**Testing & Validation:**
- All models (YAMNet, CLAP, Music2Emo, Essentia) working on test files
- always use port 8003 for this app (port 8001 is reserved for kazen — never kill it)