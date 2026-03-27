# SibyllAI Web UI

Two-phase music analysis web application with visual segment detection, threshold tuning, internal structure detection, and library-based track replacement.

## Architecture

```
sibyllai-web/
├── backend/
│   ├── api/
│   │   └── main.py          # FastAPI app — REST + WebSocket endpoints
│   └── core/
│       ├── analysis.py       # Phase 2 deep analysis (all detectors + structure)
│       └── library_index.py  # SQLite library index for track replacement
└── frontend/
    └── src/
        ├── App.tsx
        ├── components/
        │   ├── MotivePipeline.tsx    # Primary view — upload, waveform, cue cards
        │   ├── WaveformViewer.tsx    # WaveSurfer.js waveform + zoom + segments
        │   ├── CueCard.tsx           # Per-cue analysis results
        │   ├── TrackReplacement.tsx  # Library matching UI
        │   ├── CueSynch.tsx          # CSV cue marker import
        │   ├── LibraryManager.tsx    # Library index management
        │   ├── LicensingPage.tsx     # Licensing cost calculator
        │   ├── LoginScreen.tsx       # Authentication
        │   ├── ProjectsPage.tsx      # Project listing
        │   └── AddItemCombobox.tsx   # Tag editing for curated attributes
        └── lib/
            ├── api.ts          # API client
            ├── store.ts        # Zustand state management
            └── types.ts        # TypeScript type definitions
```

## Features

- **Phase 1: Fast Segmentation** (~3 seconds)
  - Two detection modes: Clean MX (RMS amplitude) or Full Mix (YAMNet classification)
  - Visual waveform display with WaveSurfer.js
  - Adjustable threshold sliders (music detection sensitivity, minimum gap, silence threshold)
  - Real-time segment preview with segment label lane

- **Phase 2: Deep Analysis** (User-confirmed segments only)
  - BPM detection (Essentia)
  - Key detection
  - Instrument classification (YAMNet — 521 audio classes, includes vocals)
  - Genre detection (YAMNet — Pop, Rock, Jazz, Classical, etc.)
  - Style/energy/era tags (CLAP — 37 categorized film-scoring tags)
  - Mood analysis (Music2Emo — valence, arousal, emotion tags)
  - Internal structure detection (section boundaries via spectral novelty)
  - Per-section lightweight analysis (energy, BPM, key)
  - WebSocket progress updates

- **Track Replacement** — Library index matching for finding replacement tracks
- **Cue Sync** — CSV cue marker import/synchronization
- **Licensing Calculator** — Cost estimation tool

## Setup & Run

### ⚠️ CRITICAL: Port Configuration ⚠️

**The backend MUST run on port 8003** — this is hardcoded in `frontend/vite.config.ts`.

Using any other port will cause upload/analysis to freeze with no error messages.

Port 8001 is reserved for kazen — never use it.

### Backend

```bash
# From project root
cd sibyllai-web/backend

# Start FastAPI server - MUST USE PORT 8003
python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8003
```

### Frontend

```bash
# From project root
cd sibyllai-web/frontend

# Install dependencies (first time only)
npm install

# Start dev server
npm run dev
```

Then open http://localhost:5174

## Detection Modes

| Mode | Method | Default Threshold | Use Case |
|------|--------|-------------------|----------|
| **Clean MX** | RMS amplitude | 0.0005 | Music-only stems (no dialogue/SFX) |
| **Full Mix** | YAMNet classification | 0.2 | Full film mix with dialogue and SFX |

## Workflow

1. **Upload** audio/video file (WAV, MP3, M4A, MP4, MOV)
2. **Choose detection mode** (Clean MX or Full Mix)
3. **Preview** segments with adjustable threshold sliders
4. **Analyze** confirmed segments (runs all detectors)
5. **View** musical profile results (BPM, key, mood, instruments, genres, style, sections)
6. **Match** cues to library tracks (Track Replacement)

## Debug Console

In-app console for capturing browser logs (toggle with `D` key):
- Intercepts `console.log/error/warn/info`
- Syncs to backend every 2 seconds
- Accessible to Claude Code at `backend/api/temp/debug.log`
- Auto-rotates (keeps last 500 lines when file exceeds 1000 lines)

```bash
curl http://localhost:8003/api/debug-logs | jq '.logs[]'
```

## API Endpoints

**Core Pipeline:**
- `POST /api/upload` — Upload file, returns file_id
- `POST /api/segment-preview` — Phase 1 (fast segmentation)
- `POST /api/analyze-cues` — Phase 2 (full analysis, background job)
- `GET /api/analysis-status/{session_id}` — Check analysis progress
- `GET /api/projects/{session_id}` — Load .sibyl.json project
- `WS /ws/progress/{session_id}` — Real-time progress updates

**Cue Management:**
- `PUT /api/projects/{session_id}/cues/{cue_id}` — Update curated attributes
- `PUT /api/projects/{session_id}/cues/{cue_id}/replacement` — Set replacement track

**Library:**
- `POST /api/library/build` — Build index from server-side folder
- `POST /api/library/build-upload` — Build/append index from uploaded files (`reset=false` appends)
- `GET /api/library/status/{job_id}` — Poll build status
- `GET /api/library/info` — Get index metadata
- `GET /api/library/sources` — Per-source stats
- `GET /api/library/sources/{source_name}/tracks` — List tracks for a source
- `DELETE /api/library/sources/{source_name}` — Remove source
- `POST /api/library/match` — Match cue to library windows
- `GET /api/library/audio/{track_id}` — Serve track audio for playback

**Debug:**
- `POST /api/debug-logs` — Receive frontend logs (auto-rotation)
- `GET /api/debug-logs` — Retrieve last 100 log lines
- `DELETE /api/cleanup/{file_id}` — Remove temp files

## Track Replacement (MVP)

The track replacement flow builds a **library index** from a folder of music files and
matches film cues to the best-fitting **time ranges** inside those tracks.

### Indexing Strategy

- Library tracks are windowed at **15s, 30s, 60s, 120s** with **50% overlap**
- Each window is analyzed with the same detectors used in cue analysis
- Results are stored in a local SQLite index (`backend/temp/library_index.sqlite`)
- Mood analysis can be toggled for indexing (faster without, better matching with)

### Feature Vector Schema (v1)

The matching vector is built from **existing analysis outputs**:

- BPM (scaled 40–200 BPM → 0–1)
- Key (24-dim one-hot for 12 pitch classes × major/minor)
- Valence + Arousal (scaled 0–5 → 0–1)
- CLAP tag scores (37 dims, ordered by `CLAP_TAG_CATEGORIES`)
- YAMNet instruments (fixed label list, ordered from class map)
- YAMNet genres (fixed label list, ordered from class map)

The index stores:

- `tracks`: track metadata (path, duration, added_at)
- `windows`: window metadata (start/end/duration) + `features_json` + `vector_json`
- `meta`: schema + label ordering

### Matching

Given a cue, we:

- Build its feature vector from the analyzed cue data in the project file
- Select the **nearest window size** (15/30/60/120) by cue length
- Rank candidate windows by cosine similarity
- Return top N matches with **time ranges** inside library tracks

### Manual Test Checklist

1. Start backend on port `8003`
2. Analyze a file and generate cues
3. Go to **Track Replacement** and build an index from a small folder
4. Select a cue and click **Find Matches**
5. Verify returned time ranges are within track duration and look reasonable

## Tech Stack

**Backend:**
- FastAPI (async Python web framework)
- YAMNet (segmentation, instrument detection, genre detection)
- Essentia (BPM/rhythm, key detection)
- CLAP (audio-text embeddings — 37 categorized film-scoring tags)
- Music2Emo (mood/valence/arousal)
- Structure detector (spectral novelty section boundaries)

**Frontend:**
- React 18 + TypeScript
- Vite (build tool)
- WaveSurfer.js v7 (waveform visualization)
- shadcn/ui + Tailwind CSS (UI components)
- Zustand (state management)

## Development

```bash
# Frontend type checking
cd frontend && npm run build

# Backend testing
cd backend && python -m pytest
```
