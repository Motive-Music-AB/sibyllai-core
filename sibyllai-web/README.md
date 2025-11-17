# SibyllAI Web UI

Two-phase music analysis web application with visual segment detection and threshold tuning.

## Architecture

```
sibyllai-web/
├── backend/         # FastAPI backend
│   ├── api/         # REST + WebSocket endpoints
│   └── core/        # Analysis modules
└── frontend/        # React + TypeScript UI
```

## Features

- **Phase 1: Fast Segmentation** (YAMNet only, ~3 seconds)
  - Visual waveform display with WaveSurfer.js
  - Adjustable threshold slider (music detection sensitivity)
  - Adjustable minimum gap (silence duration)
  - Real-time segment preview

- **Phase 2: Deep Analysis** (User-confirmed segments only)
  - BPM detection (Essentia)
  - Key detection
  - Instrument classification (YAMNet)
  - Mood analysis (Music2Emotion)
  - CLAP genre/style tags (29 categorized tags)
  - WebSocket progress updates

## Setup & Run

### ⚠️ CRITICAL: Port Configuration ⚠️

**The backend MUST run on port 8001** - this is hardcoded in `frontend/vite.config.ts`.

Using any other port will cause:
- Upload freezes with no error messages
- Analysis requests to hang indefinitely
- Silent failures in the frontend

### Backend

```bash
# From project root
cd sibyllai-web/backend

# Start FastAPI server - MUST USE PORT 8001
python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8001
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

Then open http://localhost:5173

## Workflow

1. **Upload** audio/video file (WAV, MP3, M4A, MP4, MOV)
2. **Preview** segments with adjustable threshold slider
3. **Tune** detection parameters visually
4. **Analyze** confirmed segments (runs expensive detectors)
5. **View** musical profile results (BPM, key, mood, instruments, tags)

## Debug Console

In-app console for capturing browser logs (toggle with `D` key):
- Intercepts `console.log/error/warn/info`
- Syncs to backend every 2 seconds
- Accessible to Claude Code at `backend/api/temp/debug.log`
- Auto-rotates (keeps last 500 lines when file exceeds 1000 lines)

**For Claude Code:**
```bash
curl http://localhost:8001/api/debug-logs | jq '.logs[]'
```

## API Endpoints

- `POST /api/upload` - Upload file, returns file_id
- `POST /api/segment-preview` - Phase 1 (fast YAMNet segmentation)
- `POST /api/analyze-cues` - Phase 2 (full analysis)
- `GET /api/projects/{session_id}` - Load .sibyl.json project
- `WS /ws/progress/{session_id}` - Real-time analysis progress
- `DELETE /api/cleanup/{file_id}` - Remove temp files
- `POST /api/debug-logs` - Receive frontend logs (auto-rotation)
- `GET /api/debug-logs` - Retrieve last 100 log lines

## Tech Stack

**Backend:**
- FastAPI (async Python web framework)
- YAMNet (fast music/speech segmentation)
- Essentia (BPM/rhythm analysis)
- CLAP (audio-text embeddings for genre/style)
- Music2Emotion (mood/valence/arousal)
- Chord detector (key detection)

**Frontend:**
- React 18 + TypeScript
- Vite (build tool)
- WaveSurfer.js v7 (waveform visualization)
- shadcn/ui + Tailwind CSS (UI components)
- Zustand (state management)
- Axios (HTTP client)

## Development

```bash
# Frontend type checking
cd frontend && npm run build

# Backend testing
cd backend && python -m pytest
```
