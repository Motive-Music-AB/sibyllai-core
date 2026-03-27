# CLAUDE.md - sibyllai-web

This file provides guidance to Claude Code when working with the web UI.

## Quick Start

### Running the App (Development)

**⚠️ CRITICAL: BACKEND MUST ALWAYS RUN ON PORT 8003 ⚠️**

The Vite proxy configuration is hardcoded to forward `/api` requests to `localhost:8003`. Using any other port will cause upload/analysis to freeze with no error messages.

Port 8001 is reserved for kazen — **never kill kazen or use port 8001**.

**Backend** (port 8003):
```bash
cd /Volumes/New\ 4\ TB/Dropbox/1_A_WORK/1_Projects/1_Motive-Music-AB/Motive/Github/sibyllai-core
source .venv/bin/activate
cd sibyllai-web/backend && python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8003
```

**Frontend** (port 5174):
```bash
cd sibyllai-web/frontend
npm run dev
```

Access at: http://localhost:5174

### Port Configuration

The frontend proxy is configured in `frontend/vite.config.ts`. If you need to change ports, update BOTH:
1. `vite.config.ts` proxy target
2. The uvicorn command port

## Frontend Pages

| Route | Component | Description |
|-------|-----------|-------------|
| `login` | LoginScreen | Authentication form |
| `projects` | ProjectsPage | Project listing and selection |
| `pipeline` / `analysis` | MotivePipeline | Primary view — upload, waveform, cue cards |
| `cuesynch` | CueSynch | CSV cue marker import/synchronization |
| `replacement` | TrackReplacement | Library upload, cue matching, match details |
| `licensing` | LicensingPage | Licensing cost calculator |

Layout: LoginScreen and ProjectsPage are standalone. Sub-pages use BrutalistLayout wrapper with header/footer.

## Testing Without UI

### curl API Test

```bash
# 1. Upload a test audio file
curl -X POST http://localhost:8003/api/upload \
  -F "file=@/path/to/test-audio.wav" \
  | jq

# Response: {"file_id": "abc123", "filename": "test-audio.wav", ...}

# 2. Get segment preview
curl -X POST http://localhost:8003/api/segment-preview \
  -H "Content-Type: application/json" \
  -d '{"file_id": "abc123", "music_thresh": 0.2, "min_gap": 1.0, "min_cue_length": 3.0}' \
  | jq

# 3. Run analysis on segments
curl -X POST http://localhost:8003/api/analyze-cues \
  -H "Content-Type: application/json" \
  -d '{"file_id": "abc123", "segments": [[0, 30], [45, 90]], "fps": 25}' \
  | jq

# 4. Check output JSON
cat backend/temp/<session_id>/project.sibyl.json | jq '.cues[0].musical_profile.curated'
```

### Python Test Script

Save as `test_analysis.py` and run from sibyllai-core root:

```python
#!/usr/bin/env python3
"""Test YAMNet genre and instrument detection directly."""
import sys
sys.path.insert(0, 'src')

import numpy as np
import soundfile as sf
from sibyllai_core.detectors.yamnet_segmenter import extract_instruments, extract_genres

# Load test audio
audio_path = "path/to/test.wav"  # Change this
y, sr = sf.read(audio_path)

# Mono conversion
if y.ndim > 1:
    y = np.mean(y, axis=1)

# Test instrument detection
print("=== Instruments (YAMNet) ===")
instruments = extract_instruments(y, sr=sr, top_n=10)
for name, score in instruments.items():
    print(f"  {name}: {score:.4f}")

# Test genre detection
print("\n=== Genres (YAMNet) ===")
genres = extract_genres(y, sr=sr, top_n=10)
for name, score in genres.items():
    print(f"  {name}: {score:.4f}")
```

## Data Model

### Curated Fields (displayed in UI)

| Field | Source | Example Values |
|-------|--------|----------------|
| `curated.instruments` | YAMNet | Piano, Guitar, Drums, Singing, Humming |
| `curated.genres` | YAMNet | Pop music, Rock music, Electronic music |
| `curated.style` | CLAP | orchestral, hybrid orchestral, cinematic percussion |
| `curated.moods` | Music2Emo | Energetic, Melancholic |

### Detected Fields (raw scores)

- `detected.instruments_yamnet` — All YAMNet instrument scores
- `detected.genres_yamnet` — All YAMNet genre scores
- `detected.clap_style` — CLAP film-scoring vocabulary
- `detected.clap_instrumentation` — CLAP ensemble detection
- `detected.clap_production`, `clap_energy`, `clap_era`, `clap_function`

### Sections (per-cue internal structure)

Cues >16s get internal section boundary detection via spectral novelty. Each section has:
- `index`, `start`, `end`, `start_relative`, `end_relative`, `duration`
- Lightweight analysis: `energy_label`, `clap_energy`, `bpm`, `key`

Stored in `musical_profile.sections[]`.

## API Endpoints

**Core Pipeline:**
- `POST /api/upload` — Upload audio/video file → file_id
- `POST /api/segment-preview` — Phase 1 fast segmentation
- `POST /api/analyze-cues` — Phase 2 full analysis (background job)
- `GET /api/analysis-status/{session_id}` — Check analysis progress
- `GET /api/projects/{session_id}` — Load analyzed project
- `WS /ws/progress/{session_id}` — Real-time progress updates

**Cue Management:**
- `PUT /api/projects/{session_id}/cues/{cue_id}` — Update curated attributes
- `PUT /api/projects/{session_id}/cues/{cue_id}/replacement` — Set replacement track

**Library:**
- `POST /api/library/build` — Build index from server folder
- `POST /api/library/build-upload` — Build/append index from uploaded files
- `GET /api/library/status/{job_id}` — Build progress
- `GET /api/library/info` — Index metadata
- `GET /api/library/sources` — Per-source stats
- `GET /api/library/sources/{source_name}/tracks` — List tracks for a source
- `DELETE /api/library/sources/{source_name}` — Remove source
- `POST /api/library/match` — Match cue to library → top N matches
- `GET /api/library/audio/{track_id}` — Serve track audio for playback

**Debug:**
- `POST /api/debug-logs` — Receive frontend logs (auto-rotation)
- `GET /api/debug-logs` — Retrieve last 100 log lines
- `DELETE /api/cleanup/{file_id}` — Remove temp files

## Debug Console

In-app console for capturing browser logs (toggle with `D` key):
- Intercepts `console.log/error/warn/info`
- Syncs to backend every 2 seconds
- Accessible at `backend/api/temp/debug.log`
- Auto-rotates (keeps last 500 lines when file exceeds 1000 lines)

```bash
curl http://localhost:8003/api/debug-logs | jq '.logs[]'
```

## Detection Modes

| Mode | Method | Default Threshold | Use Case |
|------|--------|-------------------|----------|
| Clean MX | RMS amplitude | 0.0005 | Music-only stems |
| Full Mix | YAMNet classification | 0.2 (music_thresh) | Full film mix with DIA/SFX |
