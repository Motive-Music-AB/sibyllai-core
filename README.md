# SibyllAI Core

A music analysis tool for film composers, designed to analyze temp music (MX) tracks and extract comprehensive musical characteristics. The primary use case is **pre-scoring analysis**: composers receive temp MX from editors and use SibyllAI to understand what musical elements (genre, instrumentation, mood, energy) the director responded to, informing their original compositions.

**Current Status:** MVP complete — web UI with two-phase analysis, library matching, and brutalist design system.

---

## How It Works (High-Level Flow)

### Web UI (Primary)

1. **Upload** audio/video file (WAV, MP3, M4A, MP4, MOV)
2. **Phase 1 — Fast Segmentation** (~3 seconds): YAMNet or RMS amplitude detects music regions, displayed on an interactive waveform with adjustable thresholds
3. **Phase 2 — Deep Analysis** (user-confirmed segments only):
   - BPM (Essentia) + Key detection
   - Instruments (YAMNet, top 15 from 521 audio classes)
   - Genres (YAMNet: Pop, Rock, Jazz, Classical, etc.)
   - Style/Energy/Era tags (CLAP, 37 categorized tags for film scoring)
   - Mood (Music2Emo: valence, arousal, emotion tags)
   - Internal structure detection (section boundaries within each cue)
4. **Output** — `project.sibyl.json` with per-cue musical profiles + project context

### CLI

```bash
python -m sibyllai_core.cli <audio_or_video_file> --fps 25 --thr 0.5
```

Same analysis pipeline, outputs to `outputs/run_NNN/project.sibyl.json`.

---

## Detection Modes

| Mode | Method | Default Threshold | Use Case |
|------|--------|-------------------|----------|
| **Clean MX** | RMS amplitude | 0.0005 | Music-only stems (no dialogue/SFX) |
| **Full Mix** | YAMNet classification | 0.2 | Full film mix with dialogue and SFX |

---

## Tech Stack

**Backend:**
- Python 3.11+
- FastAPI (async web framework)
- PyTorch 2.2.2 (pinned for compatibility — see security notice below)
- TensorFlow & Keras (for YAMNet)
- YAMNet (music segmentation, instrument detection, genre detection)
- CLAP / LAION-CLAP (genre, production style, energy, era, function — 37 categorized tags)
- Music2Emo (mood/emotion: valence, arousal, mood tags)
- Essentia (BPM/rhythm extraction, key detection)
- Librosa (audio processing, resampling)
- PyTorch Lightning + Hydra (Music2Emo model infrastructure)

**Frontend:**
- React 18 + TypeScript
- Vite (build tool)
- WaveSurfer.js v7 (waveform visualization)
- shadcn/ui + Tailwind CSS (UI components)
- Zustand (state management)

---

## Project Structure

```
src/
  sibyllai_core/
    cli.py                  # Command-line interface
    pipeline.py             # Main analysis pipeline (CLI path)
    sibyl_format.py         # .sibyl.json project file format
    detectors/
      __init__.py           # Exports: music_probability, tag_chunk, global_moods, detect_sections, Section
      yamnet_segmenter.py   # Music segmentation, instrument & genre detection
      clap.py               # CLAP audio-text embeddings (37 categorized tags)
      m2e_wrapper.py        # Music2Emo mood/emotion wrapper
      chord_detector.py     # Key detection
      structure.py          # Internal section boundary detection (spectral novelty)
      ast.py                # Audio spectrogram transformer (legacy)
    markers/                # Marker and export utilities
    thirdparty/
      music2emo/            # Integrated Music2Emo package (models, configs)

sibyllai-web/
  backend/
    api/
      main.py               # FastAPI app — all REST + WebSocket endpoints
    core/
      analysis.py            # Phase 2 deep analysis (runs all detectors)
      library_index.py       # SQLite library index for track replacement matching
  frontend/
    src/
      components/
        MotivePipeline.tsx   # Primary analysis view (upload → waveform → cue cards)
        WaveformViewer.tsx   # WaveSurfer.js waveform with zoom, segments, playback
        CueCard.tsx          # Per-cue analysis results display
        TrackReplacement.tsx # Library upload, cue matching, match details
        CueSynch.tsx         # CSV cue marker import/synchronization
        LibraryManager.tsx   # Library index management
        LicensingPage.tsx    # Licensing cost calculator
        LoginScreen.tsx      # Authentication
        ProjectsPage.tsx     # Project listing and selection
      lib/
        api.ts               # API client
        store.ts             # Zustand state management
        types.ts             # TypeScript type definitions
```

---

## Detailed Pipeline Flow

1. **Audio Extraction** — ffmpeg converts input to mono 44.1 kHz WAV
2. **Music Cue Detection** (`yamnet_segmenter.py`):
   - Clean MX mode: RMS amplitude thresholding
   - Full Mix mode: YAMNet music probability classification
   - Returns list of (start, end) time ranges
3. **Per-Cue Analysis** (7 detectors):
   - **BPM**: Essentia RhythmExtractor2013 (requires 44100 Hz)
   - **Key**: Chord/key detection
   - **Instruments**: YAMNet — top 15 from 521 audio event classes (includes vocals: Singing, Humming, Choir, etc.)
   - **Genres**: YAMNet — Pop, Rock, Jazz, Electronic, Classical, Hip hop, Country, etc.
   - **Style/Tags**: CLAP — 37 tags across genre, production, energy, era, function, instrumentation
   - **Mood**: Music2Emo — valence (0-1), arousal (0-1), emotion tags
   - **Structure**: Spectral novelty change-point detection — finds internal section boundaries (intro, verse, chorus, bridge, build-up, climax)
4. **Per-Section Lightweight Analysis** — for cues >16s, each detected section gets energy, BPM, and key
5. **Output** — `project.sibyl.json` with `musical_profile` (universal) + `project_context` (project-specific)

---

## Track Replacement (Library Matching)

Matches film cues to the best-fitting time ranges inside a composer's music library.

- Library tracks are windowed at 15s, 30s, 60s, 120s with 50% overlap
- Each window is analyzed with the same detectors used in cue analysis
- Results stored in a local SQLite index (`backend/temp/library_index.sqlite`)
- Matching uses cosine similarity on normalized feature vectors (BPM, key, CLAP, moods, instruments, genres)
- Returns top N matches with time ranges, scores, and metadata

---

## API Endpoints

**Core Pipeline:**
- `POST /api/upload` — Upload audio/video file
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
- `POST /api/library/build-upload` — Build index from uploaded files
- `GET /api/library/status/{job_id}` — Build progress
- `GET /api/library/info` — Index metadata
- `GET /api/library/sources` — Per-source stats
- `POST /api/library/match` — Match cue to library
- `GET /api/library/audio/{track_id}` — Serve track audio

**Debug:**
- `POST /api/debug-logs` — Receive frontend logs
- `GET /api/debug-logs` — Retrieve debug logs
- `DELETE /api/cleanup/{file_id}` — Remove temp files

---

## Running Locally

**Backend** (port 8003 — hardcoded in Vite proxy, do not change):
```bash
cd sibyllai-web/backend && python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8003
```

**Frontend** (port 5174):
```bash
cd sibyllai-web/frontend && npm run dev
```

Port 8001 is reserved for kazen — never use it.

---

## ⚠️ Security Notice: PyTorch Version

This project pins `torch==2.2.2` due to compatibility requirements.

- **Known vulnerability:** Remote Code Execution (RCE) via `torch.load` with `weights_only=True` (CVE-2025-32434).
- **Mitigation:** Do NOT load untrusted model files. Only use model files from trusted sources.
- We will upgrade to a patched version as soon as compatibility allows.
