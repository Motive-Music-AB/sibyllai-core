# CLAUDE.md - sibyllai-web

This file provides guidance to Claude Code when working with the web UI.

## Quick Start

### Running the App (Development)

**Backend** (port 8002 to avoid conflict with kazen on 8001):
```bash
cd /Volumes/New\ 4\ TB/Dropbox/1_A_WORK/1_Projects/1_Motive-Music-AB/Motive/Github/sibyllai-core
source .venv/bin/activate
cd sibyllai-web/backend/api
PORT=8002 uvicorn main:app --reload --port 8002
```

**Frontend** (port 5174):
```bash
cd sibyllai-web/frontend
npm run dev
```

Access at: http://localhost:5174

### Port Configuration

The frontend proxy is configured in `frontend/vite.config.ts`. If you need to change ports:
1. Update `vite.config.ts` proxy target
2. Update uvicorn port to match

## Testing Without UI

You can test the analysis pipeline directly via curl or Python script.

### Option 1: curl API Test

```bash
# 1. Upload a test audio file
curl -X POST http://localhost:8002/api/upload \
  -F "file=@/path/to/test-audio.wav" \
  | jq

# Response: {"file_id": "abc123", "filename": "test-audio.wav", ...}

# 2. Get segment preview
curl -X POST http://localhost:8002/api/segment-preview \
  -H "Content-Type: application/json" \
  -d '{"file_id": "abc123", "music_thresh": 0.2, "min_gap": 1.0, "min_cue_length": 3.0}' \
  | jq

# 3. Run analysis on segments
curl -X POST http://localhost:8002/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"file_id": "abc123", "segments": [[0, 30], [45, 90]], "fps": 25}' \
  | jq

# 4. Check output JSON
cat backend/temp/<session_id>/project.sibyl.json | jq '.cues[0].musical_profile.curated'
```

### Option 2: Python Test Script

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

Run:
```bash
cd /Volumes/New\ 4\ TB/Dropbox/1_A_WORK/1_Projects/1_Motive-Music-AB/Motive/Github/sibyllai-core
source .venv/bin/activate
python test_analysis.py
```

### Checking Backend Logs

Monitor genre/instrument detection during analysis:
```bash
tail -f /private/tmp/claude-501/-Volumes-New-4-TB-Obsidian-AgentWorkspace/tasks/<task_id>.output | grep -E "(DEBUG|WARNING|genres|instruments)"
```

Look for:
```
[DEBUG] Detected instruments for segment 1: {'Percussion': 0.02, 'Singing': 0.01, ...}
[DEBUG] Detected genres for segment 1: {'Pop music': 0.15, 'Rock music': 0.08, ...}
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

- `detected.instruments_yamnet` - All YAMNet instrument scores
- `detected.genres_yamnet` - All YAMNet genre scores
- `detected.clap_style` - CLAP film-scoring vocabulary
- `detected.clap_instrumentation` - CLAP ensemble detection
- `detected.clap_production`, `clap_energy`, `clap_era`, `clap_function`

## Recent Changes (Feb 2026)

### YAMNet Genre Detection
- Added `extract_genres()` function to yamnet_segmenter.py
- Detects: Pop, Rock, Jazz, Electronic, Classical, Hip hop, etc.
- Separate from CLAP's film-scoring "style" tags

### Vocal Detection
- YAMNet instruments now include: Singing, Humming, Speech, Choir, Rapping, Chant, Whistling, Beatbox

### UI Updates
- CueCard now shows both Genre (YAMNet) and Style (CLAP)
- Instruments display from YAMNet (not CLAP instrumentation)
