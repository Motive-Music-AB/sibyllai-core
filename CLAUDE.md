# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SibyllAI Core is a music analysis tool for film composers, designed to analyze temp music (MX) tracks and extract comprehensive musical characteristics. The primary use case is **pre-scoring analysis**: composers receive temp MX from editors and use SibyllAI to understand what musical elements (genre, instrumentation, mood, energy) the director responded to, informing their original compositions.

**Current Status:** MVP in active development on `mvp-web-ui` branch. Core detection works; web UI and export features in progress.

**Future Vision:** Reverse lookup capability - match detected musical profiles against composer's personal music library to suggest similar unused tracks/demos.

## Tech Stack

- Python 3.11+
- PyTorch 2.2.2 (pinned for compatibility; has known RCE vulnerability via torch.load - only use trusted model files)
- TensorFlow & Keras (for YAMNet)
- YAMNet (music cue segmentation + instrument detection from 521 audio event classes)
- CLAP (LAION-CLAP for genre, production style, energy, era classification - 25-30 categorized tags)
- Music2Emo (mood/emotion analysis: valence, arousal, mood tags)
- Essentia (rhythm/BPM extraction)
- Librosa (audio processing, resampling)

## Common Commands

### Installation
```bash
pip install -e .[dev]
```

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
   - Uses YAMNet to identify where music starts and stops (cue boundaries)
   - Returns list of (start, end) time ranges in seconds
   - Filters segments shorter than 3 seconds (minimum cue length)

4. **Feature Analysis** (per detected cue):
   - **BPM Analysis**: Uses Essentia RhythmExtractor2013
   - **Key Detection**: Analyzes harmonic content
   - **Instrument Detection**: Extracts top 3-5 instruments from YAMNet's 521 audio event classes
   - **Genre/Style Tags**: CLAP analysis across 25-30 categorized tags (genre, production, energy, era, function)
   - **Mood Analysis**: Music2Emo model for valence, arousal, and mood tags

5. **Output Generation**:
   - Creates incremental run folders: `outputs/run_001/`, `outputs/run_002/`, etc.
   - Saves project file: `project.sibyl.json` with comprehensive cue data
   - Structured format separates universal musical attributes from project-specific context

### Key Components

**Detectors Module** (`src/sibyllai_core/detectors/`):
- Each detector provides standalone analysis functionality
- Exports are consolidated in `__init__.py`: `music_probability`, `tag_chunk`, `global_moods`
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
     - Music2Emo moods (all detected with confidence)
     - CLAP tags organized by category: genre, production, energy, era, function
   - **`curated`**: User-approved subset of detected attributes
     - Top 3 instruments, top 2-3 per CLAP category
     - User can promote/demote/remove via UI
   - **Musical fundamentals**: BPM, key, valence (0-1), arousal (0-1)

2. **`project_context`** - Project-specific metadata (NOT used for matching):
   - Custom tags (e.g., "Hero's theme", "Director's favorite")
   - Composer notes
   - Visual color coding
   - Status (draft/review/locked)
   - Director feedback

This separation allows future reverse-lookup: "Find tracks in my library matching this cue's musical_profile"

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
