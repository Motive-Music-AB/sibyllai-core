# SibyllAI MVP Roadmap

## Goal

A simple tool for film composers to analyze temp MX tracks and export cue markers into Logic Pro.

## MVP User Flow

1. **Drag MX audio file** into the interface
2. **Analyze** — detect music cues and extract musical characteristics
3. **Review** timecode and info for each detected cue
4. **Select** which data fields to include in the export
5. **Export CSV** formatted for Logic Pro marker import
6. **Import** into Logic Pro on macOS as markers

---

## Backend — Complete ✅

All analysis runs through `sibyllai-web/backend/core/analysis.py`. Tested on 89-minute feature film MX (73 cues detected, ~7 min processing time).

| Feature | Status | Notes |
|---|---|---|
| Audio ingestion | ✅ Done | ffmpeg extracts audio from video or audio files |
| Music cue detection | ✅ Done | YAMNet `segment_music_regions()` |
| Timecode extraction | ✅ Done | HH:MM:SS:FF conversion |
| BPM detection | ✅ Done | Essentia `RhythmExtractor2013` |
| Key detection | ✅ Done | `chord_detector.analyze_chords()` |
| Instrument detection | ✅ Done | YAMNet `extract_instruments()`, top 15 |
| Genre detection | ✅ Done | YAMNet `extract_genres()`, top 10 |
| Style / energy / era / function tags | ✅ Done | CLAP, 37 categorized tags |
| Mood / valence / arousal | ✅ Done | Music2Emo |
| `.sibyl.json` output | ✅ Done | `musical_profile` + `project_context` per cue |
| CSV output | ✅ Done | Backwards-compatible `music_segments.csv` |

## Web UI — In Progress 🔄

Built with FastAPI (port 8003) + Vite/React (port 5174). Core upload/analyze/display flow is functional. Lives in `sibyllai-web/`.

| Feature | Status | Notes |
|---|---|---|
| File upload (drag & drop) | ✅ Done | |
| Waveform display + zoom | ✅ Done | WaveSurfer.js, complex zoom logic — do not simplify |
| Segment detection preview | ✅ Done | Select/deselect cues before analysis |
| Cue cards with analysis results | ✅ Done | Genre, style, instruments, BPM, key, mood |
| Playback with cue navigation | ✅ Done | Spacebar from cue start; click to scrub |
| Selectable export fields | ⬜ To do | Checkboxes for which columns to include |
| Logic Pro–compatible CSV export | ⬜ To do | Format TBD — needs confirmed LP marker spec |
| macOS packaging | ⬜ Post-MVP | Local server or Electron wrapper |

---

## Open Questions

- What is Logic Pro's exact marker CSV format? (column names, timecode format, delimiter)
- Which analysis fields are most useful to composers for the export? (BPM and key are likely essential; mood/valence optional)
