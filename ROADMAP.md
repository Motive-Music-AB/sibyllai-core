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

## What Already Exists (Backend)

| Feature | Status | Notes |
|---|---|---|
| Audio ingestion | ✅ Done | ffmpeg extracts audio from video or audio files |
| Music cue detection | ✅ Done | YAMNet segments music vs silence/dialogue |
| Timecode extraction | ✅ Done | `_tc()` converts seconds → HH:MM:SS:FF |
| BPM detection | ✅ Done | Essentia `RhythmExtractor2013` |
| Key detection | ✅ Done | Chord detector (key only) |
| Genre / mood tags | ✅ Done | CLAP (37 categorized tags) |
| Mood / valence / arousal | ✅ Done | Music2Emo |
| Instrument detection | ✅ Done | YAMNet `extract_instruments()` |
| CSV output | ✅ Done | `music_segments.csv` per run |
| .sibyl.json output | ✅ Done | Structured project format |

## What Needs to Be Built

### 1. UI — Drag & Drop Interface
- Drag-and-drop audio file input
- Gradio is already installed and is the simplest path
- Alternatively: simple web UI (FastAPI + minimal HTML)

### 2. Results View
- Display detected cues in a table: cue #, start TC, end TC, duration, BPM, key, top tags
- Show confidence scores where useful

### 3. Selectable Export Fields
- Checkboxes or toggles for which columns to include in CSV
- Minimum useful set for Logic Pro: cue name, start timecode

### 4. Logic Pro–Compatible CSV Export
- Logic Pro marker import format needs to be confirmed
- Likely requires: Position (timecode or SMPTE), Name/Label columns
- May need specific delimiter or header format

### 5. macOS Packaging (post-MVP)
- Wrap as a standalone macOS app or local web server
- Models are large — consider download-on-first-run approach

---

## Open Questions

- What is Logic Pro's exact marker CSV format? (column names, timecode format, delimiter)
- Should cue detection use YAMNet (full mix with dialogue) or amplitude detection (clean MX only)?
- Which analysis fields are most useful to composers? (BPM and key are likely essential; mood/valence optional)
- Should Music2Emo and AST remain in MVP, or simplify to CLAP + Essentia only for speed?
