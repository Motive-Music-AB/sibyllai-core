"High-level spotting pipeline."
from __future__ import annotations
import os
print("=== PIPELINE MODULE LOADED FROM:", os.path.abspath(__file__), "===")
import json, shutil, tempfile, logging
from pathlib import Path

import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import essentia.standard as es

from .detectors.yamnet_segmenter import segment_music_regions, extract_instruments
from .detectors.chord_detector import analyze_chords  # For key detection only
from .output import get_incremental_path
from .detectors import (
    music_probability,
    tag_chunk,
)
from .detectors.m2e_wrapper import global_moods

def _extract_audio(src: str | Path) -> Path:
    print("=== ENTERED _extract_audio ===")
    "Return temp mono 44.1 kHz wav path extracted with ffmpeg."
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError(
            "ffmpeg not found. Please install ffmpeg and ensure it is in your PATH."
        )
    tmp = Path(tempfile.mkdtemp(prefix="sibyllai_"))
    wav = tmp / "audio.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn",
             "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", str(wav)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        raise
    return wav


def _bpm_track(y, sr):
    print("=== ENTERED _bpm_track ===")
    if y.ndim > 1:
        y = librosa.to_mono(y.T)
    return es.RhythmExtractor2013(method="multifeature")(y)[0]


def _tc(sec: float, fps: int = 25) -> str:
    print("=== ENTERED _tc ===")
    frames = int(round(sec * fps))
    h = frames // (3600 * fps)
    m = (frames % (3600 * fps)) // (60 * fps)
    s = (frames % (60 * fps)) // fps
    f = frames % fps
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def get_incremental_run_folder(base_dir="outputs"):
    base = Path(base_dir)
    base.mkdir(exist_ok=True)
    i = 1
    while (base / f"run_{i:03d}").exists():
        i += 1
    run_dir = base / f"run_{i:03d}"
    run_dir.mkdir()
    return run_dir


# ─── public API ────────────────────────────────────────────────────────────
def analyse(src: str | Path, out_dir: str | Path, thr: float = 0.5, fps=25):
    src = Path(src)
    # Use incremental run folder
    run_dir = get_incremental_run_folder()
    print("=== ENTERED analyse ===")
    print("=== ENTERED ANALYSE FUNCTION (DEBUG MARKER) ===")
    print(f"[DEBUG] Input file received: {src}")
    if not src.exists():
        print(f"[ERROR] File does not exist: {src}")
        return
    print(f"[DEBUG] File exists: {src}")

    # 1. Extract audio from input (video or audio file)
    wav = _extract_audio(src)
    print(f"[DEBUG] Extracted WAV path: {wav}")
    y, sr = sf.read(str(wav))

    # 2. Segment music regions using YAMNet
    music_regions = segment_music_regions(wav)
    print(f"[DEBUG] Detected music regions: {music_regions}")
    if not music_regions:
        print("INFO: No music regions were detected in the input file. No output files will be generated.")
        return

    # 3. For each region, extract audio and run detectors
    rows = []
    min_duration = 3.0  # seconds
    for i, (start, end) in enumerate(music_regions):
        if (end - start) < min_duration:
            print(f"[WARNING] Skipping segment {i+1} (too short: {end - start:.2f}s)")
            continue
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk = y[start_sample:end_sample]

        print(f"[DEBUG] Segment {i+1}: {start:.2f}-{end:.2f}s")

        # Analyze the music segment directly (no Demucs processing)
        try:
            # Ensure audio is mono for analysis
            if chunk.ndim > 1:
                music_mono = librosa.to_mono(chunk.T)
            else:
                music_mono = chunk

            # Ensure minimum length for analysis (at least 1 second)
            min_samples = sr
            if len(music_mono) < min_samples:
                print(f"[WARNING] Segment {i+1} too short for analysis ({len(music_mono)} samples)")
                raise ValueError("Audio too short for analysis")
                
            # Initialize analysis variables
            bpm = "Unknown"
            moods_str = "Unknown"
            valence = 0.0
            arousal = 0.0
            music_prob = 0.0
            clap_categorized = {}
            clap_top_overall = []
            instruments = {}

            # BPM analysis
            try:
                bpm = _bpm_track(music_mono, sr)
            except Exception as e:
                print(f"[WARNING] BPM analysis failed for segment {i+1}: {e}")

            # Instrument detection via YAMNet
            try:
                instruments = extract_instruments(music_mono, sr=sr, top_n=5)
            except Exception as e:
                print(f"[WARNING] Instrument extraction failed for segment {i+1}: {e}")

            # Music probability (using AST detector) - more robust
            try:
                music_prob = music_probability(music_mono, sr)
            except Exception as e:
                print(f"[WARNING] Music probability analysis failed for segment {i+1}: {e}")

            # CLAP tags for music characteristics (categorized structure)
            try:
                clap_categorized = tag_chunk(music_mono, sr)

                # Extract top tags per category for summary
                top_tags = []
                for category, tags in clap_categorized.items():
                    sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
                    # Get top 2 per category
                    for tag, score in sorted_tags[:2]:
                        top_tags.append((tag, score, category))

                # Sort all top tags by score and take top 5 overall
                clap_top_overall = sorted(top_tags, key=lambda x: x[1], reverse=True)[:5]

            except Exception as e:
                print(f"[WARNING] CLAP analysis failed for segment {i+1}: {e}")

            # Key detection (keeping only key, not full chord analysis)
            try:
                chord_analysis = analyze_chords(audio_data=music_mono, sr=sr)
                detected_key = chord_analysis.get('key', 'Unknown')
            except Exception as e:
                print(f"[WARNING] Key detection failed for segment {i+1}: {e}")
                detected_key = "Unknown"

            # Mood analysis using Music2Emotion - most fragile, try last
            # Music2Emo requires a file path, so save temp segment
            try:
                segment_path = run_dir / f"segment_{i+1}_temp.wav"
                sf.write(segment_path, music_mono, sr)

                mood_result = global_moods(str(segment_path), threshold=thr)

                # Extract top mood predictions
                predicted_moods = mood_result.get('moods', [])
                top_moods = predicted_moods[:3] if predicted_moods else []
                moods_str = ", ".join(top_moods) if top_moods else "Unknown"

                # Extract valence and arousal
                valence = mood_result.get('valence', 0.0)
                arousal = mood_result.get('arousal', 0.0)

                # Clean up temp file
                if segment_path.exists():
                    segment_path.unlink()
            except Exception as e:
                print(f"[WARNING] Music2Emotion analysis failed for segment {i+1}: {e}")

            # Format CLAP tags for CSV output (temporary until JSON export)
            clap_summary = ", ".join([f"{tag} ({cat}): {score:.2f}"
                                      for tag, score, cat in clap_top_overall]) if clap_top_overall else "Unknown"

            # Format instruments for CSV output
            instruments_summary = ", ".join([f"{name}: {score:.2f}"
                                            for name, score in instruments.items()]) if instruments else "Unknown"

            rows.append({
                "Start": start,
                "End": end,
                "Length": end - start,
                "BPM": round(bpm, 1) if bpm != "Unknown" else "Unknown",
                "Key": detected_key,
                "Instruments": instruments_summary,
                "Instruments_Data": json.dumps(instruments) if instruments else "{}",
                "Moods": moods_str,
                "Valence": round(valence, 3),
                "Arousal": round(arousal, 3),
                "Music_Probability": round(music_prob, 3),
                "CLAP_Top_Tags": clap_summary,
                "CLAP_Full_Data": json.dumps(clap_categorized) if clap_categorized else "{}"
            })
        except Exception as e:
            print(f"[WARNING] Music analysis failed for segment {i+1}: {e}")
            # Fallback to basic info if analysis fails
            rows.append({
                "Start": start,
                "End": end,
                "Length": end - start,
                "BPM": "Unknown",
                "Key": "Unknown",
                "Instruments": "Unknown",
                "Instruments_Data": "{}",
                "Moods": "Unknown",
                "Valence": 0.0,
                "Arousal": 0.0,
                "Music_Probability": 0.0,
                "CLAP_Top_Tags": "Unknown",
                "CLAP_Full_Data": "{}"
            })

    # 4. Save per-segment results to CSV in run_dir
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(run_dir / "music_segments.csv", index=False)
        print(f"[INFO] Output written to {run_dir}")
    else:
        print("[INFO] No valid segments to output.")

    # Clean up extracted audio
    if wav.exists():
        wav.unlink()
