"High-level spotting pipeline."
from __future__ import annotations
import os
print("=== PIPELINE MODULE LOADED FROM:", os.path.abspath(__file__), "===")
import json, shutil, subprocess, tempfile, logging
from pathlib import Path

import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import essentia.standard as es
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio

from .detectors.yamnet_segmenter import segment_music_regions
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
        # Ensure chunk is stereo and 44.1kHz for Demucs
        target_sr = 44100
        if sr != target_sr:
            chunk = librosa.resample(chunk, orig_sr=sr, target_sr=target_sr)
        if chunk.ndim == 1:
            chunk = np.stack([chunk, chunk], axis=0)
        elif chunk.shape[0] == 1:
            chunk = np.vstack([chunk, chunk])
        chunk = chunk.T
        segment_wav_path = run_dir / f"segment_{i+1}.wav"
        sf.write(segment_wav_path, chunk, target_sr)
        print(f"[DEBUG] Segment {i+1}: {start:.2f}-{end:.2f}s, temp WAV: {segment_wav_path}")
        # Run Demucs in 4-stem mode
        demucs_out_dir = run_dir / f"segment_{i+1}_demucs"
        demucs_out_dir.mkdir(exist_ok=True)
        try:
            subprocess.run([
                "demucs", "-o", str(demucs_out_dir), str(segment_wav_path)
            ], check=True)
            demucs_stem_dir = demucs_out_dir / "htdemucs" / segment_wav_path.stem
            required_stems = ["drums.wav", "bass.wav", "other.wav"]
            missing = [stem for stem in required_stems if not (demucs_stem_dir / stem).exists()]
            if missing:
                print(f"[WARNING] Demucs missing stems for segment {i+1}: {missing}")
                continue
            # Load and combine stems
            try:
                drums, sr2 = sf.read(demucs_stem_dir / "drums.wav")
                bass, _ = sf.read(demucs_stem_dir / "bass.wav")
                other, _ = sf.read(demucs_stem_dir / "other.wav")
                min_len = min(len(drums), len(bass), len(other))
                drums = drums[:min_len]
                bass = bass[:min_len]
                other = other[:min_len]
                music = drums + bass + other
                max_val = np.max(np.abs(music))
                if max_val > 1.0:
                    music = music / max_val
                music_path = run_dir / f"segment_{i+1}_music_minus_vocals.wav"
                sf.write(music_path, music, sr2)
                print(f"[INFO] Combined music-minus-vocals stem saved as: {music_path}")
            except Exception as e:
                print(f"[ERROR] Could not combine stems for segment {i+1}: {e}")
                continue
            # Clean up Demucs output folder
            shutil.rmtree(demucs_out_dir, ignore_errors=True)
            # Clean up temp segment wav
            if segment_wav_path.exists():
                segment_wav_path.unlink()
            # Analyze the music segment for additional features
            try:
                # Ensure audio is mono and has sufficient length
                if music.ndim > 1:
                    music_mono = librosa.to_mono(music.T)
                else:
                    music_mono = music
                
                # Ensure minimum length for analysis (at least 1 second)
                min_samples = sr2
                if len(music_mono) < min_samples:
                    print(f"[WARNING] Segment {i+1} too short for analysis ({len(music_mono)} samples)")
                    raise ValueError("Audio too short for analysis")
                
                # Initialize analysis variables
                bpm = "Unknown"
                moods_str = "Unknown"
                valence = 0.0
                arousal = 0.0
                music_prob = 0.0
                clap_str = "Unknown"
                
                # BPM analysis
                try:
                    bpm = _bpm_track(music_mono, sr2)
                except Exception as e:
                    print(f"[WARNING] BPM analysis failed for segment {i+1}: {e}")
                
                # Music probability (using AST detector) - more robust
                try:
                    music_prob = music_probability(music_mono, sr2)
                except Exception as e:
                    print(f"[WARNING] Music probability analysis failed for segment {i+1}: {e}")
                
                # CLAP tags for music characteristics
                try:
                    clap_tags = tag_chunk(music_mono, sr2)
                    # Filter out speech-related tags since we already remove vocals
                    filtered_clap = {k: v for k, v in clap_tags.items() 
                                   if 'speech' not in k.lower() and 'voice' not in k.lower() and 'contains speech' not in k.lower()}
                    top_clap_tags = sorted(filtered_clap.items(), key=lambda x: x[1], reverse=True)[:5]
                    clap_str = ", ".join([f"{tag}:{prob:.2f}" for tag, prob in top_clap_tags])
                except Exception as e:
                    print(f"[WARNING] CLAP analysis failed for segment {i+1}: {e}")
                
                # Mood analysis using Music2Emotion - most fragile, try last
                try:
                    mood_result = global_moods(str(music_path), threshold=thr)
                    
                    # Extract top mood predictions
                    predicted_moods = mood_result.get('moods', [])
                    top_moods = predicted_moods[:3] if predicted_moods else []
                    moods_str = ", ".join(top_moods) if top_moods else "Unknown"
                    
                    # Extract valence and arousal
                    valence = mood_result.get('valence', 0.0)
                    arousal = mood_result.get('arousal', 0.0)
                except Exception as e:
                    print(f"[WARNING] Music2Emotion analysis failed for segment {i+1}: {e}")
                
                rows.append({
                    "Start": start,
                    "End": end,
                    "Length": end - start,
                    "File": f"segment_{i+1}_music_minus_vocals.wav",
                    "BPM": round(bpm, 1) if bpm else "Unknown",
                    "Moods": moods_str,
                    "Valence": round(valence, 3),
                    "Arousal": round(arousal, 3),
                    "Music_Probability": round(music_prob, 3),
                    "CLAP_Tags": clap_str
                })
            except Exception as e:
                print(f"[WARNING] Music analysis failed for segment {i+1}: {e}")
                # Fallback to basic info if analysis fails
                rows.append({
                    "Start": start,
                    "End": end,
                    "Length": end - start,
                    "File": f"segment_{i+1}_music_minus_vocals.wav",
                    "BPM": "Unknown",
                    "Moods": "Unknown",
                    "Valence": 0.0,
                    "Arousal": 0.0,
                    "Music_Probability": 0.0,
                    "CLAP_Tags": "Unknown"
                })
        except Exception as e:
            print(f"[WARNING] Demucs failed for segment {i+1}: {e}")
            continue

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
