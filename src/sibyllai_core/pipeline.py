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
            ["ffmpeg", "-y", "-i", str(src), "-vn",
             "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", str(wav)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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


# ─── public API ────────────────────────────────────────────────────────────
def analyse(src: str | Path, out_dir: str | Path, thr: float = 0.5, fps=25):
    src = Path(src)
    out_dir = Path(out_dir)
    print("=== ENTERED analyse ===")
    print("=== ENTERED ANALYSE FUNCTION (DEBUG MARKER) ===")
    print(f"[DEBUG] Input file received: {src}")
    if not src.exists():
        print(f"[ERROR] File does not exist: {src}")
        return
    print(f"[DEBUG] File exists: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract audio from input (video or audio file)
    wav = _extract_audio(src)
    print(f"[DEBUG] Extracted WAV path: {wav}")
    y, sr = sf.read(str(wav))

    # 2. Segment music regions using YAMNet
    music_regions = segment_music_regions(wav)
    print(f"[DEBUG] Detected music regions: {music_regions}")
    if not music_regions:
        logging.warning("No music detected.")
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
        # Save as (n_samples, 2) for sf.write
        chunk = chunk.T
        segment_wav_path = out_dir / f"segment_{i+1}.wav"
        sf.write(segment_wav_path, chunk, target_sr)
        print(f"[DEBUG] Segment {i+1}: {start:.2f}-{end:.2f}s, temp WAV: {segment_wav_path}")
        # Run Demucs on the segment WAV
        demucs_out_dir = out_dir / f"segment_{i+1}_demucs"
        demucs_out_dir.mkdir(exist_ok=True)
        try:
            # Use Demucs CLI for robust file output
            subprocess.run([
                "demucs", "--two-stems", "other", "-o", str(demucs_out_dir), str(segment_wav_path)
            ], check=True)
            # Demucs outputs to demucs_out_dir/demucs/segment_{i+1}/other.wav
            demucs_stem_dir = demucs_out_dir / "htdemucs" / segment_wav_path.stem
            other_stem_path = demucs_stem_dir / "other.wav"
            if not other_stem_path.exists():
                # Try 'accompaniment.wav' as fallback
                other_stem_path = demucs_stem_dir / "accompaniment.wav"
            if not other_stem_path.exists():
                print(f"[WARNING] Demucs did not produce an 'other' or 'accompaniment' stem for segment {i+1}.")
                continue
            # Load the separated music stem
            stem_chunk, stem_sr = sf.read(str(other_stem_path))
            if stem_sr != sr:
                stem_chunk = librosa.resample(stem_chunk, orig_sr=stem_sr, target_sr=sr)
            # Use the separated music stem for all detectors
            prob = music_probability(stem_chunk, sr)
            tags = tag_chunk(stem_chunk, sr)
            bpm = _bpm_track(stem_chunk, sr)
            rows.append({
                "start": start,
                "end": end,
                "prob": prob,
                "tags": tags,
                "bpm": bpm,
            })
            # Use the separated music stem for mood detection
            try:
                mood_result = global_moods(str(other_stem_path))
            except Exception as e:
                print(f"[WARNING] music2emo failed for segment {i+1}: {e}")
                print(f"[WARNING] Segment WAV kept for inspection: {other_stem_path}")
                continue
            json_path = out_dir / f"mood_segment_{i+1}.json"
            with open(json_path, "w") as f:
                json.dump(mood_result, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Demucs failed for segment {i+1}: {e}")
            print(f"[WARNING] Segment WAV kept for inspection: {segment_wav_path}")
            continue

    # 4. Save per-segment results to CSV
    df = pd.DataFrame(
        [[_tc(r["start"], fps), _tc(r["end"], fps), _tc(r["end"]-r["start"], fps),
          f'{r["prob"]:.2f}', f'{r["bpm"]:.2f}',
          ", ".join(f"{k}:{v:.2f}" for k, v in r["tags"].items() if "speech" not in k.lower())]
         for r in rows],
        columns=["Start", "End", "Length", "MusicProb", "BPM", "Tags"],
    )
    df.to_csv(get_incremental_path(out_dir, "music_segments.csv"), index=False)
    logging.info("Music segments saved → %s", out_dir / "music_segments.csv")

    # 5. Copy the extracted WAV to outputs for inspection
    shutil.copy(str(wav), str(Path(out_dir) / "audio_debug.wav"))
    shutil.rmtree(wav.parent, ignore_errors=True)
