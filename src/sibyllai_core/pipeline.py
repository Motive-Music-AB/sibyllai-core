"High-level spotting pipeline."
from __future__ import annotations
import json, shutil, subprocess, tempfile, logging
from pathlib import Path

import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import essentia.standard as es

from .detectors import (
    detect_music_regions,
    music_probability,
    tag_chunk,
    global_moods,
)

def _extract_audio(src: str | Path) -> Path:
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
        # Log or include stderr in the exception message
        # For now, relying on the default exception trace to include some info
        # A more specific error message could be crafted here if needed:
        # raise RuntimeError(f"ffmpeg failed with error: {e.stderr.decode()}") from e
        raise  # Re-raise the original exception
    return wav


def _bpm_track(y, sr):
    if y.ndim > 1:
        y = librosa.to_mono(y.T)
    return es.RhythmExtractor2013(method="multifeature")(y)[0]


def _tc(sec: float, fps: int = 25) -> str:
    frames = int(round(sec * fps))
    h = frames // (3600 * fps)
    m = (frames % (3600 * fps)) // (60 * fps)
    s = (frames % (60 * fps)) // fps
    f = frames % fps
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


# ─── public API ────────────────────────────────────────────────────────────
def analyse(src: str | Path, out_dir: str | Path, thr: float = 0.5, fps=25):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    wav = _extract_audio(src)
    y, sr = sf.read(str(wav))

    regions = detect_music_regions(wav)
    if not regions:
        logging.warning("No music detected.")
        print("INFO: No music regions were detected in the input file. No output files will be generated.")
        return

    rows = []
    for start, end in regions:
        chunk = y[int(start*sr): int(end*sr)]
        rows.append({
            "start": start,
            "end":   end,
            "prob":  music_probability(chunk, sr),
            "tags":  tag_chunk(chunk, sr),
        })

    # markers CSV
    df = pd.DataFrame(
        [[_tc(r["start"], fps), _tc(r["end"], fps), _tc(r["end"]-r["start"], fps),
          f'{r["prob"]:.2f}',
          ", ".join(f"{k}:{v:.2f}" for k, v in r["tags"].items())]
         for r in rows],
        columns=["Start", "End", "Length", "MusicProb", "Tags"],
    )
    df.to_csv(out_dir / "markers.csv", index=False)
    logging.info("Markers saved → %s", out_dir / "markers.csv")

    # bpm & moods
    bpm   = _bpm_track(y, sr)
    moods = global_moods(str(wav), threshold=thr)
    with open(out_dir / "mood.json", "w") as f:
        json.dump({"bpm": bpm, **moods}, f, indent=2)
    logging.info("BPM & mood saved → %s", out_dir / "mood.json")

    shutil.rmtree(wav.parent, ignore_errors=True)
