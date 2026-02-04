"""Full analysis module - runs expensive detectors on confirmed segments (Phase 2)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Optional

import soundfile as sf
import librosa
import essentia.standard as es

from sibyllai_core.detectors.yamnet_segmenter import extract_instruments, extract_genres
from sibyllai_core.detectors.chord_detector import analyze_chords
from sibyllai_core.detectors import music_probability, tag_chunk
from sibyllai_core.detectors.m2e_wrapper import global_moods
from sibyllai_core.sibyl_format import create_cue, create_project, save_project


def _bpm_track(y, sr):
    """
    Extract BPM using Essentia's RhythmExtractor2013.

    RhythmExtractor2013 requires exactly 44100 Hz sample rate.
    If audio is at a different rate, resample it to prevent tempo errors.
    """
    if y.ndim > 1:
        y = librosa.to_mono(y.T)

    # Resample to 44100 Hz if needed (RhythmExtractor2013 requirement)
    if sr != 44100:
        y = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sr = 44100

    return es.RhythmExtractor2013(method="multifeature")(y)[0]


def analyze_segments(
    audio_path: str | Path,
    segments: list[tuple[float, float]],
    output_dir: str | Path,
    fps: int = 25,
    thr: float = 0.5,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> dict:
    """
    Run full analysis on confirmed segments.

    Args:
        audio_path: Path to original audio file
        segments: List of (start, end) tuples in seconds
        output_dir: Directory to save results
        fps: Frames per second for timecode conversion
        thr: Mood probability threshold
        progress_callback: Optional callback(current, total, status_message)

    Returns:
        Complete .sibyl.json project dictionary
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full audio
    y, sr = sf.read(str(audio_path))

    cues = []
    rows = []  # For backwards-compatible CSV

    total_segments = len(segments)

    for i, (start, end) in enumerate(segments, 1):
        if progress_callback:
            progress_callback(i, total_segments, f"Analyzing segment {i}/{total_segments}")

        # Extract segment audio
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk = y[start_sample:end_sample]

        # Ensure mono
        if chunk.ndim > 1:
            music_mono = librosa.to_mono(chunk.T)
        else:
            music_mono = chunk

        # Initialize analysis results
        bpm = "Unknown"
        moods_str = "Unknown"
        valence = 0.0
        arousal = 0.0
        music_prob = 0.0
        clap_categorized = {}
        clap_top_overall = []
        instruments = {}
        genres = {}
        detected_key = "Unknown"

        try:
            # BPM analysis
            try:
                if progress_callback:
                    progress_callback(i, total_segments, f"Segment {i}: Detecting BPM")
                bpm = _bpm_track(music_mono, sr)
            except Exception as e:
                print(f"[WARNING] BPM analysis failed for segment {i}: {e}")

            # Instrument detection via YAMNet
            try:
                if progress_callback:
                    progress_callback(i, total_segments, f"Segment {i}: Detecting instruments")
                instruments = extract_instruments(music_mono, sr=sr, top_n=15)
                print(f"[DEBUG] Detected instruments for segment {i}: {instruments}")
            except Exception as e:
                print(f"[WARNING] Instrument extraction failed for segment {i}: {e}")

            # Genre detection via YAMNet
            try:
                if progress_callback:
                    progress_callback(i, total_segments, f"Segment {i}: Detecting genres")
                genres = extract_genres(music_mono, sr=sr, top_n=10)
                print(f"[DEBUG] Detected genres for segment {i}: {genres}")
            except Exception as e:
                print(f"[WARNING] Genre extraction failed for segment {i}: {e}")

            # Music probability
            try:
                if progress_callback:
                    progress_callback(i, total_segments, f"Segment {i}: Checking music probability")
                music_prob = music_probability(music_mono, sr)
            except Exception as e:
                print(f"[WARNING] Music probability analysis failed for segment {i}: {e}")

            # CLAP tags
            try:
                if progress_callback:
                    progress_callback(i, total_segments, f"Segment {i}: Analyzing style")
                clap_categorized = tag_chunk(music_mono, sr)

                # Extract top tags per category
                top_tags = []
                for category, tags in clap_categorized.items():
                    sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
                    for tag, score in sorted_tags[:2]:
                        top_tags.append((tag, score, category))

                clap_top_overall = sorted(top_tags, key=lambda x: x[1], reverse=True)[:5]

            except Exception as e:
                print(f"[WARNING] CLAP analysis failed for segment {i}: {e}")

            # Key detection
            try:
                if progress_callback:
                    progress_callback(i, total_segments, f"Segment {i}: Detecting key")
                chord_analysis = analyze_chords(audio_data=music_mono, sr=sr)
                detected_key = chord_analysis.get('key', 'Unknown')
            except Exception as e:
                print(f"[WARNING] Key detection failed for segment {i}: {e}")

            # Mood analysis (most fragile)
            try:
                if progress_callback:
                    progress_callback(i, total_segments, f"Segment {i}: Analyzing mood/emotion")

                segment_path = output_dir / f"segment_{i}_temp.wav"
                sf.write(segment_path, music_mono, sr)

                mood_result = global_moods(str(segment_path), threshold=thr)

                predicted_moods = mood_result.get('moods', [])
                top_moods = predicted_moods[:3] if predicted_moods else []
                moods_str = ", ".join(top_moods) if top_moods else "Unknown"

                valence = mood_result.get('valence', 0.0)
                arousal = mood_result.get('arousal', 0.0)

                if segment_path.exists():
                    segment_path.unlink()

            except Exception as e:
                print(f"[WARNING] Music2Emotion analysis failed for segment {i}: {e}")

            # Create cue structure
            cue_id = f"cue_{i:03d}"
            moods_list = moods_str.split(", ") if moods_str != "Unknown" else []

            cue = create_cue(
                cue_id=cue_id,
                start=start,
                end=end,
                fps=fps,
                bpm=bpm,
                key=detected_key,
                instruments=instruments,
                genres=genres,
                moods=moods_list,
                valence=valence,
                arousal=arousal,
                clap_categorized=clap_categorized,
            )
            cues.append(cue)

            # CSV row for backwards compatibility
            clap_summary = ", ".join([f"{tag} ({cat}): {score:.2f}"
                                      for tag, score, cat in clap_top_overall]) if clap_top_overall else "Unknown"

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
            print(f"[ERROR] Analysis failed for segment {i}: {e}")
            # Still create a cue with basic info
            cue = create_cue(
                cue_id=f"cue_{i:03d}",
                start=start,
                end=end,
                fps=fps,
                bpm="Unknown",
                key="Unknown",
                instruments={},
                genres={},
                moods=[],
                valence=0.0,
                arousal=0.0,
                clap_categorized={},
            )
            cues.append(cue)

    # Create and save project
    project = create_project(
        mx_file_path=audio_path,
        cues=cues,
        project_name=audio_path.stem,
        fps=fps,
    )

    project_file = output_dir / "project.sibyl.json"
    save_project(project, project_file)

    # Also save CSV
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "music_segments.csv", index=False)

    if progress_callback:
        progress_callback(total_segments, total_segments, "Complete!")

    return project
