"""
Music structure detection via beat-synchronous self-similarity analysis.

Identifies structural sections (intro, verse, chorus, build-up, climax, etc.)
by computing a cosine similarity matrix over beat-synchronous MFCC + chroma
features, then applying a checkerboard kernel to find novelty peaks where
musical character actually changes.

Uses librosa + scipy + sklearn (already installed).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Section(NamedTuple):
    """A detected section with start/end times and duration (in seconds)."""
    start: float
    end: float
    duration: float


def detect_sections(
    audio: np.ndarray,
    sr: int,
    min_section_length: float = 8.0,
    hop_length: int = 512,
) -> list[Section]:
    """
    Detect structural sections using beat-synchronous self-similarity novelty.

    1. Extract MFCC (timbre) + chroma (harmony) features per frame.
    2. Beat-synchronize features — averages per beat for a cleaner signal.
    3. Build a cosine similarity matrix (dense, not sparse recurrence).
    4. Apply a checkerboard kernel to find novelty peaks — points where
       the musical character *changes* on either side.
    5. Pick peaks in the novelty curve as structural boundaries.
    6. Merge short sections.

    This detects actual structural changes (timbre + harmony shifts)
    rather than just energy transients.

    Args:
        audio: Mono audio signal as numpy array.
        sr: Sample rate.
        min_section_length: Minimum section duration in seconds.
        hop_length: Hop length in samples.

    Returns:
        List of Section namedtuples sorted by start time.
    """
    import librosa
    from scipy.signal import find_peaks
    from sklearn.metrics.pairwise import cosine_similarity

    total_duration = len(audio) / sr
    if total_duration < min_section_length * 2:
        return [Section(start=0.0, end=round(total_duration, 3), duration=round(total_duration, 3))]

    # --- 1. Extract features ---
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=hop_length)
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)

    # --- 2. Beat-synchronize for cleaner representation ---
    # Averaging features per beat removes transient noise and aligns
    # the analysis to the musical pulse.
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length)
    if len(beats) < 4:
        return [Section(start=0.0, end=round(total_duration, 3), duration=round(total_duration, 3))]

    mfcc_sync = librosa.util.sync(mfcc, beats, aggregate=np.median)
    chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)

    features = np.vstack([
        librosa.util.normalize(mfcc_sync, axis=1),
        librosa.util.normalize(chroma_sync, axis=1),
    ])

    n_beats = features.shape[1]
    if n_beats < 8:
        return [Section(start=0.0, end=round(total_duration, 3), duration=round(total_duration, 3))]

    # --- 3. Cosine similarity matrix ---
    # Dense matrix showing how similar every beat is to every other beat.
    # Unlike the sparse recurrence matrix, this preserves local structure
    # needed for the checkerboard kernel.
    sim = cosine_similarity(features.T)

    # --- 4. Checkerboard novelty on similarity matrix ---
    # At each beat position, compare self-similarity BEFORE vs AFTER.
    # High novelty = the musical character changes here.
    half_k = max(4, n_beats // 8)
    novelty = np.zeros(n_beats)

    for i in range(half_k, n_beats - half_k):
        before = sim[i - half_k:i, i - half_k:i]
        after = sim[i:i + half_k, i:i + half_k]
        cross = sim[i - half_k:i, i:i + half_k]
        self_sim = (before.mean() + after.mean()) / 2
        cross_sim = cross.mean()
        novelty[i] = max(0, self_sim - cross_sim)

    if novelty.max() == 0:
        return [Section(start=0.0, end=round(total_duration, 3), duration=round(total_duration, 3))]

    novelty = novelty / novelty.max()

    # --- 5. Peak picking ---
    # Convert min_section_length to beat frames
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    avg_beat_dur = np.median(np.diff(beat_times)) if len(beat_times) > 1 else 0.5
    min_beat_frames = max(1, int(min_section_length / avg_beat_dur))

    # Threshold: mean + 1.5 std catches only significant structural changes
    threshold = float(np.mean(novelty) + 1.5 * np.std(novelty))
    threshold = max(threshold, 0.3)
    threshold = min(threshold, 0.85)

    peaks, _ = find_peaks(
        novelty,
        height=threshold,
        distance=min_beat_frames,
        prominence=0.15,
    )

    if len(peaks) == 0:
        return [Section(start=0.0, end=round(total_duration, 3), duration=round(total_duration, 3))]

    # --- 6. Convert beat-frame peaks to time boundaries ---
    peak_times = []
    for p in peaks:
        if p < len(beat_times):
            peak_times.append(float(beat_times[p]))
        else:
            peak_times.append(float(librosa.frames_to_time(beats[-1], sr=sr, hop_length=hop_length)))

    boundaries = [0.0] + peak_times + [total_duration]
    sections: list[Section] = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]
        sections.append(Section(start=round(s, 3), end=round(e, 3), duration=round(e - s, 3)))

    # --- 7. Merge short sections ---
    sections = _merge_short_sections(sections, min_section_length)

    return sections


def _merge_short_sections(
    sections: list[Section], min_length: float
) -> list[Section]:
    """Merge sections shorter than min_length into the shorter neighbour."""
    if len(sections) <= 1:
        return sections

    merged: list[Section] = list(sections)
    changed = True
    while changed:
        changed = False
        new_merged: list[Section] = []
        i = 0
        while i < len(merged):
            sec = merged[i]
            if sec.duration < min_length and len(merged) > 1:
                changed = True
                if i == 0 and i + 1 < len(merged):
                    nxt = merged[i + 1]
                    new_merged.append(Section(
                        start=sec.start,
                        end=nxt.end,
                        duration=round(nxt.end - sec.start, 3),
                    ))
                    i += 2
                elif i == len(merged) - 1 and len(new_merged) > 0:
                    prev = new_merged.pop()
                    new_merged.append(Section(
                        start=prev.start,
                        end=sec.end,
                        duration=round(sec.end - prev.start, 3),
                    ))
                    i += 1
                elif i > 0 and i + 1 < len(merged):
                    prev = new_merged[-1] if new_merged else None
                    nxt = merged[i + 1]
                    if prev and prev.duration <= nxt.duration:
                        p = new_merged.pop()
                        new_merged.append(Section(
                            start=p.start,
                            end=sec.end,
                            duration=round(sec.end - p.start, 3),
                        ))
                        i += 1
                    else:
                        new_merged.append(Section(
                            start=sec.start,
                            end=nxt.end,
                            duration=round(nxt.end - sec.start, 3),
                        ))
                        i += 2
                else:
                    new_merged.append(sec)
                    i += 1
            else:
                new_merged.append(sec)
                i += 1
        merged = new_merged

    return merged


def detect_sections_with_fallback(
    audio: np.ndarray,
    sr: int,
    min_section_length: float = 8.0,
    sensitivity: float = 1.0,
) -> list[Section]:
    """
    Detect sections with fallback for edge cases.

    Primary: beat-synchronous cosine similarity novelty.
    Fallback: agglomerative clustering on stacked MFCC + chroma.

    Args:
        audio: Mono audio signal.
        sr: Sample rate.
        min_section_length: Minimum section duration in seconds.
        sensitivity: Unused (kept for API compatibility).

    Returns:
        List of Section namedtuples.
    """
    total_duration = len(audio) / sr

    try:
        sections = detect_sections(
            audio, sr,
            min_section_length=min_section_length,
        )

        # If primary method returns only 1 section on long audio, try agglomerative
        if len(sections) <= 1 and total_duration > min_section_length * 3:
            sections = _agglomerative_fallback(audio, sr, min_section_length)

        return sections

    except Exception as e:
        print(f"[WARNING] Structure detection failed: {e}")
        # Try agglomerative as last resort
        try:
            return _agglomerative_fallback(audio, sr, min_section_length)
        except Exception:
            return [Section(start=0.0, end=round(total_duration, 3), duration=round(total_duration, 3))]


def _agglomerative_fallback(
    audio: np.ndarray,
    sr: int,
    min_section_length: float,
) -> list[Section]:
    """
    Fallback segmentation using agglomerative clustering on MFCC + chroma.

    Uses feature stacking for richer representation than MFCC alone.
    """
    import librosa

    total_duration = len(audio) / sr
    max_sections = max(2, int(total_duration / min_section_length))
    k = min(max_sections, 8)

    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        features = np.vstack([
            librosa.util.normalize(mfcc, axis=1),
            librosa.util.normalize(chroma, axis=1),
        ])

        bound_frames = librosa.segment.agglomerative(features, k=k)
        bound_times = librosa.frames_to_time(bound_frames, sr=sr)

        boundaries = sorted(set([0.0] + list(bound_times) + [total_duration]))

        sections: list[Section] = []
        for i in range(len(boundaries) - 1):
            s = boundaries[i]
            e = boundaries[i + 1]
            sections.append(Section(start=round(s, 3), end=round(e, 3), duration=round(e - s, 3)))

        sections = _merge_short_sections(sections, min_section_length)
        return sections

    except Exception as e:
        print(f"[WARNING] Agglomerative fallback failed: {e}")
        return [Section(start=0.0, end=round(total_duration, 3), duration=round(total_duration, 3))]
