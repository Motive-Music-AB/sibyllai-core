"""
Music change-point detection via spectral novelty function.

Detects structural boundaries (intro, verse, chorus, build-up, climax, etc.)
by computing spectral flux from a mel-spectrogram and picking peaks that
correspond to significant timbral/energy transitions.

Fast, no new dependencies: uses librosa + scipy (already installed).
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
    sensitivity: float = 1.0,
    n_mels: int = 128,
    hop_length: int = 512,
    kernel_size: int = 64,
) -> list[Section]:
    """
    Detect structural sections in audio using spectral novelty + peak picking.

    Args:
        audio: Mono audio signal as numpy array.
        sr: Sample rate.
        min_section_length: Minimum section duration in seconds.
        sensitivity: Peak threshold multiplier (higher = fewer peaks).
        n_mels: Number of mel bands for the spectrogram.
        hop_length: Hop length in samples (~11.6ms at 44.1kHz).
        kernel_size: Gaussian smoothing kernel width in frames.

    Returns:
        List of Section namedtuples sorted by start time.
    """
    import librosa
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    total_duration = len(audio) / sr
    if total_duration < min_section_length:
        return [Section(start=0.0, end=total_duration, duration=total_duration)]

    # 1. Mel spectrogram
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # 2. Spectral flux (L2 norm of positive frame-to-frame differences)
    diff = np.diff(S_db, axis=1)
    diff = np.maximum(diff, 0)  # only positive changes (onsets)
    flux = np.linalg.norm(diff, axis=0)

    if len(flux) < 3:
        return [Section(start=0.0, end=total_duration, duration=total_duration)]

    # 3. Gaussian smoothing
    sigma = max(1, kernel_size / 6)
    flux_smooth = gaussian_filter1d(flux, sigma=sigma)

    # 4. Normalize to [0, 1]
    fmax = flux_smooth.max()
    if fmax > 0:
        flux_norm = flux_smooth / fmax
    else:
        return [Section(start=0.0, end=total_duration, duration=total_duration)]

    # 5. Adaptive threshold
    threshold = float(np.mean(flux_norm) + sensitivity * np.std(flux_norm))
    threshold = min(threshold, 0.95)  # prevent threshold from eliminating everything

    # 6. Peak picking
    min_frames = int(min_section_length * sr / hop_length)
    peaks, _ = find_peaks(
        flux_norm,
        height=threshold,
        distance=min_frames,
        prominence=0.05,
    )

    if len(peaks) == 0:
        return [Section(start=0.0, end=total_duration, duration=total_duration)]

    # 7. Convert peak frames to time boundaries
    # The diff operation shifts indices by 1, so peaks correspond to frame (peak+1) in the original spectrogram
    peak_times = librosa.frames_to_time(peaks + 1, sr=sr, hop_length=hop_length)

    # 8. Build sections from boundaries
    boundaries = [0.0] + list(peak_times) + [total_duration]
    sections: list[Section] = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]
        sections.append(Section(start=round(s, 3), end=round(e, 3), duration=round(e - s, 3)))

    # 9. Merge short sections into neighbours
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
                    # Merge with next
                    nxt = merged[i + 1]
                    new_merged.append(Section(
                        start=sec.start,
                        end=nxt.end,
                        duration=round(nxt.end - sec.start, 3),
                    ))
                    i += 2
                elif i == len(merged) - 1 and len(new_merged) > 0:
                    # Merge with previous (last in new_merged)
                    prev = new_merged.pop()
                    new_merged.append(Section(
                        start=prev.start,
                        end=sec.end,
                        duration=round(sec.end - prev.start, 3),
                    ))
                    i += 1
                elif i > 0 and i + 1 < len(merged):
                    # Merge with shorter neighbour
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
    Detect sections with fallback strategies for edge cases.

    - Over-segmented (>20 sections): retry with higher sensitivity (1.5x).
    - Under-segmented (1 section on long audio): fallback to agglomerative segmentation.
    - Final fallback: single section covering full duration.

    Args:
        audio: Mono audio signal.
        sr: Sample rate.
        min_section_length: Minimum section duration in seconds.
        sensitivity: Initial peak threshold multiplier.

    Returns:
        List of Section namedtuples.
    """
    total_duration = len(audio) / sr

    try:
        sections = detect_sections(
            audio, sr,
            min_section_length=min_section_length,
            sensitivity=sensitivity,
        )

        # Over-segmented: retry with higher sensitivity
        if len(sections) > 20:
            sections = detect_sections(
                audio, sr,
                min_section_length=min_section_length,
                sensitivity=sensitivity * 1.5,
            )

        # Still over-segmented: bump sensitivity further
        if len(sections) > 20:
            sections = detect_sections(
                audio, sr,
                min_section_length=min_section_length,
                sensitivity=sensitivity * 2.0,
            )

        # Under-segmented on long audio: try agglomerative
        if len(sections) <= 1 and total_duration > min_section_length * 3:
            sections = _agglomerative_fallback(audio, sr, min_section_length)

        return sections

    except Exception as e:
        print(f"[WARNING] Structure detection failed: {e}")
        return [Section(start=0.0, end=round(total_duration, 3), duration=round(total_duration, 3))]


def _agglomerative_fallback(
    audio: np.ndarray,
    sr: int,
    min_section_length: float,
) -> list[Section]:
    """
    Fallback segmentation using librosa's agglomerative clustering on MFCCs.

    Used when spectral novelty finds no boundaries (e.g., very smooth audio).
    """
    import librosa

    total_duration = len(audio) / sr
    max_sections = max(2, int(total_duration / min_section_length))
    k = min(max_sections, 10)

    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        bound_frames = librosa.segment.agglomerative(mfcc, k=k)
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
