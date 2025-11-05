"""Fast segmentation module - YAMNet only (Phase 1)."""
from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path

from sibyllai_core.detectors.yamnet_segmenter import segment_music_regions


def extract_audio_fast(src: str | Path) -> Path:
    """Extract audio to temporary mono 16kHz WAV for YAMNet."""
    src = Path(src)
    tmp = Path(tempfile.mkdtemp(prefix="sibyllai_segment_"))
    wav = tmp / "audio.wav"

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    return wav


def segment_only(
    audio_path: str | Path,
    music_thresh: float = 0.2,
    min_gap: float = 5.0,
    min_cue_length: float = 3.0
) -> tuple[list[tuple[float, float]], float]:
    """
    Fast segmentation using YAMNet only.

    Args:
        audio_path: Path to audio or video file
        music_thresh: YAMNet music probability threshold (0-1)
        min_gap: Minimum gap in seconds to keep segments separate
        min_cue_length: Minimum cue length in seconds (shorter cues filtered out)

    Returns:
        Tuple of (segments, duration) where segments is [(start, end), ...]
    """
    # Extract audio for YAMNet
    wav_path = extract_audio_fast(audio_path)

    try:
        # Run YAMNet segmentation
        segments = segment_music_regions(wav_path, music_thresh, min_gap)

        # Filter segments by minimum cue length
        segments = [(start, end) for start, end in segments if (end - start) >= min_cue_length]

        # Get audio duration
        import soundfile as sf
        with sf.SoundFile(wav_path) as f:
            duration = len(f) / f.samplerate

        return segments, duration

    finally:
        # Clean up temp file
        if wav_path.exists():
            wav_path.unlink()
        if wav_path.parent.exists():
            wav_path.parent.rmdir()
