"""Fast segmentation module - YAMNet or volume-based (Phase 1)."""
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
    min_cue_length: float = 3.0,
    silence_thresh: float = 0.01,
    mix_type: str = "full_mix",
) -> tuple[list[tuple[float, float]], float]:
    """
    Fast segmentation using YAMNet (full_mix) or volume-based (clean_mx).

    Args:
        audio_path: Path to audio or video file
        music_thresh: YAMNet music probability threshold (0-1) - only used for full_mix
        min_gap: Minimum gap in seconds to keep segments separate
        min_cue_length: Minimum cue length in seconds (shorter cues filtered out)
        silence_thresh: RMS amplitude threshold (0-1) - main threshold for clean_mx
        mix_type: "clean_mx" for music-only files (volume detection),
                  "full_mix" for full film mix (YAMNet music classification)

    Returns:
        Tuple of (segments, duration) where segments is [(start, end), ...]
    """
    wav_path = extract_audio_fast(audio_path)
    use_yamnet = (mix_type != "clean_mx")

    try:
        segments = segment_music_regions(
            wav_path,
            music_thresh=music_thresh,
            min_gap=min_gap,
            silence_thresh=silence_thresh,
            use_yamnet=use_yamnet,
        )

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
