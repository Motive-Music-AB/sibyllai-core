"Speech/Music segmentation via inaSpeechSegmenter."
from pathlib import Path
import logging
import soundfile as sf
from inaSpeechSegmenter import Segmenter

__all__ = ["detect_music_regions"]

_segmenter = Segmenter()           # heavy → load once


def detect_music_regions(wav_path: str | Path):
    """
    Return list[(start_sec, end_sec)] that are music according to inaSpeech.
    Falls back to a single full-length region when the package crashes on
    “empty energy” edge-case.
    """
    wav_path = str(wav_path)
    try:
        regions = [
            (start, end)
            for lab, start, end in _segmenter(wav_path) if lab == "music"
        ]
    except TypeError:  # “arrays to stack must be passed …” bug
        dur = sf.info(wav_path).duration
        logging.warning(
            "inaSpeechSegmenter failed; using full-length region instead."
        )
        regions = [(0.0, dur)]

    return regions
