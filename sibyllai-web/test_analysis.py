#!/usr/bin/env python3
"""
Test YAMNet genre and instrument detection directly.

Usage:
    cd sibyllai-core
    source .venv/bin/activate
    python sibyllai-web/test_analysis.py /path/to/audio.wav
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import soundfile as sf
from sibyllai_core.detectors.yamnet_segmenter import extract_instruments, extract_genres
from sibyllai_core.detectors.clap import tag_chunk


def test_audio(audio_path: str):
    """Run all detectors on an audio file and print results."""
    print(f"\n{'='*60}")
    print(f"Testing: {audio_path}")
    print(f"{'='*60}\n")

    # Load audio
    y, sr = sf.read(audio_path)
    print(f"Loaded: {len(y)/sr:.1f}s @ {sr}Hz")

    # Mono conversion
    if y.ndim > 1:
        y = np.mean(y, axis=1)
        print("Converted to mono")

    # Limit to first 30 seconds for faster testing
    max_samples = 30 * sr
    if len(y) > max_samples:
        y = y[:max_samples]
        print(f"Trimmed to 30s for testing")

    print()

    # Test YAMNet instrument detection
    print("=== Instruments (YAMNet) ===")
    try:
        instruments = extract_instruments(y, sr=sr, top_n=10)
        for name, score in instruments.items():
            bar = '█' * int(score * 100)
            print(f"  {name:30} {score:.4f} {bar}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()

    # Test YAMNet genre detection
    print("=== Genres (YAMNet) ===")
    try:
        genres = extract_genres(y, sr=sr, top_n=10)
        for name, score in genres.items():
            bar = '█' * int(score * 100)
            print(f"  {name:30} {score:.4f} {bar}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()

    # Test CLAP style detection
    print("=== Style (CLAP) ===")
    try:
        clap_results = tag_chunk(y, sr)
        # Show genre category (now called "style" in our model)
        if 'genre' in clap_results:
            sorted_style = sorted(clap_results['genre'].items(), key=lambda x: x[1], reverse=True)
            for name, score in sorted_style[:5]:
                bar = '█' * int(score * 50)
                print(f"  {name:30} {score:.4f} {bar}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_analysis.py <audio_file>")
        print("\nExample:")
        print("  python test_analysis.py ~/Music/test-track.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    test_audio(audio_path)
