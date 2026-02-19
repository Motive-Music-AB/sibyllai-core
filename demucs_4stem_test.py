import subprocess
from pathlib import Path
import soundfile as sf
import numpy as np

# Path to your test audio file (should be WAV, ideally stereo 44.1kHz)
input_wav = "outputs/segment_1.wav"  # Change this to your test file
output_dir = "outputs/demucs_4stem_test"

# Run Demucs in 4-stem mode (no --two-stems argument)
subprocess.run([
    "demucs", "-o", output_dir, input_wav
], check=True)

# Demucs will output stems in output_dir/htdemucs/<input_wav_stem>/
stem_dir = Path(output_dir) / "htdemucs" / Path(input_wav).stem

print("\nDemucs 4-stem output files:")
for stem in ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]:
    stem_path = stem_dir / stem
    print(f"{stem}: {'FOUND' if stem_path.exists() else 'NOT FOUND'} ({stem_path})")

# Combine drums, bass, and other into music_minus_vocals.wav
try:
    drums, sr = sf.read(stem_dir / "drums.wav")
    bass, _ = sf.read(stem_dir / "bass.wav")
    other, _ = sf.read(stem_dir / "other.wav")
    min_len = min(len(drums), len(bass), len(other))
    drums = drums[:min_len]
    bass = bass[:min_len]
    other = other[:min_len]
    music = drums + bass + other
    max_val = np.max(np.abs(music))
    if max_val > 1.0:
        music = music / max_val
    out_path = stem_dir / "music_minus_vocals.wav"
    sf.write(out_path, music, sr)
    print(f"Combined music-minus-vocals stem saved as: {out_path}")
except Exception as e:
    print(f"[ERROR] Could not combine stems: {e}")