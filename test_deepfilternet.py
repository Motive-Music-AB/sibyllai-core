import soundfile as sf
import deepfilternet as dfn

# Input and output paths
input_path = "outputs/run_002/segment_1_music_minus_vocals.wav"
output_path = "outputs/run_002/segment_1_music_minus_vocals_enhanced.wav"

# Load audio
y, sr = sf.read(input_path)

# Enhance audio
enhanced = dfn.deepfilter(y, sr)

# Save result
sf.write(output_path, enhanced, sr)
print(f"Enhanced audio saved to: {output_path}")