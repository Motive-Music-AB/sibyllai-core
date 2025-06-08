import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import soundfile as sf
import pandas as pd
import urllib.request
import os
import sys
import subprocess

# Path to your test file in examples/data (change as needed)
AUDIO_FILE = "examples/data/trailer.mov"  # or .wav

# Function to extract audio if needed
def extract_audio_if_needed(input_file):
    if input_file.lower().endswith('.wav'):
        return input_file
    # Extract audio to temp file
    temp_wav = "examples/data/temp_extracted.wav"
    print(f"Extracting audio from {input_file} to {temp_wav} ...")
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-ac", "1", "-ar", "16000", "-vn", temp_wav
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed:", e.stderr.decode())
        sys.exit(1)
    return temp_wav

AUDIO_FILE = extract_audio_if_needed(AUDIO_FILE)

# Download class map if not present
CLASS_MAP_CSV = "yamnet_class_map.csv"
if not os.path.exists(CLASS_MAP_CSV):
    url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    urllib.request.urlretrieve(url, CLASS_MAP_CSV)

# Load class names
class_names = pd.read_csv(CLASS_MAP_CSV)["display_name"].tolist()

# Load audio
waveform, sr = sf.read(AUDIO_FILE)
if len(waveform.shape) > 1:
    waveform = np.mean(waveform, axis=1)  # Convert to mono

# Resample to 16kHz if needed
if sr != 16000:
    import librosa
    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    sr = 16000

# Convert to float32
waveform = waveform.astype(np.float32)

# Load YAMNet model
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Run model
scores, embeddings, spectrogram = yamnet_model(waveform)

# 1. Find class indices
music_idx = class_names.index("Music")
speech_idx = class_names.index("Speech")

# 2. Get per-frame probabilities
music_probs = scores[:, music_idx].numpy()
# speech_probs = scores[:, speech_idx].numpy()

# 3. Set thresholds
music_thresh = 0.2
# speech_thresh = 0.3

# 4. Get frame times
frame_hop_s = 0.48  # seconds per frame (YAMNet default)
frame_times = np.arange(len(music_probs)) * frame_hop_s

def get_segments(probs, threshold, frame_times):
    """Return list of (start_time, end_time) for contiguous regions above threshold."""
    above = probs > threshold
    segments = []
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = frame_times[i]
        elif not flag and start is not None:
            end = frame_times[i]
            segments.append((start, end))
            start = None
    if start is not None:
        segments.append((start, frame_times[-1] + frame_hop_s))
    return segments

def merge_close_segments(segments, min_gap=1.0):
    """Merge segments if the gap between them is less than min_gap seconds."""
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end < min_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged

music_segments = get_segments(music_probs, music_thresh, frame_times)
music_segments = merge_close_segments(music_segments, min_gap=1.0)
# speech_segments = get_segments(speech_probs, speech_thresh, frame_times)

print("\nMusic segments (start, end in seconds, merged):")
for seg in music_segments:
    print(f"{seg[0]:.2f} - {seg[1]:.2f}")
    # Get frame indices for this segment
    start_idx = int(seg[0] // frame_hop_s)
    end_idx = int(seg[1] // frame_hop_s)
    segment_scores = scores[start_idx:end_idx]
    mean_scores = np.mean(segment_scores.numpy(), axis=0)
    top5_i = np.argsort(mean_scores)[::-1][:5]
    print("  Top 5 predictions for this segment:")
    for i in top5_i:
        print(f"    {class_names[i]}: {mean_scores[i]:.3f}")

# print("\nSpeech segments (start, end in seconds):")
# for seg in speech_segments:
#     print(f"{seg[0]:.2f} - {seg[1]:.2f}")

    