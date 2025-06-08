import os
import subprocess
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf

def extract_audio(input_path, output_path):
    input_path = str(input_path)
    output_path = str(output_path)
    if input_path.lower().endswith('.wav'):
        return input_path
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-vn", output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path

def segment_music_regions(audio_path, music_thresh=0.2, min_gap=1.0):
    """
    Returns a list of (start_time, end_time) tuples for detected music regions in the audio file.
    """
    temp_wav = "temp_extracted.wav"
    wav_path = extract_audio(audio_path, temp_wav)
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    # Load class map from the same directory as this file
    import pandas as pd
    class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    class_map_path = os.path.join(os.path.dirname(__file__), "yamnet_class_map.csv")
    if not os.path.exists(class_map_path):
        import urllib.request
        urllib.request.urlretrieve(class_map_url, class_map_path)
    class_names = pd.read_csv(class_map_path)["display_name"].tolist()
    music_idx = class_names.index("Music")
    # Load audio
    waveform, sr = sf.read(wav_path)
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    waveform = waveform.astype(np.float32)
    # Run YAMNet
    scores, _, _ = yamnet_model(waveform)
    music_probs = scores[:, music_idx].numpy()
    frame_hop_s = 0.48
    frame_times = np.arange(len(music_probs)) * frame_hop_s
    # Segment logic
    def get_segments(probs, threshold, frame_times):
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
    music_segments = merge_close_segments(music_segments, min_gap=min_gap)
    # Clean up temp file if created
    if wav_path == temp_wav and os.path.exists(temp_wav):
        os.remove(temp_wav)
    return music_segments 