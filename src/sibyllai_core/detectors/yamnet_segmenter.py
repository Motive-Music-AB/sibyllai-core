import os
import subprocess
import tempfile
import numpy as np
import tensorflow_hub as hub
import soundfile as sf
import pandas as pd

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

def segment_music_regions(audio_path, music_thresh=0.2, min_gap=1.0, silence_thresh=0.01):
    """
    Returns a list of (start_time, end_time) tuples for detected music regions in the audio file.
    """
    tmp_handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav = tmp_handle.name
    tmp_handle.close()
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

    # Add padding to compensate for frame granularity (~0.48s)
    # Extend each segment by 0.5s at start and end to capture missed audio
    PADDING = 0.5
    music_segments = [(max(0, start - PADDING), end + PADDING) for start, end in music_segments]

    # Trim silence from segment starts using amplitude threshold
    # This prevents segments starting at 0 when there's actual silence
    def trim_silence_start(waveform, sr, start, end, amp_thresh, hop_size=0.05):
        """Trim silence from the start of a segment based on RMS amplitude."""
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        hop_samples = int(hop_size * sr)

        # Check amplitude in small windows from segment start
        for i in range(start_sample, min(end_sample, start_sample + int(2 * sr)), hop_samples):
            window = waveform[i:i + hop_samples]
            if len(window) > 0:
                rms = np.sqrt(np.mean(window ** 2))
                if rms > amp_thresh:
                    # Found audio - return this position
                    return i / sr
        # No significant audio found in first 2 seconds, return original start
        return start

    # Apply silence trimming to each segment
    trimmed_segments = []
    for start, end in music_segments:
        trimmed_start = trim_silence_start(waveform, 16000, start, end, silence_thresh)
        trimmed_segments.append((trimmed_start, end))
    music_segments = trimmed_segments

    # Clean up temp file if created
    if wav_path == temp_wav and os.path.exists(temp_wav):
        os.remove(temp_wav)
    return music_segments


# Instrument class indices from YAMNet's 521 classes
# These are the primary musical instrument categories
INSTRUMENT_CLASSES = {
    "Piano": 148,
    "Electric piano": 149,
    "Guitar": 135,
    "Electric guitar": 136,
    "Bass guitar": 137,
    "Acoustic guitar": 138,
    "Violin": 186,
    "String section": 185,
    "Bowed string instrument": 184,
    "Drum kit": 157,
    "Drum": 159,
    "Snare drum": 160,
    "Bass drum": 162,
    "Brass instrument": 180,
    "Synthesizer": None,  # YAMNet doesn't have a direct synthesizer class
    "Keyboard (musical)": None,  # Would need to check class map
}


def extract_instruments(audio_data, sr=16000, top_n=5):
    """
    Extract top N detected musical instruments from audio using YAMNet.

    Args:
        audio_data: Audio waveform as numpy array (mono)
        sr: Sample rate (will be resampled to 16kHz if needed)
        top_n: Number of top instruments to return

    Returns:
        Dictionary mapping instrument name to confidence score:
        {"Piano": 0.89, "String section": 0.76, "Drum kit": 0.65}
    """
    # Load YAMNet model
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    # Load class map
    class_map_path = os.path.join(os.path.dirname(__file__), "yamnet_class_map.csv")
    if not os.path.exists(class_map_path):
        class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
        import urllib.request
        urllib.request.urlretrieve(class_map_url, class_map_path)

    class_names_df = pd.read_csv(class_map_path)
    class_names = class_names_df["display_name"].tolist()

    # Build instrument index map
    instrument_indices = {}
    for instrument_name in class_names:
        lower_name = instrument_name.lower()
        # Check if this class name contains instrument or vocal keywords
        if any(keyword in lower_name for keyword in [
            # Instruments
            'piano', 'guitar', 'violin', 'drum', 'brass', 'string',
            'trumpet', 'saxophone', 'flute', 'clarinet', 'trombone',
            'cello', 'bass', 'harp', 'organ', 'accordion', 'harmonica',
            'synthesizer', 'keyboard', 'percussion',
            # Vocals
            'singing', 'humming', 'vocal', 'choir', 'rapping', 'chant',
            'speech', 'whistling', 'beatbox'
        ]):
            idx = class_names.index(instrument_name)
            instrument_indices[instrument_name] = idx

    # Ensure mono audio
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

    audio_data = audio_data.astype(np.float32)

    # Run YAMNet inference
    scores, _, _ = yamnet_model(audio_data)

    # Average scores across all frames
    avg_scores = np.mean(scores.numpy(), axis=0)

    # Extract instrument scores
    instrument_scores = {}
    for instrument_name, idx in instrument_indices.items():
        instrument_scores[instrument_name] = float(avg_scores[idx])

    # Sort by score and return top N
    sorted_instruments = sorted(instrument_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return dict(sorted_instruments)


def extract_genres(audio_data, sr=16000, top_n=5):
    """
    Extract top N detected music genres from audio using YAMNet.

    Args:
        audio_data: Audio waveform as numpy array (mono)
        sr: Sample rate (will be resampled to 16kHz if needed)
        top_n: Number of top genres to return

    Returns:
        Dictionary mapping genre name to confidence score:
        {"Pop music": 0.45, "Electronic music": 0.32, "Rock music": 0.18}
    """
    # Load YAMNet model
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    # Load class map
    class_map_path = os.path.join(os.path.dirname(__file__), "yamnet_class_map.csv")
    if not os.path.exists(class_map_path):
        class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
        import urllib.request
        urllib.request.urlretrieve(class_map_url, class_map_path)

    class_names_df = pd.read_csv(class_map_path)
    class_names = class_names_df["display_name"].tolist()

    # Genre-related keywords from YAMNet classes (indices 211-269 are mostly genres)
    genre_keywords = [
        'pop music', 'hip hop', 'rock music', 'heavy metal', 'punk rock',
        'grunge', 'progressive rock', 'rock and roll', 'psychedelic',
        'rhythm and blues', 'soul music', 'reggae', 'country', 'swing',
        'bluegrass', 'funk', 'folk music', 'middle eastern', 'jazz',
        'disco', 'classical music', 'opera', 'electronic music', 'house music',
        'techno', 'dubstep', 'drum and bass', 'electronica', 'electronic dance',
        'ambient music', 'trance', 'latin', 'salsa', 'flamenco', 'blues',
        'music for children', 'new-age', 'vocal music', 'a capella',
        'afrobeat', 'christian music', 'gospel', 'ska', 'traditional music',
        'indie', 'soundtrack', 'theme music', 'lullaby', 'video game music',
        'christmas music', 'dance music', 'wedding music',
        # Mood/energy descriptors from YAMNet
        'happy music', 'sad music', 'tender music', 'exciting music',
        'angry music', 'scary music'
    ]

    # Build genre index map
    genre_indices = {}
    for class_name in class_names:
        lower_name = class_name.lower()
        if any(keyword in lower_name for keyword in genre_keywords):
            idx = class_names.index(class_name)
            genre_indices[class_name] = idx

    # Ensure mono audio
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

    audio_data = audio_data.astype(np.float32)

    # Run YAMNet inference
    scores, _, _ = yamnet_model(audio_data)

    # Average scores across all frames
    avg_scores = np.mean(scores.numpy(), axis=0)

    # Extract genre scores
    genre_scores = {}
    for genre_name, idx in genre_indices.items():
        genre_scores[genre_name] = float(avg_scores[idx])

    # Sort by score and return top N
    sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return dict(sorted_genres)
