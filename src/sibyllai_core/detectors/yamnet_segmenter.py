import os
import subprocess
import tempfile
import numpy as np
import tensorflow_hub as hub
import soundfile as sf
import pandas as pd

_yamnet_model = None
_class_names = None


def _ensure_class_map() -> str:
    class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    class_map_path = os.path.join(os.path.dirname(__file__), "yamnet_class_map.csv")
    if not os.path.exists(class_map_path):
        import urllib.request
        urllib.request.urlretrieve(class_map_url, class_map_path)
    return class_map_path


def _load_yamnet():
    global _yamnet_model, _class_names
    if _yamnet_model is None:
        _yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    if _class_names is None:
        class_map_path = _ensure_class_map()
        _class_names = pd.read_csv(class_map_path)["display_name"].tolist()
    return _yamnet_model, _class_names

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

def segment_music_regions(audio_path, music_thresh=0.2, min_gap=1.0, silence_thresh=0.01, use_yamnet=True):
    """
    Returns a list of (start_time, end_time) tuples for detected music regions in the audio file.

    Args:
        audio_path: Path to audio file
        music_thresh: YAMNet music probability threshold (only used when use_yamnet=True)
        min_gap: Minimum gap in seconds to split segments
        silence_thresh: RMS amplitude threshold (used for trimming, or as main threshold when use_yamnet=False)
        use_yamnet: If True, use YAMNet music classification (for full mix with dialogue/SFX).
                    If False, use simple amplitude detection (for clean MX, music-only files).
    """
    tmp_handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav = tmp_handle.name
    tmp_handle.close()
    wav_path = extract_audio(audio_path, temp_wav)

    # Load audio
    waveform, sr = sf.read(wav_path)
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != 16000:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
        sr = 16000
    waveform = waveform.astype(np.float32)

    if use_yamnet:
        # Full Mix mode: Use YAMNet to classify music vs dialogue/SFX
        yamnet_model, class_names = _load_yamnet()
        music_idx = class_names.index("Music")
        scores, _, _ = yamnet_model(waveform)
        probs = scores[:, music_idx].numpy()
        threshold = music_thresh
        frame_hop_s = 0.48
    else:
        # Clean MX mode: Use simple amplitude detection (faster, no ML model)
        frame_hop_s = 0.05  # 50ms windows for finer resolution
        hop_samples = int(frame_hop_s * sr)
        num_frames = len(waveform) // hop_samples
        probs = np.array([
            np.sqrt(np.mean(waveform[i*hop_samples:(i+1)*hop_samples] ** 2))
            for i in range(num_frames)
        ])
        threshold = silence_thresh

    frame_times = np.arange(len(probs)) * frame_hop_s
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
    segments = get_segments(probs, threshold, frame_times)
    segments = merge_close_segments(segments, min_gap=min_gap)

    if use_yamnet:
        # YAMNet mode: Add padding to compensate for frame granularity (~0.48s)
        PADDING = 0.5
        segments = [(max(0, start - PADDING), end + PADDING) for start, end in segments]

        # Trim silence from segment starts using amplitude threshold
        def trim_silence_start(waveform, sr, start, end, amp_thresh, hop_size=0.05):
            """Trim silence from the start of a segment based on RMS amplitude."""
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            hop_samples = int(hop_size * sr)

            for i in range(start_sample, min(end_sample, start_sample + int(2 * sr)), hop_samples):
                window = waveform[i:i + hop_samples]
                if len(window) > 0:
                    rms = np.sqrt(np.mean(window ** 2))
                    if rms > amp_thresh:
                        return i / sr
            return start

        # Apply silence trimming to each segment
        trimmed_segments = []
        for start, end in segments:
            trimmed_start = trim_silence_start(waveform, 16000, start, end, silence_thresh)
            trimmed_segments.append((trimmed_start, end))
        segments = trimmed_segments

    # Clean MX mode: no padding or silence trimming needed (already amplitude-based)

    # Clean up temp file if created
    if wav_path == temp_wav and os.path.exists(temp_wav):
        os.remove(temp_wav)
    return segments


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
    yamnet_model, class_names = _load_yamnet()

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
    yamnet_model, class_names = _load_yamnet()

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
