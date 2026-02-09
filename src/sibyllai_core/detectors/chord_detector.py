import numpy as np
import librosa
from typing import Dict, Tuple
from essentia.standard import KeyExtractor

# Chord vocabulary definitions
idx2chord = ['C', 'C:min', 'C#', 'C#:min', 'D', 'D:min', 'D#', 'D#:min', 'E', 'E:min', 'F', 'F:min', 'F#',
             'F#:min', 'G', 'G:min', 'G#', 'G#:min', 'A', 'A:min', 'A#', 'A#:min', 'B', 'B:min', 'N']

root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
quality_list = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']

def idx2voca_chord():
    """Create mapping from index to chord names for the larger vocabulary."""
    idx2voca_chord = {}
    idx2voca_chord[169] = 'N'
    idx2voca_chord[168] = 'X'
    for i in range(168):
        root = i // 14
        root = root_list[root]
        quality = i % 14
        quality = quality_list[quality]
        if i % 14 != 1:
            chord = root + ':' + quality
        else:
            chord = root
        idx2voca_chord[i] = chord
    return idx2voca_chord

def extract_chord_features(audio_data: np.ndarray, sr: int, config: Dict = None) -> Tuple[np.ndarray, float, float]:
    """
    Extract chord detection features from audio data.
    
    Args:
        audio_data: Audio samples
        sr: Sample rate
        config: Configuration dictionary with feature extraction parameters
    
    Returns:
        feature: Extracted features
        feature_per_second: Features per second
        song_length_second: Song length in seconds
    """
    if config is None:
        config = {
            'mp3': {'song_hz': 44100, 'inst_len': 0.1},
            'feature': {'n_bins': 84, 'bins_per_octave': 12, 'hop_length': 512},
            'model': {'timestep': 8}
        }
    
    original_wav = audio_data
    currunt_sec_hz = 0

    # Pre-initialise so that the variable is **always** defined, preventing
    # `UnboundLocalError` when the audio is shorter than a single window.
    feature = None

    # Process fixed-length chunks first
    while len(original_wav) > currunt_sec_hz + config['mp3']['song_hz'] * config['mp3']['inst_len']:
        start_idx = int(currunt_sec_hz)
        end_idx = int(currunt_sec_hz + config['mp3']['song_hz'] * config['mp3']['inst_len'])
        tmp = librosa.cqt(
            original_wav[start_idx:end_idx],
            sr=sr,
            n_bins=config['feature']['n_bins'],
            bins_per_octave=config['feature']['bins_per_octave'],
            hop_length=config['feature']['hop_length'],
        )

        # Initialise or append depending on whether this is the first block
        feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)

        currunt_sec_hz = end_idx

    # Handle the trailing (or entire) remainder of the audio
    if len(original_wav) > currunt_sec_hz:
        tmp = librosa.cqt(
            original_wav[currunt_sec_hz:],
            sr=sr,
            n_bins=config['feature']['n_bins'],
            bins_per_octave=config['feature']['bins_per_octave'],
            hop_length=config['feature']['hop_length'],
        )

        feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)
    
    if feature is None:
        # Handle case where audio is too short
        feature = np.zeros((config['feature']['n_bins'], 1))
    
    feature = np.log(np.abs(feature) + 1e-6)
    feature_per_second = config['mp3']['inst_len'] / config['model']['timestep']
    song_length_second = len(original_wav) / config['mp3']['song_hz']
    
    return feature, feature_per_second, song_length_second

def detect_chords_simple(audio_data: np.ndarray, sr: int) -> Dict[str, any]:
    """
    Key detection using Essentia's KeyExtractor algorithm.

    Args:
        audio_data: Audio samples
        sr: Sample rate

    Returns:
        Dictionary with key analysis results
    """
    try:
        # Ensure audio is mono and float32 (Essentia requirement)
        if audio_data.ndim > 1:
            audio_mono = librosa.to_mono(audio_data.T)
        else:
            audio_mono = audio_data

        # Convert to float32 if needed
        if audio_mono.dtype != np.float32:
            audio_mono = audio_mono.astype(np.float32)

        # Use Essentia's KeyExtractor with correct sample rate
        # This prevents pitch shift errors when analyzing audio at non-44.1kHz rates
        key_extractor = KeyExtractor(sampleRate=int(sr))
        key, scale, strength = key_extractor(audio_mono)

        # Format key name (capitalize scale: major/minor)
        scale_formatted = scale.capitalize()
        detected_key = f"{key} {scale_formatted}"

        # Map to chord format for backwards compatibility
        if scale == 'minor':
            primary_chord = f"{key}:min"
        else:
            primary_chord = key

        return {
            'primary_chord': primary_chord,
            'key': detected_key,
            'chord_confidence': float(strength),
            'chord_progression': [primary_chord],
            'chord_complexity': 'simple'
        }

    except Exception as e:
        return {
            'primary_chord': 'Unknown',
            'key': 'Unknown',
            'chord_confidence': 0.0,
            'chord_progression': [],
            'chord_complexity': 'unknown',
            'error': str(e)
        }

def analyze_chords(audio_path: str = None, audio_data: np.ndarray = None, sr: int = 44100) -> Dict[str, any]:
    """
    Analyze chords in an audio file or audio data.
    
    Args:
        audio_path: Path to audio file (if audio_data is None)
        audio_data: Audio samples (if provided)
        sr: Sample rate
    
    Returns:
        Dictionary with chord analysis results
    """
    try:
        if audio_data is None:
            if audio_path is None:
                raise ValueError("Either audio_path or audio_data must be provided")
            # Load audio from file
            audio_data, sr = librosa.load(audio_path, sr=sr, mono=True)
        
        # Perform chord analysis
        chord_analysis = detect_chords_simple(audio_data, sr)
        
        return chord_analysis
        
    except Exception as e:
        return {
            'primary_chord': 'Unknown',
            'key': 'Unknown',
            'chord_confidence': 0.0,
            'chord_progression': [],
            'chord_complexity': 'unknown',
            'error': str(e)
        } 