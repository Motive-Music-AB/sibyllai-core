"Thin wrapper around third-party Music2Emo package."
from ..thirdparty.music2emo.music2emo import Music2emo

_m2e = None

def global_moods(wav_path: str, threshold: float = 0.5):
    "Return {'valence':…, 'arousal':…, 'predicted_moods':[…]} dict."
    global _m2e
    if _m2e is None:
        _m2e = Music2emo()
    return _m2e.predict(wav_path, threshold=threshold)
