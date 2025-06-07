"LAION-CLAP tag similarity helper."
import numpy as np
import librosa

_clap = None
_TAGS = ["rock", "classical", "contains speech", "lo-fi", "orchestral"]

def tag_chunk(chunk, sr: int) -> dict[str, float]:
    """
    Return {tag: similarity} for a small fixed tag list.
    """
    global _clap
    if _clap is None:
        import laion_clap
        _clap = laion_clap.CLAP_Module(enable_fusion=False)
        _clap.load_ckpt()

    if sr != 48_000:
        chunk = librosa.resample(y=chunk, orig_sr=sr, target_sr=48_000)
        sr = 48_000

    emb  = _clap.get_audio_embedding_from_data(chunk.reshape(1, -1))[0]
    temb = _clap.get_text_embedding(_TAGS)
    sims = (emb @ temb.T) / (np.linalg.norm(emb) * np.linalg.norm(temb, axis=1))
    return dict(zip(_TAGS, sims))
