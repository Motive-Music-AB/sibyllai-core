"Audio-Spectrogram-Transformer music probability helper."
import torch
import librosa
from transformers import (
    AutoProcessor,
    AutoModelForAudioClassification,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy-loaded model and processor
_proc = None
_model = None
_music_idx = None

def _load_ast_model():
    global _proc, _model, _music_idx
    if _proc is None or _model is None or _music_idx is None:
        _proc = AutoProcessor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        _model = AutoModelForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        ).to(DEVICE)
        _music_idx = _model.config.label2id["Music"]


def music_probability(chunk, sr: int) -> float:
    "Return probability [0-1] that *chunk* is music."
    _load_ast_model()
    if sr != 16_000:
        chunk = librosa.resample(y=chunk, orig_sr=sr, target_sr=16_000)
        sr = 16_000

    ins   = _proc(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
    ins   = {k: v.to(DEVICE) for k, v in ins.items()}
    logits = _model(**ins).logits
    return torch.sigmoid(logits)[0, _music_idx].item()
