"LAION-CLAP tag similarity helper with comprehensive categorized tags."
import numpy as np
import librosa

_clap = None
_text_embeddings = None

# Comprehensive tag taxonomy organized by category (29 tags total)
CLAP_TAG_CATEGORIES = {
    "genre": [
        "orchestral",
        "electronic",
        "hybrid orchestral",
        "rock",
        "jazz",
        "classical",
        "minimalist",
        "ambient",
        "cinematic percussion"
    ],
    "production": [
        "polished production",
        "raw production",
        "vintage sound",
        "modern production",
        "acoustic recording",
        "heavily processed"
    ],
    "energy": [
        "high energy",
        "low energy",
        "building tension",
        "climactic",
        "gentle",
        "aggressive"
    ],
    "era": [
        "retro",
        "futuristic",
        "timeless",
        "modern"
    ],
    "function": [
        "action sequence",
        "dramatic underscore",
        "theme music",
        "transitional"
    ]
}

# Flatten tags for embedding (preserves order by category)
_ALL_TAGS = [tag for category in CLAP_TAG_CATEGORIES.values() for tag in category]


def tag_chunk(chunk, sr: int) -> dict[str, dict[str, float]]:
    """
    Analyze audio chunk and return categorized CLAP tag similarities.

    Args:
        chunk: Audio data (mono numpy array)
        sr: Sample rate

    Returns:
        Dictionary organized by category:
        {
            "genre": {"orchestral": 0.89, "electronic": 0.45, ...},
            "production": {"polished production": 0.68, ...},
            "energy": {"building tension": 0.76, ...},
            "era": {"timeless": 0.54, ...},
            "function": {"theme music": 0.62, ...}
        }
    """
    global _clap, _text_embeddings

    # Lazy load CLAP model and pre-compute text embeddings
    if _clap is None:
        import laion_clap
        _clap = laion_clap.CLAP_Module(enable_fusion=False)
        _clap.load_ckpt()
        # Pre-compute text embeddings once (performance optimization)
        _text_embeddings = _clap.get_text_embedding(_ALL_TAGS)

    # Resample to 48kHz if needed (CLAP requirement)
    if sr != 48_000:
        chunk = librosa.resample(y=chunk, orig_sr=sr, target_sr=48_000)
        sr = 48_000

    # Compute audio embedding
    audio_emb = _clap.get_audio_embedding_from_data(chunk.reshape(1, -1))[0]

    # Compute similarities using pre-computed text embeddings
    eps = 1e-8
    audio_norm = np.linalg.norm(audio_emb) + eps
    text_norms = np.linalg.norm(_text_embeddings, axis=1) + eps
    similarities = (audio_emb @ _text_embeddings.T) / (audio_norm * text_norms)

    # Organize results by category
    result = {}
    tag_idx = 0
    for category, tags in CLAP_TAG_CATEGORIES.items():
        result[category] = {}
        for tag in tags:
            result[category][tag] = float(similarities[tag_idx])
            tag_idx += 1

    return result
