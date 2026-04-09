"LAION-CLAP tag similarity helper with comprehensive categorized tags."
import numpy as np
import librosa

_clap = None
_text_embeddings = None

# Comprehensive tag taxonomy organized by category (37 tags total).
# Each tag is either a plain string (used as both label and CLAP prompt)
# or a (label, prompt) tuple where the prompt is a more descriptive phrase
# that CLAP matches more accurately against audio embeddings.
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
        "cinematic percussion",
    ],
    "instrumentation": [
        ("brass section", "cinematic brass"),
        ("string ensemble", "string ensemble"),
        ("woodwinds", "woodwinds playing"),
        ("choir vocals", "choir vocals singing"),
        ("piano keys", "piano keys"),
        ("guitar bass", "guitar bass"),
        ("synthesizers", "synthesizers"),
        ("full orchestra", "strings and brass orchestra"),
    ],
    "production": [
        "polished production",
        "raw production",
        "vintage sound",
        "modern production",
        "acoustic recording",
        "heavily processed",
    ],
    "energy": [
        "high energy",
        "low energy",
        "building tension",
        "climactic",
        "gentle",
        "aggressive",
    ],
    "era": [
        "retro",
        "futuristic",
        "timeless",
        "modern",
    ],
    "function": [
        "action sequence",
        "dramatic underscore",
        "theme music",
        "transitional",
    ],
}


def _resolve_tag(tag):
    """Return (display_label, clap_prompt) for a tag entry."""
    if isinstance(tag, tuple):
        return tag
    return (tag, tag)


# Flatten: extract (label, prompt) pairs preserving order by category.
# _ALL_PROMPTS are sent to CLAP; _ALL_LABELS are used for output keys.
_ALL_LABELS = [_resolve_tag(tag)[0] for category in CLAP_TAG_CATEGORIES.values() for tag in category]
_ALL_PROMPTS = [_resolve_tag(tag)[1] for category in CLAP_TAG_CATEGORIES.values() for tag in category]


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
            "instrumentation": {"brass section": 0.32, ...},
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
        # Pre-compute text embeddings using descriptive prompts (performance optimization)
        _text_embeddings = _clap.get_text_embedding(_ALL_PROMPTS)

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

    # Organize results by category, using display labels as keys
    result = {}
    tag_idx = 0
    for category, tags in CLAP_TAG_CATEGORIES.items():
        result[category] = {}
        for tag in tags:
            label, _ = _resolve_tag(tag)
            result[category][label] = float(similarities[tag_idx])
            tag_idx += 1

    return result
