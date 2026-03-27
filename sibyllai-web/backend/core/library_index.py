from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

DEFAULT_WINDOW_SIZES = [15.0, 30.0, 60.0, 120.0]  # seconds
DEFAULT_HOP_RATIO = 0.5
DEFAULT_DB_PATH = Path("temp/library_index.sqlite")

BPM_MIN = 40.0
BPM_MAX = 200.0
VALENCE_SCALE = 5.0

PITCH_CLASS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


@dataclass(frozen=True)
class FeatureSchema:
    clap_order: list[str]
    instrument_labels: list[str]
    genre_labels: list[str]
    key_order: list[str]

    @property
    def vector_dim(self) -> int:
        return (
            1  # bpm
            + len(self.key_order)
            + 2  # valence, arousal
            + len(self.clap_order)
            + len(self.instrument_labels)
            + len(self.genre_labels)
        )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _get_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def _ensure_track_columns(conn: sqlite3.Connection) -> None:
    cols = _get_table_columns(conn, "tracks")
    if "filename" not in cols:
        conn.execute("ALTER TABLE tracks ADD COLUMN filename TEXT")
    if "size" not in cols:
        conn.execute("ALTER TABLE tracks ADD COLUMN size INTEGER")
    if "source_name" not in cols:
        conn.execute("ALTER TABLE tracks ADD COLUMN source_name TEXT")
    if "source_type" not in cols:
        conn.execute("ALTER TABLE tracks ADD COLUMN source_type TEXT")


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add structure-segmentation columns to windows table (backward-compatible)."""
    cols = _get_table_columns(conn, "windows")
    if "segment_type" not in cols:
        conn.execute("ALTER TABLE windows ADD COLUMN segment_type TEXT DEFAULT 'fixed'")
    if "section_index" not in cols:
        conn.execute("ALTER TABLE windows ADD COLUMN section_index INTEGER")
    if "total_sections" not in cols:
        conn.execute("ALTER TABLE windows ADD COLUMN total_sections INTEGER")


def _load_yamnet_class_names() -> list[str]:
    import pandas as pd
    import urllib.request
    from sibyllai_core.detectors import yamnet_segmenter as ys

    class_map_path = Path(ys.__file__).parent / "yamnet_class_map.csv"
    if not class_map_path.exists():
        class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
        _ensure_parent(class_map_path)
        urllib.request.urlretrieve(class_map_url, class_map_path)

    return pd.read_csv(class_map_path)["display_name"].tolist()


def _build_instrument_labels(class_names: list[str]) -> list[str]:
    keywords = [
        "piano", "guitar", "violin", "drum", "brass", "string",
        "trumpet", "saxophone", "flute", "clarinet", "trombone",
        "cello", "bass", "harp", "organ", "accordion", "harmonica",
        "synthesizer", "keyboard", "percussion",
        "singing", "humming", "vocal", "choir", "rapping", "chant",
        "speech", "whistling", "beatbox",
    ]
    labels = []
    for name in class_names:
        lower = name.lower()
        if any(k in lower for k in keywords):
            labels.append(name)
    return labels


def _build_genre_labels(class_names: list[str]) -> list[str]:
    genre_keywords = [
        "pop music", "hip hop", "rock music", "heavy metal", "punk rock",
        "grunge", "progressive rock", "rock and roll", "psychedelic",
        "rhythm and blues", "soul music", "reggae", "country", "swing",
        "bluegrass", "funk", "folk music", "middle eastern", "jazz",
        "disco", "classical music", "opera", "electronic music", "house music",
        "techno", "dubstep", "drum and bass", "electronica", "electronic dance",
        "ambient music", "trance", "latin", "salsa", "flamenco", "blues",
        "music for children", "new-age", "vocal music", "a capella",
        "afrobeat", "christian music", "gospel", "ska", "traditional music",
        "indie", "soundtrack", "theme music", "lullaby", "video game music",
        "christmas music", "dance music", "wedding music",
        "happy music", "sad music", "tender music", "exciting music",
        "angry music", "scary music",
    ]

    labels = []
    for name in class_names:
        lower = name.lower()
        if any(k in lower for k in genre_keywords):
            labels.append(name)
    return labels


def _build_schema() -> FeatureSchema:
    from sibyllai_core.detectors.clap import CLAP_TAG_CATEGORIES
    clap_order = [tag for _, tags in CLAP_TAG_CATEGORIES.items() for tag in tags]
    class_names = _load_yamnet_class_names()
    instrument_labels = _build_instrument_labels(class_names)
    genre_labels = _build_genre_labels(class_names)
    key_order = [f"{pc}_major" for pc in PITCH_CLASS] + [f"{pc}_minor" for pc in PITCH_CLASS]
    return FeatureSchema(
        clap_order=clap_order,
        instrument_labels=instrument_labels,
        genre_labels=genre_labels,
        key_order=key_order,
    )


def _normalize_key(key: str | None) -> tuple[str, str] | None:
    if not key or key == "Unknown":
        return None
    parts = key.replace("-", "b").split()
    tonic = parts[0]
    tonic = FLAT_TO_SHARP.get(tonic, tonic)
    mode = parts[1].lower() if len(parts) > 1 else "major"
    if mode not in {"major", "minor"}:
        mode = "major"
    if tonic not in PITCH_CLASS:
        return None
    return tonic, mode


def normalize_key(key: str | None) -> tuple[str, str] | None:
    return _normalize_key(key)


def _key_one_hot(key: str | None, schema: FeatureSchema) -> list[float]:
    vec = [0.0] * len(schema.key_order)
    norm = _normalize_key(key)
    if not norm:
        return vec
    tonic, mode = norm
    label = f"{tonic}_{mode}"
    try:
        idx = schema.key_order.index(label)
    except ValueError:
        return vec
    vec[idx] = 1.0
    return vec


def _scale_bpm(bpm: float | str | None) -> float:
    if bpm is None or bpm == "Unknown":
        return 0.0
    try:
        value = float(bpm)
    except (TypeError, ValueError):
        return 0.0
    value = max(BPM_MIN, min(BPM_MAX, value))
    return (value - BPM_MIN) / (BPM_MAX - BPM_MIN)


def _scale_valence_arousal(value: float | None) -> float:
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    v = max(0.0, min(VALENCE_SCALE, v))
    return v / VALENCE_SCALE


def _clap_vector(clap_categorized: dict[str, dict[str, float]], schema: FeatureSchema) -> list[float]:
    from sibyllai_core.detectors.clap import CLAP_TAG_CATEGORIES
    out: list[float] = []
    for category, tags in CLAP_TAG_CATEGORIES.items():
        cat_scores = clap_categorized.get(category, {})
        for tag in tags:
            out.append(float(cat_scores.get(tag, 0.0)))
    return out


def _dict_vector(scores: dict[str, float], labels: list[str]) -> list[float]:
    return [float(scores.get(label, 0.0)) for label in labels]


def build_vector_from_features(features: dict, schema: FeatureSchema, include_moods: bool = True) -> list[float]:
    bpm_scaled = _scale_bpm(features.get("bpm"))
    key_vec = _key_one_hot(features.get("key"), schema)
    if include_moods:
        valence = _scale_valence_arousal(features.get("valence"))
        arousal = _scale_valence_arousal(features.get("arousal"))
    else:
        valence = 0.0
        arousal = 0.0
    clap_vec = _clap_vector(features.get("clap", {}), schema)
    inst_vec = _dict_vector(features.get("instruments", {}), schema.instrument_labels)
    genre_vec = _dict_vector(features.get("genres", {}), schema.genre_labels)

    return [bpm_scaled, *key_vec, valence, arousal, *clap_vec, *inst_vec, *genre_vec]


def build_vector_from_cue(cue: dict, schema: FeatureSchema, include_moods: bool = True) -> list[float]:
    profile = cue.get("musical_profile", {})
    detected = profile.get("detected", {})
    features = {
        "bpm": profile.get("bpm"),
        "key": profile.get("key"),
        "valence": profile.get("valence"),
        "arousal": profile.get("arousal"),
        "instruments": detected.get("instruments_yamnet", {}),
        "genres": detected.get("genres_yamnet", {}),
        "clap": {
            "genre": detected.get("clap_style", {}) or detected.get("clap_genre", {}),
            "instrumentation": detected.get("clap_instrumentation", {}),
            "production": detected.get("clap_production", {}),
            "energy": detected.get("clap_energy", {}),
            "era": detected.get("clap_era", {}),
            "function": detected.get("clap_function", {}),
        },
    }
    return build_vector_from_features(features, schema, include_moods=include_moods)


def build_features_from_cue(cue: dict) -> dict:
    profile = cue.get("musical_profile", {})
    detected = profile.get("detected", {})
    return {
        "bpm": profile.get("bpm"),
        "key": profile.get("key"),
        "valence": profile.get("valence"),
        "arousal": profile.get("arousal"),
        "instruments": detected.get("instruments_yamnet", {}),
        "genres": detected.get("genres_yamnet", {}),
        "clap": {
            "genre": detected.get("clap_style", {}) or detected.get("clap_genre", {}),
            "instrumentation": detected.get("clap_instrumentation", {}),
            "production": detected.get("clap_production", {}),
            "energy": detected.get("clap_energy", {}),
            "era": detected.get("clap_era", {}),
            "function": detected.get("clap_function", {}),
        },
    }


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _iter_windows(duration: float, window_sizes: list[float], hop_ratio: float) -> Iterable[tuple[float, float, float]]:
    if duration <= 0:
        return []
    min_window = min(window_sizes)
    if duration < min_window:
        return [(0.0, duration, duration)]

    windows: list[tuple[float, float, float]] = []
    for size in window_sizes:
        hop = max(1.0, size * hop_ratio)
        start = 0.0
        while start + size <= duration:
            windows.append((start, start + size, size))
            start += hop
    return windows


def _iter_sections(
    audio: np.ndarray,
    sr: int,
    min_section_length: float = 8.0,
    sensitivity: float = 1.0,
) -> list[tuple[float, float]]:
    """Detect structural sections and return (start, end) pairs."""
    from sibyllai_core.detectors.structure import detect_sections_with_fallback

    sections = detect_sections_with_fallback(
        audio, sr,
        min_section_length=min_section_length,
        sensitivity=sensitivity,
    )
    return [(sec.start, sec.end) for sec in sections]


def _load_audio(file_path: Path) -> tuple[np.ndarray, int]:
    import librosa
    import soundfile as sf
    try:
        audio, sr = sf.read(str(file_path))
    except Exception:
        audio, sr = librosa.load(str(file_path), sr=None, mono=False)

    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T)
    return audio, int(sr)


def _bpm_track(y: np.ndarray, sr: int) -> float:
    import librosa
    import essentia.standard as es
    if y.ndim > 1:
        y = librosa.to_mono(y.T)
    if sr != 44100:
        y = librosa.resample(y, orig_sr=sr, target_sr=44100)
        sr = 44100
    return es.RhythmExtractor2013(method="multifeature")(y)[0]


def _extract_features_for_window(
    chunk: np.ndarray,
    sr: int,
    temp_dir: Path,
    include_moods: bool = True,
) -> dict:
    features: dict = {
        "bpm": "Unknown",
        "key": "Unknown",
        "valence": 0.0,
        "arousal": 0.0,
        "instruments": {},
        "genres": {},
        "clap": {},
    }

    # BPM
    try:
        features["bpm"] = _bpm_track(chunk, sr)
    except Exception:
        pass

    # Instruments / Genres
    try:
        from sibyllai_core.detectors.yamnet_segmenter import extract_instruments
        features["instruments"] = extract_instruments(chunk, sr=sr, top_n=9999)
    except Exception:
        pass

    try:
        from sibyllai_core.detectors.yamnet_segmenter import extract_genres
        features["genres"] = extract_genres(chunk, sr=sr, top_n=9999)
    except Exception:
        pass

    # CLAP tags
    try:
        from sibyllai_core.detectors.clap import tag_chunk
        features["clap"] = tag_chunk(chunk, sr)
    except Exception:
        pass

    # Key
    try:
        from sibyllai_core.detectors.chord_detector import analyze_chords
        chord_analysis = analyze_chords(audio_data=chunk, sr=sr)
        features["key"] = chord_analysis.get("key", "Unknown")
    except Exception:
        pass

    # Mood / Valence / Arousal
    if include_moods:
        try:
            from sibyllai_core.detectors.m2e_wrapper import global_moods
            import soundfile as sf
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"mood_{uuid.uuid4().hex}.wav"
            sf.write(str(temp_path), chunk, sr)
            mood_result = global_moods(str(temp_path))
            features["valence"] = mood_result.get("valence", 0.0)
            features["arousal"] = mood_result.get("arousal", 0.0)
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass

    return features


def init_index(
    db_path: Path,
    schema: FeatureSchema,
    reset: bool = True,
    include_moods: bool = True,
    window_sizes: list[float] | None = None,
    hop_ratio: float | None = None,
    source_name: str | None = None,
    source_type: str | None = None,
    segmentation_mode: str = "fixed",
    min_section_length: float = 8.0,
    sensitivity: float = 1.0,
) -> None:
    _ensure_parent(db_path)
    if reset and db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tracks (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                duration REAL NOT NULL,
                added_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS windows (
                id TEXT PRIMARY KEY,
                track_id TEXT NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                duration REAL NOT NULL,
                window_size REAL NOT NULL,
                bpm REAL,
                key TEXT,
                valence REAL,
                arousal REAL,
                vector_json TEXT NOT NULL,
                features_json TEXT NOT NULL,
                segment_type TEXT DEFAULT 'fixed',
                section_index INTEGER,
                total_sections INTEGER,
                FOREIGN KEY(track_id) REFERENCES tracks(id)
            )
            """
        )

        meta = {
            "schema_version": 2,
            "clap_order": schema.clap_order,
            "instrument_labels": schema.instrument_labels,
            "genre_labels": schema.genre_labels,
            "key_order": schema.key_order,
            "bpm_min": BPM_MIN,
            "bpm_max": BPM_MAX,
            "valence_scale": VALENCE_SCALE,
            "include_moods": include_moods,
            "window_sizes": window_sizes or DEFAULT_WINDOW_SIZES,
            "hop_ratio": hop_ratio if hop_ratio is not None else DEFAULT_HOP_RATIO,
            "built_at": datetime.utcnow().isoformat(),
            "source_name": source_name,
            "source_type": source_type,
            "segmentation_mode": segmentation_mode,
            "min_section_length": min_section_length,
            "sensitivity": sensitivity,
        }
        for key, value in meta.items():
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, json.dumps(value)),
            )


def build_library_index(
    folder: Path,
    db_path: Path = DEFAULT_DB_PATH,
    window_sizes: list[float] | None = None,
    hop_ratio: float = DEFAULT_HOP_RATIO,
    include_moods: bool = True,
    reset: bool = True,
    progress_callback: Callable[[dict], None] | None = None,
    source_name: str | None = None,
    source_type: str | None = None,
    dedupe_simple: bool = True,
    segmentation_mode: str = "fixed",
    min_section_length: float = 8.0,
    sensitivity: float = 1.0,
) -> dict:
    schema = _build_schema()
    window_sizes = window_sizes or DEFAULT_WINDOW_SIZES
    init_index(
        db_path,
        schema,
        reset=reset,
        include_moods=include_moods,
        window_sizes=window_sizes,
        hop_ratio=hop_ratio,
        source_name=source_name,
        source_type=source_type,
        segmentation_mode=segmentation_mode,
        min_section_length=min_section_length,
        sensitivity=sensitivity,
    )

    audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".aiff", ".aif"}
    files = [p for p in folder.rglob("*") if p.suffix.lower() in audio_exts]

    track_count = 0
    window_count = 0
    temp_dir = Path("temp/library_moods")

    if progress_callback:
        progress_callback({
            "status": "running",
            "phase": "scanning",
            "current_track": 0,
            "total_tracks": len(files),
            "windows_indexed": 0,
            "message": "Scanning library",
        })

    with sqlite3.connect(db_path) as conn:
        _ensure_track_columns(conn)
        _migrate_schema(conn)
        existing_keys: set[tuple[str, int]] = set()
        if dedupe_simple and not reset:
            cols = _get_table_columns(conn, "tracks")
            if "filename" in cols and "size" in cols:
                existing_keys = {
                    (row[0], row[1])
                    for row in conn.execute(
                        "SELECT filename, size FROM tracks WHERE filename IS NOT NULL AND size IS NOT NULL"
                    ).fetchall()
                }

        for idx, path in enumerate(files, start=1):
            file_name = path.name
            file_size = path.stat().st_size
            if dedupe_simple and not reset and (file_name, file_size) in existing_keys:
                if progress_callback:
                    progress_callback({
                        "status": "running",
                        "phase": "indexing",
                        "current_track": idx,
                        "total_tracks": len(files),
                        "windows_indexed": window_count,
                        "message": f"Skipped duplicate ({idx}/{len(files)})",
                    })
                continue
            try:
                audio, sr = _load_audio(path)
            except Exception:
                if progress_callback:
                    progress_callback({
                        "status": "running",
                        "phase": "indexing",
                        "current_track": idx,
                        "total_tracks": len(files),
                        "windows_indexed": window_count,
                        "message": f"Skipped unreadable file ({idx}/{len(files)})",
                    })
                continue

            duration = len(audio) / sr if sr > 0 else 0.0
            if duration <= 0.0:
                continue

            track_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO tracks (id, path, duration, added_at, filename, size, source_name, source_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    track_id,
                    str(path.resolve()),
                    duration,
                    datetime.utcnow().isoformat(),
                    file_name,
                    file_size,
                    source_name,
                    source_type,
                ),
            )
            track_count += 1
            if dedupe_simple and not reset:
                existing_keys.add((file_name, file_size))

            if segmentation_mode == "structure":
                sections = _iter_sections(audio, sr, min_section_length, sensitivity)
                total_sections = len(sections)
                for sec_idx, (sec_start, sec_end) in enumerate(sections):
                    start_idx = int(sec_start * sr)
                    end_idx = int(sec_end * sr)
                    chunk = audio[start_idx:end_idx]
                    if len(chunk) == 0:
                        continue

                    features = _extract_features_for_window(chunk, sr, temp_dir, include_moods=include_moods)
                    vector = build_vector_from_features(features, schema, include_moods=include_moods)

                    conn.execute(
                        """
                        INSERT INTO windows (
                            id, track_id, start, end, duration, window_size,
                            bpm, key, valence, arousal,
                            vector_json, features_json,
                            segment_type, section_index, total_sections
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            uuid.uuid4().hex,
                            track_id,
                            sec_start,
                            sec_end,
                            sec_end - sec_start,
                            -1.0,  # variable-length sections
                            features.get("bpm") if features.get("bpm") != "Unknown" else None,
                            features.get("key"),
                            features.get("valence"),
                            features.get("arousal"),
                            json.dumps(vector),
                            json.dumps(features),
                            "structure",
                            sec_idx,
                            total_sections,
                        ),
                    )
                    window_count += 1
            else:
                windows = _iter_windows(duration, window_sizes, hop_ratio)
                for start, end, size in windows:
                    start_idx = int(start * sr)
                    end_idx = int(end * sr)
                    chunk = audio[start_idx:end_idx]
                    if len(chunk) == 0:
                        continue

                    features = _extract_features_for_window(chunk, sr, temp_dir, include_moods=include_moods)
                    vector = build_vector_from_features(features, schema, include_moods=include_moods)

                    conn.execute(
                        """
                        INSERT INTO windows (
                            id, track_id, start, end, duration, window_size,
                            bpm, key, valence, arousal,
                            vector_json, features_json,
                            segment_type, section_index, total_sections
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            uuid.uuid4().hex,
                            track_id,
                            start,
                            end,
                            end - start,
                            size,
                            features.get("bpm") if features.get("bpm") != "Unknown" else None,
                            features.get("key"),
                            features.get("valence"),
                            features.get("arousal"),
                            json.dumps(vector),
                            json.dumps(features),
                            "fixed",
                            None,
                            None,
                        ),
                    )
                    window_count += 1

            if progress_callback:
                progress_callback({
                    "status": "running",
                    "phase": "indexing",
                    "current_track": idx,
                    "total_tracks": len(files),
                    "windows_indexed": window_count,
                    "message": f"Indexed {idx}/{len(files)} tracks",
                })

        conn.commit()

    return {
        "tracks_indexed": track_count,
        "windows_indexed": window_count,
        "db_path": str(db_path),
        "vector_dim": schema.vector_dim,
        "source_name": source_name,
        "source_type": source_type,
        "segmentation_mode": segmentation_mode,
    }


def load_schema_from_db(db_path: Path) -> FeatureSchema:
    meta = load_meta_from_db(db_path)
    return FeatureSchema(
        clap_order=meta.get("clap_order", []),
        instrument_labels=meta.get("instrument_labels", []),
        genre_labels=meta.get("genre_labels", []),
        key_order=meta.get("key_order", []),
    )


def load_meta_from_db(db_path: Path) -> dict:
    with sqlite3.connect(db_path) as conn:
        meta_rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {k: json.loads(v) for k, v in meta_rows}


def get_library_info(db_path: Path) -> dict:
    if not db_path.exists():
        return {
            "exists": False,
            "db_path": str(db_path),
            "tracks": 0,
            "windows": 0,
            "meta": {},
        }

    with sqlite3.connect(db_path) as conn:
        track_count = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        window_count = conn.execute("SELECT COUNT(*) FROM windows").fetchone()[0]

    meta = load_meta_from_db(db_path)
    return {
        "exists": True,
        "db_path": str(db_path),
        "tracks": track_count,
        "windows": window_count,
        "meta": meta,
        "segmentation_mode": meta.get("segmentation_mode", "fixed"),
    }


def get_library_sources(db_path: Path) -> list[dict]:
    """Return per-source stats: source_name, source_type, track_count, total_windows, total_duration, last_added."""
    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as conn:
        _ensure_track_columns(conn)
        rows = conn.execute(
            """
            SELECT
                t.source_name,
                t.source_type,
                COUNT(DISTINCT t.id) AS track_count,
                COUNT(w.id) AS total_windows,
                COALESCE(SUM(DISTINCT t.duration), 0) AS total_duration,
                MAX(t.added_at) AS last_added
            FROM tracks t
            LEFT JOIN windows w ON w.track_id = t.id
            GROUP BY t.source_name, t.source_type
            ORDER BY last_added DESC
            """,
        ).fetchall()

    return [
        {
            "source_name": row[0] or "Unknown",
            "source_type": row[1] or "unknown",
            "track_count": row[2],
            "total_windows": row[3],
            "total_duration": row[4],
            "last_added": row[5],
        }
        for row in rows
    ]


def _source_name_condition(source_name: str) -> tuple[str, tuple]:
    """Return SQL condition and params for matching source_name, handling NULL."""
    if source_name == "Unknown":
        return "(t.source_name IS NULL OR t.source_name = ?)", (source_name,)
    return "t.source_name = ?", (source_name,)


def get_library_tracks(db_path: Path, source_name: str) -> list[dict]:
    """Return tracks for a given source with window counts."""
    if not db_path.exists():
        return []

    condition, params = _source_name_condition(source_name)
    with sqlite3.connect(db_path) as conn:
        _ensure_track_columns(conn)
        rows = conn.execute(
            f"""
            SELECT
                t.id,
                t.filename,
                t.path,
                t.duration,
                t.size,
                t.added_at,
                COUNT(w.id) AS window_count
            FROM tracks t
            LEFT JOIN windows w ON w.track_id = t.id
            WHERE {condition}
            GROUP BY t.id
            ORDER BY t.filename
            """,
            params,
        ).fetchall()

    return [
        {
            "id": row[0],
            "filename": row[1] or Path(row[2]).name,
            "path": row[2],
            "duration": row[3],
            "size": row[4],
            "added_at": row[5],
            "window_count": row[6],
        }
        for row in rows
    ]


def delete_library_source(db_path: Path, source_name: str) -> dict:
    """Delete all tracks and windows for a source. Returns counts of deleted items."""
    if not db_path.exists():
        return {"tracks_deleted": 0, "windows_deleted": 0}

    condition, params = _source_name_condition(source_name)

    with sqlite3.connect(db_path) as conn:
        _ensure_track_columns(conn)
        track_ids = [
            row[0]
            for row in conn.execute(
                f"SELECT t.id FROM tracks t WHERE {condition}", params
            ).fetchall()
        ]
        if not track_ids:
            return {"tracks_deleted": 0, "windows_deleted": 0}

        placeholders = ",".join("?" for _ in track_ids)
        windows_deleted = conn.execute(
            f"DELETE FROM windows WHERE track_id IN ({placeholders})", track_ids
        ).rowcount
        tracks_deleted = conn.execute(
            f"DELETE FROM tracks WHERE id IN ({placeholders})", track_ids
        ).rowcount
        conn.commit()

    return {"tracks_deleted": tracks_deleted, "windows_deleted": windows_deleted}


def match_cue_to_library(
    cue_vector: list[float],
    cue_length: float,
    db_path: Path = DEFAULT_DB_PATH,
    top_n: int = 5,
    window_sizes: list[float] | None = None,
    unique_tracks: bool = True,
    include_features: bool = False,
    duration_weight: float = 0.1,
    duration_tolerance: float = 2.0,
) -> list[dict]:
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    # Auto-detect segmentation mode from DB meta
    meta = load_meta_from_db(db_path)
    seg_mode = meta.get("segmentation_mode", "fixed")

    _select_cols = """
        w.id, w.track_id, w.start, w.end, w.duration, w.window_size,
        w.vector_json, w.features_json, t.path
    """

    with sqlite3.connect(db_path) as conn:
        _migrate_schema(conn)

        if seg_mode == "structure":
            # Query all structure-type windows (variable duration)
            rows = conn.execute(
                f"""
                SELECT {_select_cols},
                       w.segment_type, w.section_index, w.total_sections
                FROM windows w
                JOIN tracks t ON t.id = w.track_id
                WHERE w.segment_type = 'structure'
                """,
            ).fetchall()
        else:
            window_sizes = window_sizes or DEFAULT_WINDOW_SIZES
            target = min(window_sizes, key=lambda w: abs(w - cue_length))
            rows = conn.execute(
                f"""
                SELECT {_select_cols},
                       w.segment_type, w.section_index, w.total_sections
                FROM windows w
                JOIN tracks t ON t.id = w.track_id
                WHERE w.window_size = ?
                """,
                (target,),
            ).fetchall()

    scored = []
    for row in rows:
        vector = json.loads(row[6])
        cosine_score = _cosine_similarity(cue_vector, vector)

        # Duration similarity penalty for structure mode
        if seg_mode == "structure" and duration_weight > 0 and cue_length > 0:
            win_duration = row[4]
            if win_duration > 0:
                ratio = max(cue_length, win_duration) / min(cue_length, win_duration)
                duration_penalty = max(0.0, 1.0 - (ratio - 1.0) / duration_tolerance)
            else:
                duration_penalty = 0.0
            final_score = (1.0 - duration_weight) * cosine_score + duration_weight * duration_penalty
        else:
            final_score = cosine_score

        entry = {
            "window_id": row[0],
            "track_id": row[1],
            "track_path": row[8],
            "start": row[2],
            "end": row[3],
            "duration": row[4],
            "window_size": row[5],
            "score": final_score,
            "segment_type": row[9],
            "section_index": row[10],
            "total_sections": row[11],
        }
        if include_features:
            entry["features"] = json.loads(row[7])
        scored.append(entry)

    scored.sort(key=lambda x: x["score"], reverse=True)
    if not unique_tracks:
        return scored[:top_n]

    filtered = []
    seen = set()
    for item in scored:
        if item["track_id"] in seen:
            continue
        seen.add(item["track_id"])
        filtered.append(item)
        if len(filtered) >= top_n:
            break
    return filtered
