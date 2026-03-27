"""
SibyllAI project file format (.sibyl.json) specification and utilities.

This module defines the structure for saving and loading music analysis projects.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Any


def timecode_from_seconds(seconds: float, fps: int = 24) -> str:
    """
    Convert seconds to timecode format HH:MM:SS:FF.

    Args:
        seconds: Time in seconds
        fps: Frames per second (default 24)

    Returns:
        Timecode string in format "HH:MM:SS:FF"
    """
    frames_total = int(round(seconds * fps))
    h = frames_total // (3600 * fps)
    m = (frames_total % (3600 * fps)) // (60 * fps)
    s = (frames_total % (60 * fps)) // fps
    f = frames_total % fps
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def create_cue(
    cue_id: str,
    start: float,
    end: float,
    fps: int,
    bpm: float | str,
    key: str,
    instruments: dict[str, float],
    genres: dict[str, float],
    moods: list[str],
    valence: float,
    arousal: float,
    clap_categorized: dict[str, dict[str, float]],
    sections: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Create a cue data structure with musical_profile and project_context.

    Args:
        cue_id: Unique identifier (e.g., "cue_001")
        start/end: Time in seconds
        fps: Frames per second for timecode
        bpm: Detected BPM or "Unknown"
        key: Detected key (e.g., "C Major")
        instruments: Dict of instrument name -> confidence score (YAMNet)
        genres: Dict of genre name -> confidence score (YAMNet)
        moods: List of mood tags
        valence/arousal: Emotional dimensions (0-1)
        clap_categorized: Categorized CLAP tags (film-scoring vocabulary)

    Returns:
        Cue dictionary matching the MVP data model
    """
    # Extract top items per category for curated section
    def get_top_n(items: dict[str, float], n: int) -> list[str]:
        sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:n]]

    curated_instruments = get_top_n(instruments, 8) if instruments else []
    curated_genres = get_top_n(genres, 3) if genres else []  # Top 3 YAMNet genres
    curated_moods = moods[:2] if moods else []  # Top 2 moods

    # Extract top tags per CLAP category
    curated_clap = {}
    for category, tags in clap_categorized.items():
        if category == "genre":
            curated_clap["genre"] = get_top_n(tags, 2)
        elif category == "instrumentation":
            curated_clap["instrumentation"] = get_top_n(tags, 3)
        elif category == "production":
            curated_clap["production"] = get_top_n(tags, 1)
        elif category == "energy":
            curated_clap["energy"] = get_top_n(tags, 2)
        elif category == "era":
            curated_clap["era"] = get_top_n(tags, 1)
        elif category == "function":
            curated_clap["function"] = get_top_n(tags, 1)

    return {
        "id": cue_id,
        "name": "",  # User-editable
        "start": start,
        "end": end,
        "start_tc": timecode_from_seconds(start, fps),
        "end_tc": timecode_from_seconds(end, fps),

        "musical_profile": {
            "detected": {
                "instruments_yamnet": instruments,
                "genres_yamnet": genres,  # YAMNet genre detection (pop, rock, jazz, etc.)
                "moods": {mood: 1.0 for mood in moods},  # Music2Emo doesn't provide scores per mood
                "clap_style": clap_categorized.get("genre", {}),  # CLAP film-scoring style (orchestral, hybrid, etc.)
                "clap_instrumentation": clap_categorized.get("instrumentation", {}),
                "clap_production": clap_categorized.get("production", {}),
                "clap_energy": clap_categorized.get("energy", {}),
                "clap_era": clap_categorized.get("era", {}),
                "clap_function": clap_categorized.get("function", {}),
            },
            "curated": {
                "instruments": curated_instruments,
                "genres": curated_genres,  # YAMNet genres (pop, rock, electronic, etc.)
                "moods": curated_moods,
                "style": curated_clap.get("genre", []),  # CLAP film-scoring style (orchestral, hybrid, cinematic)
                "instrumentation": curated_clap.get("instrumentation", []),
                "production": curated_clap.get("production", []),
                "energy": curated_clap.get("energy", []),
                "era": curated_clap.get("era", []),
                "function": curated_clap.get("function", []),
            },
            "bpm": bpm if bpm != "Unknown" else None,
            "key": key if key != "Unknown" else None,
            "valence": valence,
            "arousal": arousal,
            "sections": sections or [],
        },

        "project_context": {
            "custom_tags": [],
            "notes": "",
            "color": "#3B82F6",  # Default blue
            "status": "draft",
        }
    }


def create_project(
    mx_file_path: str | Path,
    cues: list[dict],
    project_name: str | None = None,
    fps: int = 24,
) -> dict[str, Any]:
    """
    Create a complete .sibyl project structure.

    Args:
        mx_file_path: Path to the MX audio file
        cues: List of cue dictionaries
        project_name: Optional project name (defaults to filename)
        fps: Frames per second

    Returns:
        Complete project dictionary
    """
    mx_path = Path(mx_file_path)
    if project_name is None:
        project_name = mx_path.stem

    return {
        "version": "1.0",
        "project": {
            "name": project_name,
            "mx_file": str(mx_path.absolute()),
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "fps": fps,
        },
        "cues": cues
    }


def save_project(project: dict, output_path: str | Path) -> None:
    """
    Save project to .sibyl.json file.

    Args:
        project: Project dictionary
        output_path: Path to save file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(project, f, indent=2, ensure_ascii=False)


def load_project(project_path: str | Path) -> dict:
    """
    Load project from .sibyl.json file.

    Args:
        project_path: Path to .sibyl.json file

    Returns:
        Project dictionary
    """
    with open(project_path, 'r', encoding='utf-8') as f:
        return json.load(f)
