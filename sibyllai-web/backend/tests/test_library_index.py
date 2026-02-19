import json
import sqlite3
from pathlib import Path

from core.library_index import _iter_windows, match_cue_to_library


def test_iter_windows_short_duration():
    windows = _iter_windows(duration=8.0, window_sizes=[15.0, 30.0], hop_ratio=0.5)
    assert windows == [(0.0, 8.0, 8.0)]


def test_iter_windows_overlap_counts():
    windows = _iter_windows(duration=60.0, window_sizes=[15.0], hop_ratio=0.5)
    # 15s windows with 7.5s hop: starts at 0, 7.5, 15, 22.5, 30, 37.5, 45
    assert len(windows) == 7
    assert windows[0] == (0.0, 15.0, 15.0)
    assert windows[-1] == (45.0, 60.0, 15.0)


def test_match_cue_to_library_sorted(tmp_path: Path):
    db_path = tmp_path / "library.sqlite"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE tracks (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                duration REAL NOT NULL,
                added_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE windows (
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
                features_json TEXT NOT NULL
            )
            """
        )

        conn.execute(
            "INSERT INTO tracks (id, path, duration, added_at) VALUES (?, ?, ?, ?)",
            ("t1", "/tmp/track_one.wav", 120.0, "now"),
        )
        conn.execute(
            "INSERT INTO tracks (id, path, duration, added_at) VALUES (?, ?, ?, ?)",
            ("t2", "/tmp/track_two.wav", 120.0, "now"),
        )

        # Two windows, same size, different vectors
        conn.execute(
            """
            INSERT INTO windows (id, track_id, start, end, duration, window_size, vector_json, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("w1", "t1", 0.0, 15.0, 15.0, 15.0, json.dumps([1.0, 0.0, 0.0]), "{}"),
        )
        conn.execute(
            """
            INSERT INTO windows (id, track_id, start, end, duration, window_size, vector_json, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("w2", "t2", 0.0, 15.0, 15.0, 15.0, json.dumps([0.0, 1.0, 0.0]), "{}"),
        )
        conn.commit()

    cue_vector = [1.0, 0.0, 0.0]
    results = match_cue_to_library(cue_vector=cue_vector, cue_length=14.0, db_path=db_path, top_n=2)
    assert results[0]["track_id"] == "t1"
    assert results[0]["score"] >= results[1]["score"]
