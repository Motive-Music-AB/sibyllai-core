import os
from pathlib import Path

def get_incremental_path(out_dir, base_name):
    """
    Returns a Path in out_dir with base_name, incrementing _1, _2, etc. if needed.
    """
    out_dir = Path(out_dir)
    stem, ext = os.path.splitext(base_name)
    candidate = out_dir / base_name
    i = 1
    while candidate.exists():
        candidate = out_dir / f"{stem}_{i}{ext}"
        i += 1
    return candidate 