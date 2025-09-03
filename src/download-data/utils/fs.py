from pathlib import Path
from typing import Set
from .config import RAW_DIR, FILE_REGEX

def ensure_dirs() -> None:
    """Create required directory structure if it does not exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "sleep-cassette").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "sleep-telemetry").mkdir(parents=True, exist_ok=True)

def local_files_for_subset(subset: str) -> Set[str]:
    """Return the set of valid existing file names in the local subset directory."""
    subdir = RAW_DIR / f"sleep-{subset}"
    if not subdir.is_dir():
        return set()
    return {
        fn.name
        for fn in subdir.iterdir()
        if fn.is_file() and FILE_REGEX.search(fn.name)
    }
