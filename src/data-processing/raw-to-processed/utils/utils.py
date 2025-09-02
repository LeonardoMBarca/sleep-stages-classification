import re
from typing import List, Optional
from rich.progress import (
    Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)

def _make_progress() -> Progress:
    """Create a standard Progress instance for console logging."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False
    )

def slugify(s: str) -> str:
    """Normalizes a channel label to something stable: lowercase, alphanumeric+underscore."""
    try:
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-z0-9_]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s
    except Exception:
        return str(s) if s is not None else ""

def normspace(s: str) -> str:
    try:
        return re.sub(r"\s+", " ", (s or "").strip())
    except Exception:
        return s or ""

def best_match_idx(pool: List[str], candidates: List[str]) -> Optional[int]:
    """Flexible match (slugify) between any item in the pool and the list of candidates."""
    try:
        pool_slug = [slugify(p) for p in pool]
        cand_slug = [slugify(c) for c in candidates]

        for cs in cand_slug:
            if cs in pool_slug:
                return pool_slug.index(cs)

        for i, ps in enumerate(pool_slug):
            for cs in cand_slug:
                if cs in ps or ps in cs:
                    return i

        return None
    except Exception:
        return None
