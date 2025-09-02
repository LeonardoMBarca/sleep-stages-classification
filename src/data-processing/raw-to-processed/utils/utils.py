import re
from typing import List, Optional
try:
    from rich.progress import (
        Progress, TextColumn, BarColumn, TaskProgressColumn,
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
    )
    def _make_progress():
        return Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=False,
        )
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def _make_progress():
        class Dummy:
            def add_task(self, *a, **k): return 0
            def update(self, *a, **k): pass
        yield Dummy()

def _process_one_pair(args):
    """
    Runs in a child process: processes 1 pair (PSG, HYP).
    Does not use a progress bar to avoid cluttering stdout.
    Creates its own (lightweight) Logger within the worker.
    """
    psg, hyp, sid, nid = args
    try:
        from labeler import process_record as _proc
        from logger import Logger as _Logger
        _logger = _Logger()
        df = _proc(_logger, psg, hyp, subject_id=sid, night_id=nid, progress=None)
        return sid, df 
    except Exception as e:
        
        return sid, None

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
