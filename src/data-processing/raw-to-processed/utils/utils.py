import re
import polars as pl
from typing import List, Optional
from pathlib import Path
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

def _pair_key(psg_path: str, hyp_path: str) -> tuple[str, str]:
    """Stable BaseNames (for save shards and registry in hash)."""
    from os.path import basename
    return basename(psg_path), basename(hyp_path)

def _subset_from_sid(subject_id: str) -> str:
    sid = (subject_id or "").strip().upper()
    if sid.startswith("SC"):
        return "sleep-cassette"
    if sid.startswith("ST"):
        return "sleep-telemetry"
    return "sleep-cassette"

def _subset_from_sid(subject_id: str) -> str:
    sid = (subject_id or "").strip().upper()
    if sid.startswith("SC"):
        return "sleep-cassette"
    if sid.startswith("ST"):
        return "sleep-telemetry"
    return "sleep-cassette"

def _write_shard(logger, df: pl.DataFrame, out_root: Path) -> Path | None:
    """
    Writes one shard per pair using subject-only partitioning and SC/ST-style filenames.
    """
    try:
        if df is None or df.is_empty():
            return None
        sid = str(df["subject_id"][0])
        nid = str(df["night_id"][0])
        subset = _subset_from_sid(sid)
        out_dir = Path(out_root) / subset / f"subject_id={sid}"
        out_dir.mkdir(parents=True, exist_ok=True)
        shard = out_dir / f"{sid}{nid}.parquet"
        df.write_parquet(shard)
        logger.log(f"[WRITE_SHARD] Saved: {shard} (rows={df.height})")
        return shard
    except Exception as e:
        logger.log(f"[WRITE_SHARD] Failed: {e}", "error")
        return None

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
