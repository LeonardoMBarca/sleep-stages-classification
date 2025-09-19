import re
import numpy as np
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

def _downsample_emg_to_1hz_rms(x_epochs_100hz: np.ndarray, fs: float = 100.0) -> np.ndarray:
    x = np.asarray(x_epochs_100hz, dtype=np.float64)
    n_ep, n_samp = x.shape
    sec = int(round(fs))
    usable = (n_samp // sec) * sec       
    x = x[:, :usable].reshape(n_ep, -1, sec) 
    env_1hz = np.sqrt((x**2).mean(axis=2))
    return env_1hz

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

def _write_shard(logger, df: pl.DataFrame, out_root: Path, subjects_info: dict = None) -> Path | None:
    """
    Writes one shard per pair using subject-only partitioning and SC/ST-style filenames.
    """
    import time
    start_time = time.time()
    try:
        if df is None or df.is_empty():
            logger.log(f"[WRITE_SHARD] Skipping empty DataFrame", "warning")
            return None
            
        sid = str(df["subject_id"][0])
        nid = str(df["night_id"][0])
        subset = _subset_from_sid(sid)
        out_dir = Path(out_root) / subset / f"subject_id={sid}"
        
        logger.log(f"[WRITE_SHARD] Writing {sid}-{nid}: {df.height} rows, {df.width} columns")
        
        out_dir.mkdir(parents=True, exist_ok=True)
        shard = out_dir / f"{sid}{nid}.parquet"
        
        if subjects_info and sid in subjects_info:
            df = df.with_columns([
                pl.lit(subjects_info[sid].get("age", None)).alias("age"),
                pl.lit(subjects_info[sid].get("sex", None)).alias("sex")
            ])
        
        df.write_parquet(shard)
        elapsed = time.time() - start_time
        file_size = shard.stat().st_size / (1024*1024)  
        logger.log(f"[WRITE_SHARD] ✓ Saved {shard.name}: {df.height} rows, {file_size:.2f}MB in {elapsed:.2f}s")
        return shard
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.log(f"[WRITE_SHARD] ✗ Failed after {elapsed:.2f}s: {e}", "error")
        import traceback
        logger.log(f"[WRITE_SHARD] Error traceback: {traceback.format_exc()}", "error")
        return None

def _process_one_pair(args):
    """
    Runs in a child process: processes 1 pair (PSG, HYP).
    Does not use a progress bar to avoid cluttering stdout.
    Creates its own (lightweight) Logger within the worker.
    """
    if len(args) == 4:
        psg, hyp, sid, nid = args
        root_dir = None
    else:
        psg, hyp, sid, nid, root_dir = args
    
    import time
    import traceback
    from os.path import basename, getsize, dirname
    
    start_time = time.time()
    try:
        from labeler import process_record as _proc
        from logger import Logger as _Logger
        _logger = _Logger()
        
        psg_size = getsize(psg) / (1024*1024)  
        hyp_size = getsize(hyp) / (1024*1024) 
        _logger.log(f"[WORKER] Starting {sid}-{nid}: PSG={basename(psg)} ({psg_size:.1f}MB), HYP={basename(hyp)} ({hyp_size:.1f}MB)")
        
        if root_dir is None:
            root_dir = dirname(dirname(psg)) 
        
        df = _proc(_logger, psg, hyp, subject_id=sid, night_id=nid, progress=None, root_dir=root_dir)
        
        elapsed = time.time() - start_time
        if df is not None and hasattr(df, 'height'):
            _logger.log(f"[WORKER] ✓ Completed {sid}-{nid}: {df.height} epochs in {elapsed:.2f}s")
        else:
            _logger.log(f"[WORKER] ⚠ Completed {sid}-{nid}: empty result in {elapsed:.2f}s", "warning")
            
        return sid, df
        
    except Exception as e:
        elapsed = time.time() - start_time
        from logger import Logger as _Logger
        _logger = _Logger()
        _logger.log(f"[WORKER] ✗ Failed {sid}-{nid} after {elapsed:.2f}s: {e}", "error")
        _logger.log(f"[WORKER] Error details: {traceback.format_exc()}", "error")
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
