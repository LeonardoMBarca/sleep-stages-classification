import time
import logging
import math
import requests
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
from typing import List, Optional, Tuple
from pathlib import Path

from utils import CHUNK, HTTP_DEFAULT_TIMEOUT, HTTP_HEADERS, SUBSETS, RAW_DIR, ensure_dirs

logger = logging.getLogger("downloader")

MAX_RETRIES = 5
BACKOFF_BASE = 1.6
CONNECT_TIMEOUT = 10
READ_TIMEOUT = max(HTTP_DEFAULT_TIMEOUT * 6, 600)
FINAL_TIMEOUT: Tuple[int, int] = (CONNECT_TIMEOUT, READ_TIMEOUT)
PROGRESS_INTERVAL = 5.0

def _session() -> requests.Session:
    """Create a configured requests Session with connection pool and retry policy."""
    s = requests.Session()
    retry = Retry(
        total=0,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET"],
        respect_retry_after_header=True,
    )
    adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def _head_size(session: requests.Session, url: str) -> Optional[int]:
    """Return Content-Length from a HEAD request if available."""
    try:
        r = session.head(url, headers=HTTP_HEADERS, timeout=(CONNECT_TIMEOUT, 30), allow_redirects=True)
        if r.status_code >= 400:
            return None
        v = r.headers.get("Content-Length")
        if v and v.isdigit():
            return int(v)
        return None
    except requests.exceptions.RequestException:
        return None

def _download_once(session: requests.Session, url: str, dst: Path, expected_size: Optional[int], subset: str, fname: str) -> str:
    """Perform a single GET with resume, log progress, validate size, and return status string."""
    headers = dict(HTTP_HEADERS)
    mode = "wb"
    start_pos = 0
    if dst.exists():
        pos = dst.stat().st_size
        if pos > 0:
            headers["Range"] = f"bytes={pos}-"
            mode = "ab"
            start_pos = pos

    started = time.monotonic()
    last_tick = started
    last_bytes = start_pos

    with session.get(url, headers=headers, stream=True, timeout=FINAL_TIMEOUT) as r:
        if r.status_code == 200 and "Range" in headers:
            mode = "wb"
            start_pos = 0
        if r.status_code not in (200, 206):
            return f"HTTP {r.status_code}"

        with open(dst, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                now = time.monotonic()
                if now - last_tick >= PROGRESS_INTERVAL:
                    written = dst.stat().st_size
                    delta_b = written - last_bytes
                    delta_t = max(now - last_tick, 1e-6)
                    rate = (delta_b / delta_t) / (1024 * 1024)
                    pct = None
                    if expected_size and expected_size > 0:
                        pct = 100.0 * (written / expected_size)
                    if pct is None:
                        logger.info(f"PROG {subset}/{fname} bytes={written} rate={rate:.2f}MB/s")
                    else:
                        pct_c = min(100.0, pct)
                        logger.info(f"PROG {subset}/{fname} {pct_c:.1f}% rate={rate:.2f}MB/s")
                    last_tick = now
                    last_bytes = written

    final_size = dst.stat().st_size
    if expected_size and expected_size > 0 and final_size != expected_size:
        return f"INCOMPLETE expected={expected_size} got={final_size}"
    elapsed = time.monotonic() - started
    mb = final_size / (1024 * 1024)
    rate = mb / max(elapsed, 1e-6)
    logger.info(f"OK {subset}/{fname} size={final_size}B time={elapsed:.1f}s avg={rate:.2f}MB/s")
    return "OK"

def download_file(base_url: str, subset: str, fname: str) -> str:
    """Download a single file with resume, progress logs, and size validation."""
    url = urljoin(base_url, fname)
    dest_dir = RAW_DIR / f"sleep-{subset}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / fname
    session = _session()
    attempt = 0
    expected_size = _head_size(session, url)
    if expected_size:
        logger.info(f"HEAD {subset}/{fname} size={expected_size}B")

    try:
        while True:
            attempt += 1
            try:
                res = _download_once(session, url, dst, expected_size, subset, fname)
                if res == "OK":
                    return f"OK {subset}/{fname}"
                if res.startswith("HTTP "):
                    code = int(res.split()[1])
                    if code in (429, 500, 502, 503, 504) and attempt <= MAX_RETRIES:
                        sleep_s = BACKOFF_BASE ** attempt
                        logger.warning(f"RETRY {subset}/{fname} {res} sleep={sleep_s:.1f}s")
                        time.sleep(sleep_s)
                        continue
                    logger.error(f"ERR {subset}/{fname} ({res})")
                    return f"ERR {subset}/{fname} ({res})"
                if "INCOMPLETE" in res and attempt <= MAX_RETRIES:
                    sleep_s = BACKOFF_BASE ** attempt
                    logger.warning(f"RETRY {subset}/{fname} {res} sleep={sleep_s:.1f}s")
                    time.sleep(sleep_s)
                    continue
                logger.error(f"ERR {subset}/{fname} ({res})")
                return f"ERR {subset}/{fname} ({res})"
            except requests.exceptions.RequestException as e:
                if attempt <= MAX_RETRIES:
                    sleep_s = BACKOFF_BASE ** attempt
                    logger.warning(f"RETRY {subset}/{fname} {type(e).__name__}: {e} sleep={sleep_s:.1f}s")
                    time.sleep(sleep_s)
                    continue
                logger.error(f"ERR {subset}/{fname} ({type(e).__name__}: {e})")
                return f"ERR {subset}/{fname} ({type(e).__name__}: {e})"
    finally:
        session.close()

def _is_transient(err: str) -> bool:
    """Return True if an error string indicates a transient failure that merits retry."""
    if "HTTP 429" in err:
        return True
    for code in ("500", "502", "503", "504"):
        if f"HTTP {code}" in err:
            return True
    for key in ("ReadTimeout", "ConnectTimeout", "ConnectionError", "ChunkedEncodingError", "ProtocolError"):
        if key in err:
            return True
    if "INCOMPLETE" in err:
        return True
    return False

def _chunks(seq: List[str], size: int) -> List[List[str]]:
    """Split sequence into chunks of given size."""
    return [seq[i:i + size] for i in range(0, len(seq), size)]

def run_download_for_subset(
    subset: str,
    missing: List[str],
    workers: Optional[int] = None,
    batch_size: Optional[int] = None,
    round_retries: int = 3,
) -> List[str]:
    """Download all missing items for a subset, retrying transient failures across rounds."""
    ensure_dirs()
    if not missing:
        logger.info(f"SKIP {subset}: nothing to download")
        return [f"SKIP {subset}: all files already exist"]

    base_url = urljoin(str(SUBSETS["base_root"]), SUBSETS[subset])
    results: List[str] = []
    remaining = sorted(missing)
    w = int(workers) if workers is not None else int(SUBSETS.get("workers", 8))
    b = int(batch_size) if batch_size is not None else max(2, w * 2)

    logger.info(f"START subset={subset} total={len(remaining)} workers={w} batch_size={b} rounds={round_retries}")

    round_idx = 0
    while remaining and round_idx <= max(0, round_retries):
        round_idx += 1
        logger.info(f"ROUND subset={subset} #{round_idx} pending={len(remaining)}")
        next_round: List[str] = []
        for batch in _chunks(remaining, b):
            batch_results: List[str] = []
            with ThreadPoolExecutor(max_workers=w, thread_name_prefix=f"dwl-{subset}") as ex:
                futs = [ex.submit(download_file, base_url, subset, f) for f in batch]
                for fut in as_completed(futs):
                    batch_results.append(fut.result())
            results.extend(batch_results)
            errs = [r for r in batch_results if r.startswith("ERR ")]
            to_retry = []
            for e in errs:
                if _is_transient(e):
                    fname = e.split("/", 1)[1].split(" ", 1)[0]
                    to_retry.append(fname)
            if to_retry:
                logger.info(f"REQUEUE subset={subset} next={len(to_retry)}")
            next_round.extend(to_retry)
        remaining = sorted(set(next_round))

    done = len([r for r in results if r.startswith("OK ")])
    failed = len([r for r in results if r.startswith("ERR ")])
    logger.info(f"END subset={subset} ok={done} err={failed}")
    return results
