import re
from pathlib import Path
from .http import DEFAULT_HEADERS, DEFAULT_TIMEOUT

def _compute_base_dir() -> Path:
    p = Path(__file__).resolve()
    for up in (p.parents[3:] if len(p.parents) >= 4 else p.parents):
        if (up / "datalake").exists():
            return up
    return Path(__file__).resolve().parents[2]

BASE_DIR = _compute_base_dir()
DATALAKE_DIR = BASE_DIR / "datalake"

RAW_DIR = DATALAKE_DIR / "raw"
PROCESSED_DIR = DATALAKE_DIR / "processed"
HASH_FILE = PROCESSED_DIR / "hash_files_processed.txt"

BASE_ROOT = "https://physionet.org/files/sleep-edfx/1.0.0/"
SUBSETS = {
    "cassette": "sleep-cassette/",
    "telemetry": "sleep-telemetry/",
    "base_root": BASE_ROOT,
    "workers": 8,
    "http_timeout": DEFAULT_TIMEOUT,
    "http_headers": DEFAULT_HEADERS,
}

FILE_REGEX = re.compile(r"(PSG\.edf|Hypnogram\.edf)$", re.IGNORECASE)

CHUNK = 1 << 20
HTTP_DEFAULT_TIMEOUT = DEFAULT_TIMEOUT
HTTP_HEADERS = DEFAULT_HEADERS
