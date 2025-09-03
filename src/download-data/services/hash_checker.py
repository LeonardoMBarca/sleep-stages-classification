from typing import Set
import hashlib
from utils import HASH_FILE

def filename_hash(filename: str) -> str:
    """Return sha256 hex digest of the exact filename string."""
    h = hashlib.sha256()
    h.update(filename.encode("utf-8"))
    return h.hexdigest()

def load_hash_values() -> Set[str]:
    """Load processed filename hashes from file; return an empty set if none."""
    if not HASH_FILE.exists():
        return set()
    hashes: Set[str] = set()
    with open(HASH_FILE, "r", encoding="utf-8") as f:
        for line in f:
            v = line.strip()
            if v:
                hashes.add(v)
    return hashes
