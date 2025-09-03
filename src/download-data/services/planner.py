import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Dict, List, Set, Any
import logging

from utils import FILE_REGEX, SUBSETS, local_files_for_subset
from .hash_checker import load_hash_values, filename_hash

logger = logging.getLogger("planner")

def fetch_listing(base_url: str) -> List[str]:
    """Fetch and parse remote directory listing; return filenames matching regex."""
    r = requests.get(base_url, timeout=SUBSETS["http_timeout"], headers=SUBSETS["http_headers"])
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = [a.get("href", "") for a in soup.find_all("a", href=True)]
    files = [f for f in links if FILE_REGEX.search(f)]
    logger.info(f"LIST {base_url} files={len(files)}")
    return files

def _processed_hashes(use_hash: bool) -> Set[str]:
    """Return processed filename-hash set; empty if hashing is disabled."""
    if not use_hash:
        return set()
    return load_hash_values()

def plan_missing(subset: str, use_hash: bool = True, limit: int | None = None) -> Dict[str, Any]:
    """Compare remote vs local vs processed-hash and return missing filenames."""
    subset_dir_remote = SUBSETS[subset]
    base_url = urljoin(str(SUBSETS["base_root"]), subset_dir_remote)

    remote = set(fetch_listing(base_url))
    local = local_files_for_subset(subset)
    processed = _processed_hashes(use_hash)
    remote_marked_processed = {fname for fname in remote if filename_hash(fname) in processed}

    missing_sorted = sorted(remote - local - remote_marked_processed)
    if limit is not None:
        missing_sorted = missing_sorted[:limit]

    logger.info(
        f"PLAN subset={subset} remote={len(remote)} local={len(local)} "
        f"processed={len(remote_marked_processed)} missing={len(missing_sorted)}"
    )
    return {"subset": subset, "base_url": base_url, "missing": missing_sorted}
