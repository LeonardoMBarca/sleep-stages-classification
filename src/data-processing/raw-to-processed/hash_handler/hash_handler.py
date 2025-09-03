import hashlib
import threading
from pathlib import Path
from typing import Iterable, Set, List

class HashHandler:
    """
    Manages a file of hashes (one SHA-256 hash per file), where each hash represents the SHA-256 of the filename.
    """

    def __init__(self, hash_file: Path):
        self.hash_file = Path(hash_file)
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def hash_name(filename: str) -> str:
        """Returns SHA-256 (hex) of filename"""
        h = hashlib.sha256()
        h.update(filename.encode("utf-8"))

        return h.hexdigest()
    
    def load(self) -> Set[str]:
        """Upload all lines (hashes) of file. If not exists, return empty set()."""
        if not self.hash_file.exists():

            return set()

        with self.hash_file.open("r", encoding="utf-8") as f:

            return {ln.strip() for ln in f if ln.strip()}
        
    def add_hashes(self, hashes: Iterable[str]) -> None:
        """Add hashes (without duplicate existent lines)"""
        hashes = [h for h in set(hashes) if h]
        
        if not hashes:

            return
        
        with self._lock:
            current = self.load()
            to_write = [h for h in hashes if h not in current]

            if not to_write:
                return
            
            with self.hash_file.open("a", encoding="utf-8") as f:
                for h in to_write:
                    f.write(h + "\n")
    
    def add_names(self, filenames: Iterable[str]) -> None:
        self.add_hashes(self.hash_name(fn) for fn in filenames)

    def contains_name(self, filename: str, loaded_set: Set[str] | None = None) -> bool:
        """Test if the hash of name already is in the file."""
        h = self.hash_name(filename)
        
        if loaded_set is None:
            
            return h in self.load()
        
        return h in loaded_set