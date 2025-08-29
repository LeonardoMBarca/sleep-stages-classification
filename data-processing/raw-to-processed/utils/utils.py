import re

from typing import List, Optional

def slugify(s: str) -> str:
    """Normaliza um rótulo de canal para algo estável: minúsculas, alfanumérico+underscore."""
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def normspace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def best_match_idx(pool: List[str], candidates: List[str]) -> Optional[int]:
    """Match flexível (slugify) entre qualquer item do pool e a lista de candidatos."""
    pool_slug = [slugify(p) for p in pool]
    cand_slug = [slugify(c) for c in candidates]
    # exato
    for cs in cand_slug:
        if cs in pool_slug:
            return pool_slug.index(cs)
    # contains (duas direções)
    for i, ps in enumerate(pool_slug):
        for cs in cand_slug:
            if cs in ps or ps in cs:
                return i
    return None

def log(msg: str):
    print(f"[LOG] {msg}")