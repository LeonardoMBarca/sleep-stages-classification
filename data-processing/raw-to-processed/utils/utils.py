import re
import glob
import os

from typing import List, Optional, Tuple

def _norm_label(s: str) -> str:
    """
    Function to "clean" the Hypnogram channel names and descriptions before any comparison.
    - Removes extra spaces at the beginning and end
    - Converts multiple spaces/blanksin a row to a single space
    - Ensures that even null values (None) become an empty string
    """
    return re.sub(r"\s+", " ", (s or "").strip())

def _best_match(label_pool: List[str], candidates: List[str]) -> Optional[int]:
    """Returns the index of the 1st label in label_pool that matches some candidate (case-insensitive, contains)"""
    low_pool = [l.lower() for l in label_pool]

    for c in candidates:
        c_low = c.lower()

        if c_low in low_pool:
            return low_pool.index(c_low)
    
    for i, lab in enumerate(low_pool):
        for c in candidates:
            if c.lower() in lab:
                return i
    
    return None

def _pair_psg_hyp(root_dir: str) -> List[Tuple[str, str, str, str]]:
    """Find *-PSG.edf and *-Hypnogram.edf pairs in the tree"""
    pairs = []

    for psg in glob.glob(os.path.join(root_dir, "**", "*-PSG.edf"), recursive=True):
        base = os.path.basename(psg)
        stem = base[:-8] # Remove "-PSG.edf"
        folder = os.path.dirname(psg)

        # Heuristic 1: SCxxxx / STxxxx (6 chars)
        prefix6 = stem[:6]
        cands = sorted(glob.glob(os.path.join(folder, f"{prefix6}*-Hypnogram.edf")))

        # Heuristic 2: exact name of stem
        if not cands:
            cands = sorted(glob.glob(os.path.join(folder, f"{stem}-Hypnogram.edf")))

        if not cands:
            continue

        hyp = cands[0]

        # subject_id and night_id
        m = re.match(r"^(SC|ST)(\d{4})([A-Z]\d)?", stem, flags=re.IGNORECASE)

        if m:
            subject_id = f"{m.group(1).upper()}{m.group(2)}"
            night_id = m.group(3).upper() if m.group(3) else "N0"
        
        else:
            subject_id, night_id = stem, "N0"

        pairs.append((psg, hyp, subject_id, night_id))
    
    return pairs