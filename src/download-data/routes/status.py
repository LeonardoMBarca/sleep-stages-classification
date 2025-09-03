from fastapi import APIRouter, Query
from typing import Literal, Dict, Any
from utils import ensure_dirs, local_files_for_subset

router = APIRouter()

@router.get("/")
def status(subset: Literal["cassette", "telemetry", "both"] = Query("both")) -> Dict[str, Any]:
    """Return local file counts by subset."""
    ensure_dirs()
    if subset == "both":
        return {
            "cassette": {"count": len(local_files_for_subset("cassette"))},
            "telemetry": {"count": len(local_files_for_subset("telemetry"))}
        }
    return {subset: {"count": len(local_files_for_subset(subset))}}
