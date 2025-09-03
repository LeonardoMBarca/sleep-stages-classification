import logging
from fastapi import APIRouter, Query
from typing import Literal, List, Dict, Any, Optional
from threading import Thread

from services import plan_missing, run_download_for_subset
from utils import ensure_dirs, RAW_DIR

router = APIRouter()
logger = logging.getLogger("route.download")

def _run_job(plans_: List[Dict[str, Any]], workers: Optional[int], batch_size: Optional[int], round_retries: int) -> None:
    """Execute download job for one or two subsets in a detached thread."""
    for p in plans_:
        try:
            logger.info(f"TASK subset={p['subset']} missing={len(p['missing'])}")
            run_download_for_subset(
                p["subset"],
                p["missing"],
                workers=workers,
                batch_size=batch_size,
                round_retries=round_retries,
            )
        except Exception as e:
            logger.exception(f"TASK ERROR subset={p['subset']}: {type(e).__name__}: {e}")

@router.get("/")
async def download(
    subset: Literal["cassette", "telemetry", "both"] = Query("cassette"),
    ignore_hash: bool = Query(False),
    limit: Optional[int] = Query(None, ge=1),
    sync: bool = Query(True),
    workers: Optional[int] = Query(None, ge=1, le=8),
    batch_size: Optional[int] = Query(None, ge=1),
    round_retries: int = Query(3, ge=0, le=8),
) -> Dict[str, Any]:
    """Plan and execute downloads; optionally block until completion."""
    ensure_dirs()

    if subset == "both":
        plans = [
            plan_missing("cassette", use_hash=not ignore_hash, limit=limit),
            plan_missing("telemetry", use_hash=not ignore_hash, limit=limit),
        ]
    else:
        plans = [plan_missing(subset, use_hash=not ignore_hash, limit=limit)]

    if sync:
        exec_results: Dict[str, List[str]] = {}
        for p in plans:
            try:
                exec_results[p["subset"]] = run_download_for_subset(
                    p["subset"],
                    p["missing"],
                    workers=workers,
                    batch_size=batch_size,
                    round_retries=round_retries,
                )
            except Exception as e:
                exec_results[p["subset"]] = [f"ERR {p['subset']}: {type(e).__name__}: {e}"]
        return {
            "status": "completed",
            "subset": subset,
            "ignore_hash": ignore_hash,
            "plan": [
                {
                    "subset": p["subset"],
                    "missing_count": len(p["missing"]),
                    "missing_samples": p["missing"][:10],
                    "base_url": p["base_url"],
                }
                for p in plans
            ],
            "results": exec_results,
            "raw_dir": str(RAW_DIR),
        }

    t = Thread(target=_run_job, args=(plans, workers, batch_size, round_retries), daemon=True)
    t.start()

    return {
        "status": "started",
        "subset": subset,
        "ignore_hash": ignore_hash,
        "plan": [
            {
                "subset": p["subset"],
                "missing_count": len(p["missing"]),
                "missing_samples": p["missing"][:10],
                "base_url": p["base_url"],
            }
            for p in plans
        ],
        "raw_dir": str(RAW_DIR),
    }
