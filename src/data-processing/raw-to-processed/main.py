import os, sys
import polars as pl
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logger")))

from labeler import pair_psg_hyp, process_record
from utils import _make_progress, _process_one_pair
from logger import Logger

logger = Logger()

def build_tabular_dataset(root_dir: str, out_sc: str | Path, out_st: str | Path,
                          workers: int = None, show_progress: bool = True) -> None:
    """
    Processes all PSG/Hyp pairs and saves TWO parquet files:
        - Sleep Cassette (SC)
        - Sleep Telemetry (ST)
    Executes in parallel per file pair.
    """
    logger.log(f"[BUILD_TABULAR_DATASET] Starting processing in: {root_dir}")

    out_sc = Path(out_sc) / "sleep-cassette" / "sleep_cassette_dataset.parquet"
    out_st = Path(out_st) / "sleep-telemetry" / "sleep_telemetry_dataset.parquet"

    sc_df = None
    st_df = None

    try:
        pairs = pair_psg_hyp(logger, root_dir)
        if not pairs:
            raise FileNotFoundError("No *-PSG.edf and *-Hypnogram.edf pairs found.")

        if workers is None:
            workers = max(1, (os.cpu_count() or 4) - 1)

        logger.log(f"[BUILD_TABULAR_DATASET] Using {workers} workers")

        sc_dfs, st_dfs = [], []

        if show_progress:
            try:
                with _make_progress() as progress:
                    task = progress.add_task("Files (parallel)", total=len(pairs))
                    with ProcessPoolExecutor(max_workers=workers, mp_context=None) as ex:
                        futures = [ex.submit(_process_one_pair, p) for p in pairs]
                        for fut in as_completed(futures):
                            try:
                                sid, df = fut.result()
                                if df is not None and df.height > 0:
                                    if sid.upper().startswith("SC"):
                                        sc_dfs.append(df)
                                    elif sid.upper().startswith("ST"):
                                        st_dfs.append(df)
                                    else:
                                        logger.log(f"[BUILD_TABULAR_DATASET] Unknown prefix for subject_id={sid}", "warning")
                                else:
                                    logger.log(f"[BUILD_TABULAR_DATASET] Empty/failed DF for subject_id={sid}", "warning")
                            except Exception as e:
                                logger.log(f"[BUILD_TABULAR_DATASET] Worker error: {e}", "warning")
                            finally:
                                progress.update(task, advance=1)
            except Exception as e:
                logger.log(f"[BUILD_TABULAR_DATASET] Progress failed ({e}); running parallel without bar.", "warning")
                with ProcessPoolExecutor(max_workers=workers, mp_context=None) as ex:
                    futures = [ex.submit(_process_one_pair, p) for p in pairs]
                    for fut in as_completed(futures):
                        try:
                            sid, df = fut.result()
                            if df is not None and df.height > 0:
                                if sid.upper().startswith("SC"):
                                    sc_dfs.append(df)
                                elif sid.upper().startswith("ST"):
                                    st_dfs.append(df)
                                else:
                                    logger.log(f"[BUILD_TABULAR_DATASET] Unknown prefix for subject_id={sid}", "warning")
                            else:
                                logger.log(f"[BUILD_TABULAR_DATASET] Empty/failed DF for subject_id={sid}", "warning")
                        except Exception as e:
                            logger.log(f"[BUILD_TABULAR_DATASET] Worker error: {e}", "warning")
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=None) as ex:
                futures = [ex.submit(_process_one_pair, p) for p in pairs]
                for fut in as_completed(futures):
                    try:
                        sid, df = fut.result()
                        if df is not None and df.height > 0:
                            if sid.upper().startswith("SC"):
                                sc_dfs.append(df)
                            elif sid.upper().startswith("ST"):
                                st_dfs.append(df)
                            else:
                                logger.log(f"[BUILD_TABULAR_DATASET] Unknown prefix for subject_id={sid}", "warning")
                        else:
                            logger.log(f"[BUILD_TABULAR_DATASET] Empty/failed DF for subject_id={sid}", "warning")
                    except Exception as e:
                        logger.log(f"[BUILD_TABULAR_DATASET] Worker error: {e}", "warning")

        if not sc_dfs and not st_dfs:
            raise RuntimeError("No valid record processed.")

        if sc_dfs:
            try:
                sc_df = pl.concat(sc_dfs, how="vertical").sort(["subject_id", "night_id", "epoch_idx"])
                out_sc.parent.mkdir(parents=True, exist_ok=True)
                sc_df.write_parquet(out_sc)
                logger.log(f"[BUILD_TABULAR_DATASET] SC saved: {out_sc} (rows={sc_df.height})")
            except Exception as e:
                logger.log(f"[BUILD_TABULAR_DATASET] Failed to write SC parquet: {e}", "error")

        if st_dfs:
            try:
                st_df = pl.concat(st_dfs, how="vertical").sort(["subject_id", "night_id", "epoch_idx"])
                out_st.parent.mkdir(parents=True, exist_ok=True)
                st_df.write_parquet(out_st)
                logger.log(f"[BUILD_TABULAR_DATASET] ST saved: {out_st} (rows={st_df.height})")
            except Exception as e:
                logger.log(f"[BUILD_TABULAR_DATASET] Failed to write ST parquet: {e}", "error")

        try:
            if sc_df is not None and sc_df.height > 0:
                resume_sc = sc_df.group_by("stage").agg(pl.len().alias("count")).sort("count", descending=True)
                logger.log(f"[BUILD_TABULAR_DATASET] SC classes:\n{resume_sc.to_pandas().to_string(index=False)}")
            else:
                logger.log("[BUILD_TABULAR_DATASET] SC: no rows to summarize.", "warning")
        except Exception as e:
            logger.log(f"[BUILD_TABULAR_DATASET] Failed to compute SC resume: {e}", "error")

        try:
            if st_df is not None and st_df.height > 0:
                resume_st = st_df.group_by("stage").agg(pl.len().alias("count")).sort("count", descending=True)
                logger.log(f"[BUILD_TABULAR_DATASET] ST classes:\n{resume_st.to_pandas().to_string(index=False)}")
            else:
                logger.log("[BUILD_TABULAR_DATASET] ST: no rows to summarize.", "warning")
        except Exception as e:
            logger.log(f"[BUILD_TABULAR_DATASET] Failed to compute ST resume: {e}", "error")

    except Exception as e:
        logger.log(f"[BUILD_TABULAR_DATASET] Fatal error: {e}", "error")
        raise


if __name__ == "__main__":
    import argparse
    base = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Sleep-EDF → SC & ST (parallel, Polars)")
    ap.add_argument("--root", required=False, default=str(base / "datalake" / "raw"),
                    help="Root folder (e.g., .../datalake/raw)")
    ap.add_argument("--out-sc", required=False, default=str(base / "datalake" / "processed"),
                    help="Output base for Sleep Cassette")
    ap.add_argument("--out-st", required=False, default=str(base / "datalake" / "processed"),
                    help="Output base for Sleep Telemetry")
    ap.add_argument("--workers", type=int, default=None, help="Max workers (default: CPUs-1)")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    args = ap.parse_args()

    build_tabular_dataset(
        root_dir=args.root,
        out_sc=args.out_sc,
        out_st=args.out_st,
        workers=args.workers,
        show_progress=not args.no_progress,
    )
