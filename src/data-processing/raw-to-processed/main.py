import os, sys
import polars as pl
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from threading import Lock

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logger")))

from labeler import pair_psg_hyp, process_record
from utils import _make_progress, _process_one_pair, _pair_key, _write_shard
from hash_handler import HashHandler
from logger import Logger

HASH_FILE_DEFAULT = Path(__file__).resolve().parents[3] / "datalake" / "processed" / "hash_files_processed.txt"

logger = Logger()
processing_lock = Lock()
processing_stats = {
    'total_pairs': 0,
    'completed': 0,
    'failed': 0,
    'empty_results': 0,
    'start_time': None,
    'last_activity': None
}

def build_tabular_dataset(
        root_dir: str, 
        out_sc: str | Path, 
        out_st: str | Path,
        workers: int = None, 
        show_progress: bool = True,
        use_hash: bool = True,
        incremental: bool = True,
        write_aggregate: bool = False,
        hash_file: Path = HASH_FILE_DEFAULT
    ) -> None:
    """
    Processes all PSG/Hyp pairs and saves TWO parquet files:
        - Sleep Cassette (SC)
        - Sleep Telemetry (ST)
    Executes in parallel per file pair.
    """
    processing_stats['start_time'] = time.time()
    processing_stats['last_activity'] = time.time()
    
    logger.log(f"[BUILD_TABULAR_DATASET] ========== STARTING PROCESSING ===========")
    logger.log(f"[BUILD_TABULAR_DATASET] Root directory: {root_dir}")
    logger.log(f"[BUILD_TABULAR_DATASET] Output SC: {out_sc}")
    logger.log(f"[BUILD_TABULAR_DATASET] Output ST: {out_st}")
    logger.log(f"[BUILD_TABULAR_DATASET] Workers: {workers}")
    logger.log(f"[BUILD_TABULAR_DATASET] Use hash: {use_hash}")
    logger.log(f"[BUILD_TABULAR_DATASET] Incremental: {incremental}")
    logger.log(f"[BUILD_TABULAR_DATASET] Write aggregate: {write_aggregate}")

    out_sc = Path(out_sc) / "sleep-cassette" / "sleep_cassette_dataset.parquet"
    out_st = Path(out_st) / "sleep-telemetry" / "sleep_telemetry_dataset.parquet"

    hh = HashHandler(hash_file)
    processed_hashes = hh.load() if use_hash else set()
    logger.log(f"[HASH] use_hash={use_hash} | loaded_entries={len(processed_hashes)}")

    sc_df = None
    st_df = None

    try:
        logger.log(f"[BUILD_TABULAR_DATASET] Scanning for PSG/Hypnogram pairs...")
        pairs = pair_psg_hyp(logger, root_dir)
        if not pairs:
            logger.log(f"[BUILD_TABULAR_DATASET] ERROR: No PSG/Hypnogram pairs found in {root_dir}", "error")
            logger.log(f"[BUILD_TABULAR_DATASET] Check if directory exists and contains *-PSG.edf and *-Hypnogram.edf files", "error")
            raise FileNotFoundError("No *-PSG.edf and *-Hypnogram.edf pairs found.")
        
        processing_stats['total_pairs'] = len(pairs)
        logger.log(f"[BUILD_TABULAR_DATASET] Found {len(pairs)} PSG/Hypnogram pairs to process")
        
        for i, (psg, hyp, sid, nid) in enumerate(pairs[:3]):
            logger.log(f"[BUILD_TABULAR_DATASET] Sample pair {i+1}: {sid}-{nid} | PSG: {Path(psg).name} | HYP: {Path(hyp).name}")

        if use_hash:
            logger.log(f"[HASH] Filtering pairs using hash file: {hash_file}")
            kept_pairs = []
            skipped = 0
            for psg, hyp, sid, nid in pairs:
                bpsg, bhyp = _pair_key(psg, hyp)
                psg_hash = hh.hash_name(bpsg)
                hyp_hash = hh.hash_name(bhyp)
                if (psg_hash in processed_hashes) and (hyp_hash in processed_hashes):
                    skipped += 1
                    logger.log(f"[HASH] Skipping already processed: {sid}-{nid} (PSG: {psg_hash[:8]}..., HYP: {hyp_hash[:8]}...)")
                else:
                    kept_pairs.append((psg, hyp, sid, nid))
            logger.log(f"[HASH] Hash filtering complete: {skipped} pairs skipped, {len(kept_pairs)} pairs to process")
            pairs = kept_pairs
            processing_stats['total_pairs'] = len(pairs)

        if not pairs:
            logger.log("[BUILD_TABULAR_DATASET] Nothing to process after hash filtering - all pairs already processed")
            logger.log(f"[BUILD_TABULAR_DATASET] Processing completed in {time.time() - processing_stats['start_time']:.2f} seconds")
            return

        if workers is None:
            workers = max(1, (os.cpu_count() or 4) - 1)

        logger.log(f"[BUILD_TABULAR_DATASET] Parallel processing configuration:")
        logger.log(f"[BUILD_TABULAR_DATASET] - Workers: {workers}")
        logger.log(f"[BUILD_TABULAR_DATASET] - CPU count: {os.cpu_count()}")
        logger.log(f"[BUILD_TABULAR_DATASET] - Show progress: {show_progress}")
        logger.log(f"[BUILD_TABULAR_DATASET] Starting parallel processing of {len(pairs)} pairs...")

        sc_dfs, st_dfs = [], []

        def _submit_all(ex):
            logger.log(f"[BUILD_TABULAR_DATASET] Submitting {len(pairs)} tasks to worker pool...")
            futures = []
            for i, p in enumerate(pairs):
                try:
                    actual_root = root_dir
                    if 'sleep-cassette' in root_dir or 'sleep-telemetry' in root_dir:
                        actual_root = str(Path(root_dir).parent)
                    args_with_root = p + (actual_root,)
                    future = ex.submit(_process_one_pair, args_with_root)
                    futures.append(future)
                    if i < 3: 
                        logger.log(f"[BUILD_TABULAR_DATASET] Submitted task {i+1}: {p[2]}-{p[3]}")
                except Exception as e:
                    logger.log(f"[BUILD_TABULAR_DATASET] Failed to submit task {i+1} ({p[2]}-{p[3]}): {e}", "error")
            logger.log(f"[BUILD_TABULAR_DATASET] Successfully submitted {len(futures)} tasks")
            return futures

        if show_progress:
            try:
                with _make_progress() as progress:
                    task = progress.add_task("Files (parallel)", total=len(pairs))
                    with ProcessPoolExecutor(max_workers=workers, mp_context=None) as ex:
                        futures = _submit_all(ex)
                        for fut in as_completed(futures, timeout=300):  
                            try:
                                start_result_time = time.time()
                                sid, df = fut.result(timeout=30)  
                                
                                with processing_lock:
                                    processing_stats['last_activity'] = time.time()
                                    processing_stats['completed'] += 1
                                
                                if df is not None and df.height > 0:
                                    logger.log(f"[BUILD_TABULAR_DATASET] ✓ Processed {sid}: {df.height} epochs ({processing_stats['completed']}/{processing_stats['total_pairs']})")
                                    if incremental:
                                        processed_root = Path(out_sc).parent.parent
                                        _write_shard(logger, df, processed_root)

                                    if sid.upper().startswith("SC"):
                                        sc_dfs.append(df)
                                    elif sid.upper().startswith("ST"):
                                        st_dfs.append(df)
                                    else:
                                        logger.log(f"[BUILD_TABULAR_DATASET] Unknown prefix for subject_id={sid}", "warning")

                                    try:
                                        nid = str(df["night_id"][0])
                                        for psg0, hyp0, sid0, nid0 in pairs:
                                            if sid0 == sid and nid0 == nid:
                                                bpsg, bhyp = _pair_key(psg0, hyp0)
                                                hh.add_names([bpsg, bhyp])
                                                break
                                    except Exception as e:
                                        logger.log(f"[HASH] Failed to register processed pair for sid={sid}: {e}", "warning")

                                else:
                                    with processing_lock:
                                        processing_stats['empty_results'] += 1
                                    logger.log(f"[BUILD_TABULAR_DATASET] ⚠ Empty/failed result for {sid} ({processing_stats['completed']}/{processing_stats['total_pairs']})", "warning")
                            except TimeoutError as e:
                                with processing_lock:
                                    processing_stats['failed'] += 1
                                logger.log(f"[BUILD_TABULAR_DATASET] ✗ TIMEOUT processing task (>{fut._timeout if hasattr(fut, '_timeout') else 'unknown'}s): {e}", "error")
                            except Exception as e:
                                with processing_lock:
                                    processing_stats['failed'] += 1
                                logger.log(f"[BUILD_TABULAR_DATASET] ✗ Worker error ({processing_stats['failed']} failures so far): {e}", "error")
                                logger.log(f"[BUILD_TABULAR_DATASET] Error traceback: {traceback.format_exc()}", "error")
                            finally:
                                progress.update(task, advance=1)
            except Exception as e:
                logger.log(f"[BUILD_TABULAR_DATASET] Progress failed ({e}); running parallel without bar.", "warning")
                with ProcessPoolExecutor(max_workers=workers, mp_context=None) as ex:
                    futures = _submit_all(ex)
                    for fut in as_completed(futures, timeout=300):
                        try:
                            sid, df = fut.result(timeout=30)
                            
                            with processing_lock:
                                processing_stats['last_activity'] = time.time()
                                processing_stats['completed'] += 1
                            
                            if df is not None and df.height > 0:
                                logger.log(f"[BUILD_TABULAR_DATASET] ✓ Processed {sid}: {df.height} epochs ({processing_stats['completed']}/{processing_stats['total_pairs']})")
                                if incremental:
                                    processed_root = Path(out_sc).parent.parent
                                    _write_shard(logger, df, processed_root)

                                if sid.upper().startswith("SC"):
                                    sc_dfs.append(df)
                                elif sid.upper().startswith("ST"):
                                    st_dfs.append(df)
                                else:
                                    logger.log(f"[BUILD_TABULAR_DATASET] Unknown prefix for subject_id={sid}", "warning")

                                try:
                                    nid = str(df["night_id"][0])
                                    for psg0, hyp0, sid0, nid0 in pairs:
                                        if sid0 == sid and nid0 == nid:
                                            bpsg, bhyp = _pair_key(psg0, hyp0)
                                            hh.add_names([bpsg, bhyp])
                                            break
                                except Exception as e:
                                    logger.log(f"[HASH] Failed to register processed pair for sid={sid}: {e}", "warning")
                            else:
                                logger.log(f"[BUILD_TABULAR_DATASET] Empty/failed DF for subject_id={sid}", "warning")
                        except TimeoutError as e:
                            with processing_lock:
                                processing_stats['failed'] += 1
                            logger.log(f"[BUILD_TABULAR_DATASET] ✗ TIMEOUT processing task: {e}", "error")
                        except Exception as e:
                            with processing_lock:
                                processing_stats['failed'] += 1
                            logger.log(f"[BUILD_TABULAR_DATASET] ✗ Worker error: {e}", "error")
                            logger.log(f"[BUILD_TABULAR_DATASET] Error traceback: {traceback.format_exc()}", "error")
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=None) as ex:
                futures = _submit_all(ex)
                for fut in as_completed(futures, timeout=300):
                    try:
                        sid, df = fut.result(timeout=30)
                        
                        with processing_lock:
                            processing_stats['last_activity'] = time.time()
                            processing_stats['completed'] += 1
                        
                        if df is not None and df.height > 0:
                            logger.log(f"[BUILD_TABULAR_DATASET] ✓ Processed {sid}: {df.height} epochs ({processing_stats['completed']}/{processing_stats['total_pairs']})")
                            if incremental:
                                processed_root = Path(out_sc).parent.parent
                                _write_shard(logger, df, processed_root)

                            if sid.upper().startswith("SC"):
                                sc_dfs.append(df)
                            elif sid.upper().startswith("ST"):
                                st_dfs.append(df)
                            else:
                                logger.log(f"[BUILD_TABULAR_DATASET] Unknown prefix for subject_id={sid}", "warning")

                            try:
                                nid = str(df["night_id"][0])
                                for psg0, hyp0, sid0, nid0 in pairs:
                                    if sid0 == sid and nid0 == nid:
                                        bpsg, bhyp = _pair_key(psg0, hyp0)
                                        hh.add_names([bpsg, bhyp])
                                        break
                            except Exception as e:
                                logger.log(f"[HASH] Failed to register processed pair for sid={sid}: {e}", "warning")
                        else:
                            logger.log(f"[BUILD_TABULAR_DATASET] Empty/failed DF for subject_id={sid}", "warning")
                    except TimeoutError as e:
                        with processing_lock:
                            processing_stats['failed'] += 1
                        logger.log(f"[BUILD_TABULAR_DATASET] ✗ TIMEOUT processing task: {e}", "error")
                    except Exception as e:
                        with processing_lock:
                            processing_stats['failed'] += 1
                        logger.log(f"[BUILD_TABULAR_DATASET] ✗ Worker error: {e}", "error")
                        logger.log(f"[BUILD_TABULAR_DATASET] Error traceback: {traceback.format_exc()}", "error")

        # Log processing summary
        elapsed_time = time.time() - processing_stats['start_time']
        logger.log(f"[BUILD_TABULAR_DATASET] ========== PROCESSING SUMMARY ===========")
        logger.log(f"[BUILD_TABULAR_DATASET] Total pairs: {processing_stats['total_pairs']}")
        logger.log(f"[BUILD_TABULAR_DATASET] Completed: {processing_stats['completed']}")
        logger.log(f"[BUILD_TABULAR_DATASET] Failed: {processing_stats['failed']}")
        logger.log(f"[BUILD_TABULAR_DATASET] Empty results: {processing_stats['empty_results']}")
        logger.log(f"[BUILD_TABULAR_DATASET] Success rate: {(processing_stats['completed']/(processing_stats['total_pairs'] or 1)*100):.1f}%")
        logger.log(f"[BUILD_TABULAR_DATASET] Total time: {elapsed_time:.2f}s")
        logger.log(f"[BUILD_TABULAR_DATASET] Avg time per pair: {elapsed_time/(processing_stats['completed'] or 1):.2f}s")
        logger.log(f"[BUILD_TABULAR_DATASET] SC DataFrames collected: {len(sc_dfs)}")
        logger.log(f"[BUILD_TABULAR_DATASET] ST DataFrames collected: {len(st_dfs)}")
        
        if write_aggregate:
            logger.log(f"[BUILD_TABULAR_DATASET] Writing aggregate files...")
            if sc_dfs:
                try:
                    logger.log(f"[BUILD_TABULAR_DATASET] Concatenating {len(sc_dfs)} SC DataFrames...")
                    sc_df = pl.concat(sc_dfs, how="vertical").sort(["subject_id", "night_id", "epoch_idx"])
                    out_sc.parent.mkdir(parents=True, exist_ok=True)
                    logger.log(f"[BUILD_TABULAR_DATASET] Writing SC aggregate to: {out_sc}")
                    sc_df.write_parquet(out_sc)
                    logger.log(f"[BUILD_TABULAR_DATASET] ✓ SC aggregate saved: {out_sc} ({sc_df.height} rows, {sc_df.width} columns)")
                except Exception as e:
                    logger.log(f"[BUILD_TABULAR_DATASET] ✗ Failed to write SC parquet: {e}", "error")
                    logger.log(f"[BUILD_TABULAR_DATASET] SC error traceback: {traceback.format_exc()}", "error")
            else:
                logger.log(f"[BUILD_TABULAR_DATASET] No SC DataFrames to write", "warning")

            if st_dfs:
                try:
                    logger.log(f"[BUILD_TABULAR_DATASET] Concatenating {len(st_dfs)} ST DataFrames...")
                    st_df = pl.concat(st_dfs, how="vertical").sort(["subject_id", "night_id", "epoch_idx"])
                    out_st.parent.mkdir(parents=True, exist_ok=True)
                    logger.log(f"[BUILD_TABULAR_DATASET] Writing ST aggregate to: {out_st}")
                    st_df.write_parquet(out_st)
                    logger.log(f"[BUILD_TABULAR_DATASET] ✓ ST aggregate saved: {out_st} ({st_df.height} rows, {st_df.width} columns)")
                except Exception as e:
                    logger.log(f"[BUILD_TABULAR_DATASET] ✗ Failed to write ST parquet: {e}", "error")
                    logger.log(f"[BUILD_TABULAR_DATASET] ST error traceback: {traceback.format_exc()}", "error")
            else:
                logger.log(f"[BUILD_TABULAR_DATASET] No ST DataFrames to write", "warning")

        try:
            if sc_df is not None and sc_df.height > 0:
                resume_sc = sc_df.group_by("stage").agg(pl.len().alias("count")).sort("count", descending=True)
                logger.log(f"[BUILD_TABULAR_DATASET] SC classes:\n{resume_sc.to_pandas().to_string(index=False)}")
        except Exception as e:
            logger.log(f"[BUILD_TABULAR_DATASET] Failed to compute SC resume: {e}", "error")

        try:
            if st_df is not None and st_df.height > 0:
                resume_st = st_df.group_by("stage").agg(pl.len().alias("count")).sort("count", descending=True)
                logger.log(f"[BUILD_TABULAR_DATASET] ST classes:\n{resume_st.to_pandas().to_string(index=False)}")
        except Exception as e:
            logger.log(f"[BUILD_TABULAR_DATASET] Failed to compute ST resume: {e}", "error")

    except Exception as e:
        elapsed_time = time.time() - (processing_stats['start_time'] or time.time())
        logger.log(f"[BUILD_TABULAR_DATASET] ========== FATAL ERROR ===========", "error")
        logger.log(f"[BUILD_TABULAR_DATASET] Fatal error after {elapsed_time:.2f}s: {e}", "error")
        logger.log(f"[BUILD_TABULAR_DATASET] Processing stats at failure:", "error")
        logger.log(f"[BUILD_TABULAR_DATASET] - Total pairs: {processing_stats['total_pairs']}", "error")
        logger.log(f"[BUILD_TABULAR_DATASET] - Completed: {processing_stats['completed']}", "error")
        logger.log(f"[BUILD_TABULAR_DATASET] - Failed: {processing_stats['failed']}", "error")
        logger.log(f"[BUILD_TABULAR_DATASET] - Empty results: {processing_stats['empty_results']}", "error")
        logger.log(f"[BUILD_TABULAR_DATASET] Fatal error traceback: {traceback.format_exc()}", "error")
        raise


if __name__ == "__main__":
    import argparse
    base = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Sleep-EDF → SC & ST (parallel, Polars)")

    ap.add_argument("--root", required=False, default=str(base / "datalake" / "raw"),
                    help="Root folder (e.g., .../datalake/raw)")

    ap.add_argument("--out-sc", required=False, default=str(base / "datalake" / "processed"),
                    help="Output base for Sleep Cassette (root for shards and/or aggregate)")

    ap.add_argument("--out-st", required=False, default=str(base / "datalake" / "processed"),
                    help="Output base for Sleep Telemetry (root for shards and/or aggregate)")

    ap.add_argument("--workers", type=int, default=None, help="Max workers (default: CPUs-1)")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar")

    ap.add_argument("--use-hash", dest="use_hash", action="store_true", default=True, help="Use hash file to skip already processed pairs")
    ap.add_argument("--no-use-hash", dest="use_hash", action="store_false", help="Ignore hash file (process everything)")
    ap.add_argument("--incremental", dest="incremental", action="store_true", default=True, help="Write one parquet shard per pair")
    ap.add_argument("--no-incremental", dest="incremental", action="store_false", help="Disable incremental shards")
    ap.add_argument("--write-aggregate", dest="write_aggregate", action="store_true", default=False, help="Also write the single aggregate parquet files (SC/ST)")
    ap.add_argument("--hash-file", dest="hash_file", default=str(HASH_FILE_DEFAULT), help="Path to hash file (one sha256 per line)")

    args = ap.parse_args()

    build_tabular_dataset(
        root_dir=args.root,
        out_sc=args.out_sc,
        out_st=args.out_st,
        workers=args.workers,
        show_progress=not args.no_progress,
        use_hash=args.use_hash,
        incremental=args.incremental,
        write_aggregate=args.write_aggregate,
        hash_file=Path(args.hash_file),
    )