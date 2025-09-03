import os, sys
import polars as pl
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logger")))

from labeler import pair_psg_hyp, process_record
from utils import _make_progress, _process_one_pair, _pair_key, _write_shard
from hash_handler import HashHandler
from logger import Logger

HASH_FILE_DEFAULT = Path(__file__).resolve().parents[3] / "datalake" / "processed" / "hash_files_processed.txt"

logger = Logger()

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
    logger.log(f"[BUILD_TABULAR_DATASET] Starting processing in: {root_dir}")

    out_sc = Path(out_sc) / "sleep-cassette" / "sleep_cassette_dataset.parquet"
    out_st = Path(out_st) / "sleep-telemetry" / "sleep_telemetry_dataset.parquet"

    hh = HashHandler(hash_file)
    processed_hashes = hh.load() if use_hash else set()
    logger.log(f"[HASH] use_hash={use_hash} | loaded_entries={len(processed_hashes)}")

    sc_df = None
    st_df = None

    try:
        pairs = pair_psg_hyp(logger, root_dir)
        if not pairs:
            raise FileNotFoundError("No *-PSG.edf and *-Hypnogram.edf pairs found.")

        if use_hash:
            kept_pairs = []
            skipped = 0
            for psg, hyp, sid, nid in pairs:
                bpsg, bhyp = _pair_key(psg, hyp)
                if (hh.hash_name(bpsg) in processed_hashes) and (hh.hash_name(bhyp) in processed_hashes):
                    skipped += 1
                else:
                    kept_pairs.append((psg, hyp, sid, nid))
            logger.log(f"[HASH] Skipping {skipped} pairs already processed; keeping {len(kept_pairs)}")
            pairs = kept_pairs

        if not pairs:
            logger.log("[BUILD_TABULAR_DATASET] Nothing to do after hash filtering.")
            return

        if workers is None:
            workers = max(1, (os.cpu_count() or 4) - 1)

        logger.log(f"[BUILD_TABULAR_DATASET] Using {workers} workers")

        sc_dfs, st_dfs = [], []

        def _submit_all(ex):
            return [ex.submit(_process_one_pair, p) for p in pairs]

        if show_progress:
            try:
                with _make_progress() as progress:
                    task = progress.add_task("Files (parallel)", total=len(pairs))
                    with ProcessPoolExecutor(max_workers=workers, mp_context=None) as ex:
                        futures = _submit_all(ex)
                        for fut in as_completed(futures):
                            try:
                                sid, df = fut.result()
                                if df is not None and df.height > 0:
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
                            except Exception as e:
                                logger.log(f"[BUILD_TABULAR_DATASET] Worker error: {e}", "warning")
                            finally:
                                progress.update(task, advance=1)
            except Exception as e:
                logger.log(f"[BUILD_TABULAR_DATASET] Progress failed ({e}); running parallel without bar.", "warning")
                with ProcessPoolExecutor(max_workers=workers, mp_context=None) as ex:
                    futures = _submit_all(ex)
                    for fut in as_completed(futures):
                        try:
                            sid, df = fut.result()
                            if df is not None and df.height > 0:
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
                        except Exception as e:
                            logger.log(f"[BUILD_TABULAR_DATASET] Worker error: {e}", "warning")
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=None) as ex:
                futures = [ex.submit(_process_one_pair, p) for p in pairs]
                for fut in as_completed(futures):
                    try:
                        sid, df = fut.result()
                        if df is not None and df.height > 0:
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
                    except Exception as e:
                        logger.log(f"[BUILD_TABULAR_DATASET] Worker error: {e}", "warning")

        if write_aggregate:
            if sc_dfs:
                try:
                    sc_df = pl.concat(sc_dfs, how="vertical").sort(["subject_id", "night_id", "epoch_idx"])
                    out_sc.parent.mkdir(parents=True, exist_ok=True)
                    sc_df.write_parquet(out_sc)
                    logger.log(f"[BUILD_TABULAR_DATASET] SC (aggregate) saved: {out_sc} (rows={sc_df.height})")
                except Exception as e:
                    logger.log(f"[BUILD_TABULAR_DATASET] Failed to write SC parquet: {e}", "error")

            if st_dfs:
                try:
                    st_df = pl.concat(st_dfs, how="vertical").sort(["subject_id", "night_id", "epoch_idx"])
                    out_st.parent.mkdir(parents=True, exist_ok=True)
                    st_df.write_parquet(out_st)
                    logger.log(f"[BUILD_TABULAR_DATASET] ST (aggregate) saved: {out_st} (rows={st_df.height})")
                except Exception as e:
                    logger.log(f"[BUILD_TABULAR_DATASET] Failed to write ST parquet: {e}", "error")

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
        logger.log(f"[BUILD_TABULAR_DATASET] Fatal error: {e}", "error")
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