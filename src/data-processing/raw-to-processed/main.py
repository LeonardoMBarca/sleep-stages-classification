import os, sys
import polars as pl

from labeler import pair_psg_hyp, process_record
from utils import _make_progress
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "logger")))
from logger import Logger


logger = Logger()

def build_tabular_dataset(root_dir: str, output_path: str) -> pl.DataFrame:
    """Process all PSG/Hypnogram pairs in root_dir into a single tabular dataset (Polars)"""
    logger.log(f"[BUILD_TABULAR_DATASET] Starting processing in: {root_dir}")

    try:
        pairs = pair_psg_hyp(logger, root_dir)
        if not pairs: 
            raise FileNotFoundError("No *-SPG.edf and *-Hypnogram.edf pairs found.")
        
        all_dfs = []
        with _make_progress() as progress:
            files_task = progress.add_task("Files", total=len(pairs))
            for psg, hyp, sid, nid in pairs:
                try:
                    df = process_record(logger, psg, hyp, subject_id=sid, night_id=nid, progress=progress)
                    if df is not None and df.height > 0:
                        all_dfs.append(df)
                    
                    else:
                        logger.log(f"[BUILD_TABULAR_DATASET] No data extracted from {os.path.basename(psg)}")
                    
                except Exception as e:
                    logger.log(f"[BUILD_TABULAR_DATASET] Error processing {os.path.basename(psg)}: {e}", "warning")
                
                finally:
                    progress.update(files_task, advance=1)
            
        if not all_dfs:
            raise RuntimeError("No valid record processed.")
        
        full = pl.concat(all_dfs, how="vertical").sort(["subject_id", "night_id", "epoch_idx"])

        try:
            full.write_parquet(output_path)
            logger.log(f"[BUILD_TABULAR_DATASET] Dataset saved to: {output_path}")
        
        except Exception as e:
            logger.log(f"[BUILD_TABULAR_DATASET] Failed to write parquet: {e}", "error")
        
        try:
            resume = (
                full.group_by("stage")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
            )
            logger.log(f"[BUILD_TABULAR_DATASET] Classes resume:\n{resume}\n\n{resume.to_pandas().to_string(index=False)}")
        except Exception as e:
            logger.log(f"[BUILD_TABULAR_DATASET] Failed to compute resume: {e}", "error")
        
        return full

    except Exception as e:
        logger.log(f"[BUILD_TABULAR_DATASET] Fatal error: {e}", "error")

        raise


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sleep-EDF -> Tabular epochs (Polars)")
    ap.add_argument("--root", required=False, help="Root folder (e.g., .../datalake/raw)", default=f"{Path(__file__).resolve().parents[3]}/datalake/raw")
    ap.add_argument("--output", required=False, help="Output Parquet file", default=f"{Path(__file__).resolve().parents[3]}/datalake/processed/sleep_edf_dataset.parquet")
    args = ap.parse_args()
    build_tabular_dataset(args.root, args.output)