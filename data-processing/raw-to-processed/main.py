import os
import pandas as pd
from utils import _pair_psg_hyp
from labeler import _process_record

def build_tabular_dataset(root_dir: str, out_path: str, to_csv: bool = False) -> pd.DataFrame:
    pairs = _pair_psg_hyp(root_dir)
    if not pairs:
        raise FileNotFoundError("No pair *-PSG.edf / *-Hypnogram.edf finded.")

    all_dfs = []
    for psg, hyp, sid, nid in pairs:
        print(f"[INFO] Processing: {os.path.basename(psg)}  |  {os.path.basename(hyp)}")
        try:
            df = _process_record(psg, hyp, subject_id=sid, night_id=nid)
            if not df.empty:
                all_dfs.append(df)
            else:
                print(f"[WARN] No valid datas in {os.path.basename(psg)}")
        except Exception as e:
            print(f"[WARN] Failure in {os.path.basename(psg)}: {e}")

    if not all_dfs:
        raise RuntimeError("No valid register processed")

    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sort_values(["subject_id", "night_id", "epoch_idx"]).reset_index(drop=True)

    if to_csv:
        full.to_csv(out_path, index=False)
        print(f"[OK] CSV saved in: {out_path}")
    else:
        full.to_parquet(out_path, index=False)
        print(f"[OK] Parquet saved in: {out_path}")

    return full

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Sleep-EDF → Tabular epochs (ML clássico, robusto SC/ST)")
    ap.add_argument("--root", required=True, help="Root (ex.: .../sleep-edfx/1.0.0/sleep-cassette or .../sleep-edfx/1.0.0)")
    ap.add_argument("--out", required=True, help="Caminho de saída (ex.: sc_epochs.parquet)")
    ap.add_argument("--csv", action="store_true", help="Salvar CSV (padrão: Parquet)")
    args = ap.parse_args()
    build_tabular_dataset(args.root, args.out, to_csv=args.csv)
