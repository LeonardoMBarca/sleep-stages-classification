import os
import pandas as pd
from utils import log
from labeler import pair_psg_hyp, process_record

def build_tabular_dataset(root_dir: str, out_path: str, to_csv: bool = False) -> pd.DataFrame:
    pairs = pair_psg_hyp(root_dir)
    if not pairs:
        raise FileNotFoundError("Nenhum par *-PSG.edf / *-Hypnogram.edf encontrado.")

    all_dfs = []
    for psg, hyp, sid, nid in pairs:
        try:
            df = process_record(psg, hyp, subject_id=sid, night_id=nid)
            if not df.empty:
                all_dfs.append(df)
            else:
                log(f"[WARN] Sem dados válidos em {os.path.basename(psg)}")
        except Exception as e:
            log(f"[WARN] Falha em {os.path.basename(psg)}: {e}")

    if not all_dfs:
        raise RuntimeError("Nenhum registro válido processado.")

    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sort_values(["subject_id", "night_id", "epoch_idx"]).reset_index(drop=True)

    if to_csv:
        full.to_csv(out_path, index=False)
        log(f"[OK] CSV salvo em: {out_path}")
    else:
        full.to_parquet(out_path, index=False)
        log(f"[OK] Parquet salvo em: {out_path}")

    # Resumo final
    log("Resumo de classes:")
    log(str(full["stage"].value_counts(dropna=False)))
    return full

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Sleep-EDF → epochs tabulares (todos os canais, logs, SC/ST robusto)")
    ap.add_argument("--root", required=True, help="Pasta raiz (ex.: .../sleep-edfx/1.0.0/sleep-cassette ou .../sleep-edfx/1.0.0)")
    ap.add_argument("--out", required=True, help="Caminho de saída (ex.: sc_epochs.parquet)")
    ap.add_argument("--csv", action="store_true", help="Salvar CSV (padrão: Parquet)")
    args = ap.parse_args()
    build_tabular_dataset(args.root, args.out, to_csv=args.csv)
