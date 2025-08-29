# sleep_edf_to_epochs_tabular.py
import os
import re
import glob
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.signal import welch
import pyedflib

# =========================
# Configurações principais
# =========================
EPOCH_LEN = 30.0  # segundos

# Bandas clássicas de EEG (Hz)
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),
    "beta":  (16.0, 30.0),
}

# Alvos de features: nomes "canônicos" que tentaremos selecionar
CANON_HIGH = {
    "EEG_Fpz_Cz": ["EEG Fpz-Cz", "Fpz-Cz", "EEG FPZ-CZ"],
    "EEG_Pz_Oz":  ["EEG Pz-Oz", "Pz-Oz", "EEG PZ-OZ"],
    "EOG":        ["EOG horizontal", "Horizontal EOG", "EOG"],
}
CANON_LOW = {
    "EMG_submental": ["EMG submental", "Submental EMG", "EMG"],
    "Resp_oronasal": ["Resp oro-nasal", "Oronasal Respiration", "Respiration"],
    "Temp_rectal":   ["Temp rectal", "Rectal Temp", "Temperature"],
}

# =========================
# Utilidades
# =========================
def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _best_match(label_pool: List[str], candidates: List[str]) -> Optional[int]:
    """Retorna o índice do 1º label em label_pool que corresponda a algum candidate (case-insensitive, contains)."""
    low_pool = [l.lower() for l in label_pool]
    for c in candidates:
        c_low = c.lower()
        # tentativa exata
        if c_low in low_pool:
            return low_pool.index(c_low)
    # contains flexível
    for i, lab in enumerate(low_pool):
        for c in candidates:
            if c.lower() in lab:
                return i
    return None

def _pair_psg_hyp(root_dir: str) -> List[Tuple[str, str, str, str]]:
    """Encontra pares *-PSG.edf e *-Hypnogram.edf na árvore."""
    pairs = []
    for psg in glob.glob(os.path.join(root_dir, "**", "*-PSG.edf"), recursive=True):
        base = os.path.basename(psg)
        stem = base[:-8]  # remove "-PSG.edf"
        folder = os.path.dirname(psg)

        # Heurística 1: prefixo SCxxxx / STxxxx (6 chars)
        prefix6 = stem[:6]
        cands = sorted(glob.glob(os.path.join(folder, f"{prefix6}*-Hypnogram.edf")))
        # Heurística 2: nome exato do stem
        if not cands:
            cands = sorted(glob.glob(os.path.join(folder, f"{stem}-Hypnogram.edf")))
        if not cands:
            continue
        hyp = cands[0]

        # subject_id e night_id
        m = re.match(r"^(SC|ST)(\d{4})([A-Z]\d)?", stem, flags=re.IGNORECASE)
        if m:
            subject_id = f"{m.group(1).upper()}{m.group(2)}"
            night_id = m.group(3).upper() if m.group(3) else "N0"
        else:
            subject_id, night_id = stem, "N0"

        pairs.append((psg, hyp, subject_id, night_id))
    return pairs

# =========================
# Hipnograma → rótulos por epoch (com offset)
# =========================
def _read_hyp_epochs_aligned(psg_file: str, hyp_file: str, epoch_len: float = EPOCH_LEN) -> pd.DataFrame:
    """Alinha Hyp↔PSG por startdatetime e retorna epochs com rótulo pela regra do centro."""
    with pyedflib.EdfReader(psg_file) as fpsg:
        start_psg = fpsg.getStartdatetime()
        # duração real do PSG (soma por canal é redundante; use comprimento do 1º canal):
        ns = fpsg.getNSamples()[0]
        fs0 = fpsg.getSampleFrequencies()[0]
        dur_psg_sec = float(ns) / float(fs0)

    with pyedflib.EdfReader(hyp_file) as fhyp:
        start_hyp = fhyp.getStartdatetime()
        onsets, durations, desc = fhyp.readAnnotations()

    offset_sec = (start_hyp - start_psg).total_seconds()

    # Constrói tabela de eventos válidos
    starts, ends, stages = [], [], []
    for o, d, txt in zip(onsets, durations, desc):
        t = _norm_label(txt).upper()
        # ignora descrições vazias e durações inválidas
        if not t or (d is None) or (float(d) <= 0):
            continue
        if "SLEEP STAGE" in t:
            # normaliza estágios
            s = None
            if t.endswith(" W"):
                s = "W"
            elif t.endswith(" R"):
                s = "REM"
            elif t.endswith(" 1"):
                s = "N1"
            elif t.endswith(" 2"):
                s = "N2"
            elif t.endswith(" 3") or t.endswith(" 4"):
                s = "N3"  # 3/4 → N3
            if s is not None:
                starts.append(float(o) + offset_sec)
                ends.append(float(o) + float(d) + offset_sec)
                stages.append(s)
        # ignora Movement/Artefact/? por padrão

    if not stages:
        raise ValueError(f"Nenhum estágio válido em {os.path.basename(hyp_file)}")

    hyp = pd.DataFrame({"start": starts, "end": ends, "stage": stages}).sort_values("start").reset_index(drop=True)

    # recorta ao máximo tempo do PSG
    hyp["start"] = hyp["start"].clip(lower=0.0, upper=dur_psg_sec)
    hyp["end"]   = hyp["end"].clip(lower=0.0, upper=dur_psg_sec)
    hyp = hyp[hyp["end"] > hyp["start"]].reset_index(drop=True)

    total_time = min(hyp["end"].max(), dur_psg_sec)
    n_epochs = int(np.floor(total_time / epoch_len))

    rows = []
    for i in range(n_epochs):
        t0 = i * epoch_len
        tc = t0 + epoch_len / 2.0
        hit = hyp[(hyp["start"] <= tc) & (tc < hyp["end"])]
        rows.append({
            "epoch_idx": i,
            "t0_sec": t0,
            "stage": hit.iloc[0]["stage"] if len(hit) else None
        })

    df = pd.DataFrame(rows).dropna(subset=["stage"]).reset_index(drop=True)
    df["stage"] = pd.Categorical(df["stage"], categories=["W", "N1", "N2", "N3", "REM"])
    return df

# =========================
# PSG → segmentos por epoch (classificação por fs real)
# =========================
def _read_and_epoch_channel(reader: pyedflib.EdfReader, ch_idx: int, epoch_len: float, expected_fs: float) -> np.ndarray:
    """Lê um canal e o coloca no grid desejado (expected_fs) com down/upsample simples, retornando [n_epochs, n_samples]."""
    x = reader.readSignal(ch_idx).astype(np.float32)
    fs_i = float(reader.getSampleFrequencies()[ch_idx])
    n_target = int(round(epoch_len * expected_fs))

    if abs(fs_i - expected_fs) < 1e-6:
        n_epochs = len(x) // n_target
        x = x[: n_epochs * n_target]
        return x.reshape(n_epochs, n_target)

    # fs_i > expected_fs -> downsample por média de blocos inteiros
    if fs_i > expected_fs:
        factor = int(round(fs_i / expected_fs))
        x = x[: (len(x) // factor) * factor].reshape(-1, factor).mean(axis=1)
    else:
        # fs_i < expected_fs -> upsample por repetição
        factor = int(round(expected_fs / fs_i))
        x = np.repeat(x, factor)

    n_epochs = len(x) // n_target
    x = x[: n_epochs * n_target]
    return x.reshape(n_epochs, n_target)

def _read_psg_epochs(psg_file: str, epoch_len: float = EPOCH_LEN) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Separa canais por frequência real:
      - HIGH (>= 50 Hz) -> reamostrados para 100 Hz → [n_epochs, 3000]
      - LOW  (<=  2 Hz) -> reamostrados para   1 Hz → [n_epochs,   30]
    Retorna dois dicionários {nome_canal_normalizado: array}.
    """
    high, low = {}, {}
    with pyedflib.EdfReader(psg_file) as f:
        labels = [_norm_label(l) for l in f.getSignalLabels()]
        fs = f.getSampleFrequencies()

        for i, lab in enumerate(labels):
            fs_i = float(fs[i])
            name = lab.replace(" ", "_")
            if fs_i >= 50.0:  # high-rate
                high[name] = _read_and_epoch_channel(f, i, epoch_len, expected_fs=100.0)
            elif fs_i <= 2.0:  # low-rate
                low[name] = _read_and_epoch_channel(f, i, epoch_len, expected_fs=1.0)
            else:
                # taxas inusitadas: manda para o grupo mais próximo
                if fs_i > 2.0:
                    high[name] = _read_and_epoch_channel(f, i, epoch_len, expected_fs=100.0)
                else:
                    low[name] = _read_and_epoch_channel(f, i, epoch_len, expected_fs=1.0)

    # Harmoniza n_epochs com base no menor
    n_list = []
    for d in (high, low):
        for v in d.values():
            n_list.append(v.shape[0])
    if not n_list:
        return {}, {}
    n_epochs = min(n_list)
    for d in (high, low):
        for k in list(d.keys()):
            d[k] = d[k][:n_epochs]
    return high, low

# =========================
# Seleção de canais-alvo para features
# =========================
def _select_canonical_channels(high: Dict[str, np.ndarray], low: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Seleciona canais canônicos (se existirem) para features com nomes estáveis."""
    high_keys = list(high.keys())
    low_keys  = list(low.keys())

    # Mapear por nomes canônicos → procurar melhor correspondência no conjunto real
    selected_high = {}
    for canon, cands in CANON_HIGH.items():
        idx = _best_match(high_keys, cands)
        if idx is not None:
            selected_high[canon] = high[high_keys[idx]]

    selected_low = {}
    for canon, cands in CANON_LOW.items():
        idx = _best_match(low_keys, cands)
        if idx is not None:
            selected_low[canon] = low[low_keys[idx]]

    return selected_high, selected_low

# =========================
# Features por epoch
# =========================
def _bandpower_welch(x: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    """Potência na banda [fmin, fmax] com Welch e integração discreta."""
    nperseg = int(4 * fs)
    nperseg = max(nperseg, 8)  # garante mínimo
    noverlap = nperseg // 2
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mask = (freqs >= fmin) & (freqs < fmax)
    return float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0

def _compute_epoch_features(epoch_high: Dict[str, np.ndarray], epoch_low: Dict[str, np.ndarray]) -> Dict[str, float]:
    feats = {}
    # EEG (100 Hz): bandpowers + stats
    for eeg_key in ["EEG_Fpz_Cz", "EEG_Pz_Oz"]:
        if eeg_key in epoch_high:
            x = epoch_high[eeg_key]
            for band, (fmin, fmax) in EEG_BANDS.items():
                feats[f"{eeg_key.lower()}_{band}_pow"] = _bandpower_welch(x, fs=100.0, fmin=fmin, fmax=fmax)
            feats[f"{eeg_key.lower()}_rms"] = float(np.sqrt(np.mean(x**2)))
            feats[f"{eeg_key.lower()}_var"] = float(np.var(x))

    # EOG (100 Hz): variância/RMS
    if "EOG" in epoch_high:
        x = epoch_high["EOG"]
        feats["eog_var"] = float(np.var(x))
        feats["eog_rms"] = float(np.sqrt(np.mean(x**2)))

    # 1 Hz: EMG/Resp/Temp (resumos)
    if "EMG_submental" in epoch_low:
        x = epoch_low["EMG_submental"]
        feats["emg_rms_1hz"]  = float(np.sqrt(np.mean(x**2)))
        feats["emg_mean_1hz"] = float(np.mean(x))
        feats["emg_std_1hz"]  = float(np.std(x))

    if "Resp_oronasal" in epoch_low:
        x = epoch_low["Resp_oronasal"]
        feats["resp_mean_1hz"]  = float(np.mean(x))
        feats["resp_std_1hz"]   = float(np.std(x))
        feats["resp_min_1hz"]   = float(np.min(x))
        feats["resp_max_1hz"]   = float(np.max(x))
        feats["resp_slope_1hz"] = float(x[-1] - x[0])

    if "Temp_rectal" in epoch_low:
        x = epoch_low["Temp_rectal"]
        feats["temp_mean_1hz"]  = float(np.mean(x))
        feats["temp_std_1hz"]   = float(np.std(x))
        feats["temp_min_1hz"]   = float(np.min(x))
        feats["temp_max_1hz"]   = float(np.max(x))
        feats["temp_slope_1hz"] = float(x[-1] - x[0])

    return feats

# =========================
# Pipeline de um par PSG/Hyp
# =========================
def _process_record(psg_file: str, hyp_file: str, subject_id: str, night_id: str) -> pd.DataFrame:
    # 1) Rotulagem por epoch (regra do centro) com alinhamento temporal
    y_df = _read_hyp_epochs_aligned(psg_file, hyp_file, epoch_len=EPOCH_LEN)  # epoch_idx, t0_sec, stage

    # 2) Sinais por epoch, separados por taxa (100 Hz e 1 Hz)
    high_all, low_all = _read_psg_epochs(psg_file, epoch_len=EPOCH_LEN)

    if not high_all and not low_all:
        return pd.DataFrame()

    # 3) Seleciona canais canônicos (se existirem) para features com nomes estáveis
    high, low = _select_canonical_channels(high_all, low_all)

    # 4) Harmoniza n_epochs entre X e y
    n_list = []
    for d in (high, low):
        for v in d.values():
            n_list.append(v.shape[0])
    if not n_list:
        return pd.DataFrame()
    n_epochs_x = min(n_list)
    n_epochs = min(n_epochs_x, len(y_df))
    if n_epochs == 0:
        return pd.DataFrame()

    rows = []
    for i in range(n_epochs):
        ep_high = {k: v[i, :] for k, v in high.items()}
        ep_low  = {k: v[i, :] for k, v in low.items()}
        feats = _compute_epoch_features(ep_high, ep_low)
        row = {
            "subject_id": subject_id,
            "night_id": night_id,
            "epoch_idx": int(y_df.loc[i, "epoch_idx"]),
            "t0_sec": float(y_df.loc[i, "t0_sec"]),
            "stage": y_df.loc[i, "stage"],
        }
        row.update(feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["stage"] = pd.Categorical(df["stage"], categories=["W", "N1", "N2", "N3", "REM"])
    return df

# =========================
# Main builder
# =========================
def build_tabular_dataset(root_dir: str, out_path: str, to_csv: bool = False) -> pd.DataFrame:
    pairs = _pair_psg_hyp(root_dir)
    if not pairs:
        raise FileNotFoundError("Nenhum par *-PSG.edf / *-Hypnogram.edf encontrado.")

    all_dfs = []
    for psg, hyp, sid, nid in pairs:
        print(f"[INFO] Processando: {os.path.basename(psg)}  |  {os.path.basename(hyp)}")
        try:
            df = _process_record(psg, hyp, subject_id=sid, night_id=nid)
            if not df.empty:
                all_dfs.append(df)
            else:
                print(f"[WARN] Sem dados válidos em {os.path.basename(psg)}")
        except Exception as e:
            print(f"[WARN] Falha em {os.path.basename(psg)}: {e}")

    if not all_dfs:
        raise RuntimeError("Nenhum registro válido processado.")

    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sort_values(["subject_id", "night_id", "epoch_idx"]).reset_index(drop=True)

    if to_csv:
        full.to_csv(out_path, index=False)
        print(f"[OK] CSV salvo em: {out_path}")
    else:
        full.to_parquet(out_path, index=False)
        print(f"[OK] Parquet salvo em: {out_path}")

    return full

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Sleep-EDF → epochs tabulares (ML clássico, robusto SC/ST)")
    ap.add_argument("--root", required=True, help="Pasta raiz (ex.: .../sleep-edfx/1.0.0/sleep-cassette ou .../sleep-edfx/1.0.0)")
    ap.add_argument("--out", required=True, help="Caminho de saída (ex.: sc_epochs.parquet)")
    ap.add_argument("--csv", action="store_true", help="Salvar CSV (padrão: Parquet)")
    args = ap.parse_args()
    build_tabular_dataset(args.root, args.out, to_csv=args.csv)
