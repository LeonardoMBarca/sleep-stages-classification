import os
import pyedflib
import glob
import re

import numpy as np
import pandas as pd

from typing import Tuple, Dict, List
from scipy.signal import welch
from utils import slugify, normspace, best_match_idx, log

EPOCH_LEN = 30.0 # Each epoch has 30 seconds. (Defined in the manual: Rechtschaffen & Kales (R&K) - A manual of standardized terminology, techniques and scoring system for sleep stages of humans subjects.

# Classic bands of EEG (Hz)
# The brain dows not always oscillate at the same rhythm; it has preferred bands of activity, which are called frequency bands.

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),
    "beta":  (16.0, 30.0),
}

# Feature Targets: "Canonical" names that we will try to select. Basically a similar name replacer in case there is a discrepancy
# Division into High and Low
# High are high frequency channels (100 Hz) like EEG, EOG, ...
# Low are low frequency channels (1 Hz) like Temp rectal, ...
CANON_HIGH_HINT = {
    "EEG_Fpz_Cz": ["EEG Fpz-Cz", "Fpz-Cz", "EEG FPZ-CZ"],
    "EEG_Pz_Oz":  ["EEG Pz-Oz", "Pz-Oz", "EEG PZ-OZ"],
    "EOG":        ["EOG horizontal", "Horizontal EOG", "EOG"],
}
CANON_LOW_HINT = {
    "EMG_submental": ["EMG submental", "Submental EMG", "EMG"],
    "Resp_oronasal": ["Resp oro-nasal", "Oronasal Respiration", "Respiration"],
    "Temp_rectal":   ["Temp rectal", "Rectal Temp", "Temperature"],
    "Event_marker":  ["Event marker", "Marker", "Event"],
}

def pair_psg_hyp(root_dir: str) -> List[Tuple[str, str, str, str]]:
    pairs = []
    for psg in glob.glob(os.path.join(root_dir, "**", "*-PSG.edf"), recursive=True):
        base = os.path.basename(psg)
        stem = base[:-8]  # remove "-PSG.edf"
        folder = os.path.dirname(psg)

        prefix6 = stem[:6]  # SCxxxx / STxxxx
        cands = sorted(glob.glob(os.path.join(folder, f"{prefix6}*-Hypnogram.edf")))
        if not cands:
            cands = sorted(glob.glob(os.path.join(folder, f"{stem}-Hypnogram.edf")))
        if not cands:
            log(f"Nenhum hypnograma para {base}")
            continue
        hyp = cands[0]

        m = re.match(r"^(SC|ST)(\d{4})([A-Z]\d)?", stem, flags=re.IGNORECASE)
        if m:
            subject_id = f"{m.group(1).upper()}{m.group(2)}"
            night_id = m.group(3).upper() if m.group(3) else "N0"
        else:
            subject_id, night_id = stem, "N0"

        pairs.append((psg, hyp, subject_id, night_id))
    log(f"Pares encontrados: {len(pairs)}")
    return pairs

# =========================
# Hypnograma → rótulos por epoch (alinhado)
# =========================
def read_hyp_epochs_aligned(psg_file: str, hyp_file: str, epoch_len: float = EPOCH_LEN) -> pd.DataFrame:
    with pyedflib.EdfReader(psg_file) as fpsg:
        start_psg = fpsg.getStartdatetime()
        ns0 = fpsg.getNSamples()[0]
        fs0 = fpsg.getSampleFrequencies()[0]
        dur_psg_sec = float(ns0) / float(fs0)

    with pyedflib.EdfReader(hyp_file) as fhyp:
        start_hyp = fhyp.getStartdatetime()
        onsets, durations, desc = fhyp.readAnnotations()

    offset_sec = (start_hyp - start_psg).total_seconds()
    log(f"Alinhamento Hyp↔PSG: offset = {offset_sec:.3f} s")

    starts, ends, stages = [], [], []
    kept, ignored = 0, 0
    for o, d, txt in zip(onsets, durations, desc):
        t = normspace(txt).upper()
        if not t or (d is None) or (float(d) <= 0):
            ignored += 1
            continue
        if "SLEEP STAGE" in t:
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
                s = "N3"
            if s is None:
                ignored += 1
                continue
            starts.append(float(o) + offset_sec)
            ends.append(float(o) + float(d) + offset_sec)
            stages.append(s)
            kept += 1
        else:
            ignored += 1

    if kept == 0:
        raise ValueError(f"Nenhum estágio válido em {os.path.basename(hyp_file)}")
    log(f"Hyp events mantidos: {kept} | ignorados: {ignored}")

    hyp = pd.DataFrame({"start": starts, "end": ends, "stage": stages}).sort_values("start").reset_index(drop=True)

    # clip para dentro da duração do PSG
    hyp["start"] = hyp["start"].clip(lower=0.0, upper=dur_psg_sec)
    hyp["end"]   = hyp["end"].clip(lower=0.0, upper=dur_psg_sec)
    hyp = hyp[hyp["end"] > hyp["start"]].reset_index(drop=True)

    total_time = min(hyp["end"].max(), dur_psg_sec)
    n_epochs = int(np.floor(total_time / epoch_len))
    log(f"Duração PSG ~ {dur_psg_sec/3600:.2f} h | total_time usado ~ {total_time/3600:.2f} h | n_epochs = {n_epochs}")

    rows = []
    empty_epochs = 0
    for i in range(n_epochs):
        t0 = i * epoch_len
        tc = t0 + epoch_len / 2.0
        hit = hyp[(hyp["start"] <= tc) & (tc < hyp["end"])]
        stage = hit.iloc[0]["stage"] if len(hit) else None
        if stage is None:
            empty_epochs += 1
        rows.append({"epoch_idx": i, "t0_sec": t0, "stage": stage})

    if empty_epochs:
        log(f"Epochs sem rótulo (fora de eventos): {empty_epochs}")

    df = pd.DataFrame(rows).dropna(subset=["stage"]).reset_index(drop=True)
    df["stage"] = pd.Categorical(df["stage"], categories=["W", "N1", "N2", "N3", "REM"])
    log(f"Epochs rotulados: {len(df)}")
    return df

# =========================
# PSG → epochs por canal
# =========================
def read_and_epoch_channel(reader: pyedflib.EdfReader, ch_idx: int, epoch_len: float, expected_fs: float) -> np.ndarray:
    x = reader.readSignal(ch_idx).astype(np.float32)
    fs_i = float(reader.getSampleFrequencies()[ch_idx])
    n_target = int(round(epoch_len * expected_fs))

    if abs(fs_i - expected_fs) < 1e-6:
        n_epochs = len(x) // n_target
        x = x[: n_epochs * n_target]
        return x.reshape(n_epochs, n_target)

    if fs_i > expected_fs:
        factor = int(round(fs_i / expected_fs))
        x = x[: (len(x) // factor) * factor].reshape(-1, factor).mean(axis=1)
    else:
        factor = int(round(expected_fs / fs_i))
        x = np.repeat(x, factor)

    n_epochs = len(x) // n_target
    x = x[: n_epochs * n_target]
    return x.reshape(n_epochs, n_target)

def read_psg_epochs(psg_file: str, epoch_len: float = EPOCH_LEN) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    high, low = {}, {}
    with pyedflib.EdfReader(psg_file) as f:
        labels_raw = [normspace(l) for l in f.getSignalLabels()]
        fs = f.getSampleFrequencies()
        log("Canais detectados (label @ fs):")
        for i, (lab, fsi) in enumerate(zip(labels_raw, fs)):
            log(f"  - {i:02d}: {lab} @ {fsi} Hz")

        for i, lab in enumerate(labels_raw):
            fs_i = float(fs[i])
            name = slugify(lab)  # nome estável para colunas
            if fs_i >= 50.0:      # high
                arr = read_and_epoch_channel(f, i, epoch_len, expected_fs=100.0)
                high[name] = arr
            elif fs_i <= 2.0:     # low
                arr = read_and_epoch_channel(f, i, epoch_len, expected_fs=1.0)
                low[name] = arr
            else:
                # taxa intermediária rara → aproxima para mais próximo
                if fs_i > 2.0:
                    arr = read_and_epoch_channel(f, i, epoch_len, expected_fs=100.0)
                    high[name] = arr
                else:
                    arr = read_and_epoch_channel(f, i, epoch_len, expected_fs=1.0)
                    low[name] = arr

    # Harmoniza n_epochs pelo mínimo
    n_list = []
    for d in (high, low):
        for v in d.values():
            n_list.append(v.shape[0])
    if not n_list:
        return {}, {}, {}
    n_epochs = min(n_list)
    for d in (high, low):
        for k in list(d.keys()):
            d[k] = d[k][:n_epochs]

    log(f"n_epochs (comum) após corte: {n_epochs} | high_ch={len(high)} | low_ch={len(low)}")
    return high, low, {"high": 100.0, "low": 1.0}

# =========================
# Features
# =========================
def bandpower_welch(x: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    nperseg = max(int(4 * fs), 8)
    noverlap = nperseg // 2
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mask = (freqs >= fmin) & (freqs < fmax)
    return float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0

def compute_features_for_epoch(epoch_high: Dict[str, np.ndarray], epoch_low: Dict[str, np.ndarray]) -> Dict[str, float]:
    feats = {}

    # Alta taxa (100 Hz): para TODOS os canais high → bandpowers + stats
    for ch, x in epoch_high.items():
        # bandas clássicas (mesmo se não for EEG; manter política uniforme)
        for band, (fmin, fmax) in EEG_BANDS.items():
            feats[f"{ch}_{band}_pow"] = bandpower_welch(x, fs=100.0, fmin=fmin, fmax=fmax)
        feats[f"{ch}_rms"] = float(np.sqrt(np.mean(x**2)))
        feats[f"{ch}_var"] = float(np.var(x))

    # Baixa taxa (1 Hz): para TODOS os canais low → estatísticas simples
    for ch, x in epoch_low.items():
        feats[f"{ch}_mean_1hz"]  = float(np.mean(x))
        feats[f"{ch}_std_1hz"]   = float(np.std(x))
        feats[f"{ch}_min_1hz"]   = float(np.min(x))
        feats[f"{ch}_max_1hz"]   = float(np.max(x))
        feats[f"{ch}_slope_1hz"] = float(x[-1] - x[0])
        feats[f"{ch}_rms_1hz"]   = float(np.sqrt(np.mean(x**2)))

    return feats

# =========================
# Processamento de um par
# =========================
def process_record(psg_file: str, hyp_file: str, subject_id: str, night_id: str) -> pd.DataFrame:
    log(f"--- Processando {os.path.basename(psg_file)} | {os.path.basename(hyp_file)} ---")
    y_df = read_hyp_epochs_aligned(psg_file, hyp_file, epoch_len=EPOCH_LEN)

    high_all, low_all, fs_dict = read_psg_epochs(psg_file, epoch_len=EPOCH_LEN)
    if not high_all and not low_all:
        log("Nenhum canal utilizável encontrado.")
        return pd.DataFrame()

    # Info canônica só para log (não limita)
    high_keys = list(high_all.keys())
    low_keys  = list(low_all.keys())
    for cname, clist in CANON_HIGH_HINT.items():
        idx = best_match_idx(high_keys, clist)
        if idx is not None:
            log(f"Canal canônico HIGH encontrado: {cname} -> {high_keys[idx]}")
    for cname, clist in CANON_LOW_HINT.items():
        idx = best_match_idx(low_keys, clist)
        if idx is not None:
            log(f"Canal canônico LOW encontrado: {cname} -> {low_keys[idx]}")

    # Harmoniza X vs y pelo número comum de epochs
    n_x = min([v.shape[0] for v in high_all.values()] + [v.shape[0] for v in low_all.values()] if (high_all or low_all) else [0])
    n_epochs = min(n_x, len(y_df))
    log(f"n_epochs (final) = {n_epochs} (X={n_x}, y={len(y_df)})")
    if n_epochs == 0:
        return pd.DataFrame()

    rows = []
    for i in range(n_epochs):
        ep_high = {k: v[i, :] for k, v in high_all.items()}
        ep_low  = {k: v[i, :] for k, v in low_all.items()}
        feats = compute_features_for_epoch(ep_high, ep_low)
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
    log(f"Linhas geradas: {len(df)} | Colunas: {len(df.columns)}")
    return df