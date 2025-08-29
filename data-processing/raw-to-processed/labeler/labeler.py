import os
import pyedflib

import numpy as np
import pandas as pd

from typing import Tuple, Dict
from scipy.signal import welch
from utils import _norm_label, _best_match

EPOCH_LEN = 30.0 # Each epoch has 30 seconds. (Defined in the manual: Rechtschaffen & Kales (R&K) - A manual of standardized terminology, techniques and scoring system for sleep stages of humans subjects.

# Classic bands of EEG (Hz)
# The brain dows not always oscillate at the same rhythm; it has preferred bands of activity, which are called frequency bands.
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

# Feature Targets: "Canonical" names that we will try to select. Basically a similar name replacer in case there is a discrepancy
# Division into High and Low
# High are high frequency channels (100 Hz) like EEG, EOG, ...
# Low are low frequency channels (1 Hz) like Temp rectal, ...
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

def _read_hyp_epochs_aligned(psg_file: str, hyp_file: str, epoch_len: float) -> pd.DataFrame:
    """
    Aligns Hyp-PSG by startdatetime and returns labels by center rule
    """
    with pyedflib.EdfReader(psg_file) as fpsg:
        start_psg = fpsg.getStartdatetime()
        # Real duration of PSG (sum per channel is redundant, use lenght of 1st channel)
        ns = fpsg.getNSamples()[0]
        fs0 = fpsg.getSampleFrequencies()[0]
        dur_psg_sec = float(ns) / float(fs0)

    with pyedflib.EdfReader(hyp_file) as fhyp:
        start_hyp = fhyp.getStartdatetime()
        onset, durations, desc = fhyp.readAnnotations()

    offsec_sec = (start_hyp - start_psg).total_seconds()

    # Create valid events table
    starts, ends, stages = [], [], []
    for o, d, txt in zip(onset, durations, desc):
        t = _norm_label(txt).upper()
        # Ignore null descriptions and invalid durations
        if not t or (d is None) or (float(d) <= 0):
            continue
        if "SLEEP STAGE" in t:
            # Normalize stages
            s = None
            if t.endswith("W"):
                s = "W"
            elif t.endswith(" R"):
                s = "REM"
            elif t.endswith(" 1"):
                s = "N1"
            elif t.endswith(" 2"):
                s = "N2"
            elif t.endswith(" 3") or t.endswith(" 4"):
                s = "N3" # 3/4 -> N3
            if s is not None:
                starts.append(float(o) + offsec_sec)
                ends.append(float(o) + float(d) + offsec_sec)
                stages.append(s)
        # Ignore Movement/Artefact/? by default

    if not stages:
        raise ValueError(f"No valid internship in {os.path.basename(hyp_file)}")
    
    hyp = pd.DataFrame({"start": starts, "end": ends, "stage": stages}).sort_values("start").reset_index(drop=True)

    # Cuts PSG's time to the maximum
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

def _read_and_epoch_channel(reader: pyedflib.EdfReader, ch_idx: int, epoch_len: float, expected_fs: float) -> np.ndarray:
    """
    Reads a channel and places it in the desired grid (expected_fs) with simple down/upsample, returning [n_epochs, n_samples].
    """
    x = reader.readSignal(ch_idx).astype(np.float32)
    fs_i = float(reader.getSampleFrequencies()[ch_idx])
    n_target = int(round(epoch_len * expected_fs))

    if abs(fs_i - expected_fs) < 1e-6:
        n_epochs = len(x) // n_target
        x = x[: n_epochs * n_target]

        return x.reshape(n_epochs, n_target)
    
    # fs_i > expected_fs -> downsample by medium entire blocks
    if fs_i > expected_fs:
        factor = int(round(fs_i / expected_fs))
        x = x[: (len(x) // factor) * factor].reshape(-1, factor).mean(axis=1)
    
    else:
        # fs_i < expected_fs -> upsample by repetition
        factor = int(round(expected_fs / fs_i))
        x = np.repeat(x, factor)

    n_epochs = len(x) // n_target
    x = x[: n_epochs * n_target]
    
    return x.reshape(n_epochs, n_target)

def _read_psg_epochs(psg_file: str, epoch_len: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Separates channels by actual frequency:
        - HIGH (>= 50 Hz) -> resampled to 100 Hz → [n_epochs, 3000]
        - LOW (<= 2 Hz) -> resampled to 1 Hz → [n_epochs, 30]
    Returns two dictionaries {normalized_channel_name: array}.
    """
    high, low = {}, {}
    with pyedflib.EdfReader(psg_file) as f:
        labels = [_norm_label(l) for l in f.getSignalLabels()]
        fs = f.getSampleFrequencies()

        for i, lab in enumerate(labels):
            fs_i = float(fs[i])
            name = lab.replace(" ", "_")
            if fs_i >= 50.0: # High-rate
                high[name] = _read_and_epoch_channel(f, i, epoch_len, expected_fs=100.0)
            elif fs_i <= 2.0: # Low-rate
                low[name] = _read_and_epoch_channel(f, i, epoch_len, expected_fs=1.0)
            else:
                # Inusitated tax: send for more closely group
                if fs_i > 2.0:
                    high[name] = _read_and_epoch_channel(f, i, epoch_len, expected_fs=100.0)
                else:
                    low[name] = _read_and_epoch_channel(f, i, epoch_len, expected_fs=1.0)
    
    # Harmonizes n_epochs based on the samllest
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

def _select_canonical_channels(high: Dict[str, np.ndarray], low: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Select canonical channels (if exists) for features with stable names"""
    high_keys = list(high.keys())
    low_keys = list(low.keys())

    # Mapping by canonical names -> look for the best match in the real set
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

def _bandpower_welch(x: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    """In-band power [fmin, fmax] with Welch and discrete integration."""
    nperseg = int(4 * fs)
    nperseg = max(nperseg, 8)
    noverlap = nperseg // 2
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mask = (freqs >= fmin) & (freqs < fmax)

    return float(np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0.0)

def _compute_epoch_features(epoch_high: Dict[str, np.ndarray], epoch_low: Dict[str, np.ndarray]) -> Dict[str, float]:
    feats = {}
    # EEG (100 Hz): bandpowers + stats
    for eeg_key in ["EEG_Fpz_Cz", "EEG_Pz_Oz"]:
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

def _process_record(psg_file: str, hyp_file: str, subject_id: str, night_id: str) -> pd.DataFrame:
    # 1) Epoch labeling (center rule) with temporal alignment
    y_df = _read_hyp_epochs_aligned(psg_file, hyp_file, epoch_len=EPOCH_LEN)  # epoch_idx, t0_sec, stage

    # 2) Signals per epoch, separated by rate (100 Hz and 1 Hz)
    high_all, low_all = _read_psg_epochs(psg_file, epoch_len=EPOCH_LEN)

    if not high_all and not low_all:
        return pd.DataFrame()

    # 3) Selects canonical channels (if any) for features with stable names
    high, low = _select_canonical_channels(high_all, low_all)

    # 4) Harmonize n_epochs between X e y
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