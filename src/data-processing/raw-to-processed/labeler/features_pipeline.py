import os
import numpy as np
import polars as pl

from typing import Dict, Tuple, Optional, Any
from scipy.signal import welch
from utils import best_match_idx
from .io_edf import EPOCH_LEN, read_hyp_epochs_aligned, read_psg_epochs

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

def _sanitize_signal(x: np.ndarray) -> np.ndarray:
    """Remove NaNs and DC component from signal."""
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmean(x) if np.isfinite(x).any() else 0.0
    x = x - (m if np.isfinite(m) else 0.0)

    return np.nan_to_num(x)

def _welch_once(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute Welch's PSD for a single channel."""
    x = _sanitize_signal(x)
    nperseg = min(max(int(4 * fs), 8), len(x))
    noverlap = nperseg // 2
    freqs, psg = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)

    return freqs, psg, nperseg

def compute_features_for_epoch(logger, epoch_high: Dict[str, np.ndarray], epoch_low: Dict[str, np.ndarray],
                               fs_map_high: Dict[str, float], fs_map_low: Dict[str, float]) -> Dict[str, float]:
    """Compute features for a single epoch (reutilizando PSD por canal)."""
    try:
        logger.log(f"[COMPUTE_FEATURES_FOR_EPOCH] {len(epoch_high)} high-ch | {len(epoch_low)} low-ch")
        feats: Dict[str, float] = {}

        for ch, x in epoch_high.items():
            fs = float(fs_map_high.get(ch, 100.0))
            try:
                freqs, psd, _ = _welch_once(x, fs)
                mask_tot = (freqs >= 0.5) & (freqs < 30.0)
                total_pow = float(np.trapz(psd[mask_tot], freqs[mask_tot])) if np.any(mask_tot) else 0.0

                for band, (fmin, fmax) in EEG_BANDS.items():
                    mask = (freqs >= fmin) & (freqs < fmax)
                    bp = float(np.trapz(psd[mask], freqs[mask])) if np.any(mask) else 0.0
                    feats[f"{ch}_{band}_pow"] = bp
                    feats[f"{ch}_{band}_relpow"] = bp / (total_pow + 1e-12)

                x_s = _sanitize_signal(x)
                feats[f"{ch}_rms"] = float(np.sqrt(np.mean(x_s**2)))
                feats[f"{ch}_var"] = float(np.var(x_s))

            except Exception as e:
                logger.log(f"[COMPUTE_FEATURES_FOR_EPOCH] High channel '{ch}' failed: {e}", "warning")

        for ch, x in epoch_low.items():
            try:
                fs_low = float(fs_map_low.get(ch, 1.0))
                x0 = np.asarray(x, dtype=np.float64)
                x0 = np.nan_to_num(x0)
                feats[f"{ch}_mean_1hz"]  = float(np.mean(x0))
                feats[f"{ch}_std_1hz"]   = float(np.std(x0))
                feats[f"{ch}_min_1hz"]   = float(np.min(x0))
                feats[f"{ch}_max_1hz"]   = float(np.max(x0))
                feats[f"{ch}_rms_1hz"]   = float(np.sqrt(np.mean(x0**2)))
                if len(x0) > 1:
                    feats[f"{ch}_slope_1hz"] = float((x0[-1] - x0[0]) / ((len(x0) - 1) / fs_low))
                else:
                    feats[f"{ch}_slope_1hz"] = 0.0

                feats[f"{ch}_median_1hz"] = float(np.median(x0))
                q75, q25 = np.percentile(x0, 75), np.percentile(x0, 25)
                feats[f"{ch}_iqr_1hz"]    = float(q75 - q25)

            except Exception as e:
                logger.log(f"[COMPUTE_FEATURES_FOR_EPOCH] Low channel '{ch}' failed: {e}", "warning")

        return feats

    except Exception as e:
        logger.log(f"[COMPUTE_FEATURES_FOR_EPOCH] Fatal: {e}", "error")
        return {}


def process_record(logger, psg_file: str, hyp_file: str, subject_id: str, night_id: str, progress: Optional["Progress"] = None) -> pl.DataFrame:
    """Process a single PSG/Hypnogram pair and return a Polars DataFrame with features and labels.
    With 'progress', create a bar for the epochs processing."""
    logger.log(f"[PROCESS_RECORD] --- Processing {os.path.basename(psg_file)} | {os.path.basename(hyp_file)} ---")
    try:
        y_df = read_hyp_epochs_aligned(logger, psg_file, hyp_file, epoch_len=EPOCH_LEN)
        high_all, low_all, fs_map = read_psg_epochs(logger, psg_file, epoch_len=EPOCH_LEN)
        if not high_all and not low_all:
            logger.log("[PROCESS_RECORD] No channels found.")
            return pl.DataFrame()

        high_keys, low_keys = list(high_all.keys()), list(low_all.keys())
        canon_high, canon_low = {}, {}

        for cname, clist in CANON_HIGH_HINT.items():
            idx = best_match_idx(high_keys, clist)
            if idx is not None:
                real = high_keys[idx]
                logger.log(f"[PROCESS_RECORD] Found canonical HIGH: {cname} <- {real}")
                canon_high[cname] = high_all[real]

        for cname, clist in CANON_LOW_HINT.items():
            idx = best_match_idx(low_keys, clist)
            if idx is not None:
                real = low_keys[idx]
                logger.log(f"[PROCESS_RECORD] Found canonical LOW:  {cname} <- {real}")
                canon_low[cname] = low_all[real]

        if not canon_high: canon_high = high_all
        if not canon_low:  canon_low  = low_all

        n_x = min([v.shape[0] for v in canon_high.values()] + [v.shape[0] for v in canon_low.values()] if (canon_high or canon_low) else [0])
        n_epochs = min(n_x, y_df.height)
        logger.log(f"[PROCESS_RECORD] n_epochs (final) = {n_epochs} (X={n_x}, y={y_df.height})")
        if n_epochs == 0:
            return pl.DataFrame()
        
        task_id = None
        desc = f"Epochs {subject_id}-{night_id}"
        task_id = progress.add_task(desc, total=n_epochs)

        rows = []
        for i in range(n_epochs):
            try:
                ep_high = {k: v[i, :] for k, v in canon_high.items()}
                ep_low  = {k: v[i, :] for k, v in canon_low.items()}

                fs_map_high = {k: float(fs_map.get("high", 100.0)) for k in ep_high.keys()}
                fs_map_low  = {k: float(fs_map.get("low", 1.0))    for k in ep_low.keys()}

                feats = compute_features_for_epoch(logger, ep_high, ep_low, fs_map_high, fs_map_low)

                yrow = y_df.row(i)
                yrec = y_df.select(["epoch_idx", "t0_sec", "stage"]).row(i)
                row = {
                    "subject_id": subject_id,
                    "night_id": night_id,
                    "epoch_idx": int(yrec[0]),
                    "t0_sec": float(yrec[1]),
                    "stage": yrec[2],
                }
                row.update(feats)
                rows.append(row)

            except Exception as e:
                logger.log(f"[PROCESS_RECORD] Epoch {i} failed: {e}", "warning")
            
            finally:
                progress.update(task_id, advance=1)

        df = pl.DataFrame(rows)
        if df.height == 0:
            return df

        try:
            df = df.with_columns(pl.col("stage").cast(pl.Enum(["W", "N1", "N2", "N3", "REM"])))
        except Exception:
            df = df.with_columns(pl.col("stage").cast(pl.Categorical))

        logger.log(f"[PROCESS_RECORD] Generated rows: {df.height} | Columns: {len(df.columns)}")
        return df

    except Exception as e:
        logger.log(f"[PROCESS_RECORD] Fatal: {e}", "error")
        return pl.DataFrame()