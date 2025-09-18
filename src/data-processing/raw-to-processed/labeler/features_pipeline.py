import os, re
import numpy as np
import polars as pl
from pathlib import Path

from typing import Dict, Tuple, Optional, Any
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from utils import best_match_idx, slugify
from .io_edf import EPOCH_LEN, read_hyp_epochs_aligned, read_psg_epochs

# Classic bands of EEG (Hz)
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta":  (16.0, 30.0),
}

# Canonical channel hints (high / mid / low)
CANON_HIGH_HINT = {
    "EEG_Fpz_Cz": ["EEG Fpz-Cz", "Fpz-Cz", "EEG FPZ-CZ"],
    "EEG_Pz_Oz":  ["EEG Pz-Oz", "Pz-Oz", "EEG PZ-OZ"],
    "EOG":        ["EOG horizontal", "Horizontal EOG", "EOG"],
    "EMG_submental": ["EMG submental", "Submental EMG", "EMG"]
}
CANON_MID_HINT = {
    "Event_marker": ["Event marker", "Marker", "Event"]
}
CANON_LOW_HINT = {
    "EMG_submental": ["EMG submental", "Submental EMG", "EMG"],
    "Resp_oronasal": ["Resp oro-nasal", "Oronasal Respiration", "Respiration"],
    "Temp_rectal":   ["Temp rectal", "Rectal Temp", "Temperature"],
    "Event_marker":  ["Event marker", "Marker", "Event"],
}

_SUBJECT_INFO_CACHE = {}

def _load_subject_info(root_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load subject info from CSV files and cache it."""
    global _SUBJECT_INFO_CACHE
    
    if _SUBJECT_INFO_CACHE:
        return _SUBJECT_INFO_CACHE
    
    root_path = Path(root_dir)
    subjects_dir = None
    
    for path in [root_path] + list(root_path.parents):
        candidate = path / "subjects-info"
        if candidate.exists():
            subjects_dir = candidate
            break
    
    if not subjects_dir:
        for path in [root_path] + list(root_path.parents):
            candidate = path / "datalake" / "raw" / "subjects-info"
            if candidate.exists():
                subjects_dir = candidate
                break
    
    if not subjects_dir:
        return {}
    
    subject_info = {}
    
    sc_file = subjects_dir / "SC-subjects.csv"
    if sc_file.exists():
        try:
            sc_df = pl.read_csv(str(sc_file)).unique(subset=['subject'])
            for row in sc_df.iter_rows(named=True):
                subject_key = f"SC{row['subject']:04d}"
                subject_info[subject_key] = {
                    'age': row['age'],
                    'sex': 'F' if row['sex'] == 1 else 'M'
                }
        except Exception as e:
            print(f"Warning: Failed to load SC subjects: {e}")
    
    st_file = subjects_dir / "ST-subjects.csv"
    if st_file.exists():
        try:
            st_df = pl.read_csv(str(st_file))
            for row in st_df.iter_rows(named=True):
                subject_key = f"ST{row['subject']:04d}"
                subject_info[subject_key] = {
                    'age': row['age'],
                    'sex': 'F' if row['sex'] == 1 else 'M'
                }
        except Exception as e:
            print(f"Warning: Failed to load ST subjects: {e}")
    
    _SUBJECT_INFO_CACHE = subject_info
    return subject_info

def _get_subject_demographics(subject_id: str, root_dir: str) -> Tuple[Optional[int], Optional[str]]:
    subject_info = _load_subject_info(root_dir)

    s = subject_id.upper()

    if s in subject_info:
        info = subject_info[s]
        return info['age'], info['sex']

    m_sc3 = re.match(r"^SC4(\d{2})(\d)$", s)   
    m_sc2 = re.match(r"^SC4(\d{2})$", s) 
    if m_sc3 or m_sc2:
        ss = int((m_sc3 or m_sc2).group(1))    
        for night in (1, 2):
            num_part = 4000 + ss*1 + night  
            csv_subject_num = num_part - 4001
            key = f"SC{csv_subject_num:04d}"
            if key in subject_info:
                info = subject_info[key]
                return info['age'], info['sex']
        key = f"SC{ss:04d}"
        if key in subject_info:
            info = subject_info[key]
            return info['age'], info['sex']

    m_st3 = re.match(r"^ST7(\d{2})(\d)$", s)  
    m_st2 = re.match(r"^ST7(\d{2})$", s)     
    if m_st3 or m_st2:
        ss = int((m_st3 or m_st2).group(1))
        for night in (1, 2):
            num_part = 7000 + ss*1 + night   
            csv_subject_num = (num_part - 7001) // 10
            key = f"ST{csv_subject_num:04d}"
            if key in subject_info:
                info = subject_info[key]
                return info['age'], info['sex']
        key = f"ST{ss:04d}"
        if key in subject_info:
            info = subject_info[key]
            return info['age'], info['sex']

    return None, None

def _band_masks_and_df(freqs: np.ndarray, bands: Dict[str, Tuple[float, float]]):
    """Build boolean masks per band and frequency step for fast integration."""
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    masks = {band: ((freqs >= fmin) & (freqs < fmax)) for band, (fmin, fmax) in bands.items()}
    mask_tot = (freqs >= 0.5) & (freqs < 30.0)
    return masks, mask_tot, df

def _sanitize_signal(x: np.ndarray) -> np.ndarray:
    """Remove NaNs and DC component from a 1D signal."""
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmean(x) if np.isfinite(x).any() else 0.0
    x = x - (m if np.isfinite(m) else 0.0)
    return np.nan_to_num(x)

def _welch_channel_batch(x_epochs: np.ndarray, fs: float, nperseg: int = 256):
    """Compute Welch PSD for all epochs of a channel in batch."""
    x = np.asarray(x_epochs, dtype=np.float64)
    x = x - np.nanmean(x, axis=1, keepdims=True)
    x = np.nan_to_num(x)
    nperseg = min(int(nperseg), x.shape[1])
    noverlap = nperseg // 2
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)
    return freqs, psd

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / (b + 1e-12)

def _epochwise_resample_mean(x_epochs: np.ndarray, src_fs: float, target_fs: float) -> np.ndarray:
    x = np.asarray(x_epochs, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x_epochs must be 2D (n_epochs, n_samples_per_epoch)")

    if abs(src_fs - target_fs) < 1e-9:
        return x  

    epoch_len_sec = x.shape[1] / float(src_fs)
    n_target = int(round(epoch_len_sec * target_fs))
    if n_target <= 0:
        raise ValueError("Computed target samples per epoch <= 0")

    if src_fs > target_fs:
        factor = int(round(src_fs / target_fs))
        usable = (x.shape[1] // factor) * factor
        x_cut = x[:, :usable]
        x_reshaped = x_cut.reshape(x.shape[0], -1, factor)  
        return x_reshaped.mean(axis=2)
    else:
        factor = int(round(target_fs / src_fs))
        x_rep = np.repeat(x, factor, axis=1)
        if x_rep.shape[1] >= n_target:
            return x_rep[:, :n_target]
        pad = n_target - x_rep.shape[1]
        return np.pad(x_rep, ((0,0),(0,pad)), mode="edge")

def _spec_peak_freq(psd: np.ndarray, freqs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Peak frequency within a band, per epoch."""
    if not mask.any():
        return np.zeros(psd.shape[0], dtype=float)
    psd_band = psd[:, mask]
    f_band = freqs[mask]
    idx = np.argmax(psd_band, axis=1)
    return f_band[idx]

def _spec_entropy(psd: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalized spectral entropy within mask (0..1)."""
    if not mask.any():
        return np.zeros(psd.shape[0], dtype=float)
    p = psd[:, mask]
    s = p.sum(axis=1, keepdims=True)
    p = p / (s + 1e-12)
    h = -(p * np.log(p + 1e-12)).sum(axis=1)
    h_max = np.log(p.shape[1])
    return _safe_div(h, h_max).astype(float)

def _spec_edge_freq(psd: np.ndarray, freqs: np.ndarray, mask: np.ndarray, q: float) -> np.ndarray:
    """Spectral edge frequency (e.g., q=0.95) within mask."""
    if not mask.any():
        return np.zeros(psd.shape[0], dtype=float)
    p = psd[:, mask]
    f = freqs[mask]
    cumsum = np.cumsum(p, axis=1)
    total = cumsum[:, -1][:, None]
    target = q * total
    idx = (cumsum >= target).argmax(axis=1)
    return f[idx]

def _median_freq(psd: np.ndarray, freqs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return _spec_edge_freq(psd, freqs, mask, 0.5)

def _aperiodic_slope(psd: np.ndarray, freqs: np.ndarray, fmin: float = 2.0, fmax: float = 30.0) -> np.ndarray:
    """Linear fit slope of log10(PSD) ~ a + b*log10(f) in [fmin,fmax]."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return np.zeros(psd.shape[0], dtype=float)
    xf = np.log10(freqs[mask] + 1e-12)
    y = np.log10(psd[:, mask] + 1e-12)
    slopes = np.empty(psd.shape[0], dtype=float)
    for i in range(psd.shape[0]):
        try:
            b, a = np.polyfit(xf, y[i], 1) 
            slopes[i] = b
        except Exception:
            slopes[i] = 0.0
    return slopes

def _features_high_channel_batch(
    logger,
    ch_name: str,
    x_epochs: np.ndarray,
    fs: float,
    bands: Dict[str, Tuple[float, float]] = EEG_BANDS,
):
    """
    Compute frequency and time-domain features for all epochs of a high-FS channel.
    Gera colunas para nperseg=256 e 512.
    """
    try:
        out: Dict[str, np.ndarray] = {}

        for nseg in (256, 512):
            freqs, psd = _welch_channel_batch(x_epochs, fs, nperseg=nseg)
            masks, mask_tot, df = _band_masks_and_df(freqs, bands)

            total_pow = (psd[:, mask_tot].sum(axis=1) * df) if mask_tot.any() else np.zeros(psd.shape[0])

            suffix = f"_{nseg}"

            band_pows: Dict[str, np.ndarray] = {}
            for band, m in masks.items():
                bp = (psd[:, m].sum(axis=1) * df) if m.any() else np.zeros(psd.shape[0])
                band_pows[band] = bp
                out[f"{ch_name}_{band}_pow{suffix}"] = bp.astype(float)
                out[f"{ch_name}_{band}_relpow{suffix}"] = _safe_div(bp, total_pow).astype(float)
                out[f"{ch_name}_{band}_logpow{suffix}"] = np.log10(bp + 1e-12).astype(float)
                out[f"{ch_name}_{band}_peakfreq{suffix}"] = _spec_peak_freq(psd, freqs, m).astype(float)

            try:
                d, t, a, s, b = [band_pows.get(k, np.zeros_like(total_pow)) for k in ("delta","theta","alpha","sigma","beta")]
                out[f"{ch_name}_delta_theta_ratio{suffix}"] = _safe_div(d, t).astype(float)
                out[f"{ch_name}_theta_alpha_ratio{suffix}"] = _safe_div(t, a).astype(float)
                out[f"{ch_name}_alpha_sigma_ratio{suffix}"] = _safe_div(a, s).astype(float)
                slow = d + t
                fast = a + b
                out[f"{ch_name}_slow_fast_ratio{suffix}"]  = _safe_div(slow, fast).astype(float)
            except Exception as e:
                logger.log(f"[RATIOS] {ch_name} failed ({nseg}): {e}", "warning")

            try:
                out[f"{ch_name}_sef95{suffix}"]      = _spec_edge_freq(psd, freqs, mask_tot, 0.95).astype(float)
                out[f"{ch_name}_medfreq{suffix}"]    = _median_freq(psd, freqs, mask_tot).astype(float)
                out[f"{ch_name}_spec_entropy{suffix}"] = _spec_entropy(psd, mask_tot).astype(float)
                out[f"{ch_name}_aperiodic_slope{suffix}"] = _aperiodic_slope(psd, freqs, 2.0, 30.0).astype(float)
            except Exception as e:
                logger.log(f"[SPECTRAL_SUMMARY] {ch_name} failed ({nseg}): {e}", "warning")

        x_s = np.nan_to_num(x_epochs - np.nanmean(x_epochs, axis=1, keepdims=True))
        out[f"{ch_name}_rms"] = np.sqrt((x_s**2).mean(axis=1)).astype(float)
        out[f"{ch_name}_var"] = x_s.var(axis=1).astype(float)
        return out

    except Exception as e:
        logger.log(f"[BATCH_HIGH] Channel '{ch_name}' failed: {e}", "warning")
        n = x_epochs.shape[0]
        zeros = np.zeros(n, dtype=float)
        out = {}
        for nseg in (256, 512):
            for band in EEG_BANDS:
                out[f"{ch_name}_{band}_pow_{nseg}"] = zeros
                out[f"{ch_name}_{band}_relpow_{nseg}"] = zeros
                out[f"{ch_name}_{band}_logpow_{nseg}"] = zeros
                out[f"{ch_name}_{band}_peakfreq_{nseg}"] = zeros
            out[f"{ch_name}_sef95_{nseg}"] = zeros
            out[f"{ch_name}_medfreq_{nseg}"] = zeros
            out[f"{ch_name}_spec_entropy_{nseg}"] = zeros
            out[f"{ch_name}_aperiodic_slope_{nseg}"] = zeros
        out.update({f"{ch_name}_rms": zeros, f"{ch_name}_var": zeros})
        return out

def _features_stats_channel_batch(logger, ch_name: str, x_epochs: np.ndarray, fs_stat: float, suffix: str):
    """Time domain statistics for 'slow' signals (1 Hz, 10 Hz, ...)."""
    try:
        x0 = np.nan_to_num(np.asarray(x_epochs, dtype=np.float64))
        mean = x0.mean(axis=1).astype(float)
        std  = x0.std(axis=1).astype(float)
        mn   = x0.min(axis=1).astype(float)
        mx   = x0.max(axis=1).astype(float)
        rms  = np.sqrt((x0**2).mean(axis=1)).astype(float)
        med  = np.median(x0, axis=1).astype(float)
        q75  = np.percentile(x0, 75, axis=1).astype(float)
        q25  = np.percentile(x0, 25, axis=1).astype(float)
        iqr  = (q75 - q25).astype(float)

        mad   = np.median(np.abs(x0 - med[:, None]), axis=1).astype(float)
        p01   = np.percentile(x0, 1, axis=1).astype(float)
        p10   = np.percentile(x0, 10, axis=1).astype(float)
        p90   = np.percentile(x0, 90, axis=1).astype(float)
        p99   = np.percentile(x0, 99, axis=1).astype(float)
        try:
            kur   = kurtosis(x0, axis=1, fisher=True, bias=False).astype(float)
            skw   = skew(x0, axis=1, bias=False).astype(float)
        except Exception:
            kur = np.zeros(x0.shape[0]); skw = np.zeros(x0.shape[0])

        if x0.shape[1] > 1:
            diff = np.diff(x0, axis=1)
            diff_rms = np.sqrt((diff**2).mean(axis=1)).astype(float)
            zcr = ( (x0[:, 1:] * x0[:, :-1]) < 0 ).sum(axis=1).astype(float) / (x0.shape[1]-1)
            slope = ((x0[:, -1] - x0[:, 0]) / ((x0.shape[1] - 1) / fs_stat)).astype(float)
        else:
            diff_rms = np.zeros(x0.shape[0], dtype=float)
            zcr = np.zeros(x0.shape[0], dtype=float)
            slope = np.zeros(x0.shape[0], dtype=float)

        out = {
            f"{ch_name}_mean_{suffix}": mean,
            f"{ch_name}_std_{suffix}": std,
            f"{ch_name}_min_{suffix}": mn,
            f"{ch_name}_max_{suffix}": mx,
            f"{ch_name}_rms_{suffix}": rms,
            f"{ch_name}_slope_{suffix}": slope,
            f"{ch_name}_median_{suffix}": med,
            f"{ch_name}_iqr_{suffix}": iqr,
            f"{ch_name}_mad_{suffix}": mad,
            f"{ch_name}_p01_{suffix}": p01,
            f"{ch_name}_p10_{suffix}": p10,
            f"{ch_name}_p90_{suffix}": p90,
            f"{ch_name}_p99_{suffix}": p99,
            f"{ch_name}_kurtosis_{suffix}": kur,
            f"{ch_name}_skewness_{suffix}": skw,
            f"{ch_name}_diff_rms_{suffix}": diff_rms,
            f"{ch_name}_zcr_{suffix}": zcr,
        }

        ch_lower = ch_name.lower()
        if "resp_oronasal" in ch_lower:
            clip = (np.abs(x0) >= 900).mean(axis=1).astype(float)
            out[f"{ch_name}_clip_frac_{suffix}"] = clip
        if "temp_rectal" in ch_lower:
            oor = ((x0 < 30.0) | (x0 > 45.0)).mean(axis=1).astype(float)
            out[f"{ch_name}_oor_frac_{suffix}"] = oor

        return out

    except Exception as e:
        logger.log(f"[BATCH_STATS] Channel '{ch_name}' failed: {e}", "warning")
        n = x_epochs.shape[0]
        zeros = np.zeros(n, dtype=float)
        base = {
            f"{ch_name}_mean_{suffix}": zeros, f"{ch_name}_std_{suffix}": zeros,
            f"{ch_name}_min_{suffix}": zeros,  f"{ch_name}_max_{suffix}": zeros,
            f"{ch_name}_rms_{suffix}": zeros,  f"{ch_name}_slope_{suffix}": zeros,
            f"{ch_name}_median_{suffix}": zeros, f"{ch_name}_iqr_{suffix}": zeros,
            f"{ch_name}_mad_{suffix}": zeros,  f"{ch_name}_p01_{suffix}": zeros,
            f"{ch_name}_p10_{suffix}": zeros,  f"{ch_name}_p90_{suffix}": zeros,
            f"{ch_name}_p99_{suffix}": zeros,  f"{ch_name}_kurtosis_{suffix}": zeros,
            f"{ch_name}_skewness_{suffix}": zeros, f"{ch_name}_diff_rms_{suffix}": zeros,
            f"{ch_name}_zcr_{suffix}": zeros,
        }
        return base

def _features_low_channel_batch(logger, ch_name: str, x_epochs: np.ndarray, fs_low: float):
    return _features_stats_channel_batch(logger, ch_name, x_epochs, fs_stat=fs_low, suffix="1hz")

def _features_mid_channel_batch(logger, ch_name: str, x_epochs: np.ndarray, fs_mid: float):
    return _features_stats_channel_batch(logger, ch_name, x_epochs, fs_stat=fs_mid, suffix="10hz")

def process_record(
    logger,
    psg_file: str,
    hyp_file: str,
    subject_id: str,
    night_id: str,
    progress: Optional[Any] = None,
    root_dir: Optional[str] = None
) -> pl.DataFrame:
    """
Processes a PSG/Hyp pair into a DataFrame (Polars) with features per epoch and labels.
- Channel buckets: 100 Hz (high), 10 Hz (mid), 1 Hz (low)
- Canonical hints for stable names; fallback includes everything left (no duplication)
- Spectrals only at 100 Hz; time statistics at 10 Hz and 1 Hz
    """
    logger.log(f"[PROCESS_RECORD] Processing {os.path.basename(psg_file)} | {os.path.basename(hyp_file)}")
    try:
        y_df = read_hyp_epochs_aligned(logger, psg_file, hyp_file, epoch_len=EPOCH_LEN)

        high_all, mid_all, low_all, fs_map = read_psg_epochs(logger, psg_file, epoch_len=EPOCH_LEN)
        if not high_all and not mid_all and not low_all:
            logger.log("[PROCESS_RECORD] No channels found.")
            return pl.DataFrame()

        canon_high, canon_mid, canon_low = {}, {}, {}
        used_high_real, used_mid_real, used_low_real = set(), set(), set()

        for cname, clist in CANON_HIGH_HINT.items():
            idx = best_match_idx(list(high_all.keys()), clist)
            if idx is not None:
                real = list(high_all.keys())[idx]
                logger.log(f"[PROCESS_RECORD] Canonical HIGH: {cname} <- {real}")
                canon_high[cname] = high_all[real]
                used_high_real.add(real)

        for cname, clist in CANON_LOW_HINT.items():
            idx = best_match_idx(list(low_all.keys()), clist)
            if idx is not None:
                real = list(low_all.keys())[idx]
                logger.log(f"[PROCESS_RECORD] Canonical LOW:  {cname} <- {real}")
                canon_low[cname] = low_all[real]
                used_low_real.add(real)

        for cname, clist in (CANON_MID_HINT if 'CANON_MID_HINT' in globals() else {}).items():
            idx = best_match_idx(list(mid_all.keys()), clist)
            if idx is not None:
                real = list(mid_all.keys())[idx]
                logger.log(f"[PROCESS_RECORD] Canonical MID:  {cname} <- {real}")
                canon_mid[cname] = mid_all[real]
                used_mid_real.add(real)

        for real, arr in high_all.items():
            if real in used_high_real:
                continue
            canon_high[real] = arr

        for real, arr in mid_all.items():
            if real in used_mid_real:
                continue
            canon_mid[real] = arr

        for real, arr in low_all.items():
            if real in used_low_real:
                continue
            canon_low[real] = arr

        n_x = min(
            [v.shape[0] for v in canon_high.values()] +
            [v.shape[0] for v in canon_mid.values()]  +
            [v.shape[0] for v in canon_low.values()]
            or [0]
        )
        n_epochs = min(n_x, y_df.height)
        logger.log(f"[PROCESS_RECORD] n_epochs={n_epochs} (X={n_x}, y={y_df.height})")
        if n_epochs == 0:
            return pl.DataFrame()

        task_id = None
        if progress is not None and hasattr(progress, "add_task"):
            try:
                total_steps = max(1, len(canon_high) + len(canon_mid) + len(canon_low))
                task_id = progress.add_task(f"Features {subject_id}-{night_id}", total=total_steps)
            except Exception:
                task_id = None

        feat_cols: Dict[str, np.ndarray] = {}

        try:
            if ("EMG_submental" in canon_high) and ("EMG_submental" not in canon_low):
                fs_high = float(fs_map.get("high", 100.0))
                emg_high = canon_high["EMG_submental"][:n_epochs, :]
                emg_1hz = _epochwise_resample_mean(emg_high, src_fs=fs_high, target_fs=1.0)
                vecs_emg_1hz = _features_low_channel_batch(logger, "EMG_submental", emg_1hz, fs_low=1.0)
                feat_cols.update(vecs_emg_1hz)
                logger.log("[PROCESS_RECORD] Telemetry bridge: EMG_submental 100Hz -> features _1hz geradas")
                
        except Exception as e:
            logger.log(f"[PROCESS_RECORD] Telemetry bridge failed: {e}", "warning")

        for ch, arr in canon_high.items():
            vecs = _features_high_channel_batch(
                logger, ch, arr[:n_epochs, :], fs=float(fs_map.get("high", 100.0))
            )
            feat_cols.update(vecs)
            if task_id is not None and hasattr(progress, "update"):
                try:
                    progress.update(task_id, advance=1)
                except Exception:
                    pass

        for ch, arr in canon_mid.items():
            vecs = _features_mid_channel_batch(
                logger, ch, arr[:n_epochs, :], fs_mid=float(fs_map.get("mid", 10.0))
            )
            feat_cols.update(vecs)
            if task_id is not None and hasattr(progress, "update"):
                try:
                    progress.update(task_id, advance=1)
                except Exception:
                    pass

        for ch, arr in canon_low.items():
            vecs = _features_low_channel_batch(
                logger, ch, arr[:n_epochs, :], fs_low=float(fs_map.get("low", 1.0))
            )
            feat_cols.update(vecs)
            if task_id is not None and hasattr(progress, "update"):
                try:
                    progress.update(task_id, advance=1)
                except Exception:
                    pass

        y_small = y_df.head(n_epochs).select(["epoch_idx", "t0_sec", "stage"])

        try:
            sleep_onset_idx = int(
                y_df.filter(pl.col("stage") != "W").select("epoch_idx").min().item()
            ) if y_df.filter(pl.col("stage") != "W").height > 0 else None
        except Exception:
            sleep_onset_idx = None

        if sleep_onset_idx is not None:
            sleep_onset_sec = float(sleep_onset_idx) * float(EPOCH_LEN)
            tso_min = (y_small["t0_sec"].to_numpy() - sleep_onset_sec) / 60.0
            tso_min = np.maximum(0.0, tso_min).astype(float)
        else:
            tso_min = np.zeros(n_epochs, dtype=float)

        lookup_root = root_dir or os.path.dirname(psg_file)
        logger.log(f"[DEMOGRAPHICS] Looking up {subject_id} in {lookup_root}")
        age, sex = _get_subject_demographics(subject_id, lookup_root)
        logger.log(f"[DEMOGRAPHICS] Found: age={age}, sex={sex}")

        meta = pl.DataFrame({
            "subject_id": [subject_id] * n_epochs,
            "night_id":   [night_id] * n_epochs,
            "age":        [age] * n_epochs,
            "sex":        [sex] * n_epochs,
            "tso_min":    tso_min,
        })

        feat_df = pl.DataFrame({k: v for k, v in feat_cols.items()})
        df = pl.concat([meta, y_small, feat_df], how="horizontal")

        try:
            df = df.with_columns(pl.col("stage").cast(pl.Enum(["W", "N1", "N2", "N3", "REM"])))
        except Exception:
            df = df.with_columns(pl.col("stage").cast(pl.Categorical))

        logger.log(f"[PROCESS_RECORD] Generated rows: {df.height} | Columns: {len(df.columns)}")
        return df

    except Exception as e:
        logger.log(f"[PROCESS_RECORD] Fatal: {e}", "error")
        return pl.DataFrame()
