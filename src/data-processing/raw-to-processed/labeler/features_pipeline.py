import os
import numpy as np
import polars as pl

from typing import Dict, Tuple, Optional, Any
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from utils import best_match_idx
from .io_edf import EPOCH_LEN, read_hyp_epochs_aligned, read_psg_epochs

# Classic bands of EEG (Hz)
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta":  (16.0, 30.0),
}

# Canonical channel hints (high / low)
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

def _features_low_channel_batch(logger, ch_name: str, x_epochs: np.ndarray, fs_low: float):
    """Compute time-domain statistics and quality metrics for all epochs of a low-FS channel (1 Hz)."""
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
        else:
            diff_rms = np.zeros(x0.shape[0], dtype=float)
            zcr = np.zeros(x0.shape[0], dtype=float)

        out = {
            f"{ch_name}_mean_1hz": mean,
            f"{ch_name}_std_1hz": std,
            f"{ch_name}_min_1hz": mn,
            f"{ch_name}_max_1hz": mx,
            f"{ch_name}_rms_1hz": rms,
            f"{ch_name}_slope_1hz": ((x0[:, -1] - x0[:, 0]) / ((x0.shape[1] - 1) / fs_low)).astype(float) if x0.shape[1] > 1 else np.zeros(x0.shape[0], dtype=float),
            f"{ch_name}_median_1hz": med,
            f"{ch_name}_iqr_1hz": iqr,
            f"{ch_name}_mad_1hz": mad,
            f"{ch_name}_p01_1hz": p01,
            f"{ch_name}_p10_1hz": p10,
            f"{ch_name}_p90_1hz": p90,
            f"{ch_name}_p99_1hz": p99,
            f"{ch_name}_kurtosis_1hz": kur,
            f"{ch_name}_skewness_1hz": skw,
            f"{ch_name}_diff_rms_1hz": diff_rms,
            f"{ch_name}_zcr_1hz": zcr,
        }

        ch_lower = ch_name.lower()
        if "resp_oronasal" in ch_lower:
            clip = (np.abs(x0) >= 900).mean(axis=1).astype(float)
            out[f"{ch_name}_clip_frac_1hz"] = clip
        if "temp_rectal" in ch_lower:
            oor = ((x0 < 30.0) | (x0 > 45.0)).mean(axis=1).astype(float)
            out[f"{ch_name}_oor_frac_1hz"] = oor

        return out

    except Exception as e:
        logger.log(f"[BATCH_LOW] Channel '{ch_name}' failed: {e}", "warning")
        n = x_epochs.shape[0]
        zeros = np.zeros(n, dtype=float)
        return {
            f"{ch_name}_mean_1hz": zeros,
            f"{ch_name}_std_1hz": zeros,
            f"{ch_name}_min_1hz": zeros,
            f"{ch_name}_max_1hz": zeros,
            f"{ch_name}_rms_1hz": zeros,
            f"{ch_name}_slope_1hz": zeros,
            f"{ch_name}_median_1hz": zeros,
            f"{ch_name}_iqr_1hz": zeros,
            f"{ch_name}_mad_1hz": zeros,
            f"{ch_name}_p01_1hz": zeros,
            f"{ch_name}_p10_1hz": zeros,
            f"{ch_name}_p90_1hz": zeros,
            f"{ch_name}_p99_1hz": zeros,
            f"{ch_name}_kurtosis_1hz": zeros,
            f"{ch_name}_skewness_1hz": zeros,
            f"{ch_name}_diff_rms_1hz": zeros,
            f"{ch_name}_zcr_1hz": zeros,
        }

def process_record(logger, psg_file: str, hyp_file: str, subject_id: str, night_id: str, progress: Optional[Any] = None) -> pl.DataFrame:
    """Process one PSG/Hyp pair into a tabular Polars DataFrame with epoch features and labels."""
    logger.log(f"[PROCESS_RECORD] Processing {os.path.basename(psg_file)} | {os.path.basename(hyp_file)}")
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
                logger.log(f"[PROCESS_RECORD] Canonical HIGH: {cname} <- {real}")
                canon_high[cname] = high_all[real]

        for cname, clist in CANON_LOW_HINT.items():
            idx = best_match_idx(low_keys, clist)
            if idx is not None:
                real = low_keys[idx]
                logger.log(f"[PROCESS_RECORD] Canonical LOW:  {cname} <- {real}")
                canon_low[cname] = low_all[real]

        if not canon_high:
            canon_high = high_all
        if not canon_low:
            canon_low = low_all

        n_x = min([v.shape[0] for v in canon_high.values()] + [v.shape[0] for v in canon_low.values()] if (canon_high or canon_low) else [0])
        n_epochs = min(n_x, y_df.height)
        logger.log(f"[PROCESS_RECORD] n_epochs={n_epochs} (X={n_x}, y={y_df.height})")
        if n_epochs == 0:
            return pl.DataFrame()

        task_id = None
        if progress is not None and hasattr(progress, "add_task"):
            try:
                total_steps = max(1, len(canon_high) + len(canon_low))
                task_id = progress.add_task(f"Features {subject_id}-{night_id}", total=total_steps)
            except Exception:
                task_id = None

        feat_cols: Dict[str, np.ndarray] = {}

        for ch, arr in canon_high.items():
            vecs = _features_high_channel_batch(logger, ch, arr[:n_epochs, :], fs=float(fs_map.get("high", 100.0)))
            feat_cols.update(vecs)
            if task_id is not None and hasattr(progress, "update"):
                try:
                    progress.update(task_id, advance=1)
                except Exception:
                    pass

        for ch, arr in canon_low.items():
            vecs = _features_low_channel_batch(logger, ch, arr[:n_epochs, :], fs_low=float(fs_map.get("low", 1.0)))
            feat_cols.update(vecs)
            if task_id is not None and hasattr(progress, "update"):
                try:
                    progress.update(task_id, advance=1)
                except Exception:
                    pass

        y_small = y_df.head(n_epochs).select(["epoch_idx", "t0_sec", "stage"])

        try:
            sleep_onset_idx = int(y_df.filter(pl.col("stage") != "W").select("epoch_idx").min().item()) if y_df.filter(pl.col("stage") != "W").height > 0 else None
        except Exception:
            sleep_onset_idx = None
        if sleep_onset_idx is not None:
            sleep_onset_sec = float(sleep_onset_idx) * float(EPOCH_LEN)
            tso_min = (y_small["t0_sec"].to_numpy() - sleep_onset_sec) / 60.0
            tso_min = np.maximum(0.0, tso_min).astype(float)
        else:
            tso_min = np.zeros(n_epochs, dtype=float)

        meta = pl.DataFrame({
            "subject_id": [subject_id] * n_epochs,
            "night_id":   [night_id] * n_epochs,
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
