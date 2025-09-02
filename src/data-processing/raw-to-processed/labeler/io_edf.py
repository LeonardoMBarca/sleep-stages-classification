import os, glob, re
import numpy as np
import polars as pl
import pyedflib

from typing import Tuple, Dict, List
from utils.utils import slugify, normspace

# Each epoch has 30 seconds (R&K manual).
EPOCH_LEN = 30.0

def pair_psg_hyp(logger, root_dir: str) -> List[Tuple[str, str, str, str]]:
    """Search for pairs of PSG and Hypnogram files in the given root directory."""
    logger.log(f"[PAIR_PSG_HYP] Searching pairs in: {root_dir}")
    pairs: List[Tuple[str, str, str, str]] = []
    try:
        for psg in glob.glob(os.path.join(root_dir, "**", "*-PSG.edf"), recursive=True):
            base = os.path.basename(psg)
            stem = base[:-8]
            folder = os.path.dirname(psg)

            prefix6 = stem[:6]
            cands = sorted(glob.glob(os.path.join(folder, f"{prefix6}*-Hypnogram.edf")))
            if not cands:
                cands = sorted(glob.glob(os.path.join(folder, f"{stem}-Hypnogram.edf")))
            if not cands:
                logger.log(f"[PAIR_PSG_HYP] No hypnogram for {base}")
                continue

            hyp = cands[0]

            m = re.match(r"^(SC|ST)(\d{4})([A-Z]\d)?", stem, flags=re.IGNORECASE)
            if m:
                subject_id = f"{m.group(1).upper()}{m.group(2)}"
                night_id = m.group(3).upper() if m.group(3) else "N0"
            else:
                subject_id, night_id = stem, "N0"

            pairs.append((psg, hyp, subject_id, night_id))

        logger.log(f"[PAIR_PSG_HYP] Found pairs: {len(pairs)}")
        return pairs
    except Exception as e:
        logger.log(f"[PAIR_PSG_HYP] Fatal: {e}", "error")
        return []


def read_hyp_epochs_aligned(logger, psg_file: str, hyp_file: str, epoch_len: float = EPOCH_LEN) -> pl.DataFrame:
    """Read and align hypnogram annotations to PSG epochs. Return Polars DataFrame (epoch_idx, t0_sec, stage)."""
    try:
        logger.log(f"[READ_HYP_EPOCHS_ALIGNED] Reading hyp alignment for {os.path.basename(hyp_file)}")

        with pyedflib.EdfReader(psg_file) as fpsg:
            start_psg = fpsg.getStartdatetime()
            ns0 = fpsg.getNSamples()[0]
            fs0 = fpsg.getSampleFrequencies()[0]
            dur_psg_sec = float(ns0) / float(fs0)

        with pyedflib.EdfReader(hyp_file) as fhyp:
            start_hyp = fhyp.getStartdatetime()
            onsets, durations, desc = fhyp.readAnnotations()

        offset_sec = (start_hyp - start_psg).total_seconds()
        logger.log(f"[READ_HYP_EPOCHS_ALIGNED] Alignment Hyp↔PSG: offset = {offset_sec:.3f} s")

        starts, ends, stages = [], [], []
        kept, ignored = 0, 0
        for o, d, txt in zip(onsets, durations, desc):
            try:
                t = normspace(txt).upper()
                if not t or (d is None) or (float(d) <= 0):
                    ignored += 1
                    continue

                if "SLEEP STAGE" in t:
                    s = None
                    if t.endswith(" W"): s = "W"
                    elif t.endswith(" R"): s = "REM"
                    elif t.endswith(" 1"): s = "N1"
                    elif t.endswith(" 2"): s = "N2"
                    elif t.endswith(" 3") or t.endswith(" 4"): s = "N3"
                    if s is None:
                        ignored += 1
                        continue

                    starts.append(float(o) + offset_sec)
                    ends.append(float(o) + float(d) + offset_sec)
                    stages.append(s)
                    kept += 1
                else:
                    ignored += 1
            except Exception:
                ignored += 1
                continue

        if kept == 0:
            raise ValueError(f"No valid stage in {os.path.basename(hyp_file)}")

        logger.log(f"[READ_HYP_EPOCHS_ALIGNED] Hyp events retained: {kept} | ignored: {ignored}")

        hyp = pl.DataFrame({"start": starts, "end": ends, "stage": stages}).sort("start")
        hyp = hyp.with_columns(
            pl.col("start").clip(lower_bound=0.0, upper_bound=dur_psg_sec),
            pl.col("end").clip(lower_bound=0.0, upper_bound=dur_psg_sec)
        ).filter(pl.col("end") > pl.col("start"))

        total_time = min(hyp.select(pl.max("end")).item(), dur_psg_sec)
        n_epochs = int(np.floor(total_time / epoch_len))
        logger.log(f"[READ_HYP_EPOCHS_ALIGNED] Duration PSG ~ {dur_psg_sec/3600:.2f} h | total used ~ {total_time/3600:.2f} h | n_epochs = {n_epochs}")

        rows, empty_epochs = [], 0
        for i in range(n_epochs):
            t0 = i * epoch_len
            tc = t0 + epoch_len / 2.0
            hit = hyp.filter((pl.col("start") <= tc) & (tc < pl.col("end")))
            stage = hit.select("stage").to_series().item() if hit.height > 0 else None
            if stage is None:
                empty_epochs += 1
            rows.append({"epoch_idx": i, "t0_sec": t0, "stage": stage})

        if empty_epochs:
            logger.log(f"[READ_HYP_EPOCHS_ALIGNED] Epochs without label (outside events): {empty_epochs}")

        df = pl.DataFrame(rows).drop_nulls(subset=["stage"])
        try:
            df = df.with_columns(pl.col("stage").cast(pl.Enum(["W", "N1", "N2", "N3", "REM"])))
        except Exception:
            df = df.with_columns(pl.col("stage").cast(pl.Categorical))

        logger.log(f"[READ_HYP_EPOCHS_ALIGNED] Labeled epochs: {df.height}")
        return df

    except Exception as e:
        logger.log(f"[READ_HYP_EPOCHS_ALIGNED] Fatal: {e}", "error")
        return pl.DataFrame({"epoch_idx": [], "t0_sec": [], "stage": []})


def read_and_epoch_channel(logger, reader: pyedflib.EdfReader, ch_idx: int, epoch_len: float, expected_fs: float) -> np.ndarray:
    """Read a channel from EDF and epoch it to the expected sampling frequency using simple decimation/repeat."""
    try:
        logger.log(f"[READ_AND_EPOCH_CHANNEL] ch_idx={ch_idx} expected_fs={expected_fs} Hz")
        x = reader.readSignal(ch_idx).astype(np.float32)
        fs_i = float(reader.getSampleFrequencies()[ch_idx])
        n_target = int(round(epoch_len * expected_fs))

        if abs(fs_i - expected_fs) < 1e-6:
            n_epochs = len(x) // n_target
            x = x[: n_epochs * n_target]
            return x.reshape(n_epochs, n_target)

        if fs_i > expected_fs:
            factor = max(1, int(round(fs_i / expected_fs)))
            x = x[: (len(x) // factor) * factor].reshape(-1, factor).mean(axis=1)
        else:
            factor = max(1, int(round(expected_fs / fs_i)))
            x = np.repeat(x, factor)

        n_epochs = len(x) // n_target
        x = x[: n_epochs * n_target]
        return x.reshape(n_epochs, n_target)

    except Exception as e:
        logger.log(f"[READ_AND_EPOCH_CHANNEL] Failed: {e}", "warning")
        return np.empty((0, int(round(epoch_len * expected_fs))), dtype=np.float32)


def read_psg_epochs(logger, psg_file: str, epoch_len: float = EPOCH_LEN) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """Read PSG file and epoch all channels into high-frequency (100 Hz) and low-frequency (1 Hz) dicts."""
    logger.log(f"[READ_PSG_EPOCHS] Reading {os.path.basename(psg_file)}")
    high, low = {}, {}
    try:
        with pyedflib.EdfReader(psg_file) as f:
            labels_raw = [normspace(l) for l in f.getSignalLabels()]
            fs = f.getSampleFrequencies()
            logger.log("[READ_PSG_EPOCHS] Detected channels (label @ fs):")
            for i, (lab, fsi) in enumerate(zip(labels_raw, fs)):
                logger.log(f"[READ_PSG_EPOCHS] - {i:02d}: {lab} @ {fsi} Hz")

            for i, lab in enumerate(labels_raw):
                try:
                    fs_i = float(fs[i])
                    name = slugify(lab)
                    if fs_i >= 50.0:
                        arr = read_and_epoch_channel(logger, f, i, epoch_len, expected_fs=100.0)
                        if arr.size > 0:
                            high[name] = arr
                    elif fs_i <= 2.0:
                        arr = read_and_epoch_channel(logger, f, i, epoch_len, expected_fs=1.0)
                        if arr.size > 0:
                            low[name] = arr
                    else:
                        if fs_i > 2.0:
                            arr = read_and_epoch_channel(logger, f, i, epoch_len, expected_fs=100.0)
                            if arr.size > 0:
                                high[name] = arr
                        else:
                            arr = read_and_epoch_channel(logger, f, i, epoch_len, expected_fs=1.0)
                            if arr.size > 0:
                                low[name] = arr
                except Exception as e:
                    logger.log(f"[READ_PSG_EPOCHS] Channel '{lab}' failed: {e}", "warning")

        n_list = [v.shape[0] for v in high.values()] + [v.shape[0] for v in low.values()]
        if not n_list:
            return {}, {}, {}

        n_epochs = min(n_list)
        for d in (high, low):
            for k in list(d.keys()):
                d[k] = d[k][:n_epochs]

        logger.log(f"[READ_PSG_EPOCHS] n_epochs (common) after cut: {n_epochs} | high_ch={len(high)} | low_ch={len(low)}")
        return high, low, {"high": 100.0, "low": 1.0}

    except Exception as e:
        logger.log(f"[READ_PSG_EPOCHS] Fatal: {e}", "error")
        return {}, {}, {}
