# Sleep-Stage Fundamentals

This note captures the essentials you need before diving into the modelling code or the feature catalog. It focuses on **what is measured**, **how the raw polysomnography (PSG) is converted into per-epoch tables**, and **why specific signal patterns matter for classification**.

---

## 1. Sleep Architecture in a Nutshell

| Stage | Physiological signature | Typical EEG/EOG/EMG behaviour |
|-------|------------------------|--------------------------------|
| **W** (Wake) | Alert, eyes open/closed | Beta & alpha dominate; muscle tone high |
| **N1** | Sleep onset | Low-amplitude mixed EEG; slow eye movements |
| **N2** | Light sleep | Sleep spindles (12–16 Hz sigma) and K-complexes |
| **N3** | Deep sleep | Delta (0.5–4 Hz) dominates; high-amplitude slow waves |
| **REM** | Rapid-eye-movement sleep | Active EOG bursts, near-absent chin EMG, mixed-frequency EEG |

- Sleep studies annotate stages in **30-second epochs**. Each line in our parquet files corresponds to a single epoch with columns `subject_id`, `night_id`, `epoch_idx`, `t0_sec`, and the target label `stage`.
- We also include **time since sleep onset** (`tso_min`), which counts minutes after the first non-W epoch. It provides a “where in the night” cue without leaking future information.

---

## 2. Core Sensor Modalities

| Channel | What it measures | How we use it |
|---------|-----------------|---------------|
| **EEG Fpz–Cz / Pz–Oz** | Cortical activity at 100 Hz | Frequency bands (delta→beta) and spectral summaries encode stage-specific rhythms |
| **EOG** | Eye movements at 100 Hz | Detect REM bursts and wake transitions |
| **Submental EMG** | Chin muscle tone (1 Hz in cassette, 100 Hz in telemetry) | Differentiates REM (atonia) from wake/NREM |
| **Resp_oronasal** | Respiratory flow at 1 Hz | Context for arousals, aggregates via robust statistics |
| **Temp_rectal** | Body temperature at 1 Hz | Slow contextual drift; used mainly as quality indicator |
| **Event marker** | Technician events | Logged for completeness; flagged as potential leakage |

Both **Sleep Cassette (SC)** and **Sleep Telemetry (ST)** cohorts share the same structure, though ST adds higher-rate submental EMG and 10 Hz event-marker summaries.

---

## 3. Sampling, Windows, and PSD Extraction

- Raw EDF channels have heterogeneous sampling rates. The processing pipeline resamples:
  - High-frequency channels (EEG, EOG, EMG in ST) → **100 Hz**
  - Low-frequency channels (respiration, temperature, EMG in SC) → **1 Hz** (or 10 Hz for telemetry event markers)
- Power Spectral Density (PSD) is computed per epoch with **Welch’s method** at `nperseg = 256` and `512` (50% overlap). Using both window sizes balances sensitivity to short events and variance reduction.

### Frequency Bands

We work with **disjoint bands** to keep relative powers interpretable:

| Band | Range (Hz) | Major cues |
|------|------------|------------|
| Delta | 0.5–4 | Dominant in N3 |
| Theta | 4–8 | Transitions, N1/N2 |
| Alpha | 8–12 | Wake with eyes closed |
| Sigma | 12–16 | Sleep spindles (N2) |
| Beta | 16–30 | Wake/REM fast activity |

From each PSD we derive:

- `pow`, `logpow`, `relpow` for every band
- Band ratios such as `delta/theta`, `theta/alpha`, `alpha/sigma`, and `(delta+theta)/(alpha+beta)` (aka slow/fast)
- Global summaries: `sef95`, `medfreq`, `spec_entropy`, and `aperiodic_slope`
- `peakfreq` to highlight spindle or alpha peaks

---

## 4. Time-Domain Statistics

For 1 Hz or 10 Hz channels we generate robust summaries per epoch:

- Central tendency & spread: mean, std, median, IQR, MAD
- Extremes: min, max, percentiles (p01/p10/p90/p99)
- Shape: skewness, kurtosis
- Dynamics: `diff_rms`, `zcr`, and linear `slope`
- Quality flags: `Resp_oronasal_clip_frac_1hz`, `Temp_rectal_oor_frac_1hz`

These statistics provide additional context for breathing effort, temperature drift, and muscle tone without introducing look-ahead bias.

---

## 5. Naming Patterns in the Parquet Files

Every column follows: `CHANNEL_STAT_SUFFIX`

- `EEG_Fpz_Cz_delta_relpow_256` → Fpz–Cz EEG, delta band, relative power, Welch window 256
- `EMG_submental_median_1hz` → 1 Hz submental EMG median
- `EOG_rms_roll_mean_5` → Rolling (5-epoch) mean of EOG RMS
- `stage` ∈ {W, N1, N2, N3, REM}, label used for training
- `tso_min` → minutes since sleep onset (clamped ≥ 0)

Understanding suffixes lets you quickly reason about hundreds of engineered features without memorising each name.

---

Use this cheat-sheet alongside the dataset-specific `datalake/processed/readme.md` to map every parquet column back to its physiological meaning.
