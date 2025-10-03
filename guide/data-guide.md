# Data Guide

This short guide complements the fundamentals note with a signal-by-signal cheat sheet. It highlights what each channel represents, how it is transformed inside the pipeline, and why it matters for sleep-stage modelling.

---

## EEG — Cortical Activity (Fpz–Cz, Pz–Oz)

- **Acquisition**: Referential leads sampled at 100 Hz after resampling.
- **Derived features**:
  - Absolute, logarithmic, and relative band powers (`pow`, `logpow`, `relpow`) for delta/theta/alpha/sigma/beta using Welch windows 256 and 512.
  - Band ratios (`delta_theta_ratio`, `theta_alpha_ratio`, `alpha_sigma_ratio`, `slow_fast_ratio`).
  - Spectral edge frequency (`sef95`), median frequency, spectral entropy, and aperiodic slope.
  - Peak frequency per band (`peakfreq`) to catch spindles or alpha shifts.
  - RMS/variance of the raw 100 Hz signal per epoch.
- **Why it helps**: Sleep stages are largely defined by spectral fingerprints (delta for N3, sigma spindles for N2, beta/alpha in wakefulness).

---

## EOG — Horizontal Eye Movements

- **Acquisition**: 100 Hz, referenced to capture horizontal movement.
- **Features**: Same spectral family as EEG, plus rolling statistics (mean/std/max) over 5, 10, and 15 epoch windows.
- **Why it helps**: Rapid bursts during REM, slow drifts during N1, transitions when dozing off.

---

## Submental EMG — Chin Muscle Tone

- **Cassette**: 1 Hz time-series statistics (`median`, `p90`, `kurtosis`, etc.).
- **Telemetry**: Both 1 Hz statistics and 100 Hz spectral descriptors (parallel to EEG/EOG) thanks to the higher sampling rate.
- **Why it helps**: Muscle atonia is a strong REM indicator; tone also differentiates wake from NREM.

---

## Respiration (Resp_oronasal)

- **Sampling**: 1 Hz.
- **Features**: Robust statistics, change-in-rate proxies (`diff_rms`, `zcr`), and quality flag `clip_frac` for saturation detection.
- **Why it helps**: Respiratory stability/instability tracks arousals and can indicate state transitions.

---

## Temperature (Temp_rectal)

- **Sampling**: 1 Hz.
- **Features**: Same statistical set as respiration with `oor_frac` to flag out-of-range temperatures.
- **Why it helps**: Provides slow context and sanity checks; shifts can reflect sensor detachment or subject movement.

---

## Event Marker

- **Cassette**: 1 Hz statistics.
- **Telemetry**: 10 Hz statistics to capture richer annotation dynamics.
- **Caution**: These columns reflect technician/recording events, not physiology. Avoid training on them unless you have verified they do not leak labels.

---

## Rolling Features

Rolling summaries (mean/std/max) are applied to selected high-value signals to inject short-term history without violating causality:

- 5-epoch windows (`_roll_mean_5`, `_roll_std_5`) for fast dynamics (EEG beta, EOG RMS, EMG tone).
- 10-epoch windows for medium trends.
- 15-epoch means for very slow drift (delta power, spectral entropy, aperiodic slope).

These windows correspond to 2.5, 5, and 7.5 minutes of context respectively.

---

## Terminology Recap

- `pow`: absolute band power.
- `relpow`: power ratio relative to total 0.5–30 Hz energy.
- `logpow`: `log10(pow + ε)` for variance stabilisation.
- `peakfreq`: frequency with the highest PSD value inside the band.
- `sef95`: frequency below which 95% of power resides.
- `medfreq`: median frequency.
- `spec_entropy`: normalised spectral entropy (0–1).
- `aperiodic_slope`: slope of the fitted 1/f baseline.
- `diff_rms`: RMS of first difference (signal velocity proxy).
- `zcr`: zero-crossing rate (0–1).

Keep this guide nearby when you inspect the parquet columns or design new features—it connects each name back to the physiology it summarises.
