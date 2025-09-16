# Complete feature dictionary (one line = 30-s epoch)

> This README explains **what each column is**, **how it was generated**, and **why it can be useful** for classifying sleep stages. You'll see several repeated names with **suffix patterns** – the key is to understand these patterns so you can interpret **all** the columns at once.

---

## 0) Naming Conventions

**Keys/Labeling**

* `subject_id`, `night_id` — identify the record (e.g., `SC4001`, `E0`).
* `epoch_idx` — epoch index (0, 1, 2, …).
* `t0_sec` — epoch start time, in seconds since the beginning of the PSG.
* `stage ∈ {W,N1,N2,N3,REM}` — epoch label (wake and sleep stages).
* `tso_min` — *Time Since Sleep Onset*: minutes since the **first epoch ≠ W** (trimmed at ≥ 0).
**Why useful?** Sleep is cyclical: **N3** tends toward the beginning, **REM** increases toward the end.

**Channels (prefix)**
Examples in your data:

* `EEG_Fpz_Cz`, `EEG_Pz_Oz` (EEG, 100 Hz)
* `EOG` (electro-oculogram, 100 Hz)
* `EMG_submental`, `Resp_oronasal`, `Temp_rectal`, `Event_marker` (1 Hz)

**Important suffixes**

* `_256` / `_512` → Welch parameters: `nperseg = 256` (2.56 s @ 100 Hz) and `512` (5.12 s).
*How:* PSD per epoch via Welch (overlap 50%).
*Why 2 sizes?* `256` is more sensitive to short events; `512` has less variance → the model chooses what helps.
* `_1hz` → **slow** channel features (sampled at 1 Hz) in the time domain.
* Bands (disjoint for relpow and ratios):

* `delta` (0.5–4), `theta` (4–8), `alpha` (8–12), `sigma` (12–16), `beta` (16–30).

---

## 1) Spectral family (100 Hz channels: EEG/EOG)

### 1.1 Power per band

* `X_<band>_pow_{256|512}`
**What it is:** Average power of the band, integrating the Welch PSD over the band. **How:** `pow = ∑ PSD(f)·Δf` for f in band.
**Why useful:** Captures **slow vs fast** (N3 ↑ delta; N2 ↑ sigma; W ↑ beta).

* `X_<band>_logpow_{256|512}`
**What it is:** `log10(pow + ε)`.
**How:** Log stabilizes amplitude/variance.
**Why useful:** More “friendly” to linear models and sometimes improves separation.

* `X_<band>_relpow_{256|512}`
**What it is:** Relative power in band relative to total **0.5–30 Hz**.
**How:** `rel = pow_band / pow_total(0.5–30)`. **Why useful:** Normalizes gain/impedance; often a **champion feature**.

* `X_<band>_peakfreq_{256|512}`
**What it is:** Frequency (Hz) of the **peak** of the PSD **within the band**.
**How:** `argmax` of the PSD in the band bins.
**Why useful:** Peaks at **alpha** (8–12) and **sigma** (12–16) are informative (e.g., *sleep spindles*).

### 1.2 Band Ratios (all with `{256|512}`)

* `X_delta_theta_ratio`, `X_theta_alpha_ratio`, `X_alpha_sigma_ratio`, `X_slow_fast_ratio`
**What it is:** Quotients between powers (or relative) of bands.
**Like:** e.g. `delta_theta = delta / theta`; `slow_fast = (delta+theta)/(alpha+beta)`.
**Why useful:** Reinforces typical contrasts (N3: slow ≫ fast; W: fast ↑).

### 1.3 Global Spectral Summaries (0.5–30 Hz, `{256|512}`)

* `X_sef95` — *Spectral Edge Frequency* (95%)
**What it is:** The frequency **up to which** 95% of the spectral energy lies.
**Why useful:** Spectral “speed” (lowest N3, highest W).

* `X_medfreq` — median power frequency (50%).
**Why useful:** Similar to SEF, robust to outliers.

* `X_spec_entropy` — **normalized** spectral entropy (0–1).
**Like:** `H = -∑ p log p`, with `p = PSD/sum`. Normalizes by `log(Nbins)`.
**Why useful:** Measures spectral “disorder” (more “organized” N3 → lower entropy).

* `X_aperiodic_slope` — 1/f slope (linear fit in log–log, 2–30 Hz).
**Like:** Regression of `log10(PSD)` vs `log10(f)`.
**Why useful:** Background component (1/f) changes with stage/alert; N3 tends to have a more negative slope (lower frequency).

### 1.4 Simple Time (no `{256|512}`)

* `X_rms`, `X_var`
**What it is:** RMS and variance of the raw (centered) trace at the epoch.
**Why useful:** Global energy over time; can capture artifacts, residual EMG, etc.

> **Quick clinical reading with your numbers (epoch W):**
> `EEG_Fpz_Cz_delta_relpow_256 ≈ 0.85` is high; there may be a slow/closed EOG.
> `EOG_delta_relpow_* ≈ 0.96` suggests low-frequency dominance in the EOG—common outside of REM.

---

## 2) Low-rate family (1 Hz: EMG, Resp, Temp, Marker)

**Statistics per epoch:** `mean`, `std`, `min`, `max`, `rms`, `median`, `iqr`, `mad`, `p01`, `p10`, `p90`, `p99`, `kurtosis`, `skewness`, `diff_rms`, `zcr`, `slope` — all with `_1hz` suffix.

* `X_mean_1hz`, `X_median_1hz`
**What it is:** Mean/median signal level at the epoch.
**Why useful:** Highest EMG in W; stable temperature (quality control).

* `X_std_1hz`, `X_iqr_1hz`, `X_mad_1hz`
**What it is:** Dispersion (SD, interquartile range, median absolute deviation).
**Why useful:** Robustness to outliers; respiratory variability; and EMG instability.

* `X_min_1hz`, `X_max_1hz`, `X_rms_1hz`
**Why useful:** Amplitude useful for detecting clips/saturations and tonic shifts.

* `X_p01_1hz`, `X_p10_1hz`, `X_p90_1hz`, `X_p99_1hz`
**What it is:** Percentiles; quantify tails/extremes.
**Why useful:** Respiration has wide tails (apneas/hypopneas affect extremes).

* `X_kurtosis_1hz`, `X_skewness_1hz`
**What it is:** Shape of the distribution; *kurtosis* (tails), *skewness* (asymmetry).
**Why useful:** Very non-Gaussian signals (e.g., resp) reveal events.

* `X_diff_rms_1hz`
**What it is:** RMS of the first difference; “speed” of the 1 Hz signal.
**Why useful:** Instability/rapid changes (even at 1 Hz).

* `X_zcr_1hz` (zero-crossing rate; ∈ \[0,1])
**What it is:** Fraction of consecutive pairs that change sign (+/−).
**Why useful:** Oscillatory (limited to 1 Hz, but still signals noise/instability).

* `X_slope_1hz`
**What it is:** Approximate slope (units/sec) over the epoch.
**Why useful:** Trend — e.g., minimal warming in temperature; slowly increasing/decreasing “drift.”

**Quality metrics (when applicable)**

* `Resp_oronasal_clip_frac_1hz`
**What it is:** Fraction of samples with |value| ≥ **900** (indicative of saturation/common clipping in this dataset).
**Why useful:** *flag* to avoid blindly trusting respiration at that epoch.

* `Temp_rectal_oor_frac_1hz`
**What it is:** Fraction outside the physiological range \[30, 45] °C.
**Why useful:** *flag* of an erroneous sensor/measurement.

**Note about `Event_marker_*_1hz`**

* These are **event marker** metrics; **they are not** direct physiological measurements.
* Useful for auditing/timing, but **avoid** as a training feature until auditing *leakage* (human markers may coincide with stages).

---

## 3) How each family was generated (technical summary)

* **Welch PSD (EEG/EOG):**

* window per epoch; `nperseg ∈ {256,512}`, `noverlap = nperseg/2`, `fs = 100 Hz`.
* Disjoint bands (0.5–4, 4–8, 8–12, 12–16, 16–30).
* **pow:** integration of the PSD in the band; **relpow:** `pow_band / pow_total(0.5–30)`;
**logpow:** `log10(pow + 1e-12)`; **peakfreq:** `argmax` of the PSD in the band.
* **Ratios:** delta/theta, theta/alpha, alpha/sigma, (delta+theta)/(alpha+beta).
* **Summaries (0.5–30):** `sef95`, `medfreq`, `spec_entropy` (normalized), `aperiodic_slope` (log–log regression 2–30 Hz).
* **Time:** `rms`, `var`.

* **1 Hz (EMG/Resp/Temp/Marker):**

* Robust statistics per epoch.
* `diff_rms` and `zcr` at 1 Hz (simple indicators of dynamics).
* `slope` by simplified regression (difference/time).
* *Channel-specific quality flags* (respiration: clipping; temperature: out of range).

* **TSO (temporal context):**

* `tso_min = max(0, (t0 - t_sleep_onset)/60)` with `t_sleep_onset = epoch_idx of the 1st stage ≠ W * 30 s`.

---

## 4) Usage Tips in the Model

* Start with **relpow** (disjoint), + **ratios**, + **sef/medfreq/entropy/slope**, + **EMG median/p90/kurtosis**, + **EOG delta relpow**, + **TSO**.
* Use both `_256` and `_512`; later you can ablate (sometimes `_512` wins).
* **Quality flags** (clipping/oor) help reduce *overfitting* to dirt. * **Event\_marker**: Leave out until audited.

---

## 5) Reading your example quickly

* `stage="W"`, but `EOG_delta_relpow_* ~ 0.96` → very slow EOG (eyes closed/no REM).
* `EEG_Fpz_Cz_delta_relpow_* ~ 0.85` high — could be ocular drift/artifact or drowsy wakefulness; check sequences.
* `Resp_oronasal_clip_frac_1hz ~ 0.033` → resp signal saturated at ~3% of the epoch; good to use as *downweight*.
* `Temp_rectal_*` is stable and within range; quality OK.
* `TSO = 0` (sleep not yet initiated — 1st epoch W).