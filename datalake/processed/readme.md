# Complete Feature Dictionary (one line = 30-s epoch)

This README explains what each column is, how it was generated, and why it can be useful for classifying sleep stages. You’ll see repeated names with suffix patterns – the key is to understand these patterns so you can interpret all the columns at once.

**New in ST**: spectral features for `EMG_submental` at 100 Hz, and `Event_marker` statistics at 10 Hz.

---

## 0. Naming Conventions

### Keys / Labeling
* `subject_id`, `night_id` — Identify the record (e.g., SC400, N1; or ST701, N2).
* `age`, `sex` — Subject metadata (years; F/M).
* `epoch_idx` — Epoch index (0, 1, 2, …).
* `t0_sec` — Epoch start time in seconds (from PSG start).
* `stage` ∈ {W,N1,N2,N3,REM} — Sleep stage label.
* `tso_min` — Time Since Sleep Onset: minutes since the first epoch ≠ W (trimmed at ≥ 0).
    * **Why useful?** Sleep is cyclical: N3 tends to cluster early; REM increases toward the end.

### Channels (prefix) & typical sampling
* **EEG**: `EEG_Fpz_Cz`, `EEG_Pz_Oz` (100 Hz)
* **EOG**: `EOG` (100 Hz)
* **EMG submental**:
    * **SC**: time-domain at 1 Hz → `_1hz` suffix
    * **ST**: time-domain at 1 Hz and spectral at 100 Hz → `{256|512}` suffixes
* **Respiration**: `Resp_oronasal` (1 Hz)
* **Temperature**: `Temp_rectal` (1 Hz)
* **Event marker**:
    * **SC**: 1 Hz → `_1hz`
    * **ST**: 10 Hz → `_10hz`

If a channel exists at 100 Hz, it can have spectral features (`{256|512}`). If it only exists at a low rate (1 Hz or 10 Hz), it will have time-domain statistics with the corresponding rate suffix.

### Important suffixes
* `_{256|512}` → Welch parameters for 100 Hz channels (`nperseg` ∈ {256,512}, 50% overlap).
    * **Why two sizes?** 256 is more sensitive to short events; 512 has lower variance.
* `_{1hz|10hz}` → Low-rate time-domain statistics computed at that sampling.
* **Bands** (disjoint for `relpow`/ratios):
    * **delta** (0.5–4 Hz)
    * **theta** (4–8 Hz)
    * **alpha** (8–12 Hz)
    * **sigma** (12–16 Hz)
    * **beta** (16–30 Hz)

---

## 1. Spectral Family (100 Hz channels: EEG/EOG and, in ST, also EMG)
Applies to: EEG, EOG, and `EMG_submental` when present at 100 Hz (ST).

### 1.1 Power per band
* `X_<band>_pow_{256|512}`
    * **What**: Band power via integration of Welch PSD.
    * **Why**: Captures slow vs. fast content (N3 ↑ delta; N2 ↑ sigma; W ↑ beta).
* `X_<band>_logpow_{256|512}`
    * **What**: `log10(pow + ε)`.
    * **Why**: Stabilizes amplitude/variance for models.
* `X_<band>_relpow_{256|512}`
    * **What**: Band power relative to total in 0.5–30 Hz.
    * **Why**: Normalizes gain/impedance; often top-performing features.
* `X_<band>_peakfreq_{256|512}`
    * **What**: Peak frequency (Hz) within the band.
    * **Why**: Alpha/sigma peaks (e.g., spindles) are informative.

### 1.2 Band ratios (all with `{256|512}`)
* `X_delta_theta_ratio`, `X_theta_alpha_ratio`, `X_alpha_sigma_ratio`, `X_slow_fast_ratio`
* **Like**: `delta/theta`, `theta/alpha`, `alpha/sigma`, `(delta+theta)/(alpha+beta)`.
* **Why**: Encode classic contrasts (N3: slow ≫ fast; W: fast ↑).

### 1.3 Global spectral summaries (0.5–30 Hz, `{256|512}`)
* `X_sef95` — Spectral Edge Frequency (95%).
* `X_medfreq` — Median frequency (50%).
* `X_spec_entropy` — Normalized spectral entropy (0–1).
* `X_aperiodic_slope` — 1/f slope from linear fit in log–log (2–30 Hz).
* **Why**: Summarize spectral “speed/organization”; N3 is slower/more structured.

### 1.4 Simple time on 100 Hz channels (no `{256|512}`)
* `X_rms`, `X_var`
    * **What**: RMS and variance of the (centered) raw signal per epoch.
    * **Why**: Overall energy; helps catch artifacts/residual EMG.

**Note (ST only)**: `EMG_submental_*` includes both spectral `{256|512}` and time-domain features when EMG is available at 100 Hz.

---

## 2. Low-Rate Family (1 Hz or 10 Hz)
Statistics per epoch: `mean`, `std`, `min`, `max`, `rms`, `median`, `iqr`, `mad`, `p01`, `p10`, `p90`, `p99`, `kurtosis`, `skewness`, `diff_rms`, `zcr`, `slope`.

* `X_*_1hz` — low-rate time series at 1 Hz (e.g., `EMG_submental` in SC, `Resp_oronasal`, `Temp_rectal`, `Event_marker` in SC).
* `X_*_10hz` — low-rate time series at 10 Hz (e.g., `Event_marker` in ST).
* **Why useful**: Dispersion/shape capture variability (respiration), muscle tone (EMG), and quality signals.

### Quality metrics (when applicable)
* `Resp_oronasal_clip_frac_1hz`
    * **What**: Fraction with `|value| ≥ 900` (saturation/clipping heuristic for this dataset).
    * **Why**: Flag to down-weight unreliable respiratory epochs.
* `Temp_rectal_oor_frac_1hz`
    * **What**: Fraction outside [30, 45] °C.
    * **Why**: Flag erroneous temperature readings.

### Event marker (1 Hz in SC, 10 Hz in ST)
* Statistics follow the same naming pattern: e.g., `Event_marker_mean_1hz` or `Event_marker_mean_10hz`.
* **Caution**: These are timing/annotation aids, not physiology. Avoid as training features until you audit potential label leakage.

---

## 3. How Each Family Was Generated (Technical Summary)

### Welch PSD (100 Hz channels)
* Per-epoch; `nperseg` ∈ {256, 512}, `noverlap` = `nperseg`/2, `fs` = 100 Hz.
* **Bands**: 0.5–4, 4–8, 8–12, 12–16, 16–30 Hz.
* **pow**: integrate PSD over band.
* **relpow**: `pow_band` / `pow_total(0.5–30)`.
* **logpow**: `log10(pow + 1e-12)`.
* **peakfreq**: argmax within the band.
* **Ratios**: `delta/theta`, `theta/alpha`, `alpha/sigma`, `(delta+theta)/(alpha+beta)`.
* **Summaries (0.5–30)**: `sef95`, `medfreq`, `spec_entropy` (normalized), `aperiodic_slope` (log–log fit 2–30 Hz).
* **Time**: `rms`, `var`.

### Low-rate (1 Hz / 10 Hz)
* Robust per-epoch statistics listed above.
* `diff_rms` and `zcr` as simple dynamics.
* `slope` via simple regression (difference/time).
* Channel-specific quality flags (resp clipping; temp out-of-range).

### TSO (temporal context)
* `tso_min` = `max(0, (t0 - t_sleep_onset)/60)` with `t_sleep_onset` = start of 1st epoch with stage ≠ W.

---

## 4. Usage Tips in the Model

* Start with `relpow` + `ratios` + `SEF`/`MedFreq`/`Entropy`/`Slope` + `EMG` stats (`median`/`p90`/`kurtosis`) + `EOG` delta `relpow` + `TSO`.
* Use both `{256,512}` and ablate later.
* **ST**: consider `EMG_submental` spectral features — often helpful for REM/W separation.
* Use quality flags (`resp clipping` / `temp OOR`) to reduce overfitting to noise.
* `Event_marker`: exclude until leakage is ruled out.

---

## 5. Reading Your Examples Quickly

* **SC (epoch W)**: `EOG_delta_relpow_*` ≈ 0.96 → slow EOG (eyes closed, non-REM-like oculogram); `EEG_Fpz_Cz_delta_relpow_*` ≈ 0.85 high → drowsy wake/ocular drift possible; `Resp_oronasal_clip_frac_1hz` ≈ 0.03 flags mild saturation.
* **ST (epoch W)**: strong fast-band content in `EEG_Pz_Oz_beta_*` and `EMG_submental` beta (100 Hz spectral) — consistent with wake; `Event_marker_*_10hz` present (treat cautiously).

---