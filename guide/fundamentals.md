# Fundamental README — concepts you *need* to master to understand/use this dataset

> Objective: To give you a mental map of the **signals**, **labels**, **extraction methods**, and **best practices**. First, the theory (the "why"), then the "how this appears in the columns."

---

## 1) Structure of a sleep study (PSG + hypnogram)

* **PSG (Polysomnography)**: multimodal sleep recording:
**EEG** (electroencephalogram), **EOG** (electro-oculogram), **EMG** (muscle activity), **respiration**, **oximetry**, **temperature**, etc.
* **Hypnogram**: temporal sequence of sleep **stages** recorded by an expert (or algorithm), with markings such as "SLEEP STAGE 2 (120 s)".

### 1.1 Stages (R&K/AASM)

* **W** (Wake): wakefulness.
* **N1**: sleep onset, transition; faster, low-amplitude EEG.
* **N2**: light sleep; typical presence of **spindles (12–16 Hz)** and **K-complexes**.
* **N3** (deep sleep): **delta** (0.5–4 Hz) dominant.
* **REM**: rapid eye movements (**active EOG**), very low **submental EMG**, faster EEG.

> Classical patterns:
> N3 ↑ delta; N2 ↑ sigma (spindles); W ↑ beta; REM: active EOG + low EMG.

### 1.2 Epoch and EPOCH\_LEN

* The hypnogram is converted to **fixed epochs** (typically **30 s**).
Each Parquet line = **1 30-s epoch**.
* Basic columns: `subject_id`, `night_id` (e.g., SC4001, E0), `epoch_idx`, `t0_sec`, `stage`.

### 1.3 `subject_id` and `night_id`

* **Sleep Cassette (SC)** and **Sleep Telemetry (ST)**: different families of Sleep-EDF.
Examples: `SC4001`, `ST7021`.
* `night_id`: some people have more than one night (e.g., `E0`, `E1`).

---

## 2) Channels and what they measure

### 2.1 EEG (EEG\_Fpz\_Cz, EEG\_Pz\_Oz, …)

* Cortical electrical activity; **100 Hz** after *epoching* in the pipeline.
* **Classical bands** (typical limits):
**delta** 0.5–4 Hz, **theta** 4–8 Hz, **alpha** 8–12 Hz, **sigma** 12–16 Hz, **beta** 16–30 Hz.
* For stage classification:
N3 → delta ↑; N2 → sigma ↑; W → beta/alpha ↑ (alpha plus with eyes closed).

### 2.2 EOG

* Potential difference near the eyes. * REM: bursts/characteristics; outside of REM, may experience slowing (drifts).

### 2.3 Submental EMG

* Chin muscle activity (tone).
* REM: very low tone; W/NREM: higher.

### 2.4 Nasal Respiratory Respiration

* Respiratory flow/pressure; 1 Hz after resampling.
* Useful for context (apneas/hypopneas may occur), but be careful with saturations.

### 2.5 Rectal Temp

* Body temperature (slow, stable).
* Useful for quality/context; not a direct stage marker.

### 2.6 Event Marker

* Marking channel (technician or system clicks). * **Not physiological** — avoid using it as a training feature to avoid creating a leak.

---

## 3) Sampling, Windowing, and Welch

### 3.1 Sampling and Resampling

* The original dataset has channels with different sampling rates.
* The pipeline groups them into:

* **High-FS** → resampling to **100 Hz** (EEG/EOG, etc.).
* **Low-FS** → resampling to **1 Hz** (Resp, Temp, Marker).
* In each epoch (30 s), you have matrices **\[epoch × samples]**:
30 s × 100 Hz = 3000 samples (high); 30 s × 1 Hz = 30 samples (low).

### 3.2 PSD, Windowing, and Welch

* PSD (Power Spectral Density): How the signal energy is distributed across frequencies.
Welch: Divides the epoch into segments, applies a window (typically Hanning), performs an FFT per segment, and averages the powers.
Advantage: PSD with lower variance than a single FFT of the entire epoch.
nperseg (and what it does): Number of samples per segment.
With 100 Hz:

nperseg = 256 → segment ≈ 2.56 s
nperseg = 512 → segment ≈ 5.12 s
In the pipeline, we use both (256 and 512) to capture short events and achieve stability.

> **Overlap**: 50% overlap is typically used; the pipeline follows this (by default in `scipy.signal.welch` when we set `noverlap=nperseg//2`).

---

## 4) Frequency Bands and Power Types

### 4.1 Disjoint Bands (recommended for *relative power*)

* **delta** 0.5–4, **theta** 4–8, **alpha** 8–12, **sigma** 12–16, **beta** 16–30.
* **Disjoint** means: no double counting (the 12 Hz bin is sigma, not alpha), so **sum of relpow ≈ 1**.

> In some work, alpha=8–13 (overlap with sigma). This is ok, but for *relpow* we prefer **disjoint** (cleaner normalization).

### 4.2 `pow`, `logpow`, `relpow`

* **pow**: integral of the PSD in the band (**absolute** energy).
* **logpow**: `log10(pow + ε)` — stabilizes and provides a more “linear” scale.
* **relpow**: `pow_band / pow_total(0.5–30 Hz)` — **normalizes** the gain; very useful.

### 4.3 Band Ratios

* **delta/theta**, **theta/alpha**, **alpha/sigma**, **slow/fast**:
`slow = delta+theta`; `fast = alpha+beta`.
* Highlights classic contrasts between stages (N3: slow ≫ fast; W: fast ↑).

---

## 5) Other spectral features

* **`peakfreq`**: **peak** frequency within the band (in Hz).
Useful for detecting *spindles*(sigma) or changes in alpha.

* **`sef95` (Spectral Edge 95%)**: Frequency below which **95%** of the energy is (0.5–30 Hz).
Proxy for “how fast” the spectrum is.

* **`medfreq`**: Median frequency (50% of the energy).
Similar to SEF, but even more resistant to outliers.

* **`spec_entropy`**: **normalized** spectral entropy (0–1).
0 = very concentrated energy; 1 = well spread out.
In N3 (slower, more structured), it is usually lower than in W/REM.

* **`aperiodic_slope`**: Slope of the **1/f** component (linear fit in log–log between 2 and 30 Hz).
Captures the “base” of the spectrum; may vary with alertness and stage.

---

## 6) Features in Time (1 Hz and 100 Hz)

### 6.1 Simple Features (Time)

* **RMS**, **variance**, **mean**, **median**.
* **IQR**, **MAD** (robust to outliers).
* **Percentiles** (p01, p10, p90, p99) — signal tails.
* **Kurtosis** (tails) and **Skewness** (asymmetry).
* **diff\_rms**: RMS of the first difference → “speed”/instability.
* **ZCR** (Zero-Crossing Rate): fraction of signal changes (∈ \[0,1]).
* **Slope**: linear trend over the epoch (approximate; at 1 Hz it is simple and useful).

### 6.2 **Quality** Flags

* **Resp\_oronasal\_clip\_frac\_1hz**: Fraction of **saturated** samples (|x| ≥ 900).
If high → be careful with this channel/epoch.
* **Temp\_rectal\_oor\_frac\_1hz**: Fraction **out of physiological range** (e.g., \[30,45] °C).
Flags an incorrect sensor/measurement.

---

## 7) Temporal Context

* **`tso_min`** (*Time Since Sleep Onset*): minutes since the first epoch **≠ W**.
Why is it strong? Sleep is **structured in cycles** (\~90 min): N3 appears more at the beginning; **REM increases throughout the night**. TSO gives the model a "position in the night," which increases accuracy.

---

## 8) How the alignment is done

1. Read `startdatetime` from the PSG and Hypnogram.
2. Calculate offset = `start_hyp − start_psg`.
3. For each "SLEEP STAGE X (duration)" annotation, map the windows to the PSG time axis.
4. Construct 30-s epochs and mark the stage by the epoch center (common rule).
5. Epochs outside any stage event are discarded (unlabeled).

---

## 9) Why use two npersegs (256 and 512)?

* **nperseg=256** (2.56 s):
Improves detection of **short events** (spindles, micro-awakenings), but the PSD has **more variance**.
* **nperseg=512** (5.12 s):
More **stable** PSD (less variance), but can “smooth” short events.
* **Using both** gives the model **short and long scale**; then you can ablate and keep only what helps.

---

## 10) Usage Tips (for real accuracy)

1. **Split by subject** (GroupKFold): never mix the same `subject_id` between training and testing.
2. **Normalization by subject/night** within the **training pipeline** (prevents leakage).
3. **Class balancing** (macro-F1, kappa).
4. Post-classification **Temporal smoothing** (moving average of probabilities, or simple HMM with transition matrix).
5. **Ablation**: test keeping only disjoint `relpow` + `sef/entropy/slope` + principal EMG/EOG; compare 256 vs. 512; see if `logpow` aggregates.

---

## 11) Quick Glossary

* **EEG**: cortical electrical activity.
* **EOG**: eye movements (REM is highlighted).
* **Submental EMG**: chin muscle tone (drops in REM).
* **Oronasal Resp**: respiratory flow/pressure (susceptible to saturation).
* **Rectal Temp**: body temperature (slow, for control/quality).
* **Eventmarker**: event markings (non-physiological; beware of *leakage*).
* **Epoch**: Fixed window (30 s) used to label a stage.
* **Stage**: W, N1, N2, N3, REM.
* **Welch**: Method for estimating PSD by averaging overlapping segments.
* **nperseg**: Number of samples per segment in Welch (256 → 2.56 s @100 Hz; 512 → 5.12 s).
* **PSD**: Power spectral density (energy per frequency).
* **pow / relpow / logpow**: Absolute / relative power / log power.
* **peakfreq**: Frequency of peak power in a band.
* **sef95 / medfreq**: 95% spectral edge / median power frequency.
* **spec\_entropy**: Normalized spectral entropy (0–1). * **aperiodic\_slope**: slope of the “1/f” (bottom of the spectrum).
* **TSO (`tso_min`)**: minutes since sleep onset (1st epoch ≠ W).
* **Clip/OOR**: saturation (resp) / out of range (temperature).

---

## 12) How these ideas become columns in Parquet

* **Keys/time**: `subject_id`, `night_id`, `epoch_idx`, `t0_sec`, `stage`, `tso_min`. * **EEG/EOG (100 Hz)**: 

* `CANAL_banda_{pow|relpow|logpow|peakfreq}_{256|512}` 
* `CHANNEL_{delta_theta_ratio|theta_alpha_ratio|alpha_sigma_ratio|slow_fast_ratio}_{256|512}` 
* `CANAL_{sef95|medfreq|spec_entropy|aperiodic_slope}_{256|512}` 
* `CHANNEL_{rms|var}`
* **Low-FS (1 Hz)**: 

* `CHANNEL_{mean|std|min|max|rms|slope|median|iqr|mad}_1hz` 
* `CHANNEL_{p01|p10|p90|p99}_1hz` 
* `CHANNEL_{kurtosis|skewness|diff_rms|zcr}_1hz` 
* quality: `Resp_oronasal_clip_frac_1hz`, `Temp_rectal_oor_frac_1hz`

---

## 13) Common pitfalls (and how to avoid them)

* Band overlap (alpha 8–13 vs sigma 12–16): for relpow, prefer disjoint (sum ≈ 1).
* Leakage with Event_marker: treat as metadata/quality, not as a physiological signal.
* Saturated signals (respiration): use clip_frac as a quality feature.
* Between-subject variation: normalize per subject in the training pipeline.
* Leakage: do not compute global statistics using testing; always use per training and apply to testing.