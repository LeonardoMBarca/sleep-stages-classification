# Sleep-EDF → Tabular (SC & ST) — A "cool" pipeline guide

> This scripts takes the Sleep-EDF EDFs (both Sleep Cassette (SC) and Sleep Telemetry (ST)), aligns them with the hypnogram, extracts features per 30-s epoch, and saves separate parquets (one for SC and one for ST). The idea is to transform raw signals into informative representations to train sleep stage classification models with high accuracy, but in a fast, parallelized, and observable way (progress bar with Rich + logs).

---

## Overview (what the pipeline does)

1. **Pair Discovery** `*-PSG.edf` + `*-Hypnogram.edf`

* Automatically identifies SC vs. ST by `subject_id` (e.g., `SC4001`, `ST7021`).
* Also extracts `night_id` (e.g., `E0`, `E1`), when present.

2. **Hypnogram to PSG Alignment**

* Reads `startdatetime` from both, calculates **offset**, and maps the **stage events** to **30s epochs (R\&K)**.
* Keeps only epochs with a **valid label** (`W`, `N1`, `N2`, `N3`, `REM`).

3. Channel Reading & Epoching

* Divides channels into two groups based on the detected sampling frequency:

* High-FS (100 Hz): EEG/EOG/… (≥ 50 Hz).
* Low-FS (1 Hz): slow signals (oronasal respiration, rectal temperature, marker, etc.).
* Downsamples (block averaging) when necessary, or simply upsamples (repeat) to ensure consistent [epoch × samples] grids.

4. **Canonical Channel Selection** (when labels vary)

* Uses flexible matching to map to “canonical” names:
`EEG_Fpz_Cz`, `EEG_Pz_Oz`, `EOG`, `EMG_submental`, `Resp_oronasal`, `Temp_rectal`, `Event_marker`.
* If a canonical channel doesn't exist, the pipeline **uses the available channels** anyway (robust).

5. **Per-epoch feature extraction (30s)**

* **High-FS (EEG/EOG)** via **Welch PSD** in **two windows**: `nperseg=256` and `512` (half overlap).

* Why two? `256` (2.56 s @ 100 Hz) captures **fast details**, `512` (5.12 s) gives a more **stable** PSD. The model learns from both scales.
* **Low-FS (1 Hz)**: robust statistics + signal **quality** metrics.

6. **Temporal Context (TSO)**

* Calculates **TSO (time since sleep onset)** in minutes, per epoch: time since the **first epoch ≠ W**.
* Why? Sleep is **cyclical**; **REM** increases** at the end of the night, **N3** appears more at the beginning. TSO injects these **temporal priors** into the model.

7. **Parallelization + Progress**

* Parallel processing per pair** (ProcessPoolExecutor).
* Progress bar with **Rich** in the **main process** (workers don't print anything).

8. **Output**

* **Two Parquets**:

* `processed/sleep-cassette/sleep_cassette_dataset.parquet`
* `processed/sleep-telemetry/sleep_telemetry_dataset.parquet`
* Each line = **one epoch** with: `subject_id`, `night_id`, `epoch_idx`, `t0_sec`, `stage`, `tso_min` and **a bunch of features**.

---

## How to run

```bash
python main.py \
--root /path/to/datalake/raw \
--out-sc /path/to/datalake/processed \
--out-st /path/to/datalake/processed \
--workers 7 \
# --no-progress # use if you don't want a slash
```

* If the folder contains **only SC** → generates **only** the SC parquet.
* If it contains **only ST** → generates **only** the ST parquet.
* If it contains both → generates both.

---

## Module structure (what each file does)

### `main.py`

* Orchestrates the pipeline:

* Discovers pairs (`pair_psg_hyp`);
* Triggers parallel workers (`_process_one_pair`); * concatenates results and **saves** parquets (SC/ST);
* prints **class summary** per dataset.
* **Progress bar (Rich)**: shows progress in “completed files”.

### `utils/utils.py`

* `_make_progress()`: creates the Rich bar (or a **Dummy** if Rich is not available, to avoid crashing).
* `_process_one_pair(args)`: isolated **worker** (top-level), creates its own `Logger`, calls `process_record`, and returns `(subject_id, df)`.
* String utility functions: `slugify`, `normspace`, `best_match_idx`.

### `labeler/io_edf.py`

* `pair_psg_hyp`: Finds `PSG.edf` ↔ `Hypnogram.edf` (try prefix `SCxxxx*` and exact match).
* `read_hyp_epochs_aligned`: Aligns hypnogram events to the PSG (uses `startdatetime` for **offset**), generates epoch table with `stage`.
* `read_psg_epochs`: Reads channels, separates **high (100 Hz)** and **low (1 Hz)**, performs epoching, and trims everything to the **minimum common number of epochs**.

### `labeler/features_pipeline.py`

* **Band definition (disjoint for relpow and ratios)**

```
delta: 0.5–4, theta: 4–8, alpha: 8–12, sigma: 12–16, beta: 16–30
```

Disjoint = **no overlap** → **relpow** sum \~1 per channel, clean normalization for models.

* **High-FS** (EEG/EOG, Welch 256 and 512):

* `*_pow_{256|512}`: power per band
* `*_relpow_{256|512}`: relative power per band (over 0.5–30)
* `*_logpow_{256|512}`: log10 of the power per band (stabilizes variance)
* `*_peakfreq_{256|512}`: peak frequency in the band
* **Ratios** (with relpow):
`delta_theta_ratio`, `theta_alpha_ratio`, `alpha_sigma_ratio`, `slow_fast_ratio=(delta+theta)/(alpha+beta)`
* **Spectral summaries** (0.5–30):

* `sef95_{*}`: 95% spectral edge (frequency below which 95% of the energy is)
* `medfreq_{*}`: median power frequency
* `spec_entropy_{*}`: **normalized** spectral entropy (0–1)
* `aperiodic_slope_{*}`: 1/f slope (linear fit in log–log 2–30 Hz)
* **Time**: `rms`, `var`
* Why all this?

* `relpow` and **ratios** distinguish **N3** (high delta), **N2** (sigma), **W** (beta), **REM** (faster EEG + active EOG).
* `sef95`, `medfreq`, `entropy`, and `slope` capture **speed” and “disorder”** of the spectrum, robust across subjects.
* Two windows (256/512) provide **detail** and **stability** at the same time.

* **Low-FS (1 Hz)**: `mean, std, min, max, rms, slope, median, iqr` + **extras**:
`mad, p01, p10, p90, p99, kurtosis, skewness, diff_rms, zcr`

* **Signal quality**:

- `Resp_oronasal_clip_frac_1hz` (|x| ≥ 900) → clipping/saturation
- `Temp_rectal_oor_frac_1hz` (\[30,45] °C) → out of physiological range
Why?
- Even at 1 Hz, it is possible to measure **stability**, **trend**, and **quality** of the channel — useful for the model to “suspect” bad epochs.

* **Temporal context**:

* `tso_min`: minutes since **sleep onset** (first epoch **≠ W**).
Why?
* Sleep is **cyclical**; TSO **increases accuracy** by providing “position in the night” (late REM, early N3).

---

## Design decisions (why)

* **30-s epochs** (R&K): classic standard; maximizes comparability with Sleep-EDF and literature.
* **Relpow with disjoint bands** (α 8–12, σ 12–16):

* Avoids **double counting** at 12–13 Hz → `relpow_*` sums to \~1 → **clean normalization**.
* **Accuracy** tends to be more stable across subjects. * Two windows (256 and 512) in Welch:

* `256` responds better to **short events** (e.g., spindles, micro-wakeups),
* `512` reduces PSD variance (better **robustness**).
* The model chooses what is most valuable (you can ablate later).
* **Per-pair parallelization** + BLAS thread limitations:

* Avoids **oversubscription** (each process using BLAS with multiple threads).
* Shows **progress** per completed file (not per epoch).
* **Log/progress robustness**:

* If Rich fails/is missing, the pipeline **does not break** (uses Dummy).
* Workers have their own `Logger`, without competing with the bar.

---

## Parquet main columns (skeleton)

* Keys & labels:

* `subject_id`, `night_id`, `epoch_idx`, `t0_sec`, `stage ∈ {W,N1,N2,N3,REM}`
* `tso_min` (time since sleep onset, in minutes, ≥ 0)

* EEG/EOG (for **each available** canonical channel):

* `..._{delta|theta|alpha|sigma|beta}_{pow|relpow|logpow}_{256|512}`
* `..._{delta|theta|alpha|sigma|beta}_peakfreq_{256|512}`
* `..._{delta_theta_ratio|theta_alpha_ratio|alpha_sigma_ratio|slow_fast_ratio}_{256|512}` 
* `..._{sef95|medfreq|spec_entropy|aperiodic_slope}_{256|512}` 
* `..._{rms|var}`

* Low-FS (for **each channel** available): 

* `..._{mean|std|min|max|rms|slope|median|iqr|mad}_1hz` 
* `..._{p01|p10|p90|p99}_1hz` 
* `..._{kurtosis|skewness}_1hz` 
* `..._{diff_rms|zcr}_1hz` 
* Quality (when applicable): 
`Resp_oronasal_clip_frac_1hz`, `Temp_rectal_oor_frac_1hz`

> **Important:** Not every file will have all channels. The pipeline **does not require** that they all exist.

---

## Common Questions

* "Why doesn't my `relpow` sometimes add up to 1?"

Here we use disjoint bands, so it tends to add up. If you choose bands with overlap, the sum can exceed 1 (this isn't an error, it's a methodological choice).

* "Can TSO be negative?"
No. If there are no epochs ≠ W, `tso_min` is set to 0 for safety.

* **Is event marker useful for training?**
In general, **no** (there may be human leaks). We keep it as a **quality signal**/reference; we recommend **not using** it on model features until you audit it.

---

## Summary

* **Fast** (parallel), **observable** (Rich + logs), and **robust** (does not stop for details).
* **Features designed** to maximize **accuracy**: relpow with **disjoint bands**, rich spectral summaries, **TSO**, and **quality** metrics.
* Clean output in two parquets (SC/ST), ready for ML.

If you'd like, I can help you put together an analysis notebook with: top 20 features by importance (LightGBM), confusions by stage, and 256×512 ablations — this way, you can close the loop scientifically.