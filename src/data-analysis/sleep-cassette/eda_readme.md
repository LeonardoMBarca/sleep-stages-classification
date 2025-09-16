# EDA Conclusion Report: Feature Selection Strategy

**Objective:** To justify which features to keep and which to exclude for a real-time model (30s per epoch), based on stage discrimination, redundancy/collinearity, risk of temporal leakage, and physiological plausibility.

---

## 1. Decision Criteria

#### **Discrimination Between Stages**
- **Method:** Use effect size from non-parametric tests (e.g., Kruskal-Wallis's $\eta^2$).
- **Rule of thumb:** $\eta^2 \geq \sim0.06$ indicates at least a medium effect size.

#### **Temporal Leakage (TSO - Time Since Onset)**
- **Method:** Check `|Spearman|(feature, tso_min)`.
- **Rule:** Prioritize features with $|\rho| \leq 0.50$. A high correlation suggests the feature is merely encoding the "time of night."

#### **Redundancy & Collinearity**
- **Method:** Analyze pairs with $|\text{corr}| \approx 1$ (e.g., `*_pow_*` vs `*_logpow_*`, `rms` vs `var`, `_256` vs `_512`).
- **Rule:** Form clusters of features with $|\text{corr}| \geq 0.85$ and keep only one representative from each cluster (e.g., the one with the lowest VIF or highest interpretability).

#### **Signal Quality (QC)**
- **Method:** Identify very sparse flags (e.g., `*_clip_frac_*`, `*_oor_frac_*`).
- **Rule:** These are for auditing or sample weighting, not for direct use as predictors.

#### **Physiological Plausibility**
- **EEG:** $\delta \uparrow$ in N3, $\sigma \uparrow$ in N2, $\beta/\alpha \uparrow$ in Wake.
- **EOG:** Bursts of activity / high entropy in REM.
- **EMG:** Muscle atonia in REM.
- **Respiration:** Variability $\uparrow$ in Wake/REM.

---

## 2. Decisions by Macro-Group

### 2.1 EEG Features

**Keep (Lean Core Set; choose one option):**

-   **Option A — Prioritize Pz-Oz (often shows better discrimination):**
    -   `EEG_Pz_Oz_delta_relpow`
    -   `EEG_Pz_Oz_sigma_relpow`
    -   `EEG_Pz_Oz_beta_relpow`

-   **Option B — Consistent with your "temporal core" in Fpz-Cz:**
    -   `EEG_Fpz_Cz_delta_relpow`
    -   `EEG_Fpz_Cz_sigma_relpow`
    -   `EEG_Fpz_Cz_beta_relpow`

> **Tip:** Use a single resolution (standardize on either `_512` or `_256`) and one primary channel to avoid collinearity. If you want spatial diversity, supplement with 1–2 bands from the other channel.

**Optional (if they add value):**

-   1–2 well-chosen ratios (e.g., `slow_fast_ratio` or `delta_theta_ratio`, but not both).
-   1 global feature per channel (e.g., `spec_entropy`, `sef95`, or `aperiodic_slope` — choose only one).

**Exclude (Reasons):**

-   `*_pow_*` and `*_logpow_*`: Duplicates ($|\text{corr}| \approx 1$). Prefer `*_relpow_*`.
-   Resolution duplicates (`_256` vs `_512`): Keep only one.
-   `rms` vs `var`: Keep `rms` if necessary; `relpow` already captures energy information.
-   Too many simultaneous ratios: They carry the same information as the base bands.
-   `*_peakfreq_*`, `*_medfreq_*`: Keep only if an ablation study shows a clear performance gain.

### 2.2 EOG Features

**Keep (Strong and complementary):**

-   `EOG_rms` (intensity of eye movements).
-   `EOG_theta_relpow` (useful for transitions, N1/REM).
-   `EOG_spec_entropy` and/or `EOG_sef95` (captures the "disorder"/high-frequency activity of REM).
-   `EOG_delta_relpow` (indicates calm/absence of movement in NREM).

**Optional:**

-   `EOG_beta_relpow` (complementary, but does not accurately mark REM on its own).

**Exclude:**

-   Duplicates like `_256`/`_512` and `pow`/`logpow` if `relpow` is already included.

### 2.3 EMG / Respiration / Temperature (1 Hz Features)

**Keep:**

-   **EMG:** `EMG_submental_p90_1hz` (or `mean_1hz`; `p90` is great at separating W↔REM).
-   **Respiration:** `Resp_oronasal_std_1hz` and/or `Resp_oronasal_iqr_1hz` (captures irregularity in W/REM).

**Exclude / Use only as QC:**

-   **Temperature:** Continuous features (`Temp_rectal_*_1hz`) have low physiological relevance for staging and are susceptible to drift/time. Use only for quality control.
-   **Flags:** `*_clip_frac_1hz`, `*_oor_frac_1hz`. These are sparse; keep them out of the model and use them for masking or weighting samples.

### 2.4 Identifiers, Time, and Markers

**Exclude from model features:**

-   `subject_id`, `night_id`, `epoch_idx`, `t0_sec` (Identifiers).

**Exclude from training features:**

-   `tso_min` (encodes "when" in the night, a direct temporal leak).

**Exclude entirely:**

-   `Event_marker_*` (high risk of leakage / non-physiological).

---

## 3. Suggested Final Shortlist

**Minimalist Version (≈ 12–14 features)**

-   **EEG (Choose A or B, standardize resolution):**
    -   **A (Pz-Oz):** `delta_relpow`, `sigma_relpow`, `beta_relpow`
    -   **B (Fpz-Cz):** `delta_relpow`, `sigma_relpow`, `beta_relpow`

-   **EOG (5–6 features):**
    -   `rms`, `theta_relpow`, `spec_entropy`, `sef95`, `delta_relpow` (+ `beta_relpow` as optional).

-   **EMG (2 features):**
    -   `EMG_submental_p90_1hz`, `EMG_submental_std_1hz` (or `mean_1hz`).

-   **Resp (2 features):**
    -   `Resp_oronasal_std_1hz`, `Resp_oronasal_iqr_1hz`.

**Result:** A feature set with multimodal coverage, low redundancy, and high interpretability.

---

## 4. Drop Rules (Checklist for Code Implementation)

1.  Remove constants and identifiers.
2.  Remove `tso_min` from the training set (use only for auditing/control).
3.  For each channel+band/statistic:
    -   If `*_pow_*` and `*_logpow_*` exist → replace with `*_relpow_*` and drop both.
    -   If `_256` and `_512` exist → keep one; standardize.
4.  If both `rms` and `var` exist → keep one (prefer `rms`).
5.  For highly collinear percentiles (`min`/`p01`, `max`/`p99`, `median`≈`mean`) → keep one that serves a specific function (e.g., `p90` for tail behavior).
6.  Keep a maximum of 1–2 ratios that provide a clear gain; cut the rest.
7.  QC flags (`*_clip_frac_*`, `*_oor_frac_*`) → keep out of the prediction vector; use them for `mask`/`weights`.

---

## 5. Sufficiency & Next Steps (Short Experiments)

-   **Sufficient to decide?** Yes, for this recording. The patterns in $\eta^2$ and the correlation matrices clearly establish the value hierarchy and redundancies.
-   **Essential Validation:** Repeat this analysis across multiple subjects/nights using a grouped cross-validation strategy (by subject).
-   **Ablation Study:**
    1.  Train on the minimalist core set.
    2.  Train on the core set + 1 ratio.
    3.  Train on the core set + global features (`entropy`/`sef`/`slope`).
    -   Compare macro-F1 and per-class recall.
-   **Class Imbalance:** Use `class_weights` and report per-class metrics, especially for minority classes (N1/REM).
-   **Model Interpretation:** Use permutation importance or SHAP on the lean feature set to confirm the independent contribution of each feature.

---

## 6. Executive Summary (In Words)

**Keep:** `relpow` of 3 key bands from the same EEG channel (δ, σ, β), EOG features (`rms`, `theta_rel`, `entropy`/`SEF`, `delta_rel`), EMG `p90`, and Respiration `std`/`iqr`.

**Exclude:** `pow`/`logpow` duplicates, resolution duplicates, `var`, redundant percentiles, excessive ratios, continuous temperature, `tso_min`, `Event_marker_*`, and QC flags as predictors.

This process yields a short, interpretable, and robust feature vector ready for your online classifier.