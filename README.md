# Sleep Stages Classification

End-to-end pipeline for classifying sleep stages (W, N1, N2, N3, REM) on the Sleep-EDFx dataset. The repository covers everything from automated data acquisition, feature engineering, and exploratory analysis to model training, model registry creation, and a FastAPI dashboard that compares all final estimators side by side.

---

## Table of Contents

1. [Project Highlights](#project-highlights)
2. [Repository Layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Pipeline](#step-by-step-pipeline)
    - [1. Download Sleep-EDFx](#1-download-sleep-edfx)
    - [2. Generate Modeling Features](#2-generate-modeling-features)
    - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
    - [4. Train & Persist Models](#4-train--persist-models)
    - [5. Run the Comparison Dashboard](#5-run-the-comparison-dashboard)
5. [Models & Metrics](#models--metrics)
6. [Key Design Choices](#key-design-choices)
7. [Troubleshooting](#troubleshooting)
8. [License](#license)

---

## Project Highlights

- **Automated ingestion**: FastAPI service orchestrates robust Sleep-EDFx downloads with resume, retries, and hashing to avoid re-fetching processed files.
- **Leakage-free feature engineering**: Subject-level stratified splits plus rolling-window statistics create 63 expressive features per epoch without seeing future data.
- **Multiple models**: Classical ML (LogReg, Naive Bayes, Random Forest, LightGBM, XGBoost) and a custom residual MLP with focal loss.
- **Unified export**: One command regenerates the scaler, feature order, class mapping, and every fitted model into `final-models/`.
- **Interactive dashboard**: FastAPI + vanilla JS UI renders overall metrics, per-stage metrics, and a multi-model playback of predictions across two test subjects.

---

## Repository Layout

```
├── datalake/                  # Raw + processed datasets (created locally)
├── final-models/              # Saved estimators, scaler, metrics (generated)
├── src/
│   ├── download-data/         # FastAPI service to fetch Sleep-EDFx
│   ├── data-processing/       # Feature engineering & subject splits
│   ├── data-analysis/         # EDA notebooks (cassette & telemetry)
│   ├── model-training/        # Training notebooks and exporter script
│   ├── interface/             # Dashboard backend & client
│   └── logger/, ml-models-pipeline/  # Shared utilities
├── requirements.txt           # Dashboard/runtime dependencies
└── README.md                  # You are here
```

---

## Prerequisites

- Python 3.10+
- Recommended: virtual environment (venv/conda)
- OS packages: `build-essential`, `libomp` (for LightGBM/XGBoost), optional GPU drivers for PyTorch acceleration

Install base dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Individual modules (download, processing, notebooks) have extra requirement files inside their folders if you need to replicate those environments exactly.

---

## Step-by-Step Pipeline

### 1. Download Sleep-EDFx

**Service location:** `src/download-data`

1. Install module requirements (optional separate env):
   ```bash
   pip install -r src/download-data/requirements.txt
   ```
2. Launch the API:
   ```bash
   uvicorn src.download-data.main:app --reload
   ```
3. Trigger downloads via HTTP:
   ```bash
   curl "http://localhost:8000/download?subset=cassette&sync=true"
   curl "http://localhost:8000/download?subset=telemetry&sync=true"
   ```
   - The planner scrapes the remote listing, compares with `datalake/raw/`, skips files already hashed during processing, and downloads only missing ones.
   - Retries/transient errors are handled automatically; logs show progress and throughput.

**Result:** EDF + annotation files stored under `datalake/raw/sleep-cassette|sleep-telemetry`.

### 2. Generate Modeling Features

**Script:** `src/data-processing/processed-for-data-for-model/processor.py`

1. Ensure Sleep-EDFx raw/processed layers are available (either from the downloader or your own preparation).
2. Run the processor:
   ```bash
   python src/data-processing/processed-for-data-for-model/processor.py
   ```
3. What it does:
   - Loads the pre-processed signals using Polars for both cassette and telemetry cohorts.
   - Keeps curated bands (delta/theta/alpha/beta, ratios, spectral descriptors, RMS, wrapped demographics).
   - Adds rolling means/std/max over 5, 10, 15 epoch windows so models capture local temporal context without peeking ahead.
   - Splits subjects into train/val/test via `stratified_subject_split_by_quotas`, balancing age groups and sex, guaranteeing no subject leakage across splits.
   - Writes out `train_sleep_cassette.parquet`, `val_sleep_cassette.parquet`, `test_sleep_cassette.parquet` into `datalake/data-for-model` (plus consolidated `sleep-cassette.parquet`, `sleep-telemetry.parquet`).

**Result:** 63-feature per-epoch tables with aligned splits ready for modeling.

### 3. Exploratory Data Analysis

**Location:** `src/data-analysis/sleep-cassette/eda.ipynb` and `sleep-telemetry/eda.ipynb`

Open the notebooks to inspect:

- Class distributions, demographic summaries, and sensor value ranges.
- Rolling feature sanity checks.
- Potential correlations or derivatives to expand in future iterations.

Execution is optional for the main pipeline but recommended before training to understand quirks in Sleep-EDFx.

### 4. Train & Persist Models

You can work in the individual notebooks (`src/model-training/<model>/<model>-training.ipynb`) or recreate everything with the consolidated script:

```bash
# Regenerate scaler, feature order, stage mapping, and all trained estimators
python -m src.model_training.save_final_models --force
```

This command:

- Loads `train/val/test` parquet splits.
- Fits a global `StandardScaler` on the training partition and applies it to val/test.
- Computes class weights (inverse frequency) to address sleep-stage imbalance.
- Trains and saves:
  - Multinomial Logistic Regression (`final-models/logistic-regression-model.pkl`)
  - Gaussian Naive Bayes (`final-models/naive-bayes-model.pkl`)
  - Random Forest with balanced subsampling (`final-models/random-forest-model.pkl`)
  - LightGBM multiclass booster (`final-models/lightgbm-model.pkl`)
  - XGBoost histogram booster (`final-models/xgboost-model.json`)
  - Residual MLP (`final-models/mlp-model.pt` + `mlp-config.json`)
- Records evaluation metrics to `final-models/metrics.json`.

Each notebook mirrors that logic if you prefer an interactive workflow, including confusion matrices and per-stage classification reports.

### 5. Run the Comparison Dashboard

**Service:** `src/interface/dashboard.py`

1. Ensure the `final-models/` directory exists (run the saver script above).
2. Start the API/UI:
   ```bash
   uvicorn src.interface.dashboard:app --reload
   ```
3. Open http://127.0.0.1:8000 to explore:
   - **Metrics table**: accuracy, balanced accuracy, macro F1, log-loss by model.
   - **Stage-level metrics**: precision/recall/F1/support for each sleep stage.
   - **Combined simulation**: 400 epochs (200 per subject across two sleepers) animated with predictions from all models simultaneously, so you can spot agreements and disagreements in real time.

---

## Models & Metrics

Summary from the latest `final-models/metrics.json` (test split):

| Model                 | Accuracy | Balanced Acc. | Macro F1 | Notes |
|-----------------------|---------:|---------------:|---------:|-------|
| Logistic Regression    | 0.723    | 0.717         | 0.667    | Strong linear baseline with multinomial saga solver |
| Naive Bayes            | 0.649    | 0.639         | 0.581    | Fast, probability-calibrated baseline |
| Random Forest          | 0.778    | 0.692         | 0.700    | Balanced subsampling across trees |
| LightGBM               | 0.767    | 0.728         | 0.708    | Early-stopped gradient boosting |
| XGBoost                | 0.794    | 0.706         | 0.716    | Histogram tree booster, best overall accuracy |
| Residual MLP           | 0.703    | 0.715         | 0.656    | Residual blocks + focal loss, solid N3/REM recall |

Per-stage precision/recall/F1 metrics are available in the dashboard and via `/api/models` for deeper dives (e.g., REM F1 around 0.69 for XGBoost, N3 recall ~0.82 for LightGBM).

---

## Key Design Choices

- **Subject-level splits** prevent temporal/data leakage—no epoch from the same subject appears in multiple partitions.
- **Rolling statistics** capture short-term context without using future epochs, keeping features causal for inference.
- **Residual MLP** uses LayerNorm blocks with skip connections to ease optimization on tabular data, while focal loss and class weighting emphasize minority stages (N3, REM).
- **Unified artefact export** simplifies deployment; any downstream consumer can load `scaler.pkl`, `feature_order.json`, and one of the model files to reproduce predictions.
- **FastAPI dashboard** centralizes monitoring and comparison, making it easy to vet new experiments against the existing leaderboard.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| **Out-of-memory during LightGBM/XGBoost training** | Reduce `n_estimators` or subsampling ratios in `save_final_models.py`; run on a machine with more RAM. |
| **Download stalls/fails** | The downloader already retries transient errors. For stubborn files, rerun with `ignore_hash=true` and `round_retries` increased. |
| **Inconsistent scikit-learn pickle warning** | Ensure your runtime scikit-learn version matches the one used to generate the artefacts (re-run `save_final_models.py` in the current environment). |
| **Dashboard shows “Simulation data unavailable”** | Regenerate models (`python -m src.model_training.save_final_models --force`) and restart the API so `SIMULATION_FRAMES` is rebuilt. |
| **Slow MLP training** | CUDA is automatically used if available. Otherwise, reduce `epochs` or `hidden_dim` in the script/notebook. |

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

---

Happy experimenting! Contributions—new feature engineering ideas, additional models, or alternative evaluation dashboards—are very welcome.
