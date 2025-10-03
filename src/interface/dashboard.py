"""Interactive dashboard API for comparing trained sleep-stage models.

The application exposes a FastAPI server that

1. loads the persisted estimators from ``final-models``;
2. computes predictions on the held-out test split; and
3. provides a lightweight web interface for simulating epoch-by-epoch
   predictions and reviewing aggregate metrics.

Run locally with::

    uvicorn src.interface.dashboard:app --reload

The HTML served at ``/`` consumes two endpoints:

``GET /api/models``
    Returns metadata and evaluation metrics for each available model.

``GET /api/simulation``
    Streams the combined epoch-by-epoch predictions for a multi-model
    animation covering the first slices of two test subjects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from xgboost import XGBClassifier


BASE_PATH = Path(__file__).resolve().parents[2]
FINAL_MODELS_DIR = BASE_PATH / "final-models"
DATASET_ROOT = BASE_PATH / "datalake" / "data-for-model"
SIMULATION_SUBJECTS = 2
SIMULATION_EPOCHS_PER_SUBJECT = 200


class ResidualBlock(torch.nn.Module):
    def __init__(self, dim: int, expansion: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = torch.nn.LayerNorm(dim)
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden, dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = inputs
        outputs = self.norm(inputs)
        outputs = self.fc1(outputs)
        outputs = self.act(outputs)
        outputs = self.drop(outputs)
        outputs = self.fc2(outputs)
        outputs = self.drop(outputs)
        return outputs + residual


class SleepMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, expansion: float, dropout: float, num_classes: int) -> None:
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.blocks = torch.nn.ModuleList([
            ResidualBlock(hidden_dim, expansion, dropout) for _ in range(depth)
        ])
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        outputs = self.stem(inputs)
        for block in self.blocks:
            outputs = block(outputs)
        return self.head(outputs)


def load_json(path: Path) -> List[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset() -> pd.DataFrame:
    test_path = DATASET_ROOT / "test" / "test_sleep_cassette.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test parquet not found at {test_path}")
    df_test = pd.read_parquet(test_path, engine="fastparquet")
    df_test = df_test.copy()
    df_test["sex"] = df_test["sex"].map({"F": 0.0, "M": 1.0}).astype(np.float32)
    df_test = df_test.reset_index(drop=True)
    return df_test


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    return {
        "loss": float(log_loss(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
    }


class ModelEntry:
    def __init__(self, name: str, display_name: str, loader: callable) -> None:
        self.name = name
        self.display_name = display_name
        self.loader = loader
        self.model = None
        self.metrics: Dict[str, float] = {}
        self.classification_report: Dict[str, Dict[str, float]] = {}
        self.confusion: List[List[int]] = []
        self.full_predictions: np.ndarray | None = None


def build_model_registry():
    return {
        "lightgbm": ModelEntry("lightgbm", "LightGBM", lambda path: joblib.load(path)),
        "logistic_regression": ModelEntry("logistic_regression", "Logistic Regression", lambda path: joblib.load(path)),
        "naive_bayes": ModelEntry("naive_bayes", "Naive Bayes", lambda path: joblib.load(path)),
        "random_forest": ModelEntry("random_forest", "Random Forest", lambda path: joblib.load(path)),
        "xgboost": ModelEntry("xgboost", "XGBoost", load_xgboost),
        "mlp": ModelEntry("mlp", "Residual MLP", load_mlp),
    }


def load_xgboost(path: Path):
    model = XGBClassifier()
    model.load_model(path)
    model.set_params(objective="multi:softprob")
    return model


def load_mlp(path: Path):
    config_path = FINAL_MODELS_DIR / "mlp-config.json"
    if not config_path.exists():
        raise FileNotFoundError("Missing mlp-config.json required to rebuild the network")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = SleepMLP(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        depth=config["depth"],
        expansion=config["expansion"],
        dropout=config["dropout"],
        num_classes=config["num_classes"],
    )
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_models_and_predictions():
    if not FINAL_MODELS_DIR.exists():
        raise FileNotFoundError("final-models directory not found. Run save_final_models.py first.")

    scaler_path = FINAL_MODELS_DIR / "scaler.pkl"
    features_path = FINAL_MODELS_DIR / "feature_order.json"
    stages_path = FINAL_MODELS_DIR / "stage_mapping.json"
    for dependency in (scaler_path, features_path, stages_path):
        if not dependency.exists():
            raise FileNotFoundError(f"Required artefact missing: {dependency}")

    scaler = joblib.load(scaler_path)
    feature_order = load_json(features_path)
    stage_labels = load_json(stages_path)
    stage_to_id = {stage: idx for idx, stage in enumerate(stage_labels)}

    df_test = load_dataset()
    y_test = df_test["stage"].map(stage_to_id).to_numpy(dtype=np.int64)
    X_test = scaler.transform(df_test[feature_order]).astype(np.float32)

    registry = build_model_registry()

    available_models = {}
    for key, entry in registry.items():
        artefact_path = MODEL_FILE_MAP.get(key)
        if artefact_path is None or not artefact_path.exists():
            continue
        try:
            entry.model = entry.loader(artefact_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Failed to load {key}: {exc}")
            continue

        proba, preds = inference(entry, X_test, feature_order)
        entry.metrics = evaluate_predictions(y_test, proba, preds)
        report = classification_report(y_test, preds, output_dict=True, target_names=stage_labels)
        entry.classification_report = report
        cm = confusion_matrix(y_test, preds)
        entry.confusion = cm.astype(int).tolist()
        entry.full_predictions = preds
        available_models[key] = entry

    if not available_models:
        raise RuntimeError("No compatible models could be loaded from final-models")

    simulation_indices = select_simulation_positions(df_test)
    simulation_frames = build_simulation_frames(
        df_test,
        y_test,
        simulation_indices,
        available_models,
        stage_labels,
    )

    model_order = [
        {"id": key, "name": entry.display_name}
        for key, entry in available_models.items()
    ]

    return available_models, stage_labels, simulation_frames, model_order


def inference(entry: ModelEntry, X_test: np.ndarray, feature_order: List[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if entry.name == "mlp":
        with torch.no_grad():
            logits = entry.model(torch.from_numpy(X_test))  # type: ignore[arg-type]
            proba = torch.softmax(logits, dim=1).numpy()
    elif entry.name == "lightgbm" and feature_order is not None:
        frame = pd.DataFrame(X_test, columns=feature_order)
        proba = entry.model.predict_proba(frame)  # type: ignore[union-attr]
    else:
        proba = entry.model.predict_proba(X_test)  # type: ignore[union-attr]
    preds = proba.argmax(axis=1)
    return proba, preds


def select_simulation_positions(df_test: pd.DataFrame) -> List[int]:
    df_sorted = df_test.assign(_pos=np.arange(len(df_test))).sort_values(["subject_id", "night_id", "epoch_idx"])
    selected: List[int] = []
    subjects_seen: List[str] = []
    for subject_id, group in df_sorted.groupby("subject_id", sort=False):
        rows = group.head(SIMULATION_EPOCHS_PER_SUBJECT)["_pos"].tolist()
        selected.extend(rows)
        subjects_seen.append(subject_id)
        if len(subjects_seen) >= SIMULATION_SUBJECTS:
            break
    return selected


def build_simulation_frames(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    positions: List[int],
    models: Dict[str, ModelEntry],
    stage_labels: List[str],
) -> List[Dict[str, object]]:
    frames: List[Dict[str, object]] = []
    for step, pos in enumerate(positions):
        row = df_test.loc[pos]
        actual_label = stage_labels[int(y_test[pos])]
        predictions: Dict[str, Dict[str, object]] = {}
        overall_correct = True
        for key, entry in models.items():
            if entry.full_predictions is None:
                raise RuntimeError(f"Missing predictions for model {entry.display_name}")
            predicted_label = stage_labels[int(entry.full_predictions[pos])]
            is_correct = predicted_label == actual_label
            if not is_correct:
                overall_correct = False
            predictions[key] = {
                "predicted": predicted_label,
                "correct": is_correct,
            }
        frames.append(
            {
                "step": step,
                "subject_id": str(row["subject_id"]),
                "night_id": str(row["night_id"]),
                "epoch_idx": int(row["epoch_idx"]),
                "actual": actual_label,
                "overall_correct": overall_correct,
                "predictions": predictions,
            }
        )
    return frames


MODEL_FILE_MAP = {
    "lightgbm": FINAL_MODELS_DIR / "lightgbm-model.pkl",
    "logistic_regression": FINAL_MODELS_DIR / "logistic-regression-model.pkl",
    "naive_bayes": FINAL_MODELS_DIR / "naive-bayes-model.pkl",
    "random_forest": FINAL_MODELS_DIR / "random-forest-model.pkl",
    "xgboost": FINAL_MODELS_DIR / "xgboost-model.json",
    "mlp": FINAL_MODELS_DIR / "mlp-model.pt",
}


available_models, stage_labels, SIMULATION_FRAMES, MODEL_ORDER = load_models_and_predictions()

app = FastAPI(title="Sleep Stage Classification Dashboard")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(CLIENT_HTML)


@app.get("/api/models")
async def list_models():
    payload = []
    for key, entry in available_models.items():
        stage_metrics = {
            stage: {
                "precision": entry.classification_report.get(stage, {}).get("precision", 0.0),
                "recall": entry.classification_report.get(stage, {}).get("recall", 0.0),
                "f1": entry.classification_report.get(stage, {}).get("f1-score", 0.0),
                "support": int(entry.classification_report.get(stage, {}).get("support", 0.0)),
            }
            for stage in stage_labels
        }
        payload.append(
            {
                "id": key,
                "name": entry.display_name,
                "metrics": entry.metrics,
                "stage_metrics": stage_metrics,
                "classification_report": entry.classification_report,
                "confusion_matrix": entry.confusion,
            }
        )
    return {"stages": stage_labels, "models": payload}


@app.get("/api/simulation")
async def simulation():
    if not SIMULATION_FRAMES:
        raise HTTPException(status_code=500, detail="Simulation data unavailable")
    return {
        "models": MODEL_ORDER,
        "frames": SIMULATION_FRAMES,
    }


CLIENT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Sleep Stage Classification Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; background: #f7f9fc; color: #222; }
    h1 { margin-bottom: 0.5rem; }
    table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
    th, td { border: 1px solid #d4dce6; padding: 0.5rem; text-align: center; }
    th { background: #eef2f9; }
    button { padding: 0.4rem 1rem; margin: 0.3rem; border: none; border-radius: 4px; background: #1f6feb; color: #fff; cursor: pointer; }
    button:disabled { background: #ccc; cursor: default; }
    .simulation { margin-top: 2rem; }
    .log { background: #111; color: #0f0; padding: 1rem; min-height: 180px; overflow-y: auto; font-family: "Courier New", monospace; }
    .correct { color: #4caf50; }
    .incorrect { color: #f44336; }
  </style>
</head>
<body>
  <h1>Sleep Stage Classification Dashboard</h1>
  <p>This dashboard replays the predictions produced by the final trained models on the held-out test set.</p>

  <section id="metrics"></section>
  <section id="stage-metrics"></section>

  <section class="simulation">
    <h2>Epoch Simulation</h2>
    <p>Replay the first epochs from two different subjects and compare all models frame by frame.</p>
    <div id="controls"></div>
    <div class="log" id="log"></div>
  </section>

  <script>
    const metricsContainer = document.getElementById('metrics');
    const stageMetricsContainer = document.getElementById('stage-metrics');
    const controlsContainer = document.getElementById('controls');
    const logEl = document.getElementById('log');
    let modelCatalog = [];
    let simulationHandle = null;

    function renderMetrics(models) {
      let html = '<table><thead><tr><th>Model</th><th>Accuracy</th><th>Balanced Acc.</th><th>Macro F1</th><th>Log Loss</th></tr></thead><tbody>';
      models.forEach(model => {
        html += `<tr><td>${model.name}</td><td>${model.metrics.accuracy.toFixed(3)}</td>`;
        html += `<td>${model.metrics.balanced_accuracy.toFixed(3)}</td>`;
        html += `<td>${model.metrics.macro_f1.toFixed(3)}</td>`;
        html += `<td>${model.metrics.loss.toFixed(3)}</td></tr>`;
      });
      html += '</tbody></table>';
      metricsContainer.innerHTML = html;
    }

    function renderPerStageMetrics(models, stages) {
      let html = '<table><thead><tr><th>Model</th><th>Stage</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>';
      models.forEach(model => {
        stages.forEach(stage => {
          const stats = model.stage_metrics[stage];
          html += `<tr><td>${model.name}</td><td>${stage}</td>`;
          html += `<td>${stats.precision.toFixed(3)}</td>`;
          html += `<td>${stats.recall.toFixed(3)}</td>`;
          html += `<td>${stats.f1.toFixed(3)}</td>`;
          html += `<td>${stats.support}</td></tr>`;
        });
      });
      html += '</tbody></table>';
      stageMetricsContainer.innerHTML = '<h2>Stage-level Metrics</h2>' + html;
    }

    function setupControls(models) {
      modelCatalog = models;
      controlsContainer.innerHTML = '';
      const button = document.createElement('button');
      button.textContent = 'Play combined simulation';
      button.onclick = () => startSimulation();
      controlsContainer.appendChild(button);
    }

    async function fetchModels() {
      const response = await fetch('/api/models');
      const payload = await response.json();
      renderMetrics(payload.models);
      renderPerStageMetrics(payload.models, payload.stages);
      setupControls(payload.models);
    }

    function appendLog(message, cls = '') {
      const line = document.createElement('div');
      if (cls) line.classList.add(cls);
      line.textContent = message;
      logEl.appendChild(line);
      logEl.scrollTop = logEl.scrollHeight;
    }

    async function startSimulation() {
      if (simulationHandle) {
        clearInterval(simulationHandle);
        simulationHandle = null;
      }
      logEl.innerHTML = '';
      appendLog('Starting combined simulation...');
      const response = await fetch('/api/simulation');
      if (!response.ok) {
        appendLog('Unable to load simulation data.', 'incorrect');
        return;
      }
      const payload = await response.json();
      const frames = payload.frames;
      const models = payload.models;
      let index = 0;
      simulationHandle = setInterval(() => {
        if (index >= frames.length) {
          appendLog('Simulation finished.');
          clearInterval(simulationHandle);
          simulationHandle = null;
          return;
        }
        const frame = frames[index];
        const parts = models.map(model => {
          const info = frame.predictions[model.id];
          const mark = info.correct ? '✓' : '✗';
          return `${model.name}: ${info.predicted} ${mark}`;
        });
        const allCorrect = models.every(model => frame.predictions[model.id].correct);
        const status = allCorrect ? 'correct' : 'incorrect';
        appendLog(`Subject ${frame.subject_id} (night ${frame.night_id}) | Epoch ${frame.epoch_idx} | Actual ${frame.actual} | ${parts.join(' | ')}`, status);
        index += 1;
      }, 200);
    }

    fetchModels();
  </script>
</body>
</html>
"""
