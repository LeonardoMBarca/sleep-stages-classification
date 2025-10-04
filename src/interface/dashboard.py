
from __future__ import annotations
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
from fastapi import Query
import json
from pathlib import Path
from typing import Dict, List, cast

import threading

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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

STATE_LOCK = threading.Lock()
STATE: Dict[str, object] = {
    "ready": False,
    "error": None,
    "models": None,
    "stages": None,
    "frames": None,
    "model_order": None,
}


def _bootstrap_models() -> None:
    try:
        models, stages, frames, model_order = load_models_and_predictions()
        with STATE_LOCK:
            STATE["models"] = models
            STATE["stages"] = stages
            STATE["frames"] = frames
            STATE["model_order"] = model_order
            STATE["error"] = None
            STATE["ready"] = True
    except Exception as exc:  # pragma: no cover - defensive
        with STATE_LOCK:
            STATE["error"] = str(exc)
            STATE["ready"] = False



app = FastAPI(title="Sleep Stage Classification Dashboard")

# Serve arquivos estáticos (html, css, js)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def schedule_bootstrap() -> None:
    threading.Thread(target=_bootstrap_models, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)


@app.get("/api/models")
async def list_models():
    with STATE_LOCK:
        ready = STATE["ready"]
        error = STATE["error"]
        models = STATE["models"]
        stages = STATE["stages"]

    if not ready:
        status = 500 if error else 202
        payload = {"status": "error", "detail": error} if error else {"status": "loading"}
        return JSONResponse(payload, status_code=status)

    payload = []
    models_dict = cast(Dict[str, ModelEntry], models)
    stages_list = cast(List[str], stages)

    for key, entry in models_dict.items():
        stage_metrics = {
            stage: {
                "precision": entry.classification_report.get(stage, {}).get("precision", 0.0),
                "recall": entry.classification_report.get(stage, {}).get("recall", 0.0),
                "f1": entry.classification_report.get(stage, {}).get("f1-score", 0.0),
                "support": int(entry.classification_report.get(stage, {}).get("support", 0.0)),
            }
            for stage in stages_list
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
    return {"status": "ready", "stages": stages_list, "models": payload}


@app.get("/api/simulation")
async def simulation():
    with STATE_LOCK:
        ready = STATE["ready"]
        error = STATE["error"]
        frames = STATE["frames"]
        model_order = STATE["model_order"]

    if not ready:
        status = 500 if error else 202
        payload = {"status": "error", "detail": error} if error else {"status": "loading"}
        return JSONResponse(payload, status_code=status)

    return {
        "status": "ready",
        "models": cast(List[Dict[str, str]], model_order),
        "frames": cast(List[Dict[str, object]], frames),
    }

@app.get("/api/probabilities")
async def get_probabilities(model: str = Query(...)):
    with STATE_LOCK:
        ready = STATE["ready"]
        error = STATE["error"]
        models = STATE["models"]
        stages = STATE["stages"]
    if not ready:
        status = 500 if error else 202
        payload = {"status": "error", "detail": error} if error else {"status": "loading"}
        return JSONResponse(payload, status_code=status)
    models_dict = cast(Dict[str, ModelEntry], models)
    stages_list = cast(List[str], stages)
    entry = models_dict.get(model)
    if not entry or not hasattr(entry.model, "predict_proba"):
        return JSONResponse({"status": "error", "detail": "Modelo não encontrado ou não suporta probabilidades."}, status_code=400)
    # Recalcular X_test para garantir alinhamento
    scaler_path = FINAL_MODELS_DIR / "scaler.pkl"
    features_path = FINAL_MODELS_DIR / "feature_order.json"
    stages_path = FINAL_MODELS_DIR / "stage_mapping.json"
    scaler = joblib.load(scaler_path)
    feature_order = load_json(features_path)
    stage_labels = load_json(stages_path)
    df_test = load_dataset()
    y_test = df_test["stage"].map({stage: idx for idx, stage in enumerate(stage_labels)}).to_numpy(dtype=np.int64)
    X_test = scaler.transform(df_test[feature_order]).astype(np.float32)
    if entry.name == "mlp":
        with torch.no_grad():
            logits = entry.model(torch.from_numpy(X_test))
            proba = torch.softmax(logits, dim=1).numpy()
    elif entry.name == "lightgbm":
        frame = pd.DataFrame(X_test, columns=feature_order)
        proba = entry.model.predict_proba(frame)
    else:
        proba = entry.model.predict_proba(X_test)
    return {
        "status": "ready",
        "y_true": y_test.tolist(),
        "proba": proba.tolist(),
        "stages": stages_list
    }
