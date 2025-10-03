"""Train and persist the final models used by the dashboard interface.

This script recreates the modelling pipeline defined in the training notebooks,
fits each estimator using the train/validation splits available under
``datalake/data-for-model`` and serialises the artefacts into ``final-models``.

Saved artefacts:
    - scaler.pkl: StandardScaler fitted on the training split.
    - feature_order.json: ordered list of feature columns expected by all models.
    - *.pkl / *.json / *.pt: estimator weights for each algorithm.
    - metrics.json: summary of evaluation metrics on the test split.

The script can be executed from the project root:

    python -m src.model_training.save_final_models

It skips training for artefacts that already exist, unless ``--force`` is
provided on the command line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


BASE_PATH = Path(__file__).resolve().parents[2]
DATASET_ROOT = BASE_PATH / "datalake" / "data-for-model"
FINAL_MODELS_DIR = BASE_PATH / "final-models"

TRAIN_PATH = DATASET_ROOT / "train" / "train_sleep_cassette.parquet"
VAL_PATH = DATASET_ROOT / "val" / "val_sleep_cassette.parquet"
TEST_PATH = DATASET_ROOT / "test" / "test_sleep_cassette.parquet"

STAGES = ["N1", "N2", "N3", "REM", "W"]


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training parquet not found at {TRAIN_PATH}")
    df_train = pd.read_parquet(TRAIN_PATH, engine="fastparquet")
    df_val = pd.read_parquet(VAL_PATH, engine="fastparquet")
    df_test = pd.read_parquet(TEST_PATH, engine="fastparquet")
    return df_train, df_val, df_test


def prepare_frames(
    df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    frames = [df_train.copy(), df_val.copy(), df_test.copy()]
    for frame in frames:
        frame["sex"] = frame["sex"].map({"F": 0.0, "M": 1.0}).astype(np.float32)

    identifiers = ["subject_id", "night_id", "epoch_idx", "stage"]
    features = sorted([c for c in df_train.columns if c not in identifiers])
    return frames[0], frames[1], frames[2], features


def compute_class_weights(train_frame: pd.DataFrame) -> Dict[int, float]:
    stages = train_frame["stage"].astype(str)
    unique_stages = sorted(stages.unique())
    stage2id = {stage: idx for idx, stage in enumerate(unique_stages)}
    counts = stages.map(stage2id).value_counts().sort_index()
    weights = (len(train_frame) / (len(unique_stages) * counts)).astype(np.float64)
    return {int(idx): float(value) for idx, value in weights.items()}


def scale_frames(
    scaler: StandardScaler,
    frames: List[pd.DataFrame],
    features: List[str],
) -> List[np.ndarray]:
    arrays = []
    for frame in frames:
        arrays.append(scaler.transform(frame[features]).astype(np.float32))
    return arrays


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def train_logistic_regression(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        penalty="l2",
        C=0.9,
        max_iter=2000,
        tol=1e-4,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def train_naive_bayes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray,
) -> GaussianNB:
    grid = np.logspace(-9, -3, 13)
    best_score = -np.inf
    best_model = None
    for smoothing in grid:
        model = GaussianNB(var_smoothing=smoothing)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average="macro")
        if score > best_score:
            best_score = score
            best_model = model
    return best_model


def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=26,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray,
    val_weight: np.ndarray,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y_train)),
        n_estimators=2200,
        learning_rate=0.045,
        num_leaves=104,
        max_depth=-1,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=0.9,
        reg_alpha=0.02,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_sample_weight=[sample_weight, val_weight],
        eval_metric=["multi_logloss", "multi_error"],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
    )
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray,
    val_weight: np.ndarray,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        learning_rate=0.045,
        max_depth=8,
        n_estimators=1300,
        subsample=0.85,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        reg_lambda=1.2,
        reg_alpha=0.05,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[val_weight],
        verbose=False,
    )
    return model


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 1.2, weight=None):
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("class_weight", weight)
        else:
            self.class_weight = None

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(logits, targets, weight=self.class_weight, reduction="none")
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        pt = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


class SleepMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, expansion: float, dropout: float, num_classes: int):
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        for block in self.blocks:
            outputs = block(outputs)
        return self.head(outputs)


class ResidualBlock(torch.nn.Module):
    def __init__(self, dim: int, expansion: float, dropout: float):
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = torch.nn.LayerNorm(dim)
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.act = torch.nn.GELU()
        self.drop = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(hidden, dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.norm(inputs)
        outputs = self.fc1(outputs)
        outputs = self.act(outputs)
        outputs = self.drop(outputs)
        outputs = self.fc2(outputs)
        outputs = self.drop(outputs)
        return outputs + residual


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: Dict[int, float],
) -> Tuple[SleepMLP, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.int64)))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.int64)))

    sample_weights = np.array([class_weights[int(label)] for label in y_train], dtype=np.float64)
    sampler = torch.utils.data.WeightedRandomSampler(torch.from_numpy(sample_weights).double(), num_samples=len(sample_weights), replacement=True)

    batch_size = 512
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SleepMLP(input_dim=X_train.shape[1], hidden_dim=384, depth=4, expansion=1.4, dropout=0.2, num_classes=len(class_weights)).to(device)
    weight_tensor = torch.tensor([class_weights[idx] for idx in range(len(class_weights))], dtype=torch.float32)
    weight_tensor = weight_tensor / weight_tensor.mean()
    criterion = FocalLoss(gamma=1.15, weight=weight_tensor.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.5e-3, epochs=20, steps_per_epoch=len(train_loader), pct_start=0.35, div_factor=10.0, final_div_factor=30.0)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_metric = -np.inf
    best_state = None
    patience = 4
    wait = 0

    for epoch in range(1, 21):
        model.train()
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(features)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        model.eval()
        preds = []
        refs = []
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features.to(device))
                preds.append(outputs.argmax(dim=1).cpu().numpy())
                refs.append(targets.numpy())
        preds = np.concatenate(preds)
        refs = np.concatenate(refs)
        metric = f1_score(refs, preds, average="macro")
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(torch.device("cpu"))
    return model, {"best_macro_f1": float(best_metric)}


def evaluate_predictions(y_true: np.ndarray, proba: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    return {
        "loss": float(log_loss(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
    }


def main(force: bool = False) -> None:
    ensure_directory(FINAL_MODELS_DIR)

    df_train, df_val, df_test = load_datasets()
    df_train, df_val, df_test, features = prepare_frames(df_train, df_val, df_test)

    stage_labels = sorted(df_train["stage"].unique())
    stage_to_id = {stage: idx for idx, stage in enumerate(stage_labels)}

    y_train = df_train["stage"].map(stage_to_id).to_numpy(dtype=np.int64)
    y_val = df_val["stage"].map(stage_to_id).to_numpy(dtype=np.int64)
    y_test = df_test["stage"].map(stage_to_id).to_numpy(dtype=np.int64)

    class_weights = compute_class_weights(df_train)
    train_weights = np.array([class_weights[int(label)] for label in y_train], dtype=np.float64)
    val_weights = np.array([class_weights[int(label)] for label in y_val], dtype=np.float64)

    scaler = StandardScaler()
    scaler.fit(df_train[features])

    X_train, X_val, X_test = scale_frames(scaler, [df_train, df_val, df_test], features)

    joblib.dump(scaler, FINAL_MODELS_DIR / "scaler.pkl", compress=3)
    save_json(FINAL_MODELS_DIR / "feature_order.json", features)
    save_json(FINAL_MODELS_DIR / "stage_mapping.json", stage_labels)

    metrics: Dict[str, Dict[str, float]] = {}

    # Logistic Regression
    logistic_path = FINAL_MODELS_DIR / "logistic-regression-model.pkl"
    if force or not logistic_path.exists():
        lr_model = train_logistic_regression(X_train, y_train, train_weights)
        joblib.dump(lr_model, logistic_path, compress=3)
    else:
        lr_model = joblib.load(logistic_path)
    lr_proba = lr_model.predict_proba(X_test)
    lr_preds = lr_proba.argmax(axis=1)
    metrics["logistic_regression"] = evaluate_predictions(y_test, lr_proba, lr_preds)

    # Naive Bayes
    nb_path = FINAL_MODELS_DIR / "naive-bayes-model.pkl"
    if force or not nb_path.exists():
        nb_model = train_naive_bayes(X_train, y_train, X_val, y_val, train_weights)
        joblib.dump(nb_model, nb_path, compress=3)
    else:
        nb_model = joblib.load(nb_path)
    nb_proba = nb_model.predict_proba(X_test)
    nb_preds = nb_proba.argmax(axis=1)
    metrics["naive_bayes"] = evaluate_predictions(y_test, nb_proba, nb_preds)

    # Random Forest
    rf_path = FINAL_MODELS_DIR / "random-forest-model.pkl"
    if force or not rf_path.exists():
        rf_model = train_random_forest(X_train, y_train, train_weights)
        joblib.dump(rf_model, rf_path, compress=3)
    else:
        rf_model = joblib.load(rf_path)
    rf_proba = rf_model.predict_proba(X_test)
    rf_preds = rf_proba.argmax(axis=1)
    metrics["random_forest"] = evaluate_predictions(y_test, rf_proba, rf_preds)

    # LightGBM
    lgb_path = FINAL_MODELS_DIR / "lightgbm-model.pkl"
    if force or not lgb_path.exists():
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, train_weights, val_weights)
        joblib.dump(lgb_model, lgb_path, compress=3)
    else:
        lgb_model = joblib.load(lgb_path)
    iter_lgb = lgb_model.best_iteration_ or lgb_model.n_estimators_
    lgb_proba = lgb_model.predict_proba(X_test, num_iteration=iter_lgb)
    lgb_preds = lgb_proba.argmax(axis=1)
    metrics["lightgbm"] = evaluate_predictions(y_test, lgb_proba, lgb_preds)

    # XGBoost
    xgb_path = FINAL_MODELS_DIR / "xgboost-model.json"
    if force or not xgb_path.exists():
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val, train_weights, val_weights)
        xgb_model.save_model(xgb_path)
    else:
        xgb_model = XGBClassifier()
        xgb_model.load_model(xgb_path)
    xgb_model.set_params(**{"objective": "multi:softprob"})
    xgb_proba = xgb_model.predict_proba(X_test)
    xgb_preds = xgb_proba.argmax(axis=1)
    metrics["xgboost"] = evaluate_predictions(y_test, xgb_proba, xgb_preds)

    # MLP
    mlp_path = FINAL_MODELS_DIR / "mlp-model.pt"
    mlp_config_path = FINAL_MODELS_DIR / "mlp-config.json"
    if force or not mlp_path.exists():
        mlp_model, extra = train_mlp(X_train, y_train, X_val, y_val, class_weights)
        torch.save(mlp_model.state_dict(), mlp_path)
        save_json(mlp_config_path, {
            "input_dim": int(X_train.shape[1]),
            "hidden_dim": 384,
            "depth": 4,
            "expansion": 1.4,
            "dropout": 0.2,
            "num_classes": len(class_weights),
        })
    else:
        mlp_model = SleepMLP(input_dim=X_train.shape[1], hidden_dim=384, depth=4, expansion=1.4, dropout=0.2, num_classes=len(class_weights))
        mlp_model.load_state_dict(torch.load(mlp_path, map_location="cpu"))
    mlp_model.eval()
    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test)
        logits = mlp_model(test_tensor)
        mlp_probabilities = torch.softmax(logits, dim=1).numpy()
        mlp_preds = mlp_probabilities.argmax(axis=1)
    metrics["mlp"] = evaluate_predictions(y_test, mlp_probabilities, mlp_preds)

    save_json(FINAL_MODELS_DIR / "metrics.json", metrics)
    print("Saved models to", FINAL_MODELS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save final models")
    parser.add_argument("--force", action="store_true", help="Retrain and overwrite existing models")
    args = parser.parse_args()
    main(force=args.force)
