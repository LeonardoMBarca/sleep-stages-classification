# XGBoost Sleep Stage Classifier

## Dataset preparation
- Shares the same preprocessing pipeline as the other models: parquet splits, sex mapped to numeric values, stage labels mapped to IDs, identifiers removed, features sorted and standardized with `StandardScaler`.
- Class weights derived from inverse frequencies are passed as per-sample weights during training.

## Model configuration
- `XGBClassifier` with `objective="multi:softprob"`, 5 classes and histogram-based tree growth.
- Hyperparameters tuned for a strong yet stable model: `n_estimators=1600`, `learning_rate=0.045`, `max_depth=8`, `subsample=0.85`, `colsample_bytree=0.7`, `min_child_weight=3`, `gamma=0.1`, `reg_lambda=1.2`, `reg_alpha=0.05`.
- Runs with `tree_method="hist"`, `n_jobs=-1`, and evaluation metric set to multi-class log loss.

## Training procedure
- Fit on training data with sample weights and monitor validation log loss through `eval_set=[(x_train, y_train), (x_val, y_val)]`.
- After training, the validation log loss trace is inspected to pick the iteration with the minimum value (reported in the notebook).
- That iteration is used for final predictions on validation/test splits via `iteration_range`.

## Evaluation results
- Test log loss: **0.5884**
- Test accuracy: **0.7774**
- Test balanced accuracy: **0.7244**
- Test macro F1: **0.7123**
- Stage-level F1 (test): W 0.891, N1 0.397, N2 0.826, N3 0.757, REM 0.690.

## Reproducing the run
Open `xgboost-training.ipynb` and execute all cells. Training progress prints validation loss every 50 boosting rounds; the notebook concludes with the metric summary and confusion matrix.
