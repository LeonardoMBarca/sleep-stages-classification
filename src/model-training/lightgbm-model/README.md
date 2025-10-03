# LightGBM Sleep Stage Classifier

## Dataset preparation
- Train/validation/test parquet files from `datalake/data-for-model` are loaded with `fastparquet`.
- Sex is mapped to numerical values (F → 0.0, M → 1.0) and the sleep stage string is mapped to an integer identifier.
- All non-identifier columns are used as features. A `StandardScaler` is fitted on the training split and applied to validation/test.
- Class imbalance is handled by computing inverse-frequency weights per stage; these weights are used both during training and in the validation callbacks.

## Model configuration
- `LGBMClassifier` with `objective="multiclass"` and 5 classes.
- Key hyperparameters: `n_estimators=2200`, `learning_rate=0.045`, `num_leaves=104`, `subsample=0.85`, `colsample_bytree=0.7`, `min_child_samples=60`, `reg_lambda=0.9`, `reg_alpha=0.02`.
- Random seed fixed at 42 and all cores enabled (`n_jobs=-1`).

## Training procedure
- Sample weights derived from the class distribution are passed to `fit`.
- Validation is executed on the hold-out split with the same weighting, logging both multi-class log loss and error.
- `early_stopping(stopping_rounds=150)` and `log_evaluation(period=25)` callbacks ensure the iteration count is tuned automatically.
- The best iteration reported by LightGBM is reused when scoring the validation and test sets.

## Evaluation results
- Test log loss: **0.6127**
- Test accuracy: **0.7664**
- Test balanced accuracy: **0.7256**
- Test macro F1: **0.7058**
- Stage-level F1 (test): W 0.884, N1 0.398, N2 0.819, N3 0.747, REM 0.681.

## Reproducing the run
Open `lightgbm-training.ipynb` and execute the cells sequentially (no additional configuration is needed). The notebook saves the evaluation tables and confusion matrix in-line for quick review.
