# Gaussian Naive Bayes Sleep Stage Classifier

## Dataset preparation
- Loads the curated train/validation/test parquet files used across all models.
- Sex is mapped to numeric values (F → 0.0, M → 1.0); stages become integer IDs via a fixed mapping.
- Drops identifiers and standardizes the remaining features with `StandardScaler` (fit on training, applied to validation/test).
- Computes class-frequency weights but keeps them mainly for evaluation context; Naive Bayes does not accept sample weights.

## Model configuration
- `GaussianNB` with variance smoothing tuned over a logarithmic grid `1e-9` to `1e-3`.
- The variance smoothing value that maximizes macro F1 on the validation set is selected; the entire grid search history is stored in the notebook.

## Training procedure
- For each smoothing candidate, fit on the scaled training data and evaluate macro F1 on validation.
- Retain the best-performing model and reuse it for final validation/test inference.

## Evaluation results
- Test log loss: **5.5241**
- Test accuracy: **0.6495**
- Test balanced accuracy: **0.6394**
- Test macro F1: **0.5812**
- Stage-level F1 (test): W 0.801, N1 0.282, N2 0.716, N3 0.537, REM 0.569.

## Reproducing the run
Run `naive-bayes-training.ipynb`. The notebook performs the grid search, reports the ranking of smoothing values, and prints the final metrics and confusion matrix.
