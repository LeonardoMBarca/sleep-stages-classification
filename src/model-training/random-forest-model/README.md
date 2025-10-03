# Random Forest Sleep Stage Classifier

## Dataset preparation
- Consistent preprocessing with the other models: parquet splits from `datalake/data-for-model`, sex mapped to {0,1}, stages mapped to IDs, identifiers removed, features sorted.
- `StandardScaler` fitted on training and applied to validation/test to keep the input space centered for each tree.
- Per-sample weights computed from inverse class frequencies and supplied during fitting.

## Model configuration
- Evaluated three candidate configurations varying tree depth, number of estimators, and bootstrap sampling ratio:
  1. `n_estimators=240`, `max_depth=22`, `min_samples_split=6`, `min_samples_leaf=2`, `max_features="sqrt"`, `max_samples=0.8`.
  2. `n_estimators=300`, `max_depth=26`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features="sqrt"`, `max_samples=0.85`.
  3. `n_estimators=200`, `max_depth=20`, `min_samples_split=8`, `min_samples_leaf=3`, `max_features="sqrt"`, `max_samples=0.75`.
- All runs use `class_weight="balanced_subsample"`, bootstrap sampling enabled, and `n_jobs=8` to keep runtime manageable.

## Training procedure
- Each candidate is fitted on the training split with sample weights. Validation macro F1 determines the best configuration.
- The validation leaderboard is captured in `history_df`. The best model is reused for final scoring.

## Evaluation results
- Test log loss: **0.6027**
- Test accuracy: **0.7777**
- Test balanced accuracy: **0.6914**
- Test macro F1: **0.7001**
- Stage-level F1 (test): W 0.891, N1 0.365, N2 0.826, N3 0.742, REM 0.675.

## Reproducing the run
Launch `random-forest-training.ipynb` and execute the cells. Progress prints show which hyperparameter combination is being trained, and the notebook concludes with the metrics table, classification report, and confusion matrix.
