# Multinomial Logistic Regression Sleep Stage Classifier

## Dataset preparation
- Uses the same pre-engineered parquet splits (`train/val/test`) created in `datalake/data-for-model`.
- Converts sex to a numeric flag (F → 0.0, M → 1.0) and stages to integer IDs.
- Excludes identifier columns (`subject_id`, `night_id`, `epoch_idx`, `stage`, `stage_id`) and sorts the remaining feature columns.
- Standardizes features with `StandardScaler`, fitted on the training split and applied to validation/test.
- Derives per-sample weights from the inverse class frequency to mitigate imbalance.

## Model configuration
- `LogisticRegression` with `multi_class="multinomial"`, `solver="saga"`, `penalty="l2"`, `C=0.9`.
- Increased `max_iter` to 2000 and tolerance 1e-4 to guarantee convergence with the large feature space.
- Trains with the class-derived sample weights to emphasise minority stages.

## Training procedure
- Fit occurs on the scaled training features with weights; validation and test predictions reuse the trained scaler and encode the same feature ordering.
- No explicit early stopping is required because the optimization is convex; convergence monitoring relies on the solver tolerance.

## Evaluation results
- Test log loss: **0.7462**
- Test accuracy: **0.7232**
- Test balanced accuracy: **0.7174**
- Test macro F1: **0.6674**
- Stage-level F1 (test): W 0.860, N1 0.387, N2 0.774, N3 0.664, REM 0.652.

## Reproducing the run
Execute `logistic-regression-training.ipynb`. The notebook handles scaling, training, and prints the summary/diagnostics at the end, including the confusion matrix.
