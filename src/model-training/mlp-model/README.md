# Residual MLP Sleep Stage Classifier

## Dataset preparation
- Reads the pre-split parquet files from `datalake/data-for-model`.
- Maps sex (F → 0.0, M → 1.0) and sleep stage strings to integer IDs.
- Drops identifier columns and sorts remaining features to ensure consistent ordering.
- Applies `StandardScaler` fitted on the training set and reused for validation/test.
- Builds class weights from inverse frequencies, slightly exponentiated to emphasise the rarest stages (N3, REM).

## Model architecture
- Feed-forward network with residual MLP blocks:
  - Stem: Linear(input → 384) → LayerNorm → GELU → Dropout(0.2).
  - Four residual blocks, each LayerNorm → Linear(expansion 1.4) → GELU → Dropout → Linear → Dropout, with skip connection.
  - Head: LayerNorm → Linear(384 → 5 classes).
- Uses `FocalLoss` (`gamma=1.15`) with normalized class weights to handle imbalance.

## Training procedure
- Batch size 512, `DataLoader` workers auto-configured (`min(8, cpu_count//2)`) with pin-memory when CUDA is available.
- Optimizer: `AdamW` (lr 3e-4, weight decay 5e-4) plus `OneCycleLR` scheduler (`max_lr 1.5e-3`, 20 epochs, pct_start 0.35).
- Mixed-precision training via `torch.cuda.amp` and gradient clipping at 1.0 to stabilize updates.
- Early stopping style patience of 4 epochs using validation macro F1; best weights are restored after training.

## Evaluation results
- Test log loss: **0.6943**
- Test accuracy: **0.7318**
- Test balanced accuracy: **0.7297**
- Test macro F1: **0.6787**
- Stage-level F1 (test): W 0.898, N1 0.393, N2 0.741, N3 0.666, REM 0.697.
- Notably improves recall for deep sleep (N3 recall 0.857) compared with earlier MLP baselines.

## Reproducing the run
Execute `mlp-training.ipynb`. The runtime on CPU-only environments is several minutes due to dataset size; enable CUDA for additional speed. All diagnostics, including epoch logs, metric tables, and the confusion matrix, are produced automatically.
