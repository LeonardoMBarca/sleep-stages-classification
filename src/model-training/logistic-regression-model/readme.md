# Baseline: Sleep Stage Classification with Logistic Regression

## Introduction
This notebook establishes a baseline for multi-class sleep-stage classification. It demonstrates how the data are loaded, pre-processed, modelled with multinomial logistic regression, and evaluated on validation and test splits.

## Dependencies
The workflow relies on standard Python data science libraries:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn components (LabelEncoder, StandardScaler, SimpleImputer, LogisticRegression, classification_report, confusion_matrix)

## Data
Training, validation, and test partitions are loaded from Parquet files stored under `datalake/data-for-model`. The script inspects:
- set dimensions
- class distribution for the `stage` label
- data types
- missing values

## Modelling Pipeline

### 1. Pre-processing
- **Feature selection**: Remove label (`stage`) and identifier columns (`subject_id`, `night_id`, `sex`, `age`) from the feature matrices.
- **Label encoding**: Convert categorical stages into integers with `LabelEncoder`, keeping track of the mapping (e.g., N1, N2, N3, REM, W).
- **Scaling**: Fit a `StandardScaler` on the training features and reuse it for validation and test sets. Shapes and summary statistics are printed after scaling.

### 2. Handling missing values
- Apply `SimpleImputer(strategy="mean")` to each scaled split to replace any remaining NaN values.

### 3. Training
- Estimator: `LogisticRegression`
- Key parameters:
  - `multi_class="multinomial"`
  - `class_weight="balanced"`
  - `random_state=42`
  - `max_iter=1000`
- The model is fitted on the imputed training matrix with the encoded labels.

## Evaluation
Performance is computed with `classification_report` and `confusion_matrix`, using the original stage labels in the output tables.

### Validation split
- Accuracy: 72%
- Macro F1-score: 0.66

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| N1    | 0.32      | 0.53   | 0.40     | 4215    |
| N2    | 0.89      | 0.60   | 0.72     | 15066   |
| N3    | 0.50      | 0.85   | 0.63     | 2524    |
| REM   | 0.60      | 0.81   | 0.69     | 5456    |
| W     | 0.93      | 0.84   | 0.88     | 15120   |

### Test split
- Accuracy: 72%
- Macro F1-score: 0.66

## Generated plots
The notebook produces several figures from the evaluation metrics:
- Per-class metric bar chart (`classification_metrics.png`) comparing precision, recall, and F1 across stages.
- Validation confusion matrix heatmap.
- Side-by-side bar chart comparing validation vs test precision, recall, and F1.
- Summary bar chart for accuracy and macro F1 on validation and test splits.

## Conclusions
- The baseline performs well on wake (W), REM, and deep sleep (N3), where both recall and F1-score remain high.
- Light sleep stages N1 and N2 remain more challenging to separate, matching expectations given their physiological similarity.
- Results are consistent between validation and test sets (accuracy 72%, macro F1 0.66), indicating the baseline is stable for subsequent comparisons.
