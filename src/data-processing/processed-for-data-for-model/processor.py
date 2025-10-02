import polars as pl
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.feature_selection import f_classif
from scipy.stats import kruskal
from utils import stratified_subject_split_by_quotas, indices_from_subject_assign, load_sleep_datasets, add_rolling_features_sleep_fast

BASE_PATH = Path(__file__).resolve().parents[3]

SLEEP_CASSETTE = BASE_PATH / "datalake" / "processed" / "sleep-cassette"
SLEEP_TELEMETRY = BASE_PATH / "datalake" / "processed" / "sleep-telemetry"

CASSETTE_TRAIN = BASE_PATH / "datalake" / "data-for-model" / "train" / "train_sleep_cassette.parquet"
CASSETTE_VAL = BASE_PATH / "datalake" / "data-for-model" / "val" / "val_sleep_cassette.parquet"
CASSETTE_TEST = BASE_PATH / "datalake" / "data-for-model" / "test" / "test_sleep_cassette.parquet"

OUT_PATH_CASSETTE = BASE_PATH / "datalake" / "data-for-model" / "sleep-cassette.parquet"
OUT_PATH_TELEMETRY = BASE_PATH / "datalake" / "data-for-model" / "sleep-telemetry.parquet" 

df_cassette = load_sleep_datasets(SLEEP_CASSETTE, True)
df_telemetry = load_sleep_datasets(SLEEP_TELEMETRY, True)

META = [
    "subject_id", "night_id", "epoch_idx", "stage",
    "age", "sex", "tso_min"
]

EEG_CORE = [
    "EEG_Pz_Oz_beta_relpow_256",
    "EEG_Pz_Oz_delta_relpow_256",
    "EEG_Fpz_Cz_delta_relpow_256",
    "EEG_Fpz_Cz_sigma_relpow_256",
    "EEG_Fpz_Cz_theta_relpow_256",
    "EEG_Pz_Oz_alpha_relpow_256",
]

EEG_RATIOS = [
    "EEG_Fpz_Cz_slow_fast_ratio_256",
    # "EEG_Pz_Oz_slow_fast_ratio_256",
    "EEG_Fpz_Cz_delta_theta_ratio_256",
    # "EEG_Pz_Oz_theta_alpha_ratio_256",
    "EEG_Fpz_Cz_alpha_sigma_ratio_256",
]
EEG_DESCR = [
    "EEG_Fpz_Cz_aperiodic_slope_256",
    "EEG_Pz_Oz_aperiodic_slope_256",
    "EEG_Pz_Oz_spec_entropy_256",
    "EEG_Fpz_Cz_spec_entropy_256",
]

EOG_CORE = [
    "EOG_rms",
    "EOG_beta_relpow_256",
    "EOG_delta_relpow_256",
    "EOG_sef95_256",
]

EMG_CORE = [
    "EMG_submental_p90_1hz",
]

OPTIONAL = [
    "EEG_Fpz_Cz_beta_relpow_256",
    "EEG_Fpz_Cz_theta_alpha_ratio_256",
    "EEG_Fpz_Cz_alpha_relpow_256",
    "EOG_theta_relpow_256",
    "EMG_submental_median_1hz",
    "EEG_Pz_Oz_sef95_256",
    "EOG_spec_entropy_256"
]

def present(df: pl.DataFrame, cols: list[str]) -> list[str]:
    s = set(df.columns)
    return [c for c in cols if c in s]

KEEP = META + EEG_CORE + EEG_RATIOS + EEG_DESCR + EOG_CORE + EMG_CORE
KEEP += OPTIONAL  # <- Remove this lines if you want without the optional columns

df_cols_cassette = present(df_cassette, KEEP)
df__cols_telemetry = present(df_telemetry, KEEP)

final_parquet_file_cassette = df_cassette.select(df_cols_cassette)
final_parquet_file_telemetry = df_telemetry.select(df__cols_telemetry)

df_final_cassette = final_parquet_file_cassette.to_pandas()
df_final_telemetry = final_parquet_file_telemetry.to_pandas()

df_columns = df_final_cassette.columns
df_columns = df_columns.to_list()
for col in ["age", "epoch_idx", "stage", "night_id", "subject_id", "sex"]:
    if col in df_columns:
        df_columns.remove(col)

ROLL_FAST = [
    "EEG_Fpz_Cz_theta_relpow_256",
    "EEG_Pz_Oz_alpha_relpow_256",
    "EEG_Fpz_Cz_theta_alpha_ratio_256",
    "EEG_Fpz_Cz_sigma_relpow_256",
    "EEG_Pz_Oz_beta_relpow_256",
    "EOG_rms",
    "EOG_theta_relpow_256",
    "EMG_submental_median_1hz",
    "EMG_submental_p90_1hz",  
]

ROLL_MED = [
    "EEG_Pz_Oz_alpha_relpow_256",
    "EEG_Pz_Oz_beta_relpow_256",
    "EEG_Fpz_Cz_sigma_relpow_256",
    "EOG_rms",
    # "EEG_Pz_Oz_slow_fast_ratio_256",
]

ROLL_LONG = [
    "EEG_Pz_Oz_delta_relpow_256",
    "EEG_Pz_Oz_spec_entropy_256",
    "EEG_Fpz_Cz_aperiodic_slope_256",
]

df_final_cassette = add_rolling_features_sleep_fast(
    df_final_cassette, feature_cols=ROLL_FAST, windows=(5,), stats=("mean","std",)
)
df_final_cassette = add_rolling_features_sleep_fast(
    df_final_cassette, feature_cols=ROLL_MED, windows=(10,), stats=("mean","std",)
)
df_final_cassette = add_rolling_features_sleep_fast(
    df_final_cassette, feature_cols=ROLL_LONG, windows=(15,), stats=("mean",)
)

df_final_cassette = add_rolling_features_sleep_fast(
    df_final_cassette, feature_cols=["EOG_rms","EMG_submental_p90_1hz"], windows=(10,), stats=("max",)
)

# df_final_cassette = add_rolling_features_sleep_fast(
#     df_final_cassette,
#     feature_cols=df_columns,
#     windows=[5, 10], 
#     stats=("mean","std","max","slope","zscore")
# )

df_final_telemetry = add_rolling_features_sleep_fast(
    df_final_telemetry, feature_cols=ROLL_FAST, windows=(5,), stats=("mean","std",)
)
df_final_telemetry = add_rolling_features_sleep_fast(
    df_final_telemetry, feature_cols=ROLL_MED, windows=(10,), stats=("mean","std",)
)
df_final_telemetry = add_rolling_features_sleep_fast(
    df_final_telemetry, feature_cols=ROLL_LONG, windows=(15,), stats=("mean",)
)

df_final_telemetry = add_rolling_features_sleep_fast(
    df_final_telemetry, feature_cols=["EOG_rms","EMG_submental_p90_1hz"], windows=(10,), stats=("max",)
)

# df_final_telemetry = add_rolling_features_sleep_fast(
#     df_final_telemetry,
#     feature_cols=df_columns,
#     windows=[5, 10],
#     # windows=[5, 2, 10], 
#     stats=("mean","std","max","slope","zscore")
# )

assign = stratified_subject_split_by_quotas(
    df_final_cassette,
    ratios={"train": 0.6, "val": 0.2, "test": 0.2},
    age_bins=(0, 40, 60, 120),
    age_labels=("≤40", "41–60", "≥61"),
    random_state=7,
)

idx = indices_from_subject_assign(df_final_cassette, assign)

df_train_cassette = df_final_cassette.iloc[idx["train"]]
df_val_cassette   = df_final_cassette.iloc[idx["val"]]
df_test_cassette  = df_final_cassette.iloc[idx["test"]]

print("NULL VALUES OF CASSETTE TRAIN: ", df_train_cassette.isnull().sum().sum())
print("NULL VALUES OF CASSETTE VAL: ", df_val_cassette.isnull().sum().sum())
print("NULL VALUES OF CASSETTE TEST: ", df_test_cassette.isnull().sum().sum())

print(f"NULL VALUES OF CASSETTE: ", df_final_cassette.isnull().sum().sum())
print(f"NULL VALUES OF TELEMETRY: ", df_final_telemetry.isnull().sum().sum())

print(f"Cassette columns: {len(df_final_cassette.columns)}")
print(f"Cassette train columns: {len(df_train_cassette.columns)}")
print(f"Cassette test columns: {len(df_test_cassette.columns)}")
print(f"Cassette val columns: {len(df_val_cassette.columns)}")

print(f"Telemetry columns: {len(df_final_telemetry.columns)}")

df_train_cassette.to_parquet(CASSETTE_TRAIN, index=False)
df_val_cassette.to_parquet(CASSETTE_VAL, index=False)
df_test_cassette.to_parquet(CASSETTE_TEST, index=False)
df_final_cassette.to_parquet(OUT_PATH_CASSETTE, index=False)
df_final_telemetry.to_parquet(OUT_PATH_TELEMETRY, index=False)

print(f"Datasets uploaded in: {BASE_PATH}/datalake/data-for-model")