import polars as pl
import pandas as pd
import os

from pathlib import Path
from sklearn.feature_selection import f_classif
from scipy.stats import kruskal

BASE_PATH = Path(__file__).resolve().parents[3]

SLEEP_CASSETTE = BASE_PATH / "datalake" / "processed" / "sleep-cassette"
SLEEP_TELEMETRY = BASE_PATH / "datalake" / "processed" / "sleep-telemetry"

OUT_PATH_CASSETTE = BASE_PATH / "datalake" / "data-for-model" / "sleep-cassette.parquet" 
OUT_PATH_TELEMETRY = BASE_PATH / "datalake" / "data-for-model" / "sleep-telemetry.parquet" 

def load_sleep_cassette(root: Path, add_from_path: bool = True,  keep_file: bool = False) -> pl.DataFrame:
    try:
        files = sorted(root.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet finded in {root}")
        
        paths = [str(p) for p in files]
        
        lf = pl.concat(
            [
                pl.read_parquet(p).with_columns(pl.lit(p).alias("file"))
                for p in paths
            ],
            how="vertical_relaxed",
        ).lazy()

        if add_from_path:
            lf = lf.drop(["subject_id", "night_id"], strict=False).with_columns([
                pl.col("file").str.extract(r"subject_id=([^/\\]+)", group_index=1).alias("subject_id"),
                pl.col("file").str.extract(r"(N\d)\.parquet$", group_index=1).alias("night_id"),
            ])
            if not keep_file:
                lf = lf.drop("file", strict=False)


        df = lf.collect(engine="streaming")

        return df

    except Exception as e:
        print(f"ERROR: {e}")

df_cassette = load_sleep_cassette(SLEEP_CASSETTE, True)
df_telemetry = load_sleep_cassette(SLEEP_TELEMETRY, True)

META = [
    "subject_id", "night_id", "epoch_idx", "t0_sec", "stage",
    "age", "sex", "tso_min",
]

EEG_CORE = [
    "EEG_Pz_Oz_beta_relpow_256",
    "EEG_Pz_Oz_delta_relpow_256",
    "EEG_Fpz_Cz_sigma_relpow_256",
    "EEG_Fpz_Cz_theta_relpow_256",
    "EEG_Pz_Oz_alpha_relpow_256",
]

EEG_RATIOS = [
    "EEG_Fpz_Cz_slow_fast_ratio_256",
    "EEG_Pz_Oz_slow_fast_ratio_256",
    "EEG_Fpz_Cz_delta_theta_ratio_256",
    "EEG_Pz_Oz_theta_alpha_ratio_256",
    "EEG_Fpz_Cz_alpha_sigma_ratio_256",
]
EEG_DESCR = [
    "EEG_Fpz_Cz_aperiodic_slope_256",
    "EEG_Pz_Oz_aperiodic_slope_256",
    "EEG_Pz_Oz_spec_entropy_256",
]

EOG_CORE = [
    "EOG_rms",
    "EOG_beta_relpow_256",
    "EOG_delta_relpow_256",
    "EOG_sef95_256",
]

EMG_CORE = [
    "EMG_submental_p90_1hz",
    "EMG_submental_rms_1hz",
]

OPTIONAL = [
    "EEG_Fpz_Cz_beta_relpow_256",
    "EEG_Fpz_Cz_alpha_relpow_256",
    "EOG_theta_relpow_256",
    "EMG_submental_median_1hz"
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

df_pandas_c = final_parquet_file_cassette.to_pandas()
df_pandas_t = final_parquet_file_telemetry.to_pandas()

x_c = df_pandas_c[df_cols_cassette].drop(columns=META)
y_c = df_pandas_c["stage"]

f_vals_c, p_vals_c = f_classif(x_c, y_c)
anova_results_c = pd.DataFrame({
    "feature": x_c.columns,
    "f_value": f_vals_c,
    "p_value": p_vals_c
}).sort_values("f_value", ascending=False)

print(f"F CLASSIF RESULT CASSETTE: \n{anova_results_c}")

kruskal_results_c = []
for col in x_c.columns:
    groups = [x_c.loc[y_c == cls, col] for cls in y_c.unique()]
    stat, p = kruskal(*groups)
    kruskal_results_c.append((col, stat, p))

kruskal_results_c = pd.DataFrame(kruskal_results_c, columns=["feature", "H_value", "p_value"]) \
                        .sort_values("H_value", ascending=False)

print(f"KRUSKAL RESULT CASSETTE: \n{kruskal_results_c}")

x_t = df_pandas_t[df_cols_cassette].drop(columns=META)
y_t = df_pandas_t["stage"]

f_vals_t, p_vals_t = f_classif(x_t, y_t)
anova_results_t = pd.DataFrame({
    "feature": x_t.columns,
    "f_value": f_vals_c,
    "p_value": p_vals_c
}).sort_values("f_value", ascending=False)

print(f"F CLASSIF RESULT TELEMETRY: \n{anova_results_t}")

kruskal_results_t = []
for col in x_t.columns:
    groups = [x_t.loc[y_t == cls, col] for cls in y_t.unique()]
    stat, p = kruskal(*groups)
    kruskal_results_t.append((col, stat, p))

kruskal_results_t = pd.DataFrame(kruskal_results_t, columns=["feature", "H_value", "p_value"]) \
                        .sort_values("H_value", ascending=False)

print(f"KRUSKAL RESULT TELEMETRY: \n{kruskal_results_t}")

print(f"Cassette columns: {len(final_parquet_file_cassette.columns)}")
print(f"Telemetry columns: {len(final_parquet_file_telemetry.columns)}")

final_parquet_file_cassette.write_parquet(OUT_PATH_CASSETTE)
final_parquet_file_telemetry.write_parquet(OUT_PATH_TELEMETRY)

print(f"Datasets uploaded in: {BASE_PATH}/datalake/data-for-model")