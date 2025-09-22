import re
import polars as pl
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.feature_selection import f_classif
from scipy.stats import kruskal

BASE_PATH = Path(__file__).resolve().parents[3]

SLEEP_CASSETTE = BASE_PATH / "datalake" / "processed" / "sleep-cassette"
SLEEP_TELEMETRY = BASE_PATH / "datalake" / "processed" / "sleep-telemetry"

OUT_PATH_CASSETTE = BASE_PATH / "datalake" / "data-for-model" / "sleep-cassette.parquet" 
OUT_PATH_TELEMETRY = BASE_PATH / "datalake" / "data-for-model" / "sleep-telemetry.parquet" 

def load_sleep_datasets(
    root: Path,
    add_from_path: bool = True,
    keep_file: bool = False,
    head_buffer: int = 30,
    tail_buffer: int = 30,
    mid_buffer: int = 30,
    mid_cap: int = 90,
) -> pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame]:
    try:
        files = sorted(root.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet finded in {root}")
        dfs = []
        for p in files:
            df = pl.read_parquet(p)
            n = df.height
            stages = df.select("stage").to_series().to_list()
            k1 = None
            for i, s in enumerate(stages):
                if s != "W":
                    k1 = i
                    break
            if k1 is None:
                start_keep = max(0, n - head_buffer)
                stop_keep = n
            else:
                k2 = None
                for j in range(n - 1, -1, -1):
                    if stages[j] != "W":
                        k2 = j
                        break
                start_keep = max(0, k1 - head_buffer)
                stop_keep = min(n, (k2 + 1 + tail_buffer) if k2 is not None else n)
                if stop_keep <= start_keep:
                    stop_keep = min(n, start_keep + 1)
            kept = stop_keep - start_keep
            df_slice = df.slice(start_keep, kept).with_columns([
                pl.lit(str(p)).alias("file"),
                pl.arange(0, pl.len()).alias("_i")
            ])
            mid_w_total = 0
            mid_w_removed = 0
            mid_runs_affected = 0
            if kept > 0 and mid_cap > 0 and mid_buffer >= 0:
                st_mid = df_slice.select("stage").to_series().to_list()
                m = len(st_mid)
                runs = []
                rs = 0
                for t in range(1, m):
                    if st_mid[t] != st_mid[t-1]:
                        runs.append((rs, t-1, st_mid[t-1]))
                        rs = t
                runs.append((rs, m-1, st_mid[-1]))
                keep_mask = np.ones(m, dtype=bool)
                for rs, re_, lab in runs:
                    L = re_ - rs + 1
                    if lab == "W" and rs > 0 and re_ < m-1:
                        mid_w_total += L
                        if L > mid_cap:
                            a = rs + mid_buffer
                            b = re_ - mid_buffer + 1
                            if a < b:
                                keep_mask[a:b] = False
                                mid_w_removed += (b - a)
                                mid_runs_affected += 1
                keep_idx = np.nonzero(keep_mask)[0].tolist()
                if len(keep_idx) < m:
                    df_slice = (
                        df_slice.join(pl.DataFrame({"_i": keep_idx}), on="_i", how="inner")
                                .sort("_i")
                    )
            df_out = df_slice.drop("_i")
            dfs.append(df_out)

        lf = pl.concat(dfs, how="vertical_relaxed").lazy()
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

df_cassette = load_sleep_datasets(SLEEP_CASSETTE, True)
df_telemetry = load_sleep_datasets(SLEEP_TELEMETRY, True)

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