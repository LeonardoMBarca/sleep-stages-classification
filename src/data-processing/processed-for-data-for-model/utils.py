import polars as pl
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Sequence, Dict, List, Optional

def bin_idade(s: pd.Series,
              bins=(0, 40, 60, 120),
              labels=("≤40", "41–60", "≥61")) -> pd.Categorical:
    return pd.cut(pd.to_numeric(s, errors="coerce"), bins=bins, labels=labels, include_lowest=True, right=True)

def subject_level_table(df: pd.DataFrame,
                        age_bins=(0, 40, 60, 120),
                        age_labels=("≤40", "41–60", "≥61")) -> pd.DataFrame:
    sdf = (df.groupby("subject_id")
             .agg(sex=("sex", lambda x: x.astype(str).mode().iat[0]),
                  age=("age", lambda x: pd.to_numeric(x, errors="coerce").dropna().astype(float).median()))
             .reset_index())
    sdf["age_bin"] = bin_idade(sdf["age"], bins=age_bins, labels=age_labels)
    return sdf

def hamilton_round(counts_float: Dict[str, float], total_int: int) -> Dict[str, int]:
    base = {k: int(np.floor(v)) for k, v in counts_float.items()}
    rem = total_int - sum(base.values())
    fracs = sorted(((k, counts_float[k] - base[k]) for k in counts_float),
                   key=lambda kv: kv[1], reverse=True)
    i = 0
    while rem > 0 and i < len(fracs):
        base[fracs[i][0]] += 1
        rem -= 1
        i += 1
    return base

def stratified_subject_split_by_quotas(
    df: pd.DataFrame,
    ratios: Dict[str, float] = {"train": 0.6, "val": 0.2, "test": 0.2},
    age_bins=(0, 40, 60, 120),
    age_labels=("≤40", "41–60", "≥61"),
    random_state: int = 42,
) -> Dict[str, List[str]]:
    rng = np.random.default_rng(random_state)

    sdf = subject_level_table(df, age_bins=age_bins, age_labels=age_labels)
    sdf["sex"] = sdf["sex"].astype(str)
    sdf["age_bin"] = sdf["age_bin"].astype("category")

    assign: Dict[str, List[str]] = {k: [] for k in ratios}

    for (sex, ageb), grp in sdf.groupby(["sex", "age_bin"], observed=True):
        subjects = grp["subject_id"].tolist()
        rng.shuffle(subjects)  

        Ns = len(subjects)
        if Ns == 0:
            continue

        desired_float = {k: v * Ns for k, v in ratios.items()}
        desired_int = hamilton_round(desired_float, total_int=Ns)

        start = 0
        for split, q in desired_int.items():
            if q <= 0:
                continue
            assign[split].extend(subjects[start:start+q])
            start += q

    all_assigned = sum((assign[k] for k in assign), [])
    assert len(all_assigned) == len(set(all_assigned)), "Leakage: subject assigned to multiple splits"
    assert set(all_assigned) == set(sdf["subject_id"]), "Missing subjects in split allocation"

    return assign

def indices_from_subject_assign(df: pd.DataFrame, assign: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    out = {}
    for split, subs in assign.items():
        mask = df["subject_id"].isin(subs).values
        out[split] = np.where(mask)[0]
    return out

def load_sleep_datasets(
    root: Path,
    add_from_path: bool = True,
    keep_file: bool = False,
    head_buffer: int = 45,
    tail_buffer: int = 45,
    mid_buffer: int = 30,
    mid_cap: int = 0,
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
        
def add_rolling_features_sleep_fast(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    subject_col: str = "subject_id",
    night_col: str = "night_id",
    epoch_col: str = "epoch_idx",
    target_col: str = "stage",
    windows: Sequence[int] = (5, 10, 20),
    stats: Sequence[str] = ("mean", "std", "max"),
    batch_size: int = 64,
) -> pd.DataFrame:
    assert subject_col in df.columns and night_col in df.columns and epoch_col in df.columns
    df_sorted = df.sort_values([subject_col, night_col, epoch_col]).copy()

    if feature_cols is None:
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
        drop = {epoch_col}
        if target_col in df_sorted.columns:
            drop.add(target_col)
        feature_cols = [c for c in numeric_cols if c not in drop]

    g = df_sorted.groupby([subject_col, night_col], group_keys=False)

    new_cols_all = []

    for col in feature_cols:
        col_new = []
        s = g[col]

        for w in windows:
            roll = s.rolling(window=w, min_periods=1)

            if "mean" in stats:
                c = f"{col}_roll_mean_{w}"
                col_new.append(roll.mean().reset_index(level=[0,1], drop=True).rename(c))

            if "std" in stats:
                c = f"{col}_roll_std_{w}"
                std_series = roll.std(ddof=0).reset_index(level=[0,1], drop=True)
                std_series = std_series.fillna(0.0)
                col_new.append(std_series.rename(c))

            if "max" in stats:
                c = f"{col}_roll_max_{w}"
                col_new.append(roll.max().reset_index(level=[0,1], drop=True).rename(c))

        new_cols_all.extend(col_new)

    # concatenate in blocks to avoid fragmentation
    for i in range(0, len(new_cols_all), batch_size):
        df_sorted = pd.concat([df_sorted, pd.concat(new_cols_all[i:i+batch_size], axis=1)], axis=1)

    return df_sorted
