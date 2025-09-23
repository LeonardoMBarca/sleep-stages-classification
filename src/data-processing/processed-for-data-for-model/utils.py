import pandas as pd
import numpy as np
from typing import Dict, List

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
    assert len(all_assigned) == len(set(all_assigned)), "Leakage: sujeito em múltiplos splits"
    assert set(all_assigned) == set(sdf["subject_id"]), "Faltam sujeitos alocados"

    return assign

def indices_from_subject_assign(df: pd.DataFrame, assign: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    out = {}
    for split, subs in assign.items():
        mask = df["subject_id"].isin(subs).values
        out[split] = np.where(mask)[0]
    return out