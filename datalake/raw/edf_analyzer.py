from pyedflib import EdfReader
from pathlib import Path
import json
from collections import Counter, defaultdict
import numpy as np

def scan_dir_psg(p):
    rows = []
    for fpath in sorted(Path(p).glob("*-PSG.edf")):
        with EdfReader(str(fpath)) as f:
            labels = f.getSignalLabels()
            sfreqs = [f.getSampleFrequency(i) for i in range(f.signals_in_file)]
            rows.append({
                "file": fpath.name,
                "n_channels": f.signals_in_file,
                "labels": labels,
                "sfreqs": sfreqs,
                "start": f.getStartdatetime(),
                "duration_s": f.getFileDuration(),
            })
    return rows

def _decode_label(x):
    return x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x)

def scan_dir_hyp(p, preview_n=10):
    """
    Lê todos os *Hypnogram.edf* em p e retorna:
      - preview dos eventos (onset, duration, label)
      - labels únicas e contagem
      - duração modal dos eventos (útil pra ver 30s)
    """
    rows = []
    for fpath in sorted(Path(p).glob("*Hypnogram.edf")):
        with EdfReader(str(fpath)) as f:
            start = f.getStartdatetime()
            duration_s = f.getFileDuration()

            onsets, durations, descs = f.readAnnotations()
            descs = [_decode_label(d) for d in descs]

            preview = []
            for i in range(min(preview_n, len(onsets))):
                preview.append({
                    "onset_s": float(onsets[i]),
                    "duration_s": float(durations[i]),
                    "label": descs[i]
                })

            label_counts = Counter(descs)
            dur_rounded = [round(float(d), 1) for d in durations]
            dur_counts = Counter(dur_rounded)
            modal_dur = None
            if dur_counts:
                modal_dur = sorted(dur_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

            rows.append({
                "file": fpath.name,
                "n_events": len(onsets),
                "start": start,
                "duration_s": duration_s,
                "labels_unique": sorted(list(label_counts.keys())),
                "label_counts": dict(label_counts),
                "duration_modal_s": modal_dur,
                "preview": preview,
            })
    return rows

def fingerprint_psg(rows_psg):
    fingerprints = {}
    for r in rows_psg:
        key = json.dumps({"labels": r["labels"], "sfreqs": r["sfreqs"]}, ensure_ascii=False)
        fingerprints.setdefault(key, []).append(r["file"])
    return fingerprints

def fingerprint_hyp_by_labelset(rows_hyp):
    """
    Agrupa Hypnograms por conjunto de labels (útil pra ver variantes de codificação).
    """
    fp = defaultdict(list)
    for r in rows_hyp:
        key = json.dumps(sorted(r["labels_unique"]), ensure_ascii=False)
        fp[key].append(r["file"])
    return fp

def plot_hypnogram_edf(edf_path, epoch_sec_guess=30):
    """
    Plot simples (em escada) só pra *um* Hypnogram.edf.
    Descomente a chamada no main se quiser ver.
    """
    import matplotlib.pyplot as plt

    with EdfReader(str(edf_path)) as f:
        onsets, durations, descs = f.readAnnotations()
        descs = [_decode_label(d) for d in descs]

    map_stage = {
        'W': 4, 'N1': 3, 'N2': 2, 'N3': 1, 'R': 5,
        'REM': 5, 'NR': 2, 'NREM2': 2, 'NREM3': 1, 'NREM1': 3,
        'Stage 1 sleep': 3, 'Stage 2 sleep': 2, 'Stage 3 sleep': 1,
        'S1': 3, 'S2': 2, 'S3': 1, 'S4': 1, 'Artefact': np.nan, 'Movement time': np.nan
    }

    t = [0.0]
    y = [np.nan]
    for onset, dur, lab in zip(onsets, durations, descs):
        t.extend([float(onset), float(onset + dur)])
        val = map_stage.get(lab, np.nan)
        y.extend([val, val])

    plt.figure(figsize=(10, 3))
    plt.step(t, y, where='post')
    plt.yticks([1,2,3,4,5], ['N3','N2','N1','W','R'])
    plt.xlabel("Tempo (s)")
    plt.ylabel("Estágio")
    plt.title(Path(edf_path).name)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    base = Path(__file__).resolve().parents[0]
    p_cassette = base / "sleep-cassette"
    p_telemetry = base / "sleep-telemetry"

    rows_psg = scan_dir_psg(p_cassette) + scan_dir_psg(p_telemetry)
    fingerprints_psg = fingerprint_psg(rows_psg)

    print("\n=== PSG: Esquemas únicos (labels+fs) ===")
    for i, (k, files) in enumerate(fingerprints_psg.items(), 1):
        schema = json.loads(k)
        print(f"\n[{i}] {len(files)} arquivos")
        print("labels:", schema["labels"])
        print("fs:", schema["sfreqs"])
        print("exemplos:", files[:5])

    rows_hyp = scan_dir_hyp(p_cassette) + scan_dir_hyp(p_telemetry)

    print("\n=== HYP: Preview por arquivo ===")
    for r in rows_hyp:
        print(f"\n- {r['file']} | eventos={r['n_events']} | duração={r['duration_s']}s | modal_dur={r['duration_modal_s']}s")
        print("  labels únicas:", r["labels_unique"])
        if r["preview"]:
            print("  preview (primeiros eventos):")
            for ev in r["preview"]:
                print(f"    t={ev['onset_s']:>8.1f}s  dur={ev['duration_s']:>6.1f}s  label={ev['label']}")

    fp_hyp = fingerprint_hyp_by_labelset(rows_hyp)
    print("\n=== HYP: Conjuntos de labels únicos (fingerprint de anotações) ===")
    for i, (k, files) in enumerate(fp_hyp.items(), 1):
        labels = json.loads(k)
        print(f"\n[{i}] {len(files)} arquivos | labels: {labels}")
        print("exemplos:", files[:5])