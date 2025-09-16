from pyedflib import EdfReader
import numpy as np

EDF_PATH = "/home/leona/workspace/github-projets/sleep-stages-classification/datalake/raw/sleep-telemetry/ST7011J0-PSG.edf"

with EdfReader(EDF_PATH) as f:
    n_signals = f.signals_in_file
    labels = f.getSignalLabels()
    sfreqs = [f.getSampleFrequency(i) for i in range(n_signals)]
    start_dt = f.getStartdatetime()

    print("Canais:", labels)
    print("Frequências de amostragem por canal:", sfreqs)
    print("Início da gravação:", start_dt)

    ch = 0
    sfreq = sfreqs[ch]
    n_samples = int(sfreq * 10)

    x = f.readSignal(ch, start=0, n=n_samples).astype(float)
    t = np.arange(n_samples) / sfreq