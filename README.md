# tech-callenge-3-fiap

## Sleep-EDF Tabular Dataset (PSG + Hypnograma)
A **PSG** é o exame padrão ouro do sono, registrando sinais fisiológicos (EEG, EOG, EMG, respiração, temperatura, etc.).  
O **hipnograma** é a anotação das fases do sono em blocos de **30 segundos**.  

O script [`build_tabular_dataset`](./build_tabular_dataset.py) processa arquivos brutos (`*-PSG.edf` e `*-Hypnogram.edf`) e gera um dataset consolidado em **Parquet** (ou CSV).  
Cada **linha do dataset representa uma EPOCH de 30s**, contendo métricas derivadas dos sinais fisiológicos e o estágio do sono correspondente.  
## Estrutura do Script
```python
import argparse
...
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sleep-EDF → epochs tabulares (todos os canais, logs, SC/ST robusto)")
    ap.add_argument("--root", required=True, help="Pasta raiz (ex.: .../sleep-edfx/1.0.0/sleep-cassette ou .../sleep-edfx/1.0.0)")
    ap.add_argument("--out", required=True, help="Caminho de saída (ex.: sc_epochs.parquet)")
    ap.add_argument("--csv", action="store_true", help="Salvar CSV (padrão: Parquet)")
    args = ap.parse_args()
    build_tabular_dataset(args.root, args.out, to_csv=args.csv)
```
---
### Identificação e Estrutura
| Coluna       | Descrição                                                         |
| ------------ | ----------------------------------------------------------------- |
| `subject_id` | Identificador do indivíduo avaliado (ex.: `SC4002`).              |
| `night_id`   | Identificador da noite (`E0`, `E1`, etc.).                        |
| `epoch_idx`  | Índice da época (0, 1, 2…). Cada época dura 30s.                  |
| `t0_sec`     | Tempo inicial em segundos desde o início da gravação.             |
| `stage`      | Estágio do sono: `W` (vigília), `N1`/`N2`/`N3` (NREM), `R` (REM). |

### EEG – Atividade Cerebral
Canais: Fpz–Cz e Pz–Oz
| Coluna        | Descrição                                                    |
| ------------- | ------------------------------------------------------------ |
| `*_delta_pow` | Potência em ondas delta (0.5–4 Hz) → sono profundo.          |
| `*_theta_pow` | Potência em ondas teta (4–8 Hz) → sonolência/início do sono. |
| `*_alpha_pow` | Potência em ondas alfa (8–12 Hz) → relaxamento.              |
| `*_sigma_pow` | Potência em fusos de sono (12–16 Hz) → marcadores de N2.     |
| `*_beta_pow`  | Potência em ondas beta (16–30 Hz) → alerta.                  |
| `*_rms`       | Intensidade média do sinal (Root Mean Square).               |
| `*_var`       | Variância do sinal.                                          |

### EOG – Movimentos Oculares
| Coluna                     | Descrição                          |
| -------------------------- | ---------------------------------- |
| `eog_horizontal_delta_pow` | Potência faixa delta.              |
| `eog_horizontal_theta_pow` | Potência faixa teta.               |
| `eog_horizontal_alpha_pow` | Potência faixa alfa.               |
| `eog_horizontal_sigma_pow` | Potência em fusos.                 |
| `eog_horizontal_beta_pow`  | Potência faixa beta.               |
| `eog_horizontal_rms`       | Intensidade média do sinal ocular. |
| `eog_horizontal_var`       | Variância do sinal ocular.         |

### Respiração (Oro-nasal)
| Coluna                     | Descrição                        |
| -------------------------- | -------------------------------- |
| `resp_oro_nasal_mean_1hz`  | Média da respiração na época.    |
| `resp_oro_nasal_std_1hz`   | Variação respiratória.           |
| `resp_oro_nasal_min_1hz`   | Fluxo mínimo detectado.          |
| `resp_oro_nasal_max_1hz`   | Fluxo máximo detectado.          |
| `resp_oro_nasal_slope_1hz` | Tendência (ex.: queda → apneia). |
| `resp_oro_nasal_rms_1hz`   | Intensidade média da respiração. |

### EMG – Atividade Muscular (Submentoniano)
| Coluna                    | Descrição                   |
| ------------------------- | --------------------------- |
| `emg_submental_mean_1hz`  | Média do tônus muscular.    |
| `emg_submental_std_1hz`   | Variação do tônus.          |
| `emg_submental_min_1hz`   | Valor mínimo detectado.     |
| `emg_submental_max_1hz`   | Valor máximo detectado.     |
| `emg_submental_slope_1hz` | Tendência de variação.      |
| `emg_submental_rms_1hz`   | Intensidade média do tônus. |

### Temperatura Corporal
| Coluna                  | Descrição                         |
| ----------------------- | --------------------------------- |
| `temp_rectal_mean_1hz`  | Temperatura média.                |
| `temp_rectal_std_1hz`   | Variação térmica.                 |
| `temp_rectal_min_1hz`   | Mínimo registrado.                |
| `temp_rectal_max_1hz`   | Máximo registrado.                |
| `temp_rectal_slope_1hz` | Tendência de variação.            |
| `temp_rectal_rms_1hz`   | Intensidade média da temperatura. |

### Event Marker
| Coluna                   | Descrição                       |
| ------------------------ | ------------------------------- |
| `event_marker_mean_1hz`  | Média dos eventos no intervalo. |
| `event_marker_std_1hz`   | Variação dos eventos.           |
| `event_marker_min_1hz`   | Valor mínimo registrado.        |
| `event_marker_max_1hz`   | Valor máximo registrado.        |
| `event_marker_slope_1hz` | Tendência (aumento/diminuição). |
| `event_marker_rms_1hz`   | Intensidade média dos eventos.  |

---
