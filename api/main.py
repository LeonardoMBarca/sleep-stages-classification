#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI, Query, BackgroundTasks
from typing import Literal, List, Dict
from pathlib import Path
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from urllib.parse import urljoin

app = FastAPI(title="Sleep-EDFx Downloader API")

# -----------------------
# Config
# -----------------------
BASE_ROOT = "https://physionet.org/files/sleep-edfx/1.0.0/"
SUBSETS = {
    "cassette": "sleep-cassette/",
    "telemetry": "sleep-telemetry/",
}
FILE_REGEX = re.compile(r"(PSG\.edf|Hypnogram\.edf)$", re.IGNORECASE)

# Raiz do projeto = .../sleep-stages-classification (um nível acima de /api)
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "datalake" / "raw"          # fora da pasta /api
LOCAL_SUBDIRS = {"cassette": "cassette", "telemetry": "telemetry"}  # dentro de ./datalake/raw
WORKERS = 8
CHUNK = 1 << 20

HTTP_DEFAULT_TIMEOUT = 60
HTTP_HEADERS = {"User-Agent": "sleep-edfx-downloader/1.0 (+fastapi)"}

# -----------------------
# Helpers de diretório
# -----------------------
def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for sub in LOCAL_SUBDIRS.values():
        (RAW_DIR / sub).mkdir(parents=True, exist_ok=True)

def local_files_for_subset(subset: str) -> set:
    """Retorna o conjunto de nomes de arquivos existentes no subdir local (cassette/telemetry)."""
    subdir = RAW_DIR / LOCAL_SUBDIRS[subset]
    if not subdir.is_dir():
        return set()
    return {fn.name for fn in subdir.iterdir() if fn.is_file() and FILE_REGEX.search(fn.name)}

# -----------------------
# Listagem remota
# -----------------------
def fetch_listing(base_url: str) -> List[str]:
    r = requests.get(base_url, timeout=HTTP_DEFAULT_TIMEOUT, headers=HTTP_HEADERS)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    links = [a.get("href", "") for a in soup.find_all("a", href=True)]
    # Mantém apenas arquivos do padrão desejado
    files = [f for f in links if FILE_REGEX.search(f)]
    return files

# -----------------------
# Download (com resume)
# -----------------------
def download_file(base_url: str, subset: str, fname: str) -> str:
    url = urljoin(base_url, fname)
    dest_dir = RAW_DIR / LOCAL_SUBDIRS[subset]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / fname

    headers = dict(HTTP_HEADERS)
    pos = 0
    if dst.exists():
        # Se já existe (mesmo que parcial), fazemos resume
        pos = dst.stat().st_size
        if pos > 0:
            headers["Range"] = f"bytes={pos}-"

    try:
        with requests.get(url, headers=headers, stream=True, timeout=HTTP_DEFAULT_TIMEOUT * 2) as r:
            if r.status_code not in (200, 206):
                return f"ERR {subset}/{fname} (HTTP {r.status_code})"
            mode = "ab" if pos > 0 else "wb"
            with open(dst, mode) as f:
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if chunk:
                        f.write(chunk)
        return f"OK {subset}/{fname}"
    except requests.exceptions.RequestException as e:
        return f"ERR {subset}/{fname} ({type(e).__name__}: {e})"

# -----------------------
# Orquestração por subset
# -----------------------
def plan_missing(subset: str) -> Dict[str, List[str] | str]:
    """Compara remoto x local e devolve a lista de faltantes para o subset."""
    subset_dir_remote = SUBSETS[subset]
    base_url = urljoin(BASE_ROOT, subset_dir_remote)
    remote = set(fetch_listing(base_url))
    local = local_files_for_subset(subset)
    missing = sorted(remote - local)
    return {"subset": subset, "base_url": base_url, "missing": missing}

def run_download_for_subset(subset: str, missing: List[str]) -> List[str]:
    """Baixa apenas os arquivos faltantes do subset."""
    if not missing:
        return [f"SKIP {subset}: todos os arquivos já existem"]

    subset_dir_remote = SUBSETS[subset]
    base_url = urljoin(BASE_ROOT, subset_dir_remote)

    results = []
    # Pool local por chamada; encerra automaticamente ao final
    with ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix=f"dwl-{subset}") as ex:
        futures = [ex.submit(download_file, base_url, subset, f) for f in missing]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results

# -----------------------
# Endpoints
# -----------------------
@app.get("/download")
async def download(
    background_tasks: BackgroundTasks,
    subset: Literal["cassette", "telemetry", "both"] = Query("cassette"),
):
    """
    Dispara o download dos arquivos (PSG/Hypnogram) — **apenas os que faltam**.
    - subset=cassette   -> baixa só o que falta em ./datalake/raw/cassette
    - subset=telemetry  -> baixa só o que falta em ./datalake/raw/telemetry
    - subset=both       -> processa os dois
    Retorna o plano (quais faltam) e executa em background.
    """
    ensure_dirs()

    # Planejamento
    if subset == "both":
        plans = [plan_missing("cassette"), plan_missing("telemetry")]
    else:
        plans = [plan_missing(subset)]

    # Execução assíncrona (não bloqueia o request)
    def task(plans_):
        for p in plans_:
            try:
                run_download_for_subset(p["subset"], p["missing"])  # logs retornam dentro do pool
            except Exception as e:  # proteção adicional para não “vazar” exceções no shutdown
                print(f"[download-task] erro no subset {p['subset']}: {e}")

    background_tasks.add_task(task, plans)

    # Retorna o plano de baixa (visibilidade imediata)
    return {
        "status": "started",
        "subset": subset,
        "plan": [
            {
                "subset": p["subset"],
                "missing_count": len(p["missing"]),
                "missing_samples": p["missing"][:10],  # amostra p/ resposta curta
            }
            for p in plans
        ],
        "raw_dir": str(RAW_DIR),
    }

@app.get("/status")
def status(subset: Literal["cassette", "telemetry", "both"] = Query("both")):
    """Mostra quantos arquivos já existem localmente por subset."""
    ensure_dirs()
    if subset == "both":
        return {
            "cassette": {"count": len(local_files_for_subset("cassette"))},
            "telemetry": {"count": len(local_files_for_subset("telemetry"))},
        }
    else:
        return {subset: {"count": len(local_files_for_subset(subset))}}
