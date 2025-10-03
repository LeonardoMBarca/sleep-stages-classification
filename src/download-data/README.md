# Sleep-EDF Download Service

This module exposes a FastAPI service that plans and downloads Sleep-EDFx recordings from PhysioNet. It handles resume, transient failures, and hash-based deduplication so you can safely re-run the ingestion step without clobbering previously processed files.

---

## Quick Start

```bash
cd src/download-data
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open the interactive docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore the available endpoints.

> **Tip:** The service writes to `datalake/raw/` relative to the project root. Make sure the directory exists or let the API create it on first run.

---

## Endpoints

### `GET /download`

Plan the download for a subset and optionally execute it synchronously.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subset` | `cassette` \| `telemetry` \| `both` | `cassette` | Which Sleep-EDFx cohort to fetch |
| `ignore_hash` | bool | `false` | Ignore processed-file hashes (forces redownload) |
| `limit` | int | — | Limit number of files per subset |
| `sync` | bool | `true` | Block until completion; set `false` to run in background |
| `workers` | int (1–8) | auto | Number of parallel download workers |
| `batch_size` | int | auto | How many files each round dispatches |
| `round_retries` | int (0–8) | `3` | How many retry rounds for transient errors |

Synchronous calls return a full log of downloaded files; asynchronous calls return immediately with the planned work so you can monitor separately.

### `GET /status`

Report which files already exist locally and which are still missing.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subset` | `cassette` \| `telemetry` \| `both` | `both` | Which cohorts to inspect |

---

## File Layout

Downloaded artefacts follow this structure:

```
datalake/
└── raw/
    ├── sleep-cassette/    # PSG & annotation EDF files for SC*
    └── sleep-telemetry/   # PSG & annotation EDF files for ST*
```

Processed layers and modelling-ready parquets are produced by the `data-processing` pipeline (see project root README for next steps).

---

## Example Calls

```bash
# Fetch the first 50 telemetry recordings using 6 workers
curl "http://localhost:8000/download?subset=telemetry&limit=50&workers=6"

# Start a background job for both subsets, custom batch size
curl "http://localhost:8000/download?subset=both&workers=6&batch_size=12&sync=false"

# Inspect which cassette nights remain missing
curl "http://localhost:8000/status?subset=cassette"
```

---

## Implementation Notes

- The planner scrapes PhysioNet’s directory listing, compares it with local files, and checks previously processed hashes so you never redownload files that were already converted to parquet.
- Downloads use `requests` with connection pooling, partial downloads, and exponential backoff. Common transient HTTP errors (429/5xx, timeouts) trigger automatic retries.
- Logs appear in stdout detailing progress (percentage, MB/s) and final throughput.

For deeper context on how these raw files feed the modelling pipeline, refer back to the project-level [README](../../README.md).
