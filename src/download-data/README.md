# Sleep-EDF Download API

API for automatic download of the **Sleep-EDF** dataset files from the [PhysioNet](https://physionet.org/) repository.

## 🚀 Installation and Execution

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Access the interactive documentation: http://127.0.0.1:8000/docs

## 📋 API Routes

### GET `/download/`
Downloads the Sleep-EDF files.

**Parameters:**
- `subset` (string): `"cassette"`, `"telemetry"`, or `"both"` (default: `"cassette"`)
- `ignore_hash` (bool): Ignore file hash verification (default: `false`)
- `limit` (int): Limit the number of files to download (optional)
- `sync` (bool): Synchronous execution (blocks until complete) (default: `true`)
- `workers` (int): Number of parallel workers (1-8) (optional)
- `batch_size` (int): Batch size per worker (optional)
- `round_retries` (int): Retry attempts (0-8) (default: `3`)

### GET `/status/`
Checks the status of local files.

**Parameters:**
- `subset` (string): `"cassette"`, `"telemetry"`, or `"both"` (default: `"both"`)

## 💻 Terminal Usage Examples

### Check Status
```bash
# Status of both subsets
curl "http://localhost:8000/status/"

# Status of sleep-cassette only
curl "http://localhost:8000/status/?subset=cassette"

# Status of sleep-telemetry only
curl "http://localhost:8000/status/?subset=telemetry"
```

### Basic Download
```bash
# Download sleep-cassette (default)
curl "http://localhost:8000/download/"

# Download sleep-telemetry
curl "http://localhost:8000/download/?subset=telemetry"

# Download both subsets
curl "http://localhost:8000/download/?subset=both"
```

### Download with 6 Workers

#### Sleep-Cassette with 6 Workers
```bash
# Synchronous (blocks until complete)
curl "http://localhost:8000/download/?subset=cassette&workers=6"

# Asynchronous (non-blocking)
curl "http://localhost:8000/download/?subset=cassette&workers=6&sync=false"

# With custom batch_size
curl "http://localhost:8000/download/?subset=cassette&workers=6&batch_size=10"
```

#### Sleep-Telemetry with 6 Workers
```bash
# Synchronous
curl "http://localhost:8000/download/?subset=telemetry&workers=6"

# Asynchronous
curl "http://localhost:8000/download/?subset=telemetry&workers=6&sync=false"

# With custom retries
curl "http://localhost:8000/download/?subset=telemetry&workers=6&round_retries=5"
```

#### Both Subsets with 6 Workers
```bash
# Complete download with 6 workers
curl "http://localhost:8000/download/?subset=both&workers=6"

# Asynchronous with custom settings
curl "http://localhost:8000/download/?subset=both&workers=6&batch_size=8&sync=false"
```

### Advanced Examples
```bash
# Ignore hash verification
curl "http://localhost:8000/download/?subset=cassette&workers=6&ignore_hash=true"

# Limit to 50 files
curl "http://localhost:8000/download/?subset=telemetry&workers=6&limit=50"

# Complete configuration
curl "http://localhost:8000/download/?subset=both&workers=6&batch_size=12&round_retries=2&sync=false&ignore_hash=false"
```

## 📁 File Structure

Files are saved in:
```
datalake/raw/
├── sleep-cassette/ # Cassette subset files
└── sleep-telemetry/ # Telemetry subset files
```

## 📊 API Response

### Synchronous Download (sync=true)
```json
{ 
  "status": "completed", 
  "subset": "cassette", 
  "ignore_hash": false, 
  "plan": [ 
    { 
      "subset": "cassette", 
      "missing_count": 153, 
      "missing_samples": ["SC4001E0", "SC4002E0", ...], 
      "base_url": "https://physionet.org/..." 
    } 
  ], 
  "results": { 
  "cassette": ["Downloaded: SC4001E0-PSG.edf", ...] 
  }, 
  "raw_dir": "/path/to/datalake/raw"
}
```

### Asynchronous Download (sync=false)
```json
{ 
  "status": "started", 
  "subset": "cassette", 
  "ignore_hash": false, 
  "plan": [ 
    { 
      "subset": "cassette", 
      "missing_count": 153, 
      "missing_samples": ["SC4001E0", "SC4002E0", ...], 
      "base_url": "https://physionet.org/..." 
    } 
  ], 
  "raw_dir": "/path/to/datalake/raw"
}
```

## ⚡ Performance

- **Workers**: Controls parallelism (recommended: 4-6)
- **Batch Size**: Files per worker per batch (recommended: 8-12)
- **Sync vs Async**: Use `sync=false` for long downloads
- **Hash Check**: Use `ignore_hash=true` to skip checking (faster)