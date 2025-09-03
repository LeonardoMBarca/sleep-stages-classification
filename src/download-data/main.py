from fastapi import FastAPI, Request
from routes import download, status
import logging
import sys
import uvicorn

def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

app = FastAPI(title="Sleep-EDFx Downloader API")

_setup_logging()
logger = logging.getLogger("app")

@app.middleware("http")
async def add_logging(request: Request, call_next):
    logger.info(f"REQ {request.method} {request.url.path} {request.query_params}")
    try:
        resp = await call_next(request)
        logger.info(f"RESP {request.method} {request.url.path} {resp.status_code}")
        return resp
    except Exception as e:
        logger.exception(f"EXC {request.method} {request.url.path}: {type(e).__name__}: {e}")
        raise

app.include_router(download.router, prefix="/download", tags=["download"])
app.include_router(status.router, prefix="/status", tags=["status"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
