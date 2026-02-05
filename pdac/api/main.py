import logging
import os
import time
from pathlib import Path
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from pdac.api.core.cors import add_cors
from pdac.api.routers.health import router as health_router
from pdac.api.routers.predict import router as predict_router
from pdac.api.utils.artifacts import ensure_gdrive_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pdac.api")

app = FastAPI(title="PANOPTIC PDAC API", version="1.0.0")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request: method, path, content-type, and response status."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        method = request.method
        path = request.url.path
        content_type = request.headers.get("content-type", "(none)")
        log.info(f">>> INCOMING: {method} {path} | Content-Type: {content_type}")
        try:
            response = await call_next(request)
            elapsed = time.time() - start
            log.info(f"<<< RESPONSE: {method} {path} -> {response.status_code} ({elapsed:.2f}s)")
            return response
        except Exception as e:
            log.exception(f"!!! ERROR handling {method} {path}: {e}")
            raise


app.add_middleware(RequestLoggingMiddleware)
add_cors(app)


@app.on_event("startup")
def warmup():
    log.info("Startup: loading predictor model...")
    try:
        from pdac.api.routers.predict import get_predictor
        get_predictor()
        log.info("Startup: predictor loaded OK")
    except Exception as e:
        log.exception(f"Startup: predictor load FAILED: {e}")
        raise


app.include_router(health_router)
app.include_router(predict_router)

# Google Drive file id for molecular_embedder.pt
MOLECULAR_EMBEDDER_FILE_ID = os.getenv("MOLECULAR_EMBEDDER_FILE_ID", "1MrtrRjWYka9qdsu862mXPOvuErEtkuxL")


