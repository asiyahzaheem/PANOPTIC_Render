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
    # Do NOT load predictor or ResNet18 at startup — stay under 512MB.
    # On each /predict we: load embedder -> embed TSV -> unload embedder -> load CT model -> load GNN.
    log.info("Startup: pre-downloading molecular_embedder.pt only (~116MB if missing)...")
    try:
        from pdac.api.services.molParser import _ensure_embedder_checkpoint_exists
        _ensure_embedder_checkpoint_exists()
        log.info("Startup: molecular_embedder file ready OK")
    except Exception as e:
        log.warning(
            f"Startup: molecular_embedder pre-download failed (will retry on first predict): {e}. "
            "If using gene/value TSV, ensure Google Drive file is shared 'Anyone with the link'."
        )
    log.info("Startup: done (predictor and ResNet18 load on first /predict to save memory)")


app.include_router(health_router)
app.include_router(predict_router)

# Google Drive file id for molecular_embedder.pt
MOLECULAR_EMBEDDER_FILE_ID = os.getenv("MOLECULAR_EMBEDDER_FILE_ID", "1MrtrRjWYka9qdsu862mXPOvuErEtkuxL")


