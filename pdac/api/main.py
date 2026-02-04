import logging
import os
from pathlib import Path

from fastapi import FastAPI
from pdac.api.core.cors import add_cors
from pdac.api.routers.health import router as health_router
from pdac.api.routers.predict import router as predict_router
from pdac.api.utils.artifacts import ensure_gdrive_file

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

app = FastAPI(title="PANOPTIC PDAC API", version="1.0.0")
add_cors(app)

app.include_router(health_router)
app.include_router(predict_router)

# Google Drive file id for molecular_embedder.pt
MOLECULAR_EMBEDDER_FILE_ID = os.getenv("MOLECULAR_EMBEDDER_FILE_ID", "1MrtrRjWYka9qdsu862mXPOvuErEtkuxL")

@app.on_event("startup")
def _startup_download():
    # download into repo root to match your current layout
    ensure_gdrive_file(MOLECULAR_EMBEDDER_FILE_ID, Path("molecular_embedder.pt"))
