from __future__ import annotations
import logging
import os
import threading
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from pdac.api.core.config import get_cfg, fusion_graph_path, model_ckpt_path
from pdac.api.schemas.predict import PredictResponse
from pdac.api.services.ctEmbedder import extract_ct_embedding_single
from pdac.api.services.molParser import load_molecular_embedding_from_uploaded_tsv
from pdac.api.services.predictorService import PredictorService
from pdac.api.services.uploadService import save_upload

log = logging.getLogger("pdac.api.predict")
router = APIRouter()

_upload_dir = Path(os.getenv("UPLOAD_DIR", "/tmp/panoptic_uploads"))
_upload_dir.mkdir(parents=True, exist_ok=True)

_predictor = None
_cfg = None
_lock = threading.Lock()

def get_predictor() -> PredictorService:
    global _predictor, _cfg
    if _predictor is not None:
        return _predictor

    with _lock:
        if _predictor is None:
            if _cfg is None:
                _cfg = get_cfg()
            _predictor = PredictorService(
                fusion_graph_pt=fusion_graph_path(_cfg),
                model_pt=model_ckpt_path(_cfg),
                cfg=_cfg,
            )
    return _predictor



@router.options("/predict")
def predict_options():
    log.info("OPTIONS /predict (CORS preflight)")
    return Response(status_code=204)


@router.get("/predict")
def predict_get():
    """Called when /predict is hit with GET (e.g. browser). POST is required."""
    log.warning("GET /predict - Method Not Allowed. Use POST with multipart/form-data.")
    return {
        "error": "Method Not Allowed",
        "message": "Use POST with multipart/form-data. Required: ct_file (.nii/.nii.gz), molecular_file (.tsv). Optional: explain (none|simple|detailed)",
        "example_curl": "curl -X POST <url>/predict -F ct_file=@scan.nii.gz -F molecular_file=@data.tsv -F explain=simple",
    }


@router.post("/predict", response_model=PredictResponse)
async def predict(
    ct_file: UploadFile = File(..., description="CT scan .nii or .nii.gz"),
    molecular_file: UploadFile = File(..., description="Molecular TSV with emb_0..emb_255"),
    explain: str = Form("simple", description="none|simple|detailed"),
):
    log.info("POST /predict started")
    predictor = get_predictor()
    cfg = _cfg or get_cfg()
    try:
        log.info("Saving uploads...")
        ct_path = save_upload(_upload_dir, ct_file)
        mol_path = save_upload(_upload_dir, molecular_file)
        log.info(f"Saved: ct={ct_path.name} mol={mol_path.name}")

        log.info("Loading molecular embedding...")
        emb_vec = load_molecular_embedding_from_uploaded_tsv(mol_path, cfg)
        log.info("Extracting CT embedding...")
        z_vec = extract_ct_embedding_single(ct_path, cfg)

        log.info("Running prediction...")
        result = predictor.predict(z_vec=z_vec, emb_vec=emb_vec, explain=explain)
        try:
            ct_path.unlink(missing_ok=True)
            mol_path.unlink(missing_ok=True)
        except Exception:
            pass

        log.info("POST /predict success")
        return result
    except Exception as e:
        log.exception(f"POST /predict failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

