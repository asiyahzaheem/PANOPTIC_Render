from __future__ import annotations
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response
import os
from pathlib import Path
from pdac.api.core.config import get_cfg
from pdac.api.services.uploadService import save_upload

from pdac.api.core.config import get_cfg, artifacts_dir, fusion_graph_path, model_ckpt_path
from pdac.api.services.uploadService import save_upload
from pdac.api.services.ctEmbedder import extract_ct_embedding_single
from pdac.api.services.predictorService import PredictorService
from pdac.api.schemas.predict import PredictResponse
from pdac.api.services.molParser import load_molecular_embedding_from_uploaded_tsv
from fastapi import APIRouter
from pathlib import Path
import os
import threading

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
    return Response(status_code=204)

@router.post("/predict", response_model=PredictResponse)
async def predict(
    ct_file: UploadFile = File(..., description="CT scan .nii or .nii.gz"),
    molecular_file: UploadFile = File(..., description="Molecular TSV with emb_0..emb_255"),
    explain: str = Form("simple", description="none|simple|detailed"),
):
    try:
        ct_path = save_upload(_upload_dir, ct_file)
        mol_path = save_upload(_upload_dir, molecular_file)

        emb_vec = load_molecular_embedding_from_uploaded_tsv(mol_path, _cfg)
        z_vec = extract_ct_embedding_single(ct_path, _cfg)

        result = _predictor.predict(z_vec=z_vec, emb_vec=emb_vec, explain=explain)
        try:
            ct_path.unlink(missing_ok=True)
            mol_path.unlink(missing_ok=True)
        except Exception:
            pass

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

