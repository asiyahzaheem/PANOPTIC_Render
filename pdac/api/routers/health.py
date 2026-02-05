from fastapi import APIRouter
from pdac.api.core.config import get_cfg, fusion_graph_path, model_ckpt_path

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/info")
def info():
    """Quick reference: how to call /predict."""
    return {
        "predict": {
            "method": "POST",
            "url": "/predict",
            "content_type": "multipart/form-data",
            "required": ["ct_file (.nii/.nii.gz)", "molecular_file (.tsv)"],
            "optional": ["explain (none|simple|detailed)"],
            "curl": "curl -X POST <base_url>/predict -F ct_file=@scan.nii.gz -F molecular_file=@data.tsv -F explain=simple",
        },
        "health": {"method": "GET", "url": "/health"},
        "debug": {"method": "GET", "url": "/debug"},
    }


@router.get("/debug")
def debug():
    """Verify model and graph paths exist (confirms GNN is loaded)."""
    cfg = get_cfg()
    fg = fusion_graph_path(cfg)
    mp = model_ckpt_path(cfg)
    return {
        "fusion_graph_path": str(fg),
        "fusion_graph_exists": fg.exists(),
        "model_path": str(mp),
        "model_exists": mp.exists(),
        "temperature": cfg.get("gnn", {}).get("temperature", 1.0),
    }

