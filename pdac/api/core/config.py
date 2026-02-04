from pathlib import Path
import yaml

def get_cfg():
    cfg_path = Path("config.yaml")  # repo root
    return yaml.safe_load(cfg_path.read_text())

def project_root(cfg) -> Path:
    return Path(cfg["data"].get("project_root", ".")).resolve()

def artifacts_dir(cfg) -> Path:
    return (project_root(cfg) / cfg["data"].get("artifacts_dir", ".")).resolve()

def models_dir(cfg) -> Path:
    return (project_root(cfg) / cfg["data"].get("models_dir", ".")).resolve()

def fusion_graph_path(cfg) -> Path:
    return artifacts_dir(cfg) / cfg["data"]["fusion_graph_pt"]

def model_ckpt_path(cfg) -> Path:
    return models_dir(cfg) / cfg["data"]["gnn_model_ckpt"]

def molecular_embedder_path(cfg) -> Path:
    return artifacts_dir(cfg) / cfg["data"]["molecular_embedder_ckpt"]

def molecular_feature_cols_path(cfg) -> Path:
    return artifacts_dir(cfg) / cfg["data"]["molecular_feature_cols_txt"]
