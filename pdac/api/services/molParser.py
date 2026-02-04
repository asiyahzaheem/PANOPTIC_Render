from __future__ import annotations

from pathlib import Path
import os
import threading
import numpy as np
import pandas as pd
import torch

from pdac.api.utils.artifacts import ensure_gdrive_file
from pdac.api.core.config import get_cfg, molecular_feature_cols_path
from pdac.molecular.embeddingModel import MolecularEmbedder

# ---- global cache (per warm serverless instance)
_EMBEDDER_CACHE: dict[str, MolecularEmbedder] = {}
_CACHE_LOCK = threading.Lock()


def _load_feature_cols(cfg: dict) -> list[str]:
    cols_path = molecular_feature_cols_path(cfg)
    if not cols_path.exists():
        raise ValueError(
            f"Missing {cols_path}. Ensure molecular_feature_cols.txt is included in the repo "
            f"(and molecular_feature_cols_path(cfg) points to it)."
        )
    cols = [c.strip() for c in cols_path.read_text().splitlines() if c.strip()]
    if not cols:
        raise ValueError("molecular_feature_cols.txt is empty.")
    return cols


def _parse_embedding_tsv(df: pd.DataFrame) -> np.ndarray:
    """
    TSV with a single row and columns emb_0..emb_255 (or any emb_i numeric).
    """
    emb_cols = [c for c in df.columns if c.startswith("emb_") and c.split("_")[1].isdigit()]
    if not emb_cols:
        raise ValueError("Embedding TSV must have emb_0..emb_255 columns (single row).")
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_")[1]))
    if len(df) != 1:
        raise ValueError("Embedding TSV must contain exactly 1 row.")
    return df.loc[0, emb_cols].to_numpy(dtype=np.float32)


def _parse_expression_gene_value_tsv(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    Raw expression TSV format:
      columns: gene, value
      rows: gene symbols with expression values

    Returns vector in the exact order of feature_cols.
    """
    if "gene" not in df.columns or "value" not in df.columns:
        raise ValueError("Expression TSV must have columns: gene, value")

    # Map gene -> value
    g2v = dict(zip(df["gene"].astype(str), df["value"].astype(float)))

    x = np.zeros((len(feature_cols),), dtype=np.float32)
    for i, g in enumerate(feature_cols):
        if g in g2v:
            x[i] = float(g2v[g])
    return x


def _get_embedder_ckpt_path() -> Path:
    """
    Vercel-safe storage location (writable, not bundled into deployment).
    """
    return Path(os.getenv("MOLECULAR_EMBEDDER_LOCAL_PATH", "/tmp/molecular_embedder.pt"))


def _ensure_embedder_checkpoint_exists() -> Path:
    """
    Lazy runtime download: only downloads during a request.
    Never download at import-time.
    """
    ckpt_path = _get_embedder_ckpt_path()
    if ckpt_path.exists():
        return ckpt_path

    file_id = os.getenv("MOLECULAR_EMBEDDER_FILE_ID", "1MrtrRjWYka9qdsu862mXPOvuErEtkuxL")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_gdrive_file(file_id, ckpt_path)
    return ckpt_path


def _load_embedder(cfg: dict, input_dim: int, device: torch.device) -> MolecularEmbedder:
    """
    Loads (and caches) MolecularEmbedder.
    - checkpoint stored in /tmp
    - lazy-download only when needed
    - cached across requests within the same warm instance
    """
    emb_dim = int(cfg["molecular"]["embedding_dim"])
    hidden_dim = int(cfg["molecular"]["hidden_dim"])

    cache_key = f"{input_dim}:{emb_dim}:{hidden_dim}:{device.type}"

    # Fast path: cached
    if cache_key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[cache_key]

    with _CACHE_LOCK:
        # Double-check inside lock
        if cache_key in _EMBEDDER_CACHE:
            return _EMBEDDER_CACHE[cache_key]

        ckpt_path = _ensure_embedder_checkpoint_exists()
        ckpt = torch.load(ckpt_path, map_location="cpu")

        model = MolecularEmbedder(input_dim=input_dim, emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)

        # Robust loading for various save formats
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError(f"Unexpected checkpoint format in {ckpt_path}")

        model.load_state_dict(state, strict=True)
        model.eval()

        _EMBEDDER_CACHE[cache_key] = model
        return model


@torch.no_grad()
def load_molecular_embedding_from_uploaded_tsv(tsv_path: Path, cfg: dict | None = None) -> np.ndarray:
    """
    Returns a molecular embedding vector (256,) from either:
      1) TSV with emb_0..emb_255 (single row), OR
      2) TSV with gene/value pairs -> embeds using MolecularEmbedder.

    This function is Vercel-safe: any big checkpoint is downloaded lazily to /tmp.
    """
    if cfg is None:
        cfg = get_cfg()

    df = pd.read_csv(tsv_path, sep="\t")
    if len(df) < 1:
        raise ValueError("Uploaded molecular TSV is empty.")

    # Case 1: embedding TSV
    if any(c.startswith("emb_") for c in df.columns):
        return _parse_embedding_tsv(df)

    # Case 2: raw expression -> embed
    feature_cols = _load_feature_cols(cfg)
    x_expr = _parse_expression_gene_value_tsv(df, feature_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_embedder(cfg, input_dim=len(feature_cols), device=device)

    x = torch.tensor(x_expr, dtype=torch.float32, device=device).unsqueeze(0)  # [1, D]
    emb = model(x)[0].detach().cpu().numpy().astype(np.float32)               # [256]
    return emb
