from __future__ import annotations
from pathlib import Path
import logging
import gdown

log = logging.getLogger("artifacts")

def ensure_gdrive_file(file_id: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info(f"[BOOT] exists: {out_path}")
        return out_path

    url = f"https://drive.google.com/uc?id={file_id}"
    log.info(f"[BOOT] downloading from Google Drive -> {out_path}")
    gdown.download(url, str(out_path), quiet=False)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Download failed or empty: {out_path}")
    log.info(f"[BOOT] downloaded: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    return out_path
