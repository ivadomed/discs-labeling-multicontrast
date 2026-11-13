"""Download and cache the pretrained weights from the GitHub releases.

The release asset is a zip of a training output folder (``config.json``,
``best_model.pt``, ``skeleton.json`` and optionally ``postprocessing.json``),
with or without a top-level folder. It is extracted once into
``<repo>/data/weights/<zip name>/`` and reused afterwards.

``DISCSLABEL_WEIGHTS_URL`` (or ``--weights-url`` on ``discslabel_inference``)
replaces the default release URL, e.g. to test a zip served locally with
``python -m http.server``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import tqdm


WEIGHTS_RELEASE = "r20261113"
WEIGHTS_ZIP = f"discslabel_weights_{WEIGHTS_RELEASE}.zip"
WEIGHTS_URL = (
    "https://github.com/ivadomed/discs-labeling-multicontrast/releases/download/"
    f"{WEIGHTS_RELEASE}/{WEIGHTS_ZIP}"
)
WEIGHTS_URL_ENV = "DISCSLABEL_WEIGHTS_URL"

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Repo checkout (editable install): <repo>/data/weights. Installed wheel: ~/.discslabel/weights.
WEIGHTS_CACHE_DIR = (
    _REPO_ROOT / "data" / "weights"
    if (_REPO_ROOT / "pyproject.toml").is_file()
    else Path.home() / ".discslabel" / "weights"
)


def _zip_stem(url: str) -> str:
    name = url.rstrip("/").split("/")[-1].split("?")[0]
    return name[: -len(".zip")] if name.endswith(".zip") else name


def _find_model_root(folder: Path) -> Path:
    """Folder holding ``config.json`` inside an extracted zip (top-level folder or not)."""
    hits = [
        p.parent
        for p in folder.rglob("config.json")
        if "__MACOSX" not in p.parts and not p.name.startswith("._")
    ]
    if not hits:
        raise FileNotFoundError("the weights zip does not contain a config.json")
    return min(hits, key=lambda p: len(p.parts))


def _download(url: str, dst: Path) -> None:
    print(f"[weights] downloading {url}")
    with urllib.request.urlopen(url) as resp, open(dst, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0) or None
        with tqdm.tqdm(total=total, unit="B", unit_scale=True, desc="weights") as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def ensure_weights(url: Optional[str] = None, cache_dir: Optional[Path] = None) -> str:
    """Path to the cached model folder, downloading and extracting it if needed.

    ``url`` defaults to ``$DISCSLABEL_WEIGHTS_URL``, else the release URL.
    Each zip name gets its own cache folder, so a different URL never reuses
    stale weights.
    """
    url = url or os.environ.get(WEIGHTS_URL_ENV) or WEIGHTS_URL
    cache_dir = Path(cache_dir or WEIGHTS_CACHE_DIR)
    target = cache_dir / _zip_stem(url)
    if (target / "config.json").is_file():
        return str(target)

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Download + extract in a temp folder next to the cache, then rename: an
    # interrupted run never leaves a half-written model folder behind.
    with tempfile.TemporaryDirectory(dir=cache_dir, prefix=".tmp_") as tmp:
        tmp = Path(tmp)
        zip_path = tmp / "weights.zip"
        try:
            _download(url, zip_path)
        except Exception as e:
            raise RuntimeError(
                f"could not download the weights from {url} ({e}). Pass --model-dir to use "
                f"a local folder, or set {WEIGHTS_URL_ENV} / --weights-url."
            ) from e
        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError(f"{url} is not a zip file.")
        extract_dir = tmp / "extracted"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        root = _find_model_root(extract_dir)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(root), str(target))
    print(f"[weights] installed into {target}")
    return str(target)
