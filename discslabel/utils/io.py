"""File-system helpers."""

import functools
import os
import subprocess


def check_and_make_dir(dir_path: str) -> None:
    """Create ``dir_path`` if it does not exist. Error if the path is a file."""
    if os.path.exists(dir_path):
        if os.path.isfile(dir_path):
            raise ValueError(f"Error, the provided path ({dir_path}) is a file path, not a folder path.")
    else:
        os.makedirs(dir_path, exist_ok=True)


@functools.lru_cache(maxsize=1)
def code_version() -> str:
    """Commit of the running code: ``git rev-parse HEAD`` of the repo checkout,
    with a ``-dirty`` suffix when tracked files have uncommitted changes.

    Falls back to the installed package version (no git checkout), else ``"unknown"``.
    """
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", repo, *args], capture_output=True, text=True, timeout=10
        )

    try:
        head = git("rev-parse", "HEAD")
        if head.returncode == 0:
            dirty = git("diff", "--quiet", "HEAD").returncode == 1
            return head.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        from importlib.metadata import version

        return version("discslabel")
    except Exception:
        return "unknown"


def write_prediction_sidecar(nifti_path: str, weights_name: str) -> None:
    """Write the BIDS-style JSON sidecar next to a predicted NIfTI.

    ``<name>.nii.gz`` → ``<name>.json``, recording which weights and which
    code version (``code_version``) produced it.
    """
    import json
    from datetime import datetime

    base = nifti_path
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    sidecar = {
        "SpatialReference": "orig",
        "GeneratedBy": [
            {
                "Name": "discslabel prediction",
                "Version": code_version(),
                "weights": weights_name,
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ],
    }
    with open(base + ".json", "w") as f:
        json.dump(sidecar, f, indent=4)
