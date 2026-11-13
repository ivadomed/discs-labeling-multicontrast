"""File-system helpers."""

import os


def check_and_make_dir(dir_path: str) -> None:
    """Create ``dir_path`` if it does not exist. Error if the path is a file."""
    if os.path.exists(dir_path):
        if os.path.isfile(dir_path):
            raise ValueError(f"Error, the provided path ({dir_path}) is a file path, not a folder path.")
    else:
        os.makedirs(dir_path, exist_ok=True)


def write_prediction_sidecar(nifti_path: str, weights_name: str) -> None:
    """Write the BIDS-style JSON sidecar next to a predicted NIfTI.

    ``<name>.nii.gz`` → ``<name>.json``, recording which weights produced it.
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
                "weights": weights_name,
                "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ],
    }
    with open(base + ".json", "w") as f:
        json.dump(sidecar, f, indent=4)
