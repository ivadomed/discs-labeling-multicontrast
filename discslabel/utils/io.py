"""File-system helpers."""

import os


def check_and_make_dir(dir_path: str) -> None:
    """Create ``dir_path`` if it does not exist. Error if the path is a file."""
    if os.path.exists(dir_path):
        if os.path.isfile(dir_path):
            raise ValueError(f"Error, the provided path ({dir_path}) is a file path, not a folder path.")
    else:
        os.makedirs(dir_path, exist_ok=True)
