"""Uncompressed, memory-mappable cache of the prepared training volumes.

Reading a prepared ``.nii.gz`` means gunzipping the whole volume on every
draw (~0.3 s for a whole-spine image), although a training sample only needs
one patch of it. The cache stores each case once, uncompressed, so the
dataset can memory-map it and read just the crop:

    <stem>_image.npy  float16, image / (max |image| / 1000)
    <stem>_seg.npy    uint8 bit mask, bit ``c`` = anatomy channel ``c``
                      (``discslabel.SEG_CHANNELS``); only when a SEG_* exists
    <stem>_meta.npz   shape, sparse point-wise label (coords + values), disc
                      foreground coords + ids for the crop sampler, and the
                      fingerprint of the source files

The image is rescaled before the float16 cast: the z-score normalisation
(``mean`` / ``std`` of the strictly positive voxels) is invariant to a
positive scale, so the training input is unchanged up to float16 rounding.

A case is rebuilt when its source files (path, size, mtime), the
``spine_disc_min_label`` threshold or ``CACHE_VERSION`` change. The meta file
is written last, so an interrupted build is redone on the next run.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import tqdm

CACHE_VERSION = 1
CACHE_DIRNAME = "cache"
IMAGE_SCALE_TARGET = 1000.0

_SOURCE_KEYS = ("image", "label", "seg_spine", "seg_cord", "seg_canal")


def default_cache_root(manifest_path: str) -> str:
    """``<prepared>/<name>/cache`` next to ``manifest.json``."""
    return os.path.join(os.path.dirname(os.path.abspath(manifest_path)), CACHE_DIRNAME)


def resolve_cache_root(setting, manifest_path: str) -> Optional[str]:
    """Config ``data_cache``: ``true`` / missing → default folder, ``false`` /
    ``null`` → no cache, a string → that folder."""
    if setting is None or setting is False:
        return None
    if setting is True:
        return default_cache_root(manifest_path)
    return str(setting)


def _stem(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def case_paths(cache_root: str, split: str, entry: dict) -> Dict[str, str]:
    prefix = os.path.join(cache_root, split, _stem(entry["image"]))
    return {
        "image": prefix + "_image.npy",
        "seg": prefix + "_seg.npy",
        "meta": prefix + "_meta.npz",
    }


def _fingerprint(entry: dict, spine_disc_min_label: int) -> str:
    sources = {}
    for key in _SOURCE_KEYS:
        path = entry.get(key)
        if path:
            st = os.stat(path)
            sources[key] = [os.path.abspath(path), st.st_size, int(st.st_mtime_ns)]
    return json.dumps(
        {
            "version": CACHE_VERSION,
            "spine_disc_min_label": int(spine_disc_min_label),
            "sources": sources,
        },
        sort_keys=True,
    )


def is_cached(cache_root: str, split: str, entry: dict, spine_disc_min_label: int) -> bool:
    meta_path = case_paths(cache_root, split, entry)["meta"]
    if not os.path.isfile(meta_path):
        return False
    try:
        with np.load(meta_path) as meta:
            return str(meta["fingerprint"]) == _fingerprint(entry, spine_disc_min_label)
    except Exception:
        return False


def _build_case(job: Tuple[str, str, dict, int]) -> int:
    """Worker job: write the cache files of one case; returns bytes written."""
    from discslabel.dataset import load_case_nifti

    cache_root, split, entry, spine_disc_min_label = job
    paths = case_paths(cache_root, split, entry)
    case = load_case_nifti(entry, spine_disc_min_label)

    np.save(paths["image"], case["image"])
    written = case["image"].nbytes
    if case["seg_bits"] is not None:
        np.save(paths["seg"], case["seg_bits"])
        written += case["seg_bits"].nbytes
    elif os.path.isfile(paths["seg"]):
        os.remove(paths["seg"])

    tmp = paths["meta"][: -len(".npz")] + ".tmp.npz"
    np.savez(
        tmp,
        shape=np.asarray(case["shape"], dtype=np.int64),
        label_coords=case["label_coords"],
        label_values=case["label_values"],
        fg_coords=case["fg_coords"],
        fg_ids=case["fg_ids"],
        has_disc_seg=np.asarray(case["has_disc_seg"]),
        has_seg=np.asarray(case["seg_bits"] is not None),
        fingerprint=np.asarray(_fingerprint(entry, spine_disc_min_label)),
    )
    os.replace(tmp, paths["meta"])
    return written


def build_cache(
    manifest_path: str,
    splits: Iterable[str],
    cache_root: str,
    spine_disc_min_label: int,
    num_workers: int = 0,
) -> None:
    """Cache every labelled entry of ``splits`` that is missing or stale."""
    from discslabel.parallel import imap_ordered, make_executor

    with open(manifest_path) as f:
        manifest = json.load(f)

    jobs: List[Tuple[str, str, dict, int]] = []
    n_total = 0
    for split in splits:
        os.makedirs(os.path.join(cache_root, split), exist_ok=True)
        for entry in manifest.get(split, []):
            if not entry.get("label"):
                continue
            n_total += 1
            if not is_cached(cache_root, split, entry, spine_disc_min_label):
                jobs.append((cache_root, split, entry, int(spine_disc_min_label)))

    print(f"[cache] {cache_root}: {n_total - len(jobs)}/{n_total} cases up to date")
    if not jobs:
        return

    executor = make_executor(num_workers)
    written = 0
    try:
        for n_bytes in tqdm.tqdm(
            imap_ordered(executor, _build_case, jobs, lookahead=max(2, 2 * num_workers)),
            total=len(jobs),
            desc="building data cache",
        ):
            written += n_bytes
    finally:
        if executor is not None:
            executor.shutdown()
    print(f"[cache] wrote {len(jobs)} cases ({written / 2**30:.1f} GiB)")


def load_case_cached(cache_root: str, split: str, entry: dict) -> Dict[str, object]:
    """Memory-mapped view of one cached case (same keys as ``load_case_nifti``)."""
    paths = case_paths(cache_root, split, entry)
    with np.load(paths["meta"]) as meta:
        case = {
            "shape": tuple(int(s) for s in meta["shape"]),
            "label_coords": meta["label_coords"],
            "label_values": meta["label_values"],
            "fg_coords": meta["fg_coords"],
            "fg_ids": meta["fg_ids"],
            "has_disc_seg": bool(meta["has_disc_seg"]),
            "has_seg": bool(meta["has_seg"]),
        }
    case["image"] = np.load(paths["image"], mmap_mode="r")
    case["seg_bits"] = np.load(paths["seg"], mmap_mode="r") if case.pop("has_seg") else None
    return case
