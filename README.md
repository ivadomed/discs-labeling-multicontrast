# discs-labeling-multicontrast

UNet heatmap-regression baseline for intervertebral disc landmark detection on **mid-sagittal MRI slices** extracted from 3D volumes.

Point-wise 3D disc labels (values `1..25`, with the anchors `3 = C2C3`, `8 = C7T1`, `20 = T12L1`, `25 = L5S1`) and the 3D MR images are reoriented to RPI and resampled to 1 mm iso offline. **Everything else happens inside the GPU data loader**: the batch is passed through SmaugLab's 3D GPU augmentations, the mid-sagittal slice is picked, the point-wise 3D label is projected onto that slice, and Gaussian heatmaps are synthesised as the training targets.

Model output — **5 heatmap channels**:

| Channel | Target |
| ------- | ------ |
| 0 | union of every visible disc (any label `1..25`) |
| 1 | C2C3 (`label = 3`) |
| 2 | C7T1 (`label = 8`) |
| 3 | T12L1 (`label = 20`) |
| 4 | L5S1 (`label = 25`) |

The data-management layout (dataset JSONs + download / prepare shell scripts) follows [SmaugLab-benchmark](https://github.com/NathanMolinier/SmaugLab-benchmark); training and inference are plain Python entry points installed as console scripts.

## Installation

Requirements:

- Python 3.10+
- [`git-annex`](https://git-annex.branchable.com/) and `jq` (for the download script)
- [SmaugLab](https://github.com/neuropoly/SmaugLab) (GPU augmentations) and [SmaugLab-benchmark](https://github.com/NathanMolinier/SmaugLab-benchmark) (reorient / resample CLIs). Both are declared as `pyproject.toml` dependencies.
- PyTorch with CUDA — SmaugLab's augmentation stack is a kornia pipeline that needs a CUDA device at train time.

```bash
git clone https://github.com/NathanMolinier/discs-labeling-multicontrast
cd discs-labeling-multicontrast
pip install -e .
```

The package is versioned automatically from git tags via `poetry-dynamic-versioning` (same mechanism as SmaugLab-benchmark — `pyproject.toml` carries `version = "0.0.0"` as a placeholder; the real version is substituted at build time).

## Quick start

```bash
export DISCSLABEL_DATA=/path/to/scratch/discslabel

# 1. Download the BIDS data
./scripts/download_datasets.sh discslabel/datasets/<your-dataset>.json

# 2. Reorient to RPI and resample to 1x1x1 mm (no 2D projection yet)
./scripts/prepare_datasets.sh discslabel/datasets/<your-dataset>.json

# 3. Train the UNet (3D volumes in -> 2D heatmap loss out, everything on GPU)
discslabel_train \
  --manifest "$DISCSLABEL_DATA/prepared/<your-dataset>/manifest.json" \
  --save-model-dir "$DISCSLABEL_DATA/weights/<your-dataset>" \
  --config discslabel/configs/default.json

# 4. Run inference on the test split
discslabel_inference \
  --weights "$DISCSLABEL_DATA/weights/<your-dataset>/best_model.pt" \
  --manifest "$DISCSLABEL_DATA/prepared/<your-dataset>/manifest.json" \
  --split test \
  --out-csv "$DISCSLABEL_DATA/prepared/<your-dataset>/predictions.csv" \
  --save-image-dir "$DISCSLABEL_DATA/prepared/<your-dataset>/visualize" \
  --config discslabel/configs/default.json
```

## Pipeline shape

```
┌──────────────────────────────────────── offline, CPU ───────────────────────┐
│  BIDS NIfTIs ──► reorient to RPI ──► resample to 1 mm iso                    │
│                 (smaugbench_reorient_image / smaugbench_resample_image)      │
│                                                                              │
│  Output: $DISCSLABEL_DATA/prepared/<name>/{train,valid,test}/{images,labels} │
│          + manifest.json                                                     │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────── online, per batch, GPU ─────────────────────┐
│  Dataset:  loads 3D image + 3D point-wise label, pads/crops to patch_size,   │
│            yields (B, 1, D, H, W) tensors                                    │
│                                                                              │
│  GpuBatchProcessor (nn.Module on CUDA):                                      │
│   1. SmaugLab AugTransformsGPU on 3D (image, label)    [kornia pipeline]    │
│   2. mid-sagittal slice = image[..., D // 2, :, :]      → (B, 1, H, W)      │
│   3. label projection:   label_3d.amax(dim=D)           → (B, 1, H, W)      │
│   4. 5-channel heatmaps: depthwise Gaussian conv2d      → (B, 5, H, W)      │
│                                                                              │
│  UNet(1 → 5) ──► focal / L1 / MSE loss                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Repository layout

```
discslabel/
├── configs/        # training configs (default.json)
├── datasets/       # dataset JSONs (SmaugLab-benchmark schema)
├── utils/
│   ├── io.py       # directory helpers
│   └── metrics.py  # heatmap decoding + visualization
├── __init__.py     # exports NUM_CLASSES, HEATMAP_CHANNELS, LANDMARK_CHANNEL_LABELS
├── dataset.py      # DiscVolumeDataset — 3D NIfTI reader with pad/crop
├── gpu_pipeline.py # GpuBatchProcessor — SmaugLab aug + mid-sag + heatmaps
├── losses.py       # focal / L1 / MSE losses
├── model.py        # UNet backbone (1 input ch, 5 output heatmaps)
├── prepare.py      # discslabel_prepare: copy BIDS files, emit manifest.json
├── train.py        # discslabel_train: GPU training loop
└── inference.py    # discslabel_inference: GPU inference + metrics
scripts/
├── download_datasets.sh   # git clone + git annex get for a dataset JSON
└── prepare_datasets.sh    # copy + reorient (RPI) + resample (1x1x1 mm)
```

## Dataset JSON schema

Each dataset is described by a JSON file under `discslabel/datasets/`:

```json
{
    "TYPE": "LANDMARK",
    "CONTRASTS": "T1w_T2w",
    "SOURCE": ["git@data.neuro.polymtl.ca:datasets/<bids-ds>.git"],
    "COMMIT": ["<commit-sha>"],
    "LABELS": {"3": "disc_C2C3", "8": "disc_C7T1", "20": "disc_T12L1", "25": "disc_L5S1"},
    "TRAINING":   [{ "IMAGE": "<bids-ds>/sub-001/anat/sub-001_T2w.nii.gz",
                     "LABEL": "<bids-ds>/derivatives/labels/sub-001/anat/sub-001_T2w_label-discs.nii.gz" }],
    "VALIDATION": [{ "IMAGE": "...", "LABEL": "..." }],
    "TESTING":    [{ "IMAGE": "...", "LABEL": "..." }]
}
```

- `SOURCE[i]` and `COMMIT[i]` are paired; each repo is checked out at the matching commit.
- `IMAGE` / `LABEL` paths are relative to `$DISCSLABEL_DATA/bids/`, and the first path segment names the repo (used by `git annex get`).
- All files must be `.nii.gz`.
- `LABEL` is a point-wise 3D volume with one voxel per disc at the posterior tip, carrying the disc-level value (`1..25`).

## Why the projection runs on GPU

Keeping the 3D volumes until the data loader has two concrete benefits:

- **3D augmentations.** Rotations, elastic deformations, noise and intensity transforms from [SmaugLab](https://github.com/neuropoly/SmaugLab) can be applied to the paired `(image, label)` 3D tensors on the GPU. If we projected to 2D offline we'd lose any R-L information those augmentations might rely on and we'd have to re-implement each transform in 2D.
- **No CPU bottleneck.** Projection and heatmap generation are reduced to one `amax` call and one depthwise `conv2d`. Everything stays on-device, so the DataLoader workers don't have to materialise large heatmap tensors — they only load and pad the NIfTIs.

The mid-sagittal projection drops the R-L voxel coordinate from each labelled voxel; posterior-tip disc annotations sit on the midline, so collapsing with `amax` yields one labelled pixel per disc on the mid-sagittal plane.

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `DISCSLABEL_DATA` | yes | Root folder for `bids/`, `prepared/<name>/` and training `weights/` |
| `DISCSLABEL_JOBS` | no | CPU parallelism for reorient / resample (defaults to detected core count) |

## Training config

Pass `--config` to `discslabel_train` and `discslabel_inference` to override the defaults in `discslabel/configs/default.json`. Common knobs:

- `patch_size` — 3D volume shape fed to the GPU augmentations (default `[96, 320, 320]`, i.e. 96 R-L, 320 P-A, 320 I-S). Volumes smaller than the patch are zero-padded; larger ones are center-cropped.
- `heatmap_sigma`, `heatmap_radius` — Gaussian spread and kernel radius for the on-GPU heatmap synthesis.
- `loss_name` — `focalLoss` (default), `L1`, or `MSE`.
- `aug_config` — path to a SmaugLab GPU augmentation JSON (same format as the ones used by SmaugLab-benchmark). Set to `null` to disable augmentation.

## License

See [LICENSE](LICENSE).
