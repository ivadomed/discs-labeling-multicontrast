# Discs labeling multicontrast

UNet heatmap-regression method for intervertebral disc landmark detection on **mid-sagittal MRI slices** extracted from 3D volumes.

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
- PyTorch with CUDA if you want to train with GPU augmentations.

```bash
git clone https://github.com/NathanMolinier/discs-labeling-multicontrast
cd discs-labeling-multicontrast

# Inference / preprocessing only:
pip install -e .

# Training (pulls SmaugLab for the 3D GPU augmentation pipeline):
pip install -e ".[training]"
```

[SmaugLab](https://github.com/neuropoly/SmaugLab) is an **optional** dependency — needed only at training time when `aug_config` points to a GPU-augmentation JSON. Inference and preprocessing never import it.

The package is versioned automatically from git tags via `poetry-dynamic-versioning` (same mechanism as SmaugLab-benchmark — `pyproject.toml` carries `version = "0.0.0"` as a placeholder; the real version is substituted at build time).

## Quick start

The whole pipeline is driven from one JSON config:

```bash
export DISCSLABEL_DATA=/path/to/scratch/discslabel
./scripts/run_pipeline.sh discslabel/configs/default.json
```

`run_pipeline.sh` reads the config and runs:

```
download_datasets.sh  →  prepare_datasets.sh  →  discslabel_train  →  discslabel_inference
```

Each step runs when either `steps.<name>` is `true` **or** its output is missing from disk — so a fresh checkout always catches up, and setting a step to `false` only skips when the previous output is already present.

Running the steps one at a time is still supported:

```bash
./scripts/download_datasets.sh discslabel/datasets/<your-dataset>.json
./scripts/prepare_datasets.sh discslabel/datasets/<your-dataset>.json
discslabel_train \
  --manifest "$DISCSLABEL_DATA/prepared/<your-dataset>/manifest.json" \
  --save-model-dir "$DISCSLABEL_DATA/weights/<your-dataset>" \
  --config discslabel/configs/default.json
discslabel_inference \
  --weights "$DISCSLABEL_DATA/weights/<your-dataset>/best_model.pt" \
  --manifest "$DISCSLABEL_DATA/prepared/<your-dataset>/manifest.json" \
  --split test \
  --out-csv "$DISCSLABEL_DATA/weights/<your-dataset>/predictions_test.csv" \
  --save-image-dir "$DISCSLABEL_DATA/weights/<your-dataset>/visualize_test" \
  --config discslabel/configs/default.json
```

## Pipeline shape

```
┌──────────────────────────────────────── offline, CPU ───────────────────────┐
│  BIDS NIfTIs ──► reorient to RPI ──► resample to 1 mm iso                    │
│                 (discslabel_reorient_image / discslabel_resample_image,       │
│                  with -int label for point-wise disc landmarks)              │
│                                                                              │
│  Output: $DISCSLABEL_DATA/prepared/<name>/{train,valid,test}/{images,labels} │
│          + manifest.json                                                     │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────── online, per batch, GPU ─────────────────────┐
│  Dataset:  loads 3D image + 3D point-wise label, pads/crops to patch_size,   │
│            yields (B, 1, D, H, W) tensors                                    │
│                                                                              │
│  GpuBatchProcessor (nn.Module on CUDA):                                      │
│   1. SmaugLab AugTransformsGPU on 3D (image, seg)      [kornia pipeline]    │
│      where seg = combined SPINE / CORD / CANAL mask                           │
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
├── prepare_datasets.sh    # copy + reorient (RPI) + resample (1x1x1 mm)
└── run_pipeline.sh        # drive download → prepare → train → inference from one JSON
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
    "TRAINING":   [{ "IMAGE":     "<bids-ds>/sub-001/anat/sub-001_T2w.nii.gz",
                     "LABEL":     "<bids-ds>/derivatives/labels/sub-001/anat/sub-001_T2w_label-discs.nii.gz",
                     "SEG_SPINE": "<bids-ds>/derivatives/labels/sub-001/anat/sub-001_T2w_label-spine_dseg.nii.gz",
                     "SEG_CORD":  "<bids-ds>/derivatives/labels/sub-001/anat/sub-001_T2w_label-SC_seg.nii.gz",
                     "SEG_CANAL": "<bids-ds>/derivatives/labels/sub-001/anat/sub-001_T2w_label-canal_seg.nii.gz" }],
    "VALIDATION": [{ "IMAGE": "...", "LABEL": "...", "SEG_SPINE": "...", "SEG_CORD": "...", "SEG_CANAL": "..." }],
    "TESTING":    [{ "IMAGE": "...", "LABEL": "...", "SEG_SPINE": "...", "SEG_CORD": "...", "SEG_CANAL": "..." }]
}
```

- `SOURCE[i]` and `COMMIT[i]` are paired; each repo is checked out at the matching commit.
- `IMAGE` / `LABEL` paths are relative to `$DISCSLABEL_DATA/bids/`, and the first path segment names the repo (used by `git annex get`).
- All files must be `.nii.gz`.
- `LABEL` is a point-wise 3D volume with one voxel per disc at the posterior tip, carrying the disc-level value (`1..25`).
- `SEG_SPINE` is a dense discrete segmentation of the whole spine (vertebrae at low label values, discs at high label values; the two populations are separated by the `spine_disc_min_label` threshold in the config, default `60`). `SEG_CORD` and `SEG_CANAL` are binary structure masks. The prepare step reorients and resamples all three with nearest-neighbour interpolation, and the dataset reader packs them into a **4-channel binary tensor** — one channel per structure, no value overlap between channels:

  | Channel | Source | Rule |
  | ------- | ------ | ---- |
  | 0 — vertebrae | `SEG_SPINE` | `0 < value < spine_disc_min_label` |
  | 1 — discs | `SEG_SPINE` | `value >= spine_disc_min_label` |
  | 2 — spinal cord | `SEG_CORD` | `value > 0` |
  | 3 — spinal canal | `SEG_CANAL` | `value > 0` |

  This 4-channel tensor is fed as the **second parameter** to SmaugLab's `AugTransformsGPU`: every channel receives the same 3D spatial transforms with nearest resampling. The point-wise disc landmark volume (`LABEL`) is not augmented. Each SEG_* field is optional per entry.

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

## Pipeline config

`discslabel/configs/default.json` is the single JSON consumed by both the pipeline driver and the training entry point. It carries three things:

```jsonc
{
    "data_json": "whole-spine",                 // basename resolved under discslabel/datasets/,
                                                // or an absolute path
    "out_name":  "whole-spine",                 // sub-folder under $DISCSLABEL_DATA/prepared/
    "run_name":  "whole-spine_UNet",            // sub-folder under $DISCSLABEL_DATA/weights/

    "steps": {
        "download":  true,
        "prepare":   true,
        "train":     true,
        "inference": false
    },

    "orientation": "RPI",   // preprocessing axes
    "resolution":  "1x1x1", // preprocessing resolution (mm)
    "label_sigma": 1.5,     // sigma (in input voxels) for the 'label' resample mode

    "model_name": "UNet",
    "patch_size": [96, 320, 320],   // 3D volume shape (R-L, P-A, I-S after RPI)
    "heatmap_sigma": 8.0,
    "heatmap_radius": 20,
    "loss_name": "focalLoss",       // focalLoss | L1 | MSE
    "aug_config": null,             // path to a SmaugLab GPU-augmentation JSON

    "inference_split": "test",
    "inference_save_images": true,
    "cuda_id": 0

    // + the usual training hyperparams (lr, scheduler, batch_size, epochs, …)
}
```

Common knobs:

- `patch_size` — 3D volume shape fed to the GPU augmentations. Volumes smaller than the patch are zero-padded; larger ones are center-cropped.
- `heatmap_sigma`, `heatmap_radius` — Gaussian spread and kernel radius for the on-GPU heatmap synthesis.
- `label_sigma` — Gaussian spread **in millimetres** of the sphere used to carry point labels through the resample step. The sigma is converted to per-axis voxel sigmas via the input image zooms, so the sphere stays isotropic in physical space even on thick-slice MRI (anisotropic voxels). Set `label_debug_dir` in the config (or pass `--debug-dir` to `discslabel_resample_image`) to write a mid-sagittal 4-panel PNG per processed file — use it to confirm the sphere is big enough that its peak survives the resample.
- `aug_config` — path to a SmaugLab GPU augmentation JSON (same format as the ones used by SmaugLab-benchmark). Set to `null` to disable augmentation.

The config is archived next to the trained weights (`$DISCSLABEL_DATA/weights/<run_name>/config.json`), together with the resolved `data.json` and a `pip freeze` snapshot, so each checkpoint carries the exact context used to produce it.

## License

See [LICENSE](LICENSE).
