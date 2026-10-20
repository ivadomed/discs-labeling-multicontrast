# discs-labeling-multicontrast

UNet heatmap-regression baseline for intervertebral disc landmark detection on **mid-sagittal MRI slices** extracted from 3D volumes.

Point-wise 3D disc labels (values `1..25`, with the anchors `3 = C2C3`, `8 = C7T1`, `20 = T12L1`, `25 = L5S1`) are projected onto the mid-sagittal plane and converted into Gaussian heatmaps. The model outputs **5 heatmap channels**:

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
- PyTorch with CUDA if you want to train on GPU

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

# 2. Extract mid-sagittal slices and write train/valid/test CSVs
./scripts/prepare_datasets.sh discslabel/datasets/<your-dataset>.json

# 3. Train the UNet
discslabel_train \
  --train-csv "$DISCSLABEL_DATA/prepared/<your-dataset>/train.csv" \
  --valid-csv "$DISCSLABEL_DATA/prepared/<your-dataset>/valid.csv" \
  --save-model-dir "$DISCSLABEL_DATA/weights/<your-dataset>" \
  --config discslabel/configs/default.json

# 4. Run inference on the test split
discslabel_inference \
  --weights "$DISCSLABEL_DATA/weights/<your-dataset>/best_model.pt" \
  --test-csv "$DISCSLABEL_DATA/prepared/<your-dataset>/test.csv" \
  --out-csv "$DISCSLABEL_DATA/prepared/<your-dataset>/predictions.csv" \
  --save-image-dir "$DISCSLABEL_DATA/prepared/<your-dataset>/visualize" \
  --config discslabel/configs/default.json
```

## Repository layout

```
discslabel/
├── configs/        # example training configs (default.json)
├── datasets/       # dataset JSONs (SmaugLab-benchmark schema)
├── utils/
│   ├── io.py       # directory helpers
│   └── metrics.py  # heatmap decoding + visualization
├── __init__.py     # exports NUM_CLASSES, HEATMAP_CHANNELS, LANDMARK_CHANNEL_LABELS
├── dataset.py      # DiscLandmarkDataset (reads a prepared CSV)
├── losses.py       # focal / L1 / MSE losses
├── model.py        # UNet backbone with 5 output channels
├── transforms.py   # Rescale, flips, Gaussian heatmap generator, ToTensor
├── prepare.py      # discslabel_prepare: BIDS NIfTI -> mid-sagittal slice + CSV
├── train.py        # discslabel_train:   training loop with early stopping
└── inference.py    # discslabel_inference: heatmap decoding + metrics
scripts/
├── download_datasets.sh   # git clone + git annex get for a dataset JSON
└── prepare_datasets.sh    # thin wrapper around discslabel_prepare
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

## How the mid-sagittal projection works

For every `(IMAGE, LABEL)` pair, `discslabel_prepare`:

1. Loads both NIfTIs and reorients them to **RAS+** (so axis 0 = R-L, axis 1 = A-P, axis 2 = S-I).
2. Takes the slice at `shape[0] // 2` on the R-L axis — this is the mid-sagittal plane.
3. For the labels, groups non-zero voxels by their label value and collapses each group to the median `(A-P, S-I)` location (this picks one 2D landmark per disc, robust against isolated label voxels).
4. Rotates/flips the slice so superior is up and anterior is left, and stores it as a `.npy` file.
5. Writes one CSV row per subject: `slice_path, slice_h, slice_w, label_1, x_1, y_1, label_2, x_2, y_2, ...`.

The resulting CSVs are what `DiscLandmarkDataset` reads at training time, which turns the `(label, x, y)` triples into the 5-channel Gaussian heatmap target.

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `DISCSLABEL_DATA` | yes | Root folder for `bids/`, `prepared/<name>/` and training `weights/` |

## Training config

Pass `--config` to `discslabel_train` and `discslabel_inference` to override the defaults in `discslabel/configs/default.json`. Common knobs:

- `image_height`, `image_width` — model input size (defaults to 512 x 512).
- `heatmap_sigma`, `heatmap_radius` — Gaussian spread and kernel radius.
- `loss_name` — `focalLoss` (default), `L1`, or `MSE`.
- `vertical_flip_prob` — leave at `0.0` for disc labelling (vertical flips scramble the anchor identities).
- `intensity_jitter` — multiplicative + additive jitter in normalised intensity units.

## License

See [LICENSE](LICENSE).
