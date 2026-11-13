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

1. Create a virtual environment and activate it:
  - venv
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   - conda env
   ```
   conda create -n myenv python=3.10
   conda activate myenv
   ```

2. Clone the repo and install the package:
```bash
git clone https://github.com/NathanMolinier/discs-labeling-multicontrast
cd discs-labeling-multicontrast

# Inference / preprocessing only:
pip install -e .

# Training (pulls SmaugLab for the 3D GPU augmentation pipeline):
pip install -e ".[training]"
```

3. Install pytorch following the instructions on their [website](https://pytorch.org/). For example, for CUDA 11.8:
```bash
python3 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

4. Define a folder where data and weights will be stored:
```bash
export DISCSLABEL_DATA=/path/to/scratch/discslabel
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
download_datasets.sh  →  prepare_datasets.sh  →  discslabel_train  →  discslabel_skeleton  →  discslabel_tune  →  discslabel_inference  →  discslabel_evaluate
```

(`run_pipeline_3d.sh` runs the same chain with the `_3d` entry points.)

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
  --model-dir "$DISCSLABEL_DATA/weights/<your-dataset>" \
  --input <image.nii.gz or folder of images> \
  --out-dir <output folder> \
  [--save-preview]
```

`--model-dir` is the training output folder. It must hold `config.json` and `best_model.pt` (use `--checkpoint` to pick another file). When `skeleton.json` is in that folder, the predictions are corrected with the skeleton by default (`--skeleton off|flag|correct` overrides this), and `postprocessing.json` is used when present. The `<name>_discs.png` previews are only written with `--save-preview`.

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
    "LABELS": {
        "LABEL":     {"3": "disc_C2C3", "8": "disc_C7T1", "20": "disc_T12L1", "25": "disc_L5S1"},
        "SEG_SPINE": {"11": "C1", "12": "C2", "...": "...", "63": "C2-C3", "...": "...", "100": "L5-S"},
        "SEG_CORD":  {"1": "spinal_cord"},
        "SEG_CANAL": {"1": "spinal_canal"}
    },
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
- `LABELS` maps each file key (`LABEL`, `SEG_SPINE`, `SEG_CORD`, `SEG_CANAL`) to a `{value: name}` dict describing the voxel values of that file. It is documentation only; the code does not read it.
- `IMAGE` and `LABEL` are required in every entry; `SEG_SPINE`, `SEG_CORD` and `SEG_CANAL` are optional and may be missing for some entries (e.g. `data_t1w_t2w_PSIR_STIR.json`).
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

  This 4-channel tensor is fed as the **second parameter** to SmaugLab's `AugTransformsGPU`: every channel receives the same 3D spatial transforms with nearest resampling. The point-wise disc landmark volume (`LABEL`) is not augmented. Each SEG_* field is optional per entry: a missing segmentation leaves its channel(s) empty (all zeros), so every sample carries a seg tensor of the same shape, mixed batches collate, and every sample goes through the augmentations.
- **Mid-sagittal slice without `SEG_SPINE` (2D pipeline).** The dense disc mask from `SEG_SPINE` is used to put the crop's centre R-L slice on a disc-bearing slice. Without it, the point-wise `LABEL` does not say which sagittal slices cross the discs, so the volume's middle sagittal slice (`D // 2`, the slice inference uses) is used instead. In training it is drawn uniformly among the `2 * midline_jitter + 1` slices centred on it (`midline_jitter`, default `2` → 5 slices); validation uses the middle slice exactly. The `LABEL` points still set the H/W position of the foreground crops. The chosen slice always lands on the patch centre, even when the volume has fewer R-L slices than `patch_size[0]`.

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
        "skeleton":  true,     // fit the inter-disc distance prior on the train labels
        "tune":      true,     // grid-search decoding params on the valid split
        "inference": false,
        "evaluate":  false
    },

    "orientation": "RPI",   // preprocessing axes
    "resolution":  "1x1x1", // preprocessing resolution (mm)
    "label_sigma": 1.5,     // sigma (in input voxels) for the 'label' resample mode

    "model_name": "UNet",
    "patch_size": [96, 320, 320],   // 3D volume shape (R-L, P-A, I-S after RPI)
    "heatmap_sigma": 8.0,
    "heatmap_radius": 20,
    "loss_name": "focalLoss",       // focalLoss | L1 | MSE, or a weighted mix:
                                    // {"focalLoss": 1.0, "MSE": 100.0} or "focalLoss+MSE"
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
- `artifact_config` — GPU MRI artifacts (`discslabel/gpu_artifacts.py`), pure-torch ports of the torchio transforms used by SmaugLab's CPU `ArtifactTransform`: `motion`, `ghosting`, `spike`, `bias_field`, `blur`, `noise`, `swap`. Each batch item is corrupted with probability `p`; with `random_pick` one random enabled artifact is applied, otherwise all of them in that order. Each artifact is `true` (defaults), `false`, or a dict of parameters. `retain_stats` restores each sample's mean/std so inputs stay z-scored. Image-only: labels and heatmaps are untouched. Remove the key (or set `null`) to disable.
- `lowres_si_config` — with probability `p`, draw a final S-I spacing uniformly in `spacing` (`[min, max]` in mm, e.g. `[1, 5]`), downsample the S-I axis from the current spacing (the config `resolution`, 1 mm by default) to it with `down_mode` (`area` = slab averaging, `nearest` = slice dropping) and upsample back with `up_mode`, to simulate thick-slice acquisitions. Draws at or below the current spacing leave the sample unchanged. `axis` can also be `PA` or `RL`.

Both run on the 3D batch on the GPU, right after the SmaugLab augmentations (low resolution first, then artifacts), in the 2D and 3D pipelines. To check what they produce, run:

```bash
discslabel_debug_augmentations --manifest $DISCSLABEL_DATA/prepared/<out_name>/manifest.json \
    --config discslabel/pipeline_2d/configs/default.json --out-dir debug_aug [--save-nifti]
# or on raw images (reoriented / resampled like inference):
discslabel_debug_augmentations --input /path/to/image_or_folder --out-dir debug_aug
```

It writes one PNG per volume (one row per augmentation forced on, original in column 0, `--n-draws` random draws in the others, plus a `training_mix` row with the configured probabilities), optionally the augmented NIfTIs, and prints the time each augmentation takes per volume.

### Data loading speed

- `data_cache` (default `true`) — at the start of training, every train / valid case is written once to `$DISCSLABEL_DATA/prepared/<out_name>/cache/` as uncompressed `.npy` files: the image as float16 (rescaled, which the z-score ignores), the anatomy segmentations as one uint8 bit mask, and the point-wise label and disc locations as sparse coordinates. The data loader memory-maps them and reads only the crop (~0.03 s per sample instead of ~0.7 s to gunzip a whole-spine NIfTI). Cases are rebuilt when their prepared files or `spine_disc_min_label` change. Expect about 2 bytes per voxel per image, plus 1 byte per voxel when segmentations exist. Set it to `false` to read the NIfTIs, or to a path to put the cache elsewhere (e.g. local SSD).
- The data loader returns compact tensors (float16 image, uint8 label, uint8 seg bit mask, ~6x fewer bytes than float32) and the z-score normalisation and seg unpacking run on the GPU (`discslabel.dataset.batch_to_device`).
- Full-image validation loads its ground truth once before training. The 2D pipeline also keeps the z-scored mid-sagittal slice stacks in memory, so a validation round only runs the model and the decoding. The 3D pipeline reloads each volume from the cache.

### Training schedule (nnU-Net style)

- `num_epochs` × `num_iterations_per_epoch` — fixed training length; batches are drawn with replacement and there is **no early stopping**. The learning rate follows nnU-Net's polynomial decay `lr * (1 - epoch / num_epochs) ** 0.9`.
- `optimizer` — `AdamW` (default), `Adam` or `SGD` (Nesterov, `momentum`); `weight_decay` applies to all three. Gradients are clipped at norm 12.
- `oversample_foreground_percent` — fraction of every batch whose crop is centred on a voxel of a uniformly drawn disc (nnU-Net uses 0.33); the rest are uniformly random crops. The 2D pipeline always keeps the R-L centre on a disc-bearing slice.
- `deep_supervision` — extra heatmap losses on the 1/2 and 1/4 decoder outputs (targets max-pooled so peaks stay at 1).
- `val_every`, `val_max_images`, `val_tolerance_mm`, `val_ema_alpha` — every `val_every` epochs the model runs the full inference decoding on the validation images and is scored with the label-aware F1 (`f1_at_tolerance`: label right **and** within `val_tolerance_mm`). `best_model.pt` is saved when the exponential moving average of that score reaches a new best (details in `best_model.json`); `final_model.pt` is the last epoch.

### Inference and post-processing

Inference stitches every sliding-window prediction into one heatmap with a Gaussian importance map, then decodes peaks once. `discslabel_tune` / `discslabel_tune_3d` grid-search `peak_threshold` × `min_peak_distance` (`tune_peak_thresholds`, `tune_min_peak_distances`) on the validation split and write `postprocessing.json` + `postprocessing_grid.csv` next to the weights; inference uses `postprocessing.json` automatically when present, otherwise the config's `peak_threshold` / `inference_min_peak_distance`.

### Skeleton QC and correction (optional)

The first labelling step numbers the disc peaks one by one away from the detected anchor, so a missed disc shifts every label past it, and a spurious or far-off peak does the opposite. `discslabel_skeleton` fits an anatomical prior on the **training** labels and writes `skeleton.json` next to the weights:

- a template gap between every pair of adjacent discs (median spine), with the spread of each gap measured after normalising each subject by its own size (scale-normalised log ratio, robust MAD sigma);
- the distribution of subject scales and of the chain's turning angle at each disc;
- a `3d` projection (3D pipeline) and a `sagittal` one (2D pipeline, R-L component dropped).

```bash
discslabel_skeleton --manifest "$DISCSLABEL_DATA/prepared/<out_name>/manifest.json" \
    --out-dir "$DISCSLABEL_DATA/weights/<run_name>"
```

At inference, `--skeleton {off,flag,correct}` (or `inference_skeleton` in the config, default `off`) runs the check after labelling. It finds the most plausible labelling of the peak chain under the skeleton, using dynamic programming: the anchor keeps its label; every pair of kept peaks costs `z²/2` for how far their distance is from the template; and every skipped label (missed disc) or dropped peak (off / spurious disc) costs a fixed penalty set by `skeleton_z_threshold`. With `correct`, off-chain peaks are removed, labels are fixed, and missed discs between two kept discs are placed by monotone interpolation along the chain (drawn in yellow in the PNG). With `flag`, the prediction is left unchanged and the proposed changes are only reported.

Either mode writes `<name>_discs_qc.json` for each image and a `qc_summary.csv` for the run. Each image's `status` is `ok` or `flagged`, with these flags:

| Flag | Meaning |
|---|---|
| `no_anchor` | no anchor channel cleared the threshold, so the prediction is empty |
| `too_few_discs` | fewer than 2 labelled discs, so nothing could be checked |
| `missed_discs` | labels missing between detected discs (imputed in `correct` mode) |
| `outlier_discs` | peaks that do not fit the chain (removed in `correct` mode) |
| `relabeled` | discs whose label changes once the chain is repaired |
| `abnormal_gap` | an inter-disc distance still outside the training distribution (z-threshold Bonferroni-corrected for the image's number of gaps, ~1% false flags on a normal spine) |
| `abnormal_angle` | the chain bends at a disc more than in any training spine |
| `abnormal_scale` | spine size outside the training range (median ± 4 robust sigma) |

When `skeleton.json` is in the weights folder, `discslabel_tune[_3d]` also scores the correction on the validation split for each `tune_skeleton_z_thresholds` value, using the best decoding parameters. It stores the best value under `"skeleton"` in `postprocessing.json`, along with `improves` (whether it beat plain decoding); inference uses that value. **Evaluation with and without the skeleton.** With `evaluate_skeleton_compare: true` (the default), the evaluate step runs the model once on the evaluation images and decodes each heatmap twice: the plain predictions go to `prediction_<run_name>` and the skeleton-corrected ones (`--skeleton correct`) go to `prediction_<run_name>_skeleton`. This uses `--plain-out-dir` on `discslabel_inference[_3d]`. The two folders are evaluated as separate runs (`metrics_<run_name>` and `metrics_<run_name>_skeleton`), so both show up in `discslabel_aggregate_evaluation`. `discslabel_compare_evaluation` then prints the metrics side by side with the differences, including F1@`val_tolerance_mm`, and lists the images whose F1 improved or got worse. It writes `comparison.csv` / `comparison.json` into the skeleton metrics folder. You can also run it by hand on any runs: `discslabel_compare_evaluation --metrics-dirs <reference> <other> [...]`.

**Error previews.** `discslabel_evaluate` also saves, in `<metrics_dir>/errors/<stem>.png`, a sagittal view with the ground truth (left) and the prediction (right) for every image that has an error: a missed disc, an extra or mislabelled disc, or a disc farther than `--tolerance-mm` (the pipeline passes `val_tolerance_mm`). Colours: green = correct, orange = missed, red = too far, magenta = extra or mislabelled. The images are drawn in their RAS grid on the slice at the median R-L position of the GT discs. The images come from `--image-dir` (default `<gt-dir>/../images`). Use `--previews all` to save every image (in `previews/`) or `--previews none` to skip them.

**Soft matching.** A disc whose heatmap blob stays just under `peak_threshold` never reaches the chain, so it can only come back as an interpolated guess — or not at all at the ends of the chain. With `skeleton_soft_floor` (default `0.1`, `null` to disable; `correct` mode only) the chain is built from *soft candidates* instead: local maxima of a local-mass score (the all-discs channel filtered with the training target Gaussian, normalised so a perfect blob scores 1), kept down to that floor. The mass score also rescues blobs that fall between voxels or are split into close sub-peaks. Each peak gets a confidence `w` (0 at the floor, 1 from `peak_threshold` up): dropping it costs `w` × the drop penalty and keeping it costs `(1 − w)` × 0.75 × the drop penalty. Confident peaks behave as before; a weak peak is kept only where the skeleton expects a disc. Kept weak peaks are listed under `changes.recovered` in `<name>_discs_qc.json` (`n_recovered` in `qc_summary.csv`) and raise the `weak_discs` flag. `discslabel_tune[_3d]` scores every `tune_skeleton_soft_floors` value (`null` = off) together with `tune_skeleton_z_thresholds` and stores the best `soft_floor` in `postprocessing.json`, which inference then uses — re-run it after enabling this.

Other knobs: `skeleton_max_missing` (most consecutive missed discs that can be filled, default 3) and `skeleton_max_skipped` (most consecutive peaks that can be dropped inside the chain, default 3).

Tuning, inference and evaluation run their CPU work in a pool of worker processes while the main process keeps the GPU busy: workers preprocess (load, reorient, resample, z-score) the next images, extract peaks / decode / write the NIfTI + PNG of the previous ones, score the tuning grid points in parallel, and score GT/prediction pairs in `discslabel_evaluate`. The pool size comes from `inference_num_workers` (`"auto"` = available CPUs − 1, capped at 8) and can be overridden with `--num-workers` on `discslabel_tune[_3d]`, `discslabel_inference[_3d]` and `discslabel_evaluate`; `0` runs everything inline (useful for debugging). Training validation keeps running inline.

The config is archived next to the trained weights (`$DISCSLABEL_DATA/weights/<run_name>/config.json`), together with the resolved `data.json` and a `pip freeze` snapshot, so each checkpoint carries the exact context used to produce it.

## License

See [LICENSE](LICENSE).
