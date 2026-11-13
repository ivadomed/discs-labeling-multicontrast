# Discs labeling multicontrast

UNet heatmap-regression method for intervertebral disc landmark detection on **mid-sagittal MRI slices** extracted from 3D volumes.

Point-wise 3D disc labels (values `1..25`, with the anchors `3 = C2C3`, `8 = C7T1`, `20 = T12L1`, `25 = L5S1`) and the 3D MR images are reoriented to RPI and resampled to 1 mm iso offline.

<img src="https://github.com/user-attachments/assets/1974c748-93f7-43b9-8f66-af5703cf484e" alt="Image description" width="50%">


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

## Inference

Running the inference:

```bash
discslabel_inference \
  --input <image.nii.gz or folder of images> \
  --out-dir <output folder> \
  [--model-dir "custom weights folder"] \
  [--save-preview] [--save-qc]
```

Without `--model-dir`, `discslabel_inference` downloads the pretrained weights from the GitHub release the first time and caches them in `data/weights/` at the repo root (git-ignored). `--weights-url` or the `DISCSLABEL_WEIGHTS_URL` environment variable replaces the release URL, e.g. to test a zip served locally with `python -m http.server`.

## Training

The whole pipeline is driven from one JSON config (e.g. for the 2D model):

```bash
export DISCSLABEL_DATA=/path/to/scratch/discslabel
./scripts/run_pipeline.sh discslabel/pipeline_2d/configs/default_mse.json
```

`run_pipeline.sh` reads the config and runs:

```
download_datasets.sh  →  prepare_datasets.sh  →  discslabel_train  →  discslabel_skeleton  →  discslabel_tune  →  discslabel_inference  →  discslabel_evaluate
```

(`run_pipeline_3d.sh` runs the same chain with the `_3d` entry points.)

Each step runs when either `steps.<name>` is `true` **or** its output is missing from disk — so a fresh checkout always catches up, and setting a step to `false` only skips when the previous output is already present.

## License

See [LICENSE](LICENSE).
