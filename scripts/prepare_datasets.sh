#!/bin/bash

# Prepare 3D NIfTIs for discslabel: copy BIDS images+labels into per-split
# folders, reorient to RPI, and resample to 1 mm isotropic.
#
# the 2D mid-sagittal projection and the Gaussian heatmap targets are built
# inside the data loader on the GPU, not here.
#
# Required environment:
#   DISCSLABEL_DATA   root folder that contains `bids/` and will receive `prepared/`
#
# Positional args:
#   $1   dataset JSON (absolute path or basename resolved under discslabel/datasets/)
#   $2   (optional) subfolder name under prepared/ — defaults to the JSON basename
#   $3   (optional) orientation code, defaults to RPI
#   $4   (optional) resolution, defaults to 1x1x1 mm

set -e
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

if [ -z "$DISCSLABEL_DATA" ]; then
    echo "Please set the DISCSLABEL_DATA environment variable."
    exit 1
fi

PROJECT_ROOT="$(realpath "$(dirname "$0")/..")"
echo "discs-labeling-multicontrast project path: $PROJECT_ROOT"

data_json="$1"
if [ ! -f "$data_json" ]; then
    data_json="$PROJECT_ROOT/discslabel/datasets/$(basename "$1").json"
    if [ ! -f "$data_json" ]; then
        echo "Error: Could not find data JSON file."
        exit 1
    fi
fi

out_name="${2:-$(basename "${data_json%.json}")}"
ORIENTATION="${3:-RPI}"
RESOLUTION="${4:-1x1x1}"

# CPU parallelism (same env-var contract as SmaugLab-benchmark).
if [[ "$OSTYPE" == "darwin"* ]]; then
    CORES=$(sysctl -n hw.ncpu)
else
    CORES=$(getconf _NPROCESSORS_ONLN)
fi
JOBS=${DISCSLABEL_JOBS:-$CORES}

# 1. Copy BIDS files into prepared/<name>/{train,valid,test}/{images,labels} and
#    write prepared/<name>/manifest.json.
discslabel_prepare --data-json "$data_json" --out-name "$out_name"

OUT_ROOT="$DISCSLABEL_DATA/prepared/$out_name"

# 2. Reorient and resample every split. Reuses smaugbench's CLIs (same contract as
#    SmaugLab-benchmark's prepare_datasets.sh — nearest for labels, linear for
#    images).
for split in train valid test; do
    IMAGES_DIR="$OUT_ROOT/$split/images"
    LABELS_DIR="$OUT_ROOT/$split/labels"

    [ -d "$IMAGES_DIR" ] || continue
    if [ -z "$(ls -A "$IMAGES_DIR" 2>/dev/null)" ]; then
        echo "[prepare] split '$split' has no images, skipping."
        continue
    fi

    echo "[prepare] reorienting split '$split' to $ORIENTATION"
    smaugbench_reorient_image -i "$IMAGES_DIR" -o "$IMAGES_DIR" -ori "$ORIENTATION" -r -w "$JOBS"
    if [ -d "$LABELS_DIR" ] && [ -n "$(ls -A "$LABELS_DIR" 2>/dev/null)" ]; then
        smaugbench_reorient_image -i "$LABELS_DIR" -o "$LABELS_DIR" -ori "$ORIENTATION" -r -w "$JOBS"
    fi

    echo "[prepare] resampling split '$split' to $RESOLUTION"
    smaugbench_resample_image -i "$IMAGES_DIR" -o "$IMAGES_DIR" -res "$RESOLUTION" -int linear -r -w "$JOBS"
    if [ -d "$LABELS_DIR" ] && [ -n "$(ls -A "$LABELS_DIR" 2>/dev/null)" ]; then
        smaugbench_resample_image -i "$LABELS_DIR" -o "$LABELS_DIR" -res "$RESOLUTION" -int nn -r -w "$JOBS"
    fi
done

echo "[prepare] done. Manifest at $OUT_ROOT/manifest.json"
