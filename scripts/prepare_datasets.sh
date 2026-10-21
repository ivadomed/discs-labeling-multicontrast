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
#   $5   (optional) label_sigma in MM for the 'label' resample mode, defaults to 1.5
#   $6   (optional) label debug dir — if set, writes a per-file PNG checker there

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
LABEL_SIGMA="${5:-1.5}"
DEBUG_DIR="${6:-}"

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

# 2. Reorient and resample every split.
#    Images use linear interpolation; point-wise disc labels use the dedicated
#    'label' mode added in discslabel_resample_image, which smooths each point
#    to a Gaussian sphere before resampling then recovers the sphere centre.
for split in train valid test; do
    IMAGES_DIR="$OUT_ROOT/$split/images"
    LABELS_DIR="$OUT_ROOT/$split/labels"

    [ -d "$IMAGES_DIR" ] || continue
    if [ -z "$(ls -A "$IMAGES_DIR" 2>/dev/null)" ]; then
        echo "[prepare] split '$split' has no images, skipping."
        continue
    fi

    echo "[prepare] reorienting split '$split' to $ORIENTATION"
    discslabel_reorient_image -i "$IMAGES_DIR" -o "$IMAGES_DIR" -ori "$ORIENTATION" -r -w "$JOBS"
    if [ -d "$LABELS_DIR" ] && [ -n "$(ls -A "$LABELS_DIR" 2>/dev/null)" ]; then
        discslabel_reorient_image -i "$LABELS_DIR" -o "$LABELS_DIR" -ori "$ORIENTATION" -r -w "$JOBS"
    fi
    # Anatomy segmentations (SEG_SPINE / SEG_CORD / SEG_CANAL) share the same
    # 3D grid as the image; reorient them alongside it and resample with NN.
    for SEG_FOLDER in segs_spine segs_cord segs_canal; do
        SEG_DIR="$OUT_ROOT/$split/$SEG_FOLDER"
        if [ -d "$SEG_DIR" ] && [ -n "$(ls -A "$SEG_DIR" 2>/dev/null)" ]; then
            discslabel_reorient_image -i "$SEG_DIR" -o "$SEG_DIR" -ori "$ORIENTATION" -r -w "$JOBS"
        fi
    done

    echo "[prepare] resampling split '$split' to $RESOLUTION"
    discslabel_resample_image -i "$IMAGES_DIR" -o "$IMAGES_DIR" -res "$RESOLUTION" -int linear -r -w "$JOBS"
    if [ -d "$LABELS_DIR" ] && [ -n "$(ls -A "$LABELS_DIR" 2>/dev/null)" ]; then
        # 'label' interpolation: Gaussian-sphere around each point, resample
        # linearly, then extract the sphere centre. This preserves the exact
        # disc identity across the resample without the aliasing you get from
        # plain nearest-neighbour on single-voxel landmarks.
        label_args=(-i "$LABELS_DIR" -o "$LABELS_DIR" -res "$RESOLUTION"
                    -int label --label-sigma "$LABEL_SIGMA" -r -w "$JOBS")
        if [ -n "$DEBUG_DIR" ]; then
            label_args+=(--debug-dir "$DEBUG_DIR/$split")
        fi
        discslabel_resample_image "${label_args[@]}"
    fi
    # Anatomy segmentations use nearest-neighbour interpolation since they
    # carry an integer class per voxel and we don't want fractional values.
    for SEG_FOLDER in segs_spine segs_cord segs_canal; do
        SEG_DIR="$OUT_ROOT/$split/$SEG_FOLDER"
        if [ -d "$SEG_DIR" ] && [ -n "$(ls -A "$SEG_DIR" 2>/dev/null)" ]; then
            discslabel_resample_image -i "$SEG_DIR" -o "$SEG_DIR" -res "$RESOLUTION" -int nn -r -w "$JOBS"
        fi
    done
done

echo "[prepare] done. Manifest at $OUT_ROOT/manifest.json"
