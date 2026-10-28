#!/bin/bash

# 3D counterpart to run_pipeline.sh. Reads the same pipeline JSON config, runs
# the enabled pipeline steps, and archives the config next to the training
# weights. The train and inference steps call the 3D entry points
# (`discslabel_train_3d`, `discslabel_inference_3d`); download and prepare are
# identical to the 2D runner since the on-disk data is dimension-agnostic.
#
# Required environment:
#   DISCSLABEL_DATA   root folder for `bids/`, `prepared/<name>/`, `weights/<name>/`

set -e
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

if [ -z "$DISCSLABEL_DATA" ]; then
    echo "Please set the DISCSLABEL_DATA environment variable."
    exit 1
fi

config_json="$1"
if [ -z "$config_json" ] || [ ! -f "$config_json" ]; then
    echo "Usage: $0 <pipeline-config.json>"
    exit 1
fi
config_json="$(realpath "$config_json")"

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
PROJECT_ROOT="$(realpath "$(dirname "$0")/..")"

jreq() {
    local val
    val=$(jq -r "$1 // empty" "$config_json")
    if [ -z "$val" ]; then
        echo "config: $1 is required"
        exit 1
    fi
    printf '%s' "$val"
}

jopt() { jq -r "if $1 == null then $2 else $1 end" "$config_json"; }

DATA_JSON=$(jreq '.data_json')
OUT_NAME=$(jopt  '.out_name' '"'$(basename "${DATA_JSON%.json}")'"')
RUN_NAME=$(jopt  '.run_name' '"'$OUT_NAME'_UNet3D"')

ORIENTATION=$(jopt '.orientation'  '"RPI"')
RESOLUTION=$(jopt  '.resolution'   '"1x1x1"')
LABEL_SIGMA=$(jopt '.label_sigma'  '1.5')
DO_DEBUG=$(jopt '.debug' 'false')

DO_DOWNLOAD=$(jopt '.steps.download'  'false')
DO_PREPARE=$(jopt  '.steps.prepare'   'false')
DO_TRAIN=$(jopt    '.steps.train'     'false')
DO_INFERENCE=$(jopt '.steps.inference' 'false')

INFERENCE_SPLIT=$(jopt       '.inference_split'       '"test"')
INFERENCE_SAVE_IMAGES=$(jopt '.inference_save_images' 'false')

if [ ! -f "$DATA_JSON" ]; then
    cand="$PROJECT_ROOT/discslabel/datasets/$(basename "${DATA_JSON%.json}").json"
    if [ ! -f "$cand" ]; then
        echo "Error: Could not find data JSON file: $DATA_JSON (also tried $cand)"
        exit 1
    fi
    DATA_JSON="$cand"
fi
DATA_JSON="$(realpath "$DATA_JSON")"

BIDS_ROOT="$DISCSLABEL_DATA/bids"
PREP_ROOT="$DISCSLABEL_DATA/prepared/$OUT_NAME"
MANIFEST="$PREP_ROOT/manifest.json"
WEIGHTS_DIR="$DISCSLABEL_DATA/weights/$RUN_NAME"
if [ "$DO_DEBUG" = "true" ]; then
    DEBUG_DIR="$DISCSLABEL_DATA/debug/$OUT_NAME"
else
    DEBUG_DIR=""
fi

_bids_complete() {
    [ -d "$BIDS_ROOT" ] || return 1
    local missing=0
    while IFS= read -r src; do
        [ -z "$src" ] && continue
        local dsn
        dsn=$(basename "$src" .git)
        if [ ! -d "$BIDS_ROOT/$dsn" ]; then
            missing=1
            break
        fi
    done < <(jq -r '.SOURCE[]? // empty' "$DATA_JSON")
    return $missing
}

echo ""
echo "Pipeline steps (config, 3D): download=$DO_DOWNLOAD prepare=$DO_PREPARE train=$DO_TRAIN inference=$DO_INFERENCE"
echo "Config:   $config_json"
echo "Data:     $DATA_JSON"
echo "Prepared: $PREP_ROOT"
echo "Weights:  $WEIGHTS_DIR"
echo ""

if [ "$DO_DOWNLOAD" = "true" ] || ! _bids_complete; then
    if [ "$DO_DOWNLOAD" != "true" ]; then
        echo "==> download_datasets.sh (auto: BIDS data missing)"
    else
        echo "==> download_datasets.sh"
    fi
    "$SCRIPT_DIR/download_datasets.sh" "$DATA_JSON"
else
    echo "==> skipping download (BIDS data already present and steps.download=false)"
fi

if [ "$DO_PREPARE" = "true" ] || [ ! -f "$MANIFEST" ]; then
    if [ "$DO_PREPARE" != "true" ]; then
        echo "==> prepare_datasets.sh (auto: $MANIFEST missing)"
    else
        echo "==> prepare_datasets.sh"
    fi
    "$SCRIPT_DIR/prepare_datasets.sh" "$DATA_JSON" "$OUT_NAME" "$ORIENTATION" "$RESOLUTION" "$LABEL_SIGMA" "$DEBUG_DIR"
else
    echo "==> skipping prepare ($MANIFEST already exists and steps.prepare=false)"
fi

mkdir -p "$WEIGHTS_DIR"
cp "$config_json" "$WEIGHTS_DIR/config.json"
cp "$DATA_JSON"   "$WEIGHTS_DIR/data.json"
pip freeze > "$WEIGHTS_DIR/requirements.txt" 2>/dev/null || true
echo "Config archived to $WEIGHTS_DIR/{config.json,data.json,requirements.txt}"

if [ "$DO_TRAIN" = "true" ]; then
    echo "==> discslabel_train_3d"
    discslabel_train_3d \
        --manifest "$MANIFEST" \
        --save-model-dir "$WEIGHTS_DIR" \
        --config "$config_json"
fi

if [ "$DO_INFERENCE" = "true" ]; then
    echo "==> discslabel_inference_3d"
    discslabel_inference_3d \
        --weights "$WEIGHTS_DIR/best_model.pt" \
        --input "$PREP_ROOT/$INFERENCE_SPLIT/images" \
        --out-dir "$WEIGHTS_DIR/predictions_${INFERENCE_SPLIT}" \
        --config "$config_json"
fi

echo ""
echo "Pipeline complete."
