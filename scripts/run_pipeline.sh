#!/bin/bash

# Read a pipeline JSON config, run the enabled pipeline steps, and archive the
# config next to the training weights.
#
# Steps:
#   download  — git clone + git annex get the BIDS data referenced by data_json
#   prepare   — copy to prepared/, reorient to RPI, resample to 1 mm iso
#   train     — call discslabel_train on the prepared manifest
#   inference — call discslabel_inference on the chosen split
#
# Each step runs when either:
#   - steps.<name> is true in the config, OR
#   - the step's output is missing from disk (so a fresh checkout always catches up).
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

# Require a value — error out when missing.
jreq() {
    local val
    val=$(jq -r "$1 // empty" "$config_json")
    if [ -z "$val" ]; then
        echo "config: $1 is required"
        exit 1
    fi
    printf '%s' "$val"
}

# Optional value — $2 is the jq fallback (quoted when a string).
# `if == null` so an explicit `false` is kept rather than overwritten.
jopt() { jq -r "if $1 == null then $2 else $1 end" "$config_json"; }

# -------- Read the config --------
DATA_JSON=$(jreq '.data_json')
OUT_NAME=$(jopt  '.out_name' '"'$(basename "${DATA_JSON%.json}")'"')
RUN_NAME=$(jopt  '.run_name' '"'$OUT_NAME'_UNet"')

ORIENTATION=$(jopt '.orientation'  '"RPI"')
RESOLUTION=$(jopt  '.resolution'   '"1x1x1"')
LABEL_SIGMA=$(jopt '.label_sigma'  '1.5')
DO_DEBUG=$(jopt '.debug' 'false')

DO_DOWNLOAD=$(jopt  '.steps.download'  'false')
DO_PREPARE=$(jopt   '.steps.prepare'   'false')
DO_TRAIN=$(jopt     '.steps.train'     'false')
DO_INFERENCE=$(jopt '.steps.inference' 'false')
DO_EVALUATE=$(jopt  '.steps.evaluate'  'false')

INFERENCE_SPLIT=$(jopt       '.inference_split'       '"test"')
INFERENCE_SAVE_IMAGES=$(jopt '.inference_save_images' 'false')

# -------- Resolve data_json to a file --------
if [ ! -f "$DATA_JSON" ]; then
    cand="$PROJECT_ROOT/discslabel/datasets/$(basename "${DATA_JSON%.json}").json"
    if [ ! -f "$cand" ]; then
        echo "Error: Could not find data JSON file: $DATA_JSON (also tried $cand)"
        exit 1
    fi
    DATA_JSON="$cand"
fi
DATA_JSON="$(realpath "$DATA_JSON")"

# -------- Derived paths --------
BIDS_ROOT="$DISCSLABEL_DATA/bids"
PREP_ROOT="$DISCSLABEL_DATA/prepared/$OUT_NAME"
MANIFEST="$PREP_ROOT/manifest.json"
WEIGHTS_DIR="$DISCSLABEL_DATA/weights/$RUN_NAME"
EVAL_JSON="$PROJECT_ROOT/discslabel/datasets/evaluation.json"
EVAL_ROOT="$DISCSLABEL_DATA/evaluation"
EVAL_PRED_DIR="$EVAL_ROOT/prediction_$RUN_NAME"
EVAL_METRICS_DIR="$EVAL_ROOT/metrics_$RUN_NAME"
if [ "$DO_DEBUG" = "true" ]; then
    DEBUG_DIR="$DISCSLABEL_DATA/debug/$OUT_NAME"
else
    DEBUG_DIR=""
fi

# Does `bids/` already hold every repo referenced by data_json?
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
echo "Pipeline steps (config): download=$DO_DOWNLOAD prepare=$DO_PREPARE train=$DO_TRAIN inference=$DO_INFERENCE evaluate=$DO_EVALUATE"
echo "Config:   $config_json"
echo "Data:     $DATA_JSON"
echo "Prepared: $PREP_ROOT"
echo "Weights:  $WEIGHTS_DIR"
echo ""

# -------- 1. Download --------
if [ "$DO_DOWNLOAD" = "true" ] || ! _bids_complete; then
    if [ "$DO_DOWNLOAD" != "true" ]; then
        echo "==> download_datasets.sh (auto: BIDS data missing)"
    else
        echo "==> download_datasets.sh"
    fi
    "$SCRIPT_DIR/download_datasets.sh" "$DATA_JSON"
    # Evaluation BIDS repos travel with the training download so a one-shot
    # ``steps.download=true`` run brings down everything needed for both.
    if [ -f "$EVAL_JSON" ]; then
        echo "==> download_datasets.sh (evaluation)"
        "$SCRIPT_DIR/download_datasets.sh" "$EVAL_JSON"
    fi
else
    echo "==> skipping download (BIDS data already present and steps.download=false)"
fi

# -------- 2. Prepare --------
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

# -------- 3. Archive the pipeline config next to the weights --------
mkdir -p "$WEIGHTS_DIR"
cp "$config_json" "$WEIGHTS_DIR/config.json"
cp "$DATA_JSON"   "$WEIGHTS_DIR/data.json"
pip freeze > "$WEIGHTS_DIR/requirements.txt" 2>/dev/null || true
echo "Config archived to $WEIGHTS_DIR/{config.json,data.json,requirements.txt}"

# -------- 4. Train --------
if [ "$DO_TRAIN" = "true" ]; then
    echo "==> discslabel_train"
    discslabel_train \
        --manifest "$MANIFEST" \
        --save-model-dir "$WEIGHTS_DIR" \
        --config "$config_json"
fi

# -------- 5. Inference --------
if [ "$DO_INFERENCE" = "true" ]; then
    echo "==> discslabel_inference"
    discslabel_inference \
        --weights "$WEIGHTS_DIR/best_model.pt" \
        --input "$PREP_ROOT/$INFERENCE_SPLIT/images" \
        --out-dir "$WEIGHTS_DIR/predictions_${INFERENCE_SPLIT}" \
        --config "$config_json"
fi

# -------- 6. Evaluate --------
if [ "$DO_EVALUATE" = "true" ]; then
    if [ ! -f "$EVAL_JSON" ]; then
        echo "Error: evaluate step enabled but $EVAL_JSON not found"
        exit 1
    fi
    if [ ! -f "$WEIGHTS_DIR/best_model.pt" ]; then
        echo "Error: evaluate step enabled but $WEIGHTS_DIR/best_model.pt not found"
        exit 1
    fi
    echo "==> prepare_evaluation.sh"
    "$SCRIPT_DIR/prepare_evaluation.sh" "$EVAL_JSON"

    echo "==> discslabel_inference (evaluation)"
    discslabel_inference \
        --weights "$WEIGHTS_DIR/best_model.pt" \
        --input "$EVAL_ROOT/images" \
        --out-dir "$EVAL_PRED_DIR" \
        --config "$config_json"

    echo "==> discslabel_evaluate"
    discslabel_evaluate \
        --gt-dir "$EVAL_ROOT/labels" \
        --pred-dir "$EVAL_PRED_DIR" \
        --out-dir "$EVAL_METRICS_DIR" \
        --run-name "$RUN_NAME"
    echo "Metrics for $RUN_NAME in $EVAL_METRICS_DIR"
    echo "Run 'discslabel_aggregate_evaluation' to compare all runs in $EVAL_ROOT/evaluation.json"
fi

echo ""
echo "Pipeline complete."
