#!/bin/bash

# Idempotently stage the evaluation dataset for inference + metrics.
#
# Steps (each one skipped if its output is already present):
#   1. Clone + git-annex get the BIDS repos listed in evaluation.json.
#   2. Copy every TESTING entry's IMAGE and LABEL into a flat folder:
#        $DISCSLABEL_DATA/evaluation/images/<dataset>__<basename>.nii.gz
#        $DISCSLABEL_DATA/evaluation/labels/<dataset>__<basename>.nii.gz
#      <dataset> is the first path segment of the BIDS-relative path, so
#      sub-ids that happen to collide across datasets stay separate. GT
#      labels get the same stem as the image so discslabel_evaluate pairs
#      them trivially.
#
# No reorient / resample is applied — ``discslabel_inference`` does its own
# preprocessing based on the pipeline config.
#
# Usage:
#   prepare_evaluation.sh <evaluation-json>
#
# Required environment:
#   DISCSLABEL_DATA   root folder that holds `bids/` and will receive
#                     `evaluation/{images,labels}/`.

set -e
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

if [ -z "$DISCSLABEL_DATA" ]; then
    echo "Please set the DISCSLABEL_DATA environment variable."
    exit 1
fi

eval_json="$1"
if [ -z "$eval_json" ] || [ ! -f "$eval_json" ]; then
    echo "Usage: $0 <evaluation-json>"
    exit 1
fi
eval_json="$(realpath "$eval_json")"

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
BIDS_ROOT="$DISCSLABEL_DATA/bids"
EVAL_ROOT="$DISCSLABEL_DATA/evaluation"
EVAL_IMAGES="$EVAL_ROOT/images"
EVAL_LABELS="$EVAL_ROOT/labels"

# 1. Download evaluation BIDS data (idempotent: clones are skipped when the
#    folder already exists; `git annex get` is a no-op for files already here).
_eval_bids_complete() {
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
    done < <(jq -r '.SOURCE[]? // empty' "$eval_json")
    return $missing
}

if ! _eval_bids_complete; then
    echo "==> download_datasets.sh (evaluation)"
    "$SCRIPT_DIR/download_datasets.sh" "$eval_json"
else
    echo "==> skipping evaluation download (BIDS repos already present)"
fi

# 2. Copy IMAGE + LABEL into the flat evaluation folders, keyed by a
#    dataset-prefixed stem so cross-dataset sub-id collisions can't clobber.
mkdir -p "$EVAL_IMAGES" "$EVAL_LABELS"
copied=0
skipped=0
missing=0
while IFS=$'\t' read -r image_rel label_rel; do
    [ -z "$image_rel" ] && continue
    [ -z "$label_rel" ] && continue

    dataset="${image_rel%%/*}"
    basename_full="$(basename "$image_rel")"
    stem="${basename_full%.nii.gz}"
    stem="${stem%.nii}"
    dst_stem="${dataset}__${stem}"

    src_image="$BIDS_ROOT/$image_rel"
    src_label="$BIDS_ROOT/$label_rel"
    dst_image="$EVAL_IMAGES/${dst_stem}.nii.gz"
    dst_label="$EVAL_LABELS/${dst_stem}.nii.gz"

    if [ ! -f "$src_image" ] || [ ! -f "$src_label" ]; then
        missing=$((missing + 1))
        continue
    fi

    if [ -f "$dst_image" ] && [ -f "$dst_label" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    [ ! -f "$dst_image" ] && cp "$src_image" "$dst_image"
    [ ! -f "$dst_label" ] && cp "$src_label" "$dst_label"
    copied=$((copied + 1))
done < <(jq -r '.TESTING[] | [.IMAGE, .LABEL] | @tsv' "$eval_json")

echo "==> evaluation staged in $EVAL_ROOT (copied=$copied skipped=$skipped missing_bids=$missing)"
