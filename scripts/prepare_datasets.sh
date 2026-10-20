#!/bin/bash

# Prepare mid-sagittal slices and landmark CSVs from BIDS NIfTIs.
#
# This is a thin wrapper around `discslabel_prepare` so the three-step
# pipeline (download -> prepare -> train) matches SmaugLab-benchmark's shape.

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

out_name="${2:-}"
if [ -z "$out_name" ]; then
    discslabel_prepare --data-json "$data_json"
else
    discslabel_prepare --data-json "$data_json" --out-name "$out_name"
fi
