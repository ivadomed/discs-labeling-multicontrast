#!/bin/bash

# Download BIDS datasets referenced by a discslabel dataset JSON.
#
# The JSON schema is:
#   SOURCE[i] / COMMIT[i]       paired git-annex repo URL and checkout commit
#   TRAINING / VALIDATION /
#   TESTING                     lists of {IMAGE, LABEL} paths relative to
#                               $DISCSLABEL_DATA/bids
#
# Required environment:
#   DISCSLABEL_DATA   root folder that will contain bids/ and prepared/

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

bids="$DISCSLABEL_DATA"/bids
mkdir -p "$bids"
CURR_DIR="$(realpath .)"
cd "$bids"

while IFS=$'\t' read -r source commit; do
    ds="$source"
    dsn=$(basename "$ds" .git)
    if [ ! -d "$dsn" ]; then
        git clone "$ds"
    else
        echo "Dataset $dsn already exists. Skipping clone."
    fi
    cd "$dsn"
    git checkout "$commit"
    cd ..
done < <(jq -r '[.SOURCE, .COMMIT] | transpose | .[] | @tsv' "$data_json")

for key in IMAGE LABEL SEG_SPINE SEG_CORD SEG_CANAL; do
    for split in TRAINING VALIDATION TESTING; do
        for path in $(jq -r ".${split} | .[].${key} // empty" "$data_json"); do
            IFS='/' read -r rep_path rel_path <<< "$path"
            git -C "$rep_path" annex get "$rel_path"
        done
    done
done

cd "$CURR_DIR"
