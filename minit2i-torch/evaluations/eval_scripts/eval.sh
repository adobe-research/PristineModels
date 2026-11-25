#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    printf 'Usage: bash eval.sh IMAGE_DIR OUTPUT_JSON [evaluation options]\n' >&2
    exit 2
fi

IMAGE_DIR="$1"
OUTPUT_JSON="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/../eval_imagenet.py" "$IMAGE_DIR" \
    --output-json "$OUTPUT_JSON" \
    --temp-dir /tmp \
    "$@"