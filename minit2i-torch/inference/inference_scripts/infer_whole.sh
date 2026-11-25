#!/usr/bin/env bash
set -eu

CHECKPOINT="${1:?Usage: bash infer_whole.sh CHECKPOINT OUTPUT_DIR [CFG] [STEPS] [BATCH_SIZE]}"
OUTPUT_DIR="${2:?Pass a local output directory}"

CFG="${3:-4}"
STEPS="${4:-200}"
BATCH_SIZE="${5:-128}"
SEED=42
N_GPUS=8
BENCHMARKS=(imagenet clearbg compositional implicit)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pids=()
for ((rank = 0; rank < N_GPUS; rank++)); do
    CUDA_VISIBLE_DEVICES="$rank" python3 "$SCRIPT_DIR/../infer_whole.py" \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --cfg_scale "$CFG" --n_T "$STEPS" --seed "$SEED" \
        --benchmarks "${BENCHMARKS[@]}" \
        --rank "$rank" --world_size "$N_GPUS" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"