#!/usr/bin/env bash
set -eu

CHECKPOINT="${1:?Usage: bash infer_pristine_split.sh CHECKPOINT OUTPUT_DIR MODE [STEPS] [BATCH_SIZE]  (MODE: safe|unsafe)}"
OUTPUT_DIR="${2:?Pass a local output directory}"
MODE="${3:?Pass the model mode: safe or unsafe}"

CFG=3
STEPS="${4:-200}"
BATCH_SIZE="${5:-128}"
SEED=42
N_GPUS=8
BENCHMARKS=(imagenet clearbg compositional implicit)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pids=()
for ((rank = 0; rank < N_GPUS; rank++)); do
    CUDA_VISIBLE_DEVICES="$rank" python3 "$SCRIPT_DIR/../infer_pristine_split.py" \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$OUTPUT_DIR" \
        --mode "$MODE" \
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
