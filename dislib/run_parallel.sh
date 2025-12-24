#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs_out

# Define reps you want to run
reps=(1 2)

for rep in "${reps[@]}"; do
  echo "=== Starting rep=${rep} ==="

  pids=()

  # rep=${rep} batch: launch in background
  CUDA_VISIBLE_DEVICES=0 nohup uv run dislib/replicate_diet.py --aug none      --rep "${rep}" --backbone cnn > "logs_out/none_rep${rep}.txt"      2>&1 & pids+=("$!")
  CUDA_VISIBLE_DEVICES=0 nohup uv run dislib/replicate_diet.py --aug crop      --rep "${rep}" --backbone cnn > "logs_out/crop_rep${rep}.txt"      2>&1 & pids+=("$!")
  CUDA_VISIBLE_DEVICES=1 nohup uv run dislib/replicate_diet.py --aug sup       --rep "${rep}" --backbone cnn > "logs_out/sup_rep${rep}.txt"       2>&1 & pids+=("$!")
  CUDA_VISIBLE_DEVICES=1 nohup uv run dislib/replicate_diet.py --aug geom_crop --rep "${rep}" --backbone cnn > "logs_out/geom_crop_rep${rep}.txt" 2>&1 & pids+=("$!")

  # Barrier: only continue when all jobs of this rep finished
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  echo "=== Finished rep=${rep} ==="
done

echo "All reps finished."

