#!/bin/bash

set -euo pipefail
mkdir -p logs_out

nohup bash -c 'CUDA_VISIBLE_DEVICES=0 uv run dislib/replicate_diet.py --aug none --rep 0 --backbone cnn' > logs_out/none.txt 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 uv run dislib/replicate_diet.py --aug crop --rep 0 --backbone cnn' > logs_out/crop.txt 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 uv run dislib/replicate_diet.py --aug simclr2 --rep 0 --backbone cnn' > logs_out/simclr2.txt 2>&1 &
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 uv run dislib/replicate_diet.py --aug geom_crop --rep 0 --backbone cnn' > logs_out/geom_crop.txt 2>&1 &

disown -a || true

