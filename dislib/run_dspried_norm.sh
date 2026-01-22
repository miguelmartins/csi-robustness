#!/bin/bash
CUDA_VISIBLE_DEVICES=1 uv run dislib/replicate_diet.py --dataset dsprites --aug crop --rep 3
CUDA_VISIBLE_DEVICES=1 uv run dislib/replicate_diet.py --dataset dsprites --aug sup --rep 3
CUDA_VISIBLE_DEVICES=1 uv run dislib/replicate_diet.py --dataset dsprites --aug sup2 --rep 3
CUDA_VISIBLE_DEVICES=1 uv run dislib/replicate_diet.py --dataset dsprites --aug geom_crop --rep 3
