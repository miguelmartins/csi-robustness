#!/bin/bash
CUDA_VISIBLE_DEVICES=0 uv run dislib/unsup_diet.py --setting 1 --aug "none"
CUDA_VISIBLE_DEVICES=0 uv run dislib/unsup_diet.py --setting 1 --aug "crop"
CUDA_VISIBLE_DEVICES=0 uv run dislib/unsup_diet.py --setting 1 --aug "sup"
CUDA_VISIBLE_DEVICES=0 uv run dislib/unsup_diet.py --setting 1 --aug "simclr"
CUDA_VISIBLE_DEVICES=0 uv run dislib/unsup_diet.py --setting 1 --aug "simclr2"
CUDA_VISIBLE_DEVICES=0 uv run dislib/unsup_diet.py --setting 1 --aug "simclr3"
