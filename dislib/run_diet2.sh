#!/bin/bash
CUDA_VISIBLE_DEVICES=1 uv run dislib/unsup_diet.py --setting 2 --aug "none"
CUDA_VISIBLE_DEVICES=1 uv run dislib/unsup_diet.py --setting 2 --aug "crop"
CUDA_VISIBLE_DEVICES=1 uv run dislib/unsup_diet.py --setting 2 --aug "sup"
CUDA_VISIBLE_DEVICES=1 uv run dislib/unsup_diet.py --setting 2 --aug "simclr"
CUDA_VISIBLE_DEVICES=1 uv run dislib/unsup_diet.py --setting 2 --aug "simclr2"
CUDA_VISIBLE_DEVICES=1 uv run dislib/unsup_diet.py --setting 2 --aug "simclr3"
