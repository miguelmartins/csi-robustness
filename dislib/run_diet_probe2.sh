#!/bin/bash
CUDA_VISIBLE_DEVICES=0 uv run dislib/probe_diet.py --setting 2 --aug "none"
CUDA_VISIBLE_DEVICES=0 uv run dislib/probe_diet.py --setting 2 --aug "crop"
CUDA_VISIBLE_DEVICES=0 uv run dislib/probe_diet.py --setting 2 --aug "sup"
CUDA_VISIBLE_DEVICES=0 uv run dislib/probe_diet.py --setting 2 --aug "simclr"
CUDA_VISIBLE_DEVICES=0 uv run dislib/probe_diet.py --setting 2 --aug "simclr2"
CUDA_VISIBLE_DEVICES=0 uv run dislib/probe_diet.py --setting 2 --aug "simclr3"
