#!/bin/bash
rep = 0
uv run dislib/replicate_diet.py --dataset shapes3d --aug none      --rep "${rep}" --backbone cnn
uv run dislib/replicate_diet.py --dataset shapes3d --aug crop      --rep "${rep}" --backbone cnn
uv run dislib/replicate_diet.py --dataset shapes3d --aug sup      --rep "${rep}" --backbone cnn
uv run dislib/replicate_diet.py --dataset shapes3d --aug sup2      --rep "${rep}" --backbone cnn
uv run dislib/replicate_diet.py --dataset shapes3d --aug simclr2      --rep "${rep}" --backbone cnn
uv run dislib/replicate_diet.py --dataset shapes3d --aug simclr3      --rep "${rep}" --backbone cnn
