#!/bin/bash 
# ===============================
# DATASET: dsprites
# ===============================
uv run dislib/pgd.py --dataset dsprites --aug none --rep 0
uv run dislib/pgd.py --dataset dsprites --aug none --rep 1
uv run dislib/pgd.py --dataset dsprites --aug none --rep 2
uv run dislib/pgd.py --dataset dsprites --aug none --rep 3

uv run dislib/pgd.py --dataset dsprites --aug crop --rep 0
uv run dislib/pgd.py --dataset dsprites --aug crop --rep 1
uv run dislib/pgd.py --dataset dsprites --aug crop --rep 2
uv run dislib/pgd.py --dataset dsprites --aug crop --rep 3

uv run dislib/pgd.py --dataset dsprites --aug sup --rep 0
uv run dislib/pgd.py --dataset dsprites --aug sup --rep 1
uv run dislib/pgd.py --dataset dsprites --aug sup --rep 2
uv run dislib/pgd.py --dataset dsprites --aug sup --rep 3

uv run dislib/pgd.py --dataset dsprites --aug sup2 --rep 0
uv run dislib/pgd.py --dataset dsprites --aug sup2 --rep 1
uv run dislib/pgd.py --dataset dsprites --aug sup2 --rep 2
uv run dislib/pgd.py --dataset dsprites --aug sup2 --rep 3


# ===============================
# DATASET: cars3d
# ===============================
uv run dislib/pgd.py --dataset cars3d --aug none --rep 0
uv run dislib/pgd.py --dataset cars3d --aug none --rep 1
uv run dislib/pgd.py --dataset cars3d --aug none --rep 2

uv run dislib/pgd.py --dataset cars3d --aug crop --rep 0
uv run dislib/pgd.py --dataset cars3d --aug crop --rep 1
uv run dislib/pgd.py --dataset cars3d --aug crop --rep 2

uv run dislib/pgd.py --dataset cars3d --aug sup --rep 0
uv run dislib/pgd.py --dataset cars3d --aug sup --rep 1
uv run dislib/pgd.py --dataset cars3d --aug sup --rep 2

uv run dislib/pgd.py --dataset cars3d --aug sup2 --rep 0
uv run dislib/pgd.py --dataset cars3d --aug sup2 --rep 1
uv run dislib/pgd.py --dataset cars3d --aug sup2 --rep 2

uv run dislib/pgd.py --dataset cars3d --aug simclr2 --rep 0
uv run dislib/pgd.py --dataset cars3d --aug simclr2 --rep 1
uv run dislib/pgd.py --dataset cars3d --aug simclr2 --rep 2
