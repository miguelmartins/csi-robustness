#!/bin/bash

DATASET="shapes3d"

for REP in 0 1 2; do
  uv run dislib/identifiability.py --rep "$REP" --aug "none"     --dataset "$DATASET"
  uv run dislib/identifiability.py --rep "$REP" --aug "crop"     --dataset "$DATASET"
  uv run dislib/identifiability.py --rep "$REP" --aug "sup"      --dataset "$DATASET"
  uv run dislib/identifiability.py --rep "$REP" --aug "sup2"     --dataset "$DATASET"
  uv run dislib/identifiability.py --rep "$REP" --aug "simclr2"  --dataset "$DATASET"
done

