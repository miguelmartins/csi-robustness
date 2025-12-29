#!/bin/bash
uv run dislib/identifiability.py --rep 0 --aug "none" --dataset shapes3d
uv run dislib/identifiability.py --rep 0 --aug "crop" --dataset shapes3d
uv run dislib/identifiability.py --rep 0 --aug "sup" --dataset shapes3d
uv run dislib/identifiability.py --rep 0 --aug "sup2" --dataset shapes3d
uv run dislib/identifiability.py --rep 0 --aug "simclr2" --dataset shapes3d

uv run dislib/identifiability.py --rep 1 --aug "none" --dataset shapes3d
uv run dislib/identifiability.py --rep 1 --aug "crop" --dataset shapes3d
uv run dislib/identifiability.py --rep 1 --aug "sup" --dataset shapes3d
uv run dislib/identifiability.py --rep 1 --aug "sup2" --dataset shapes3d
uv run dislib/identifiability.py --rep 1 --aug "simclr2" --dataset shapes3d

uv run dislib/identifiability.py --rep 2 --aug "none" --dataset shapes3d
uv run dislib/identifiability.py --rep 2 --aug "crop" --dataset shapes3d
uv run dislib/identifiability.py --rep 2 --aug "sup" --dataset shapes3d
uv run dislib/identifiability.py --rep 2 --aug "sup2" --dataset shapes3d
uv run dislib/identifiability.py --rep 2 --aug "simclr2" --dataset shapes3d
