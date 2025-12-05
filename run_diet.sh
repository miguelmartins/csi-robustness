#!/bin/bash
uv run dislib/diet.py --setting 0 --aug "none"
uv run dislib/diet.py --setting 0 --aug "crop"
uv run dislib/diet.py --setting 0 --aug "sup"
uv run dislib/diet.py --setting 0 --aug "simclr"
uv run dislib/diet.py --setting 0 --aug "simclr2"
uv run dislib/diet.py --setting 0 --aug "simclr3"
