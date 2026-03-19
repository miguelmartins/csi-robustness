#!/bin/bash 
uv run dislib/pgd.py --dataset smallnorb --aug none --rep 0
uv run dislib/pgd.py --dataset smallnorb --aug none --rep 1
uv run dislib/pgd.py --dataset smallnorb --aug none --rep 2


uv run dislib/pgd.py --dataset smallnorb --aug crop --rep 0
uv run dislib/pgd.py --dataset smallnorb --aug crop --rep 1
uv run dislib/pgd.py --dataset smallnorb --aug crop --rep 2

uv run dislib/pgd.py --dataset smallnorb --aug sup --rep 0
uv run dislib/pgd.py --dataset smallnorb --aug sup --rep 1
uv run dislib/pgd.py --dataset smallnorb --aug sup --rep 2

uv run dislib/pgd.py --dataset smallnorb --aug sup2 --rep 0
uv run dislib/pgd.py --dataset smallnorb --aug sup2 --rep 1
uv run dislib/pgd.py --dataset smallnorb --aug sup2 --rep 2


uv run dislib/pgd.py --dataset smallnorb --aug simclr2 --rep 0
uv run dislib/pgd.py --dataset smallnorb --aug simclr2 --rep 1
uv run dislib/pgd.py --dataset smallnorb --aug simclr2 --rep 2


