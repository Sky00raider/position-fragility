"""Pipeline CLI. Every step reads and writes jsonl under runs/<run_name>/ and is resumable.

    python -m src.pipeline prepare-data
    python -m src.pipeline baseline  --config configs/X.yaml
    python -m src.pipeline eligible  --config configs/X.yaml
    python -m src.pipeline inject    --config configs/X.yaml
    python -m src.pipeline continue  --config configs/X.yaml
    python -m src.pipeline score     --config configs/X.yaml
    python -m src.pipeline estimate  --config configs/X.yaml --tok-per-s 1500 --n-items 800
"""
from __future__ import annotations

import argparse
import sys

from . import data


def cmd_prepare_data(args):
    data.prepare(splits=args.splits)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252
    ap = argparse.ArgumentParser(prog="python -m src.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-data", help="download MuSiQue, filter 3/4-hop, build contexts")
    p.add_argument("--splits", nargs="+", default=["dev", "train"])
    p.set_defaults(fn=cmd_prepare_data)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
