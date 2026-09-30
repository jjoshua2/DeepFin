#!/usr/bin/env python3
"""Freeze Syzygy path/size/mtime metadata without reading tablebase contents."""
from __future__ import annotations

import argparse
from pathlib import Path

from chess_anti_engine.eval.arena_durable import prepare_tablebase_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--syzygy", required=True,
                        help="ordered, colon-separated Syzygy directories")
    args = parser.parse_args()
    try:
        digest = prepare_tablebase_inventory(args.output, args.syzygy)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"{digest}  {args.output}", flush=True)


if __name__ == "__main__":
    main()
