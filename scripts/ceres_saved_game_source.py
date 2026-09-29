#!/usr/bin/env python3
"""Replay one pinned Ceres saved game through tracked ZIP publication."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

from chess_anti_engine.source.ceres_saved_game import load_saved_game, write_saved_game
from chess_anti_engine.tablebase import open_strict_match_tablebase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--game-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--syzygy-path", help="Exact saved path pair for strict six-man WDL/DTZ")
    args = parser.parse_args()
    handle = (contextlib.closing(open_strict_match_tablebase(args.syzygy_path, max_pieces=6))
              if args.syzygy_path else contextlib.nullcontext(None))
    with handle as match_tablebase:
        game = load_saved_game(args.source_zip, args.source_sha256, args.game_id,
                               syzygy_path=args.syzygy_path, match_tablebase=match_tablebase)
        receipt = write_saved_game(game, args.output, syzygy_path=args.syzygy_path,
                                   match_tablebase=match_tablebase)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
