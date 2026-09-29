#!/usr/bin/env python3
"""Independently reopen a tracked Ceres saved-game fixture archive."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

from chess_anti_engine.source.ceres_saved_game import read_saved_game_archive
from chess_anti_engine.tablebase import open_strict_match_tablebase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--syzygy-path", help="Exact saved path pair for strict six-man WDL/DTZ")
    args = parser.parse_args()
    handle = (contextlib.closing(open_strict_match_tablebase(args.syzygy_path, max_pieces=6))
              if args.syzygy_path else contextlib.nullcontext(None))
    with handle as match_tablebase:
        game = read_saved_game_archive(args.output, syzygy_path=args.syzygy_path,
                                       match_tablebase=match_tablebase)
    print(json.dumps({
        "status": "PASS_CERES_TRACKED_SAVED_GAME_READBACK",
        "termination": game.game["termination"],
        "game_id": game.game["game_id"],
        "rows": len(game.rows),
        "model_sha256": game.model_sha256,
        "source_archive_sha256": game.source_archive_sha256,
        "training_ready": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
