#!/usr/bin/env python3
"""Independently reopen a tracked Ceres saved-game fixture archive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_anti_engine.source.ceres_saved_game import read_saved_game_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    game = read_saved_game_archive(args.output)
    print(json.dumps({
        "status": "PASS_CERES_TRACKED_SAVED_NATURAL_GAME_READBACK",
        "game_id": game.game["game_id"],
        "rows": len(game.rows),
        "model_sha256": game.model_sha256,
        "source_archive_sha256": game.source_archive_sha256,
        "training_ready": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
