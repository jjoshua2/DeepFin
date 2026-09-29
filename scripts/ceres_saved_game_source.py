#!/usr/bin/env python3
"""Replay one pinned Ceres saved natural game through tracked ZIP publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_anti_engine.source.ceres_saved_game import load_saved_natural_game, write_saved_game


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--game-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    game = load_saved_natural_game(args.source_zip, args.source_sha256, args.game_id)
    receipt = write_saved_game(game, args.output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
