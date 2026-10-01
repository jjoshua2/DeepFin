#!/usr/bin/env python3
"""Freeze the inputs for one fixed, resumable 576-pair arena.

This command never reads tablebase data files. It binds their frozen
path/size/mtime inventory; that is metadata identity, not content identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_anti_engine.eval.arena_durable import prepare_source_seal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--argv-json", required=True, type=Path,
                        help="JSON array of the exact future arena argv, excluding "
                             "--durable-seal and --durable-seal-sha256")
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--openings-fen", required=True, type=Path)
    parser.add_argument("--production-config", required=True, type=Path)
    parser.add_argument("--tablebase-catalog", required=True, type=Path)
    parser.add_argument("--syzygy", required=True)
    args = parser.parse_args()
    argv = json.loads(args.argv_json.read_text())
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        parser.error("--argv-json must contain a JSON array of strings")
    try:
        digest = prepare_source_seal(
            args.output, argv=argv, candidate=args.candidate,
            reference=args.reference, openings=args.openings_fen,
            config=args.production_config,
            catalog_path=args.tablebase_catalog, syzygy_path=args.syzygy,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"{digest}  {args.output}", flush=True)


if __name__ == "__main__":
    main()
