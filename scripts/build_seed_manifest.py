#!/usr/bin/env python3
"""Build a ``<list>.manifest.json`` seed-provenance manifest from a live seed list.

The manifest maps each blind-spot seed FEN's position identity (first 4 FEN
fields) to a stable ``(seed_id, seed_family_id)`` pair. ``finalize.py`` resolves
a game's ``start_fen`` against this manifest (via
``chess_anti_engine.selfplay.seed_manifest.resolve_seed_ids``) to stamp
provenance onto each training replay row.

seed_id here is the enumeration index and is only stable WITHIN one list
version — the retire step rewrites the list, shifting indices. For any
cross-version analysis use ``content_id`` (the content hash finalize stamps in
distributed production), which is stable by construction. seed_family_id is a
v1 PLACEHOLDER equal to seed_id — real near-transposition family grouping
comes later.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import chess

from chess_anti_engine.selfplay.opening import _load_fen_list, seed_board_from_line
from chess_anti_engine.selfplay.seed_manifest import content_seed_id, position_key


def _derived_content_ids(list_path: str) -> dict[str, str]:
    """content_id -> parent seed FEN for every position along each seed's
    baked refute line (``FEN | uci uci ...`` suffix in the raw list).

    sf_refute games (opening_source_code=3) can start from a position partway
    down the refute line, so their shard ``seed_id`` hashes a derived position,
    not the seed itself. This map lets offline analysis collapse those ids back
    to the parent seed (values match ``by_content_id``'s seed FENs).
    """
    out: dict[str, str] = {}
    with open(list_path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            fen_part, _, moves_part = raw.partition("|")
            fen = fen_part.split("#")[0].strip()
            if not fen or not moves_part.strip():
                continue
            try:
                board = chess.Board(fen)
            except ValueError:
                continue
            positions = [board.fen()]
            for mv in moves_part.split():
                try:
                    board.push_uci(mv)
                except ValueError:
                    break
                positions.append(board.fen())
            seed_fen = positions[-1]  # terminal = the seed identity
            for pos_fen in positions:
                out[str(content_seed_id(pos_fen))] = seed_fen
    return out


def _default_list_path() -> str | None:
    """Best-effort read of ``selfplay.opening_fen_list_path`` from the prod config."""
    try:
        from chess_anti_engine.utils.config_yaml import load_yaml_file
        cfg = load_yaml_file("configs/pbt2_small.yaml")
    except Exception:
        return None
    selfplay = cfg.get("selfplay", {})
    path = selfplay.get("opening_fen_list_path") if isinstance(selfplay, dict) else None
    return str(path) if path else None


def build_manifest(list_path: str) -> dict[str, object]:
    lines = _load_fen_list(list_path)
    seeds: list[dict[str, object]] = []
    by_key: dict[str, list[int]] = {}
    by_content_id: dict[str, str] = {}  # production seed_id (hash) -> seed FEN, for offline joins
    dup_count = 0
    bad_count = 0
    for seed_id, line in enumerate(lines):
        # seed_family_id is a v1 PLACEHOLDER (== seed_id); real near-transposition
        # grouping comes later.
        seed_family_id = seed_id
        # The seed IDENTITY is the TERMINAL position of the line (base FEN with
        # any `| uci ...` history replayed) as chess.Board.fen() renders it —
        # exactly what finalize.py sees as start_fen. Keying the raw base-FEN
        # string instead breaks BOTH joins for suffixed seeds (wrong position)
        # and for cosmetic-ep FENs (Board.fen() drops a no-legal-capture ep).
        try:
            seed_fen = seed_board_from_line(line).fen()
        except ValueError:
            bad_count += 1
            continue
        key = position_key(seed_fen)
        # content_id is what finalize actually stamps in distributed mode (the
        # worker's list path is ephemeral, so the path-keyed manifest never
        # resolves there). Enrichment analysis joins shard seed_id -> this.
        content_id = content_seed_id(seed_fen)
        seeds.append({
            "seed_id": seed_id,
            "seed_family_id": seed_family_id,
            "content_id": content_id,
            "fen": seed_fen,
            "line": line,
            "position_key": key,
        })
        by_content_id[str(content_id)] = seed_fen
        if key in by_key:
            # Two list entries share a position identity: keep the FIRST.
            dup_count += 1
            continue
        by_key[key] = [seed_id, seed_family_id]
    if dup_count:
        print(f"warning: {dup_count} duplicate position_key(s); kept the first each")
    if bad_count:
        print(f"warning: {bad_count} unparseable seed line(s) skipped")
    # Intermediate refute-line positions hash to their own content_ids in
    # shards (sf_refute games can start partway down the line); map each back
    # to the parent seed FEN so analysis can collapse them.
    derived = _derived_content_ids(list_path)
    return {
        "version": 2,
        "list_path": os.path.abspath(list_path),
        "n_seeds": len(lines),
        "seeds": seeds,
        "by_key": by_key,
        "by_content_id": by_content_id,
        "by_derived_content_id": derived,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-path",
        default=None,
        help="Seed list path (default: selfplay.opening_fen_list_path in configs/pbt2_small.yaml)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output manifest path (default: <list-path>.manifest.json)",
    )
    args = parser.parse_args()

    list_path = args.list_path or _default_list_path()
    if not list_path:
        parser.error("--list-path is required (could not read a default from configs/pbt2_small.yaml)")

    out_path = args.out or (str(list_path) + ".manifest.json")
    manifest = build_manifest(str(list_path))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")

    n_seeds = manifest["n_seeds"]
    by_key = manifest["by_key"]
    n_unique_keys = len(by_key) if isinstance(by_key, dict) else 0
    print(f"n_seeds={n_seeds} n_unique_keys={n_unique_keys} out={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
