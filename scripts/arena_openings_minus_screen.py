#!/usr/bin/env python3
"""Confirmation openings: a split's EVAL lines minus the openings a screen arena played.

A screen arena run as ``arena_standard.py --openings <split>/eval.pgn
--opening-plies <eval_max_plies> --games G --seed S`` draws its G/2 opening pairs
with the arena's own sampler and rng (``rng = np.random.default_rng(S)``, then
``load_paired_openings``). This script reproduces that draw with the same two
calls, removes those openings from the split's ``eval.jsonl``, and writes the
remaining lines as full-history ``<root_fen> | <uci ...>`` rows, the grammar
``arena_standard.py --openings-fen`` replays WITH its move stack (a bare FEN
would drop the history the net reads).

An opening is keyed by its final position (EPD) and its move list, the key the
arena's FEN loader dedups on. Every screen opening must match an EVAL line (else
the subtraction is incomplete and the script refuses).

The rebuilt screen set is only as right as ``--screen-games`` / ``--screen-seed``
/ ``--opening-plies``: a wrong value rebuilds a DIFFERENT set, and nothing in the
rebuild can see that. ``--screen-log <games.jsonl>`` (repeatable; the screens'
``--games-out`` logs) closes that gap once the screens have been played: the
script refuses unless the rebuilt set equals each log's set of played
(``opening_fen``, ``opening_uci``) keys. Before the screens exist, the output is
only as good as the arguments.

``--check-games`` / ``--check-seed`` (the confirmation's own ``--games`` and
``--seed``) then load the written file exactly as the arena will
(``load_fen_openings``) and refuse unless every drawn pair has its history and
none is in the rebuilt screen set (which, with ``--screen-log``, is the set the
screens actually played).

Output: the file, and one JSON line on stdout with the counts and the file's
sha256. Exit 0 on success, 2 on a refusal (a file that fails the check is
removed).

    PYTHONPATH=. python scripts/arena_openings_minus_screen.py --openings-dir <split> \\
        --screen-games 400 --screen-seed 2026112003 \\
        --check-games 1600 --check-seed 2026112005 --out <run>/openings/confirmation.fen
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import numpy as np

from scripts.arena_standard import load_fen_openings, load_paired_openings

OpeningKey = tuple[str, tuple[str, ...]]


class Refusal(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(2)
        self.message = message


def opening_key(board: chess.Board) -> OpeningKey:
    return board.epd(), tuple(m.uci() for m in board.move_stack)


def screen_openings(eval_pgn: Path, *, games: int, seed: int, opening_plies: int) -> list[chess.Board]:
    """The openings ``arena_standard.run_arena`` draws for this screen: same rng, same sampler."""
    return load_paired_openings(eval_pgn, n_pairs=games // 2, max_plies=opening_plies,
                                rng=np.random.default_rng(seed))


def eval_lines(eval_jsonl: Path) -> dict[OpeningKey, str]:
    """EVAL lines by opening key, in file order."""
    out: dict[OpeningKey, str] = {}
    for raw in eval_jsonl.read_text().splitlines():
        line = str(json.loads(raw)["line"]).strip()
        fen, _, moves = line.partition("|")
        board = chess.Board(fen.strip())
        for uci in moves.split():
            board.push_uci(uci)
        out[opening_key(board)] = line
    return out


def played_openings(games_log: Path) -> tuple[set[Any], bool]:
    """The opening keys of every game row in an arena ``--games-out`` log, and whether they carry moves.

    matched_sims rows record ``opening_uci`` (the full key); matched_time rows do
    not, and are keyed by final EPD alone -- unique within a split's EVAL lines,
    which refuses two lines whose final positions share an orbit.
    """
    rows = [r for r in (json.loads(raw) for raw in games_log.read_text().splitlines()) if r.get("kind") == "game"]
    with_moves = bool(rows) and all("opening_uci" in r for r in rows)
    keys: set[Any] = set()
    for r in rows:
        epd = chess.Board(r["opening_fen"]).epd()
        keys.add((epd, tuple(r["opening_uci"])) if with_moves else epd)
    return keys, with_moves


def build(openings_dir: Path, out: Path, *, screen_games: int, screen_seed: int,
          opening_plies: int | None = None, check_games: int | None = None,
          check_seed: int | None = None, screen_logs: list[Path] | None = None) -> dict[str, Any]:
    if opening_plies is None:
        opening_plies = int(json.loads((openings_dir / "summary.json").read_text())["eval_max_plies"])
    full = eval_lines(openings_dir / "eval.jsonl")
    screen = screen_openings(openings_dir / "eval.pgn", games=screen_games, seed=screen_seed,
                             opening_plies=opening_plies)
    skeys = {opening_key(b) for b in screen}
    log_keys: dict[str, str] = {}
    for log in screen_logs or []:
        played, with_moves = played_openings(log)
        log_keys[str(log)] = "epd_and_moves" if with_moves else "epd_only"
        rebuilt = skeys if with_moves else {epd for epd, _ in skeys}
        if played != rebuilt:
            raise Refusal(f"{log}: the screen played {len(played)} opening(s), the rebuilt set has {len(rebuilt)}; "
                          f"{len(played - rebuilt)} played but not rebuilt, {len(rebuilt - played)} rebuilt but not "
                          "played -- --screen-games / --screen-seed / --opening-plies do not reproduce this screen")
    unmatched = sorted(skeys - full.keys())
    if unmatched:
        raise Refusal(f"{len(unmatched)} screen opening(s) match no EVAL line (first: {unmatched[0]}); "
                      "the subtraction would be incomplete -- check --opening-plies against summary.json")
    keep = [line for key, line in full.items() if key not in skeys]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(line + "\n" for line in keep))
    report: dict[str, Any] = {
        "openings_dir": str(openings_dir), "opening_plies": opening_plies,
        "screen_games": screen_games, "screen_seed": screen_seed,
        "eval_lines": len(full), "screen_openings": len(screen), "screen_unique": len(skeys),
        "confirmation_rows": len(keep), "out": str(out),
        "screen_logs_checked": log_keys,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    if check_games is not None:
        if check_seed is None:
            raise Refusal("--check-games needs --check-seed (the confirmation's --seed)")
        drawn = load_fen_openings(out, n_pairs=check_games // 2, rng=np.random.default_rng(check_seed))
        report.update(check_games=check_games, check_seed=check_seed, check_pairs=len(drawn),
                      check_with_history=sum(1 for b in drawn if b.move_stack),
                      check_overlap_with_screen=sum(1 for b in drawn if opening_key(b) in skeys))
        problems = []
        if len(drawn) != check_games // 2:
            problems.append(f"{len(drawn)} pairs loadable, {check_games // 2} needed")
        # Zero by construction (the screen set was just removed); a guard against a future edit of `keep`.
        # What was PLAYED is proven only by --screen-log.
        if report["check_overlap_with_screen"]:
            problems.append(f"{report['check_overlap_with_screen']} confirmation pair(s) overlap the screen")
        if report["check_with_history"] != len(drawn):
            problems.append(f"{len(drawn) - report['check_with_history']} pair(s) without move history")
        if problems:
            out.unlink()                                 # never leave a refused file to be used
            print(json.dumps(report), flush=True)
            raise Refusal("confirmation check failed: " + "; ".join(problems))
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--openings-dir", required=True, help="split-openings output (eval.pgn, eval.jsonl, summary.json)")
    p.add_argument("--screen-games", type=int, required=True, help="the screen arena's --games")
    p.add_argument("--screen-seed", type=int, required=True, help="the screen arena's --seed")
    p.add_argument("--opening-plies", type=int, default=None,
                   help="the screen arena's --opening-plies (default: summary.json eval_max_plies)")
    p.add_argument("--check-games", type=int, default=None, help="the confirmation arena's --games")
    p.add_argument("--check-seed", type=int, default=None, help="the confirmation arena's --seed")
    p.add_argument("--screen-log", action="append", default=[],
                   help="a screen arena's --games-out log (repeatable): refuse unless the rebuilt screen set "
                        "equals its played openings")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    try:
        report = build(Path(a.openings_dir), Path(a.out), screen_games=a.screen_games, screen_seed=a.screen_seed,
                       opening_plies=a.opening_plies, check_games=a.check_games, check_seed=a.check_seed,
                       screen_logs=[Path(x) for x in a.screen_log])
    except Refusal as exc:
        print(f"REFUSED: {exc.message}", file=sys.stderr)
        return 2
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
