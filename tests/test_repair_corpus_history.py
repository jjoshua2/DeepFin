"""The legacy-corpus history repair: exact equivalence with live play, or quarantine.

A synthetic one-worker stream (four games across two shards, written the way
the generator writes them, minus the plies dedup would have dropped) is
repaired through the real tool, and every written row is compared against the
TRUE game the test generated it from:

* the deriver's planes of the repaired row == the C play path's planes of the
  live board (the PR #497 gate, re-run on repaired rows);
* ``history_root_fen`` / ``history_uci`` / ``history_root_reason`` == the
  generator's own ``history_for`` on the live board, as STRINGS -- which is
  what catches a root written with the default ``fen()``;
* the quarantine set is exactly the rows whose window spans an ambiguous or
  unbridgeable gap, or needs a book the corpus cannot verify;
* every label is copied BYTE-FOR-BYTE (R2 is parked): ``phases`` and every
  other original field of the output row equal the input's;
* the repetition TAGS, ``input_key`` and ``search_key`` equal an independent
  computation on the live board;
* with ``--relabel window`` the parked re-label path sends ``position fen
  <root> moves ...`` for EVERY flagged row, ONLY for flagged rows, with a
  fresh ``ucinewgame`` each.

The oracle is the true game, never the tool's own output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest import mock

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.selfplay.opening import OpeningConfig, sample_starting_board
from scripts import derive_corpus_targets as derive
from scripts import gen_sf_rooted_corpus as corpus
from scripts import repair_corpus_history as repair
from tests.test_gen_sf_rooted_corpus import ScriptedEngine, uci_double

WORKER = 3
SEED = 7
BOOK_LINE = "e2e4 e7e5 g1f3 b8c6"
BOOK_PLIES = 4
STAIRCASE = "all:3"

#: Game A: a shuffle repetition at the front (ply 4 repeats ply 0), a double
#: pawn step whose ep square is pseudo-legal only (d2d4 with no black pawn on
#: c4/e4), an AMBIGUOUS 3-ply dedup gap (the b1 knight reaches c4 via a3 or
#: d2, capturing -- so the position after it has clock 0 and is the ROOT of
#: the run that follows), a bridgeable 1-ply gap (only the queen can play
#: d1d2), an UNBRIDGEABLE 4-ply gap, then an irreversible move and a fresh
#: repetition at the end.
GAME_A = (
    "f3g1 c6b8 g1f3 b8c6 "                       # 0-3
    "d2d4 g8f6 f1e2 f8e7 e1g1 e8g8 h2h3 c6a5 "    # 4-11
    "b2b3 a5c4 "                                 # 12-13
    "b1d2 h7h6 d2c4 "                            # 14-16 ambiguous gap
    "f8e8 c1e3 e7f8 d1d2 f8e7 a1d1 e7f8 "        # 17-23 (20 = bridgeable gap)
    "f1e1 f8e7 e1f1 e7f8 f1e1 f8e7 e1f1 e7f8 "   # 24-31
    "f1e1 f8e7 e1f1 e7f8 "                       # 32-35 unbridged gap
    "g2g3 g7g6 g1g2 g8g7 e2d3 f8e7 d3e2 e7f8 e2d3 f8e7 d3e2 e7f8"  # 36-47
)
GAP_AMBIGUOUS_A = (14, 15, 16)
GAP_BRIDGED_A = (20,)
GAP_UNBRIDGED_A = (32, 33, 34, 35)
DROPPED_A = frozenset(GAP_AMBIGUOUS_A + GAP_BRIDGED_A + GAP_UNBRIDGED_A)

#: Game B: its ply-0 row was dedup-served, so the book start is joined to the
#: first banked row by a bridge; a knight shuffle repeats; it SPANS the two
#: shards (rows 1-9 in the first, the rest in the second).
GAME_B = "f1c4 g8f6 b1c3 f8c5 d2d3 d7d6 c1g5 h7h6 g5h4 g7g5 h4g3 c8e6 c3d5 e6d5 c4d5 c6d4 f3d4 c5d4 c2c3 d4b6 d5b3 b6c5 b3d5 c5b6"
DROPPED_B = frozenset({0})
SPLIT_B = 10

#: Game C: banked from a DIFFERENT opening than the book gives its game id, so
#: the resampled start does not match its ply-0 row.  Rows whose window
#: reaches into the book are quarantined; rows rooted at a banked clock-0
#: position are repaired from the chain alone.
GAME_C_START = "d2d4 d7d5 c2c4 e7e6"
GAME_C = "b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3 b8d7 f1d3 c7c6 e1g1 f8e8 d1c2 d7f8 a1d1 e7d6 h2h3 c8d7"

#: Game D: five plies, every window shorter than 7 + the book -- the short
#: TRUE history that is kept, never dropped.
GAME_D = "f1c4 g8f6 d2d3 f8c5 b1c3"

#: Game E: the commonest shuffle -- a clock-0 position (the ROOT of every
#: later window) recurring four plies later, then again.  A repeat scan that
#: starts one position late tags none of these rows (review finding F1).
GAME_E = "d2d3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8"

#: Game F: the banking gate's FAR repetition (each side cycles two knights, so
#: a position recurs EIGHT plies later) followed by an IRREVERSIBLE move
#: (d2d3 at ply 16) and seven reversible plies.  Rows 17-23 have a repetition
#: and then that pawn move inside their 8 frames: the unfixed C encoder clears
#: its hash stack at the irreversible move and loses the earlier frames'
#: repetition planes, the fixed one keeps the per-slot flags recorded at push
#: time -- so ``row_key`` differs between the regimes on exactly these rows,
#: and only there (a repetition with no irreversible move in the frames keys
#: identically either way).  The deriver runs FIXED.
GAME_F = (
    "f3g1 c6b8 b1c3 g8f6 g1f3 b8c6 c3b1 f6g8 f3g1 c6b8 b1c3 g8f6 g1f3 b8c6 c3b1 f6g8 "
    "d2d3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8"
)

#: Game H: the ONLY repeat is the clock-0 root recurring once (ply 5 == ply 1),
#: and row 8's production window is rooted exactly there -- so a scan that
#: starts at ``keys[1:]`` finds nothing (review G5).  Rows stop at ply 8.
GAME_H = "d2d3 g8f6 f3g1 f6g8 g1f3 a8b8 f3h4 b8a8 h4f3"

#: Game I: an ambiguous 3-ply gap ending on a capture whose root CANNOT have
#: been reached by a double pawn step (the e2 bishop sits behind the e4 pawn, the d-pawn is on d3),
#: so the rows rooted there are rebuilt ANCHORLESS and written (review G2's
#: other half: game A's anchorless rows have a d4 pawn with d3/d2 empty and
#: are quarantined ``root_ep_unverifiable``).
GAME_I = "f1e2 c6a5 d2d3 a5c4 b1d2 h7h6 d2c4 g8f6 e1g1 f8e7 c1e3 e8g8 d1d2 a8b8 a1b1 b8a8 b1a1"
GAP_AMBIGUOUS_I = (4, 5, 6)
#: A bridgeable 1-ply gap AFTER game I's anchorless root: only the queen can
#: play d1d2, so rows 13-16 are ``chained+bridged`` anchorless rebuilds.
GAP_BRIDGED_I = (12,)

GAMES: dict[int, tuple[str | None, str, frozenset[int]]] = {
    # game_id -> (start override, moves, dropped plies)
    10: (None, GAME_A, DROPPED_A),
    24: (None, GAME_B, DROPPED_B),
    38: (GAME_C_START, GAME_C, frozenset()),
    52: (None, GAME_D, frozenset()),
    66: (None, GAME_E, frozenset()),
    70: (None, GAME_F, frozenset()),
    84: (None, GAME_H, frozenset()),
    106: (None, GAME_I, frozenset(GAP_AMBIGUOUS_I + GAP_BRIDGED_I)),
    # In the UNLISTED shard only: a complete game and a torn one.
    80: (None, GAME_D, frozenset()),
    94: (None, GAME_E, frozenset()),
    # A worker killed before it closed its FIRST shard: represented only by
    # an unlisted shard holding this complete game.
    120: (None, GAME_D, frozenset()),
}
LISTED_GAMES = (10, 24, 38, 52, 66, 70, 84, 106)
UNLISTED_COMPLETE_GAME = 80
UNLISTED_TORN_GAME = 94
WORKER_UNLISTED_ONLY = 4
UNLISTED_ONLY_GAME = 120
#: The row of game D whose label the generator searched cold (a wedge retry):
#: the ONLY per-row cold marker.
COLD_RETRY_ROW = (52, 1)
#: A later row of the same worker: its run block says the TT was not carried
#: because the worker's wedge counter never resets -- NOT a cold label.
TT_CLEARED_ROW = (52, 3)
REPO_ROOT = Path(__file__).resolve().parents[1]


# ── the two encoders (the PR #497 gate's, restated) ──────────────────────────


def live_planes(board: chess.Board) -> np.ndarray:
    return np.asarray(
        encode_cboard(
            CBoard.from_board(board),
            input_history_encoding=derive.INPUT_HISTORY_ENCODING,
            input_extra_features=derive.INPUT_EXTRA_FEATURES,
        ),
        dtype=np.float32,
    )


DERIVER = derive.TargetDeriver(
    derive.DeriveOptions(
        scheme=derive.parse_scheme("uniform-d3"), temp=1.0, cp_slope=1.0,
        cp_draw_width=1.0, limit=0, seed=0, rows_per_shard=8, max_envelope_misses=0,
    ),
)


def derived_planes(board: chess.Board) -> np.ndarray:
    return DERIVER._encode(board)


# ── the synthetic corpus ─────────────────────────────────────────────────────


def book_path(tmp_path: Path) -> Path:
    path = tmp_path / "book.pgn"
    path.write_text("1. e4 e5 2. Nf3 Nc6 *\n", encoding="utf-8")
    return path


def opening_config(tmp_path: Path) -> OpeningConfig:
    return OpeningConfig(
        opening_book_path=str(book_path(tmp_path)), opening_book_max_plies=BOOK_PLIES,
        opening_book_max_games=10, opening_book_prob=1.0,
    )


def true_boards(game_id: int, tmp_path: Path, *, worker: int = WORKER) -> list[chess.Board]:
    """Position ``i`` (before move ``i``) of the game, each with its live stack."""
    start_override, moves, _ = GAMES[game_id]
    if start_override is None:
        start = sample_starting_board(
            rng=corpus.book_rng(seed=SEED, worker_id=worker, game_id=game_id),
            cfg=opening_config(tmp_path),
        ).board
    else:
        start = chess.Board()
        for uci in start_override.split():
            start.push_uci(uci)
    boards = [start.copy(stack=True)]
    board = start
    for uci in moves.split():
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, (game_id, uci, board.fen())
        board.push(move)
        boards.append(board.copy(stack=True))
    return boards


def fake_phases(board: chess.Board, played: str, *, nonce: int = 1000) -> list[dict[str, Any]]:
    """A schema-1 staircase block whose rank 1 is the played move.

    ``nonce`` is a PER-ROW number written into the node counts and the cp of
    every line, so two rows at the same position carry different label bytes
    and a rebuilt-from-the-position block cannot masquerade as the copy
    (review G4).
    """
    legal = [m.uci() for m in board.legal_moves]
    ranked = [played, *[m for m in legal if m != played]]
    lines = [
        [rank, move, float(100 - rank) + nonce / 1e6, nonce + rank]
        for rank, move in enumerate(ranked, start=1)
    ]
    return [{
        "index": 0, "width_requested": "all", "width_realized": len(legal),
        "width_streamed": len(legal), "depth_requested": 3, "searchmoves": None,
        "per_depth": [{
            "depth": 3, "complete": True, "emissions": len(legal),
            "nodes_at_depth": nonce, "lines": lines,
        }],
        "nodes_at_depth": {"3": nonce},
        "anomalies": {},
    }]


def row_nonce(game_id: int, ply: int) -> int:
    return 100_000 + game_id * 1_000 + ply


def row_for(
    game_id: int, ply: int, board: chess.Board, played: str, *, config_sha: str,
    worker: int = WORKER,
) -> dict[str, Any]:
    return {
        "schema": 1,
        **({"cold_tt_retry": True} if (game_id, ply) == COLD_RETRY_ROW else {}),
        "run": {
            "run_id": "test_repair", "config_sha256": config_sha,
            corpus.KEY_TT_CARRIED: (game_id, ply) != TT_CLEARED_ROW,
        },
        "fen": board.fen(),
        "dedup_key": corpus.dedup_key(board),
        "worker_id": worker,
        "game_id": game_id,
        "ply": ply,
        "stm": "w" if board.turn else "b",
        "piece_count": chess.popcount(board.occupied),
        "game_phase": "opening",
        "played_move": played,
        "selection": {"temp": 1.0, "legal_moves": board.legal_moves.count()},
        "phases": fake_phases(board, played, nonce=row_nonce(game_id, ply)),
        "result": 0.0, "result_pgn": "1/2-1/2", "adjudication": None,
    }


def game_rows(game_id: int, tmp_path: Path, *, worker: int = WORKER) -> list[dict[str, Any]]:
    _, moves, dropped = GAMES[game_id]
    boards = true_boards(game_id, tmp_path, worker=worker)
    config_sha = corpus.stamp_sha256(requested_config(tmp_path))
    return [
        row_for(game_id, ply, boards[ply], uci, config_sha=config_sha, worker=worker)
        for ply, uci in enumerate(moves.split()) if ply not in dropped
    ]


def write_shard(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    writer = repair.ZstdLines(path)
    for row in rows:
        writer.write(row)
    writer.close()


def requested_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "out_dir": str(tmp_path / "in"), "games": 4, "workers": 1, "staircase": STAIRCASE,
        "seed": SEED, "temp_plies": 20, "temp_high": 1.0, "temp_low": 0.3,
        "max_plies": 400, "shard_rows": 8192, "sf_hash_mb": 64,
        "sf_read_timeout_s": 300.0, "sf_search_timeout_s": 2.0,
        "dedup_cache_max": 2_000_000, "syzygy_path": "/nonexistent/syzygy", "nice": 0,
        "cp_slope": 0.006, "cp_draw_width": 120.0, "book": str(book_path(tmp_path)),
        "book_plies": BOOK_PLIES, "book_max_games": 10, "run_id": "test_repair",
        "stockfish": "/nonexistent/stockfish",
    }


SHARDS = ("w03-00000.jsonl.zst", "w03-00001.jsonl.zst")
#: On disk, NOT in the progress inventory -- a paused run's in-flight shard.
UNLISTED_SHARD = "w03-00002.jsonl.zst"
UNLISTED_ONLY_SHARD = "w04-00000.jsonl.zst"
UNLISTED_SHARDS = (UNLISTED_SHARD, UNLISTED_ONLY_SHARD)


def build_corpus(tmp_path: Path) -> Path:
    """Two listed shards for worker 3 (game B spanning them) plus an unlisted one."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    rows_a = game_rows(10, tmp_path)
    rows_b = game_rows(24, tmp_path)
    head_b = [r for r in rows_b if r["ply"] < SPLIT_B]
    tail_b = [r for r in rows_b if r["ply"] >= SPLIT_B]
    shard_rows = {
        SHARDS[0]: [*rows_a, *head_b],
        SHARDS[1]: [
            *tail_b, *game_rows(38, tmp_path), *game_rows(52, tmp_path),
            *game_rows(66, tmp_path), *game_rows(70, tmp_path),
            *game_rows(84, tmp_path), *game_rows(106, tmp_path),
        ],
    }
    for name, rows in shard_rows.items():
        write_shard(in_dir / name, rows)
    # The unlisted shard: a complete game and the head of one that never ended.
    write_shard(
        in_dir / UNLISTED_SHARD,
        [*game_rows(UNLISTED_COMPLETE_GAME, tmp_path), *game_rows(UNLISTED_TORN_GAME, tmp_path)[:4]],
    )
    # Worker 4 closed no shard: its only shard is unlisted and holds one
    # complete game (the torn tail lost everything after it).
    write_shard(
        in_dir / UNLISTED_ONLY_SHARD,
        game_rows(UNLISTED_ONLY_GAME, tmp_path, worker=WORKER_UNLISTED_ONLY),
    )
    with open(in_dir / "w03.progress.jsonl", "w", encoding="utf-8") as progress:
        for name, rows in shard_rows.items():
            progress.write(json.dumps({
                "path": str(in_dir / name), "rows": len(rows), "codec": "zstd",
                "games": sorted({r["game_id"] for r in rows}),
            }) + "\n")
    requested = requested_config(tmp_path)
    manifest = {
        "schema": 1, "row_schema": 1, "config_requested": requested,
        "config_sha256": corpus.stamp_sha256(requested),
        "engine": {"path": "/nonexistent/stockfish", "sha256": "0" * 64, "id_name": "fake"},
        "staircase_parsed": [{"width": "all", "depth": 3}],
        "banked_rows_min_piece_count": corpus.MIN_BANKED_PIECES,
        "adjudication_max_piece_count": corpus.ADJUDICATION_MAX_PIECES,
    }
    (in_dir / corpus.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return in_dir


def fake_factory(engine: ScriptedEngine) -> repair.SearcherFactory:
    def spawn(stats: corpus.SearchStats) -> corpus.StaircaseSearcher:
        return corpus.StaircaseSearcher(
            engine=uci_double(engine), staircase=corpus.parse_staircase(STAIRCASE),
            cp_slope=0.006, cp_draw_width=120.0, stats=stats,
        )
    return spawn


def args_for(in_dir: Path, out_dir: Path | None, **extra: Any) -> argparse.Namespace:
    """CLI args; ``relabel`` defaults to the parser's default (off)."""
    argv = ["--in", str(in_dir)]
    if out_dir is not None:
        argv += ["--out", str(out_dir)]
    for key, value in extra.items():
        argv.append(f"--{key.replace('_', '-')}")
        if value is not True:
            argv.append(str(value))
    return repair.build_parser().parse_args(argv)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return list(corpus.iter_shard_rows(path))


# ── the oracle: what each row MUST be, from the true game ────────────────────


def repeat_in(board: chess.Board, plies: int) -> bool:
    """A transposition key repeats among the last ``plies`` positions and this one."""
    walk = board.copy(stack=True)
    keys = [walk._transposition_key()]
    for _ in range(plies):
        walk.pop()
        keys.append(walk._transposition_key())
    return len(set(keys)) != len(keys)


def expected_class(
    game_id: int, ply: int, live: chess.Board, *, relabel: str = repair.RELABEL_OFF,
) -> tuple[str, str | None]:
    """``(class, must-be-bridged)`` for a banked row, from the true game alone."""
    start_override, _, dropped = GAMES[game_id]
    history = corpus.history_for(live)
    root = ply - history.plies
    span = range(max(root, 0), ply)
    gaps = {
        10: {REASON_AMB: GAP_AMBIGUOUS_A, REASON_UNB: GAP_UNBRIDGED_A},
        106: {REASON_AMB: GAP_AMBIGUOUS_I},
    }.get(game_id, {})
    for reason, plies in gaps.items():
        if set(span) & set(plies):
            return repair.QUARANTINE_PREFIX + reason, None
    if root < 0 and start_override is not None:
        return repair.QUARANTINE_PREFIX + repair.REASON_BOOK_MISMATCH, None
    unresolved = set().union(*gaps.values()) if gaps else set()
    # Anchorless: an unresolved ply before a known root, or the ply-0 row of a
    # game whose book cannot be used (the position before ply 0 is unknown).
    if (root >= 1 and (root - 1) in unresolved) or (root == 0 and start_override is not None):
        # Anchorless: the root itself must provably carry no ep square.
        root_board = live.copy(stack=True)
        for _ in range(history.plies):
            root_board.pop()
        mover = not root_board.turn
        rank = 3 if mover == chess.WHITE else 4
        behind = -1 if mover == chess.WHITE else 1
        for sq in chess.SquareSet(root_board.pieces(chess.PAWN, mover) & chess.BB_RANKS[rank]):
            f = chess.square_file(sq)
            if root_board.piece_at(chess.square(f, rank + behind)) is None and root_board.piece_at(chess.square(f, rank + 2 * behind)) is None:
                return repair.QUARANTINE_PREFIX + repair.REASON_ROOT_EP, None
    bridged = bool(set(span) & set(dropped) - set(GAP_AMBIGUOUS_A) - set(GAP_UNBRIDGED_A) - set(GAP_AMBIGUOUS_I))
    if relabel == repair.RELABEL_WINDOW:
        flagged = repeat_in(live, history.plies)
    elif relabel == repair.RELABEL_SEGMENT:
        flagged = repeat_in(live, min(live.halfmove_clock, history.plies))
    else:
        flagged = False
    return (repair.CLASS_RELABELED if flagged else repair.CLASS_REPAIRED), (
        "bridged" if bridged else None
    )


def expected_tags(live: chess.Board) -> dict[str, Any]:
    """The tags from the live stack: frames, segment, current-position count."""
    plies = len(live.move_stack)
    walk = live.copy(stack=True)
    keys = [walk._transposition_key()]
    for _ in range(plies):
        walk.pop()
        keys.append(walk._transposition_key())
    frames = keys[: corpus.HISTORY_WINDOW_PLIES + 1]
    segment = keys[: live.halfmove_clock + 1]
    return {
        "rep_in_frames8": len(set(frames)) != len(frames),
        "rep_in_segment": len(set(segment)) != len(segment),
        "cur_position_repeat_count": segment.count(keys[0]),
    }


REASON_AMB = repair.REASON_AMBIGUOUS
REASON_UNB = repair.REASON_UNBRIDGED


def row_map(out_dir: Path) -> dict[tuple[int, int], str]:
    entries = read_rows(
        out_dir / repair.PROVENANCE_DIR / repair.ROW_MAP_TEMPLATE.format(worker_id=WORKER),
    )
    return {(int(e["game_id"]), int(e["ply"])): str(e["class"]) for e in entries}


# ── the run under test ───────────────────────────────────────────────────────


def pinned(tmp_path: Path) -> Any:
    """Production mode pins the PRODUCTION book by hash; the fixture's book
    stands in for it through this one explicit seam.  Every test that runs
    production mode on the fixture says so by using it."""
    return mock.patch.object(repair, "PRODUCTION_BOOK_SHA256", repair.sha256_of(book_path(tmp_path)))


def run_repair(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    in_dir = build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    engine = ScriptedEngine(multipv=1)
    with pinned(tmp_path):
        manifest = repair.run(args_for(in_dir, out_dir, **extra), searcher_factory=fake_factory(engine))
    written = {
        (int(r["game_id"]), int(r["ply"])): r
        for shard in SHARDS
        for r in read_rows(out_dir / shard)
    }
    inputs = {
        (int(r["game_id"]), int(r["ply"])): r
        for shard in SHARDS
        for r in read_rows(in_dir / shard)
    }
    truth = {
        game_id: true_boards(game_id, tmp_path) for game_id in LISTED_GAMES
    }
    return {
        "in_dir": in_dir, "out_dir": out_dir, "engine": engine, "manifest": manifest,
        "written": written, "inputs": inputs, "truth": truth, "map": row_map(out_dir),
        "tmp_path": tmp_path,
    }


@pytest.fixture
def repaired(tmp_path: Path) -> dict[str, Any]:
    """The DEFAULT run -- PRODUCTION MODE: labels preserved, rows tagged."""
    return run_repair(tmp_path)


@pytest.fixture
def relabeled(tmp_path: Path) -> dict[str, Any]:
    """The parked R2 path, switched on explicitly (audit mode only)."""
    return run_repair(tmp_path, audit_mode=True, relabel=repair.RELABEL_WINDOW)


def test_a_production_repair_is_a_whole_corpus_and_an_audit_slice_says_it_is_not(tmp_path: Path) -> None:
    """The completeness contract: a production-mode repair covers the whole
    recorded inventory with nothing refused or skipped, so ``summary.json``
    says ``run_finished: true`` (the deriver's ``corpus_complete``) and no
    partial block; an ``--audit-mode --shards`` slice says ``false``, names
    the slice, and both records say ``production: false``."""
    in_dir = build_corpus(tmp_path)
    out_dir = tmp_path / "out_production"
    with pinned(tmp_path):
        manifest = repair.run(args_for(in_dir, out_dir), searcher_factory=fake_factory(ScriptedEngine()))
    assert manifest["production"] is True
    summary = json.loads((out_dir / corpus.SUMMARY_NAME).read_text())
    assert summary["run_finished"] is True
    assert summary["production"] is True
    assert "partial_repair" not in summary
    record = derive.read_corpus_record(out_dir)
    assert record.facts["run_finished"] is True
    assert record.complete is True

    out_dir2 = tmp_path / "out_slice"
    with pinned(tmp_path):
        sliced = repair.run(
            args_for(in_dir, out_dir2, audit_mode=True, shards="0"),
            searcher_factory=fake_factory(ScriptedEngine()),
        )
    assert sliced["production"] is False
    assert [s["path"] for s in sliced["shards"]] == [SHARDS[0]]
    summary = json.loads((out_dir2 / corpus.SUMMARY_NAME).read_text())
    assert summary["run_finished"] is False
    assert summary["production"] is False
    assert summary["audit"] == {"reasons": ["--audit-mode", "--shards"]}
    assert summary["partial_repair"] == {
        "workers": [WORKER], "shards": [0], "listed_input_shards": 2, "repaired_shards": 1,
    }

    # ⚑ A FULL-INVENTORY audit-mode run is not a slice and is still not the
    # corpus: `run_finished` -- the only field a consumer reads -- is false.
    out_dir3 = tmp_path / "out_audit_full"
    with pinned(tmp_path):
        full = repair.run(
            args_for(in_dir, out_dir3, audit_mode=True, relabel=repair.RELABEL_WINDOW),
            searcher_factory=fake_factory(ScriptedEngine(multipv=1)),
        )
    assert full["production"] is False
    assert full["audit"] == {"reasons": ["--audit-mode", "--relabel"]}
    summary = json.loads((out_dir3 / corpus.SUMMARY_NAME).read_text())
    assert summary["run_finished"] is False
    assert summary["audit"] == {"reasons": ["--audit-mode", "--relabel"]}
    assert "partial_repair" not in summary
    assert derive.read_corpus_record(out_dir3).facts["run_finished"] is False


@pytest.mark.parametrize(
    ("flag", "extra"),
    [
        ("--shards", {"shards": "0"}),
        ("--workers", {"workers": WORKER}),
        ("--book", {"book": "BOOK"}),
        ("--relabel", {"relabel": repair.RELABEL_WINDOW}),
        ("--report-json", {"report_json": "REPORT"}),
        ("--book-sha256", {"book_sha256": "0" * 64}),
    ],
)
def test_production_mode_refuses_each_optional_flag_by_name(
    tmp_path: Path, flag: str, extra: dict[str, Any],
) -> None:
    """One refusal per optional flag, before anything is read or written; the
    same flag is accepted under --audit-mode (the other tests run them)."""
    in_dir = build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    extra = {
        k: {"BOOK": str(book_path(tmp_path)), "REPORT": str(tmp_path / "report.json")}.get(v, v)
        for k, v in extra.items()
    }
    with pinned(tmp_path), pytest.raises(repair.RepairError, match=f"{flag}.*--audit-mode"):
        repair.run(args_for(in_dir, out_dir, **extra), searcher_factory=fake_factory(ScriptedEngine()))
    assert not out_dir.exists()


def test_production_mode_requires_the_pinned_production_book(tmp_path: Path) -> None:
    """The fixture's book is not the production book: unpatched, production
    mode refuses it naming the pin; a --book-sha256 EQUAL to the pin is not
    an override and is accepted (the book still has to hash to it)."""
    in_dir = build_corpus(tmp_path)
    assert repair.sha256_of(book_path(tmp_path)) != repair.PRODUCTION_BOOK_SHA256
    with pytest.raises(repair.RepairError, match="production_pin.*" + repair.PRODUCTION_BOOK_SHA256):
        repair.run(args_for(in_dir, tmp_path / "out"), searcher_factory=fake_factory(ScriptedEngine()))
    assert not (tmp_path / "out").exists()
    with pinned(tmp_path):
        manifest = repair.run(
            args_for(in_dir, tmp_path / "out", book_sha256=repair.PRODUCTION_BOOK_SHA256),
            searcher_factory=fake_factory(ScriptedEngine()),
        )
    assert manifest["book"]["expected_from"] == "--book-sha256"
    assert manifest["book"]["matches_production_pin"] is True
    assert manifest["book"]["production_pin"]["historical_hash"] is None


def banked_rows() -> list[tuple[int, int]]:
    return [
        (game_id, ply)
        for game_id in LISTED_GAMES
        for ply in range(len(GAMES[game_id][1].split())) if ply not in GAMES[game_id][2]
    ]


def test_the_row_map_names_every_input_row_exactly_once(repaired: dict[str, Any]) -> None:
    assert sorted(repaired["map"]) == sorted(banked_rows())
    manifest = repaired["manifest"]
    assert manifest["rows_in"] == len(banked_rows())
    assert sum(manifest["classes"].values()) == manifest["rows_in"]
    assert manifest["rows_out"] == len(repaired["written"])
    assert manifest["rows_out"] == manifest["rows_in"] - manifest["quarantined"]


def test_every_row_gets_the_class_the_true_game_dictates(repaired: dict[str, Any]) -> None:
    """⚑ The oracle is the true game: gaps, book verdict -- not the tool."""
    check_classes(repaired, relabel=repair.RELABEL_OFF)


def check_classes(result: dict[str, Any], *, relabel: str) -> None:
    repaired = result
    mismatches = []
    seen_classes: set[str] = set()
    for game_id, ply in banked_rows():
        live = repaired["truth"][game_id][ply]
        want, bridged = expected_class(game_id, ply, live, relabel=relabel)
        got = repaired["map"][(game_id, ply)]
        seen_classes.add(got)
        if got != want:
            mismatches.append((game_id, ply, want, got))
            continue
        if not got.startswith(repair.QUARANTINE_PREFIX):
            row = repaired["written"][(game_id, ply)]
            kind = row["repair"]["history"]
            if bridged and "bridged" not in kind:
                mismatches.append((game_id, ply, "bridged", kind))
            root = ply - corpus.history_for(live).plies
            # A bridged ply outside the window (the anchor ply, at most one)
            # may legitimately be reported; one inside must be.
            if "bridged" in kind and not (set(range(max(root - 1, 0), ply)) & set(GAMES[game_id][2])):
                mismatches.append((game_id, ply, "not bridged", kind))
    assert not mismatches, mismatches
    # Every branch of the classifier was actually exercised.
    want_classes = {
        repair.CLASS_REPAIRED,
        repair.QUARANTINE_PREFIX + REASON_AMB, repair.QUARANTINE_PREFIX + REASON_UNB,
        repair.QUARANTINE_PREFIX + repair.REASON_BOOK_MISMATCH,
        repair.QUARANTINE_PREFIX + repair.REASON_ROOT_EP,
    }
    if relabel != repair.RELABEL_OFF:
        want_classes.add(repair.CLASS_RELABELED)
    assert seen_classes == want_classes


def test_labels_are_copied_byte_for_byte_and_rows_are_tagged_and_keyed(repaired: dict[str, Any]) -> None:
    """⚑⚑ R2 IS PARKED: no original field changes; the additions are exactly the schema-2 set."""
    added = {
        "history_root_fen", "history_uci", "history_plies", "history_root_reason",
        "input_key", "search_key", "rep_in_frames8", "rep_in_segment",
        "cur_position_repeat_count", "label_regime", "repair",
    }
    tag_hits = {"rep_in_frames8": 0, "rep_in_segment": 0, "cur_position_repeat_count": 0}
    for key, row in repaired["written"].items():
        source = repaired["inputs"][key]
        label = f"game {key[0]} ply {key[1]}"
        # ⚑ RAW SERIALISED BYTES, not ``==`` (0 == 0.0 would pass), and the
        # source carries a per-row nonce so a block rebuilt from the position
        # cannot match by shape.
        assert source["phases"][0]["per_depth"][0]["nodes_at_depth"] == row_nonce(*key)
        for name, value in source.items():
            if name == "run":
                assert row["run"] == {**value, repair.KEY_HISTORY_REP_FIX: True}, label
            elif name != "schema":
                assert json.dumps(row[name], sort_keys=True) == json.dumps(value, sort_keys=True), (label, name)
        assert set(row) - set(source) == added, label
        if key == COLD_RETRY_ROW:
            assert row["label_regime"] == repair.LABEL_REGIME_COLD_BLIND, label
        else:
            # TT_CLEARED_ROW included: the run block's per-worker flag is
            # carried verbatim and is NOT read as a per-row cold marker.
            assert row["label_regime"] == repair.LABEL_REGIME_CARRIED_BLIND, label
        if key == TT_CLEARED_ROW:
            assert row["run"][corpus.KEY_TT_CARRIED] is False, label
        assert row["repair"]["label"] == repair.LABEL_ORIGINAL, label
        live = repaired["truth"][key[0]][key[1]]
        assert row["input_key"] == corpus.row_key(live), label
        assert row["search_key"] == corpus.search_key(live), label
        want = expected_tags(live)
        got = {name: row[name] for name in want}
        assert got == want, (label, got, want)
        tag_hits["rep_in_frames8"] += int(row["rep_in_frames8"])
        tag_hits["rep_in_segment"] += int(row["rep_in_segment"])
        tag_hits["cur_position_repeat_count"] += int(row["cur_position_repeat_count"] >= 2)
    # The tags are not vacuous: every one fires somewhere, and game E (the
    # clock-0 root recurring) is tagged on its shuffle rows.
    assert all(v > 0 for v in tag_hits.values()), tag_hits
    game_e = {ply: repaired["written"][(66, ply)] for ply in range(8, len(GAME_E.split()))}
    assert all(r["rep_in_segment"] and r["rep_in_frames8"] for r in game_e.values()), game_e
    assert max(r["cur_position_repeat_count"] for r in game_e.values()) >= 3
    manifest = repaired["manifest"]
    assert manifest["tags"]["rep_in_segment"] == tag_hits["rep_in_segment"]
    assert manifest["tags"]["rep_in_frames8"] == tag_hits["rep_in_frames8"]
    assert manifest["relabel"]["mode"] == repair.RELABEL_OFF
    assert manifest["classes"].get(repair.CLASS_RELABELED, 0) == 0
    assert "label_regime" not in manifest  # the counts are the record
    assert manifest["label_regimes"] == {
        repair.LABEL_REGIME_CARRIED_BLIND: manifest["rows_out"] - 1,
        repair.LABEL_REGIME_COLD_BLIND: 1,
    }
    assert manifest["engine"] is None
    assert manifest[repair.KEY_HISTORY_REP_FIX] is True
    assert rep_fix.current() is True


def test_the_repeat_scan_sees_the_root_position(tmp_path: Path) -> None:
    """F1: the repeat partner IS the window root (index 0 of the replay)."""
    boards = true_boards(66, tmp_path)
    root = boards[1]  # after d2d3: clock 0, and it recurs at plies 5 and 9
    history = corpus.RowHistory(
        fen=boards[5].fen(), root_fen=root.fen(en_passant="fen"),
        uci=tuple(GAME_E.split()[1:5]), reason=corpus.HISTORY_ROOT_IRREVERSIBLE,
    )
    tags = repair.repeats_in(history, halfmove_clock=boards[5].halfmove_clock)
    assert tags == repair.RepeatTags(banked_window=True, frames=True, segment=True, cur_count=2)
    # One ply short of the recurrence: nothing repeats yet.
    short = corpus.RowHistory(
        fen=boards[4].fen(), root_fen=root.fen(en_passant="fen"),
        uci=tuple(GAME_E.split()[1:4]), reason=corpus.HISTORY_ROOT_IRREVERSIBLE,
    )
    assert repair.repeats_in(short, halfmove_clock=boards[4].halfmove_clock) == repair.RepeatTags(
        banked_window=False, frames=False, segment=False, cur_count=1,
    )
    # Threefold at ply 9, and the frames window (8 positions) still holds it.
    long = corpus.RowHistory(
        fen=boards[9].fen(), root_fen=root.fen(en_passant="fen"),
        uci=tuple(GAME_E.split()[1:9]), reason=corpus.HISTORY_ROOT_IRREVERSIBLE,
    )
    assert repair.repeats_in(long, halfmove_clock=boards[9].halfmove_clock).cur_count == 3


def test_the_root_only_repeat_is_tagged_on_the_production_window(repaired: dict[str, Any]) -> None:
    """G5: game H row 8 -- its window is rooted at the clock-0 position and
    that root's single recurrence (ply 5) is the ONLY repeat anywhere in it,
    so the tags are true only if the scan includes index 0."""
    boards = repaired["truth"][84]
    history = corpus.history_for(boards[8])
    assert 8 - history.plies == 1, history
    walk = boards[8].copy(stack=True)
    keys = [walk._transposition_key()]
    for _ in range(history.plies):
        walk.pop()
        keys.append(walk._transposition_key())
    partners = [i for i, k in enumerate(keys) if k == keys[-1]]  # keys[-1] is the root
    assert partners == [3, len(keys) - 1], partners
    assert len(set(keys[:-1])) == len(keys) - 1, "no repeat should exist without the root"
    tags = repair.repeats_in(history, halfmove_clock=boards[8].halfmove_clock)
    assert tags == repair.RepeatTags(banked_window=True, frames=True, segment=True, cur_count=1)
    row = repaired["written"][(84, 8)]
    assert row["rep_in_frames8"] is True
    assert row["rep_in_segment"] is True


def test_every_written_row_encodes_exactly_like_live_play(repaired: dict[str, Any]) -> None:
    """⚑⚑ THE DELIVERABLE, on repaired rows: window strings AND planes equal live."""
    for (game_id, ply), row in repaired["written"].items():
        live = repaired["truth"][game_id][ply]
        want = corpus.history_for(live)
        label = f"game {game_id} ply {ply}"
        assert row["schema"] == corpus.ROW_SCHEMA, label
        assert row["history_root_fen"] == want.root_fen, label
        assert list(row["history_uci"]) == list(want.uci), label
        assert row["history_plies"] == want.plies, label
        assert row["history_root_reason"] == want.reason, label
        assert row["repair"]["source_schema"] == 1, label
        rebuilt = derive.board_from_row(row)
        assert rebuilt.fen() == live.fen(), label
        assert np.array_equal(derived_planes(rebuilt), live_planes(live)), label
        # G6: the C play-path encoder on the REBUILT board too, under the
        # production regime -- the comparison ``input_key`` rests on.
        assert rep_fix.current() is True
        assert np.array_equal(live_planes(rebuilt), live_planes(live)), label
        assert row["input_key"] == corpus.input_tensor_key(live_planes(rebuilt)), label
        assert CBoard.from_board(rebuilt).is_repetition() == CBoard.from_board(live).is_repetition(), label


def test_the_cases_the_gate_exists_for_are_present_in_the_written_rows(repaired: dict[str, Any]) -> None:
    """A gate over rows that happen to avoid every hard case proves nothing."""
    written = repaired["written"]
    kinds = {row["repair"]["history"] for row in written.values()}
    assert kinds == {
        repair.HISTORY_CHAINED, repair.HISTORY_CHAINED_BOOK,
        repair.HISTORY_CHAINED_BRIDGED, repair.HISTORY_CHAINED_BOOK_BRIDGED,
    }
    reasons = {row["history_root_reason"] for row in written.values()}
    assert reasons == {corpus.HISTORY_ROOT_GAME_START, corpus.HISTORY_ROOT_IRREVERSIBLE}
    # A root carrying a PSEUDO-LEGAL ep square (after ...e7e5 no white pawn
    # can capture): the default fen() would print '-' and the row would differ
    # from live in exactly the string the "default fen()" mutant changes.
    ep_roots = [
        row for row in written.values()
        if row["history_root_fen"].split(" ")[3] != "-"
        and chess.Board(row["history_root_fen"]).fen().split(" ")[3] == "-"
    ]
    assert ep_roots, "no written row has a pseudo-legal-only ep root"
    # Short TRUE histories: windows shorter than 7 plies, kept.
    short = [row for row in written.values() if row["history_plies"] < corpus.HISTORY_WINDOW_PLIES]
    assert short
    assert all(r["history_root_reason"] == corpus.HISTORY_ROOT_GAME_START for r in short)
    # Rows repaired from the chain alone in the book-MISMATCH game.
    assert any(game_id == 38 for game_id, _ in written)
    # The anchorless rebuild (a run whose root is the position right after an
    # ambiguous gap): written in game I (root provably ep-free), quarantined
    # in game A (a d4 pawn with d3/d2 empty could have double-stepped).
    manifest = repaired["manifest"]
    assert manifest["anchorless"] > 0
    assert manifest["classes"][repair.QUARANTINE_PREFIX + repair.REASON_ROOT_EP] > 0
    assert any(game_id == 106 and ply >= 14 for game_id, ply in written)
    assert not any(game_id == 10 and 21 <= ply <= 31 for game_id, ply in written)
    assert manifest["book_games"] == {
        repair.BOOK_EXACT: 6, repair.BOOK_BRIDGED: 1, repair.BOOK_MISMATCH: 1,
    }
    # Game B's rows straddle the shard boundary and every one of them survived.
    assert {ply for game_id, ply in written if game_id == 24} == set(range(1, len(GAME_B.split())))


def test_only_flagged_rows_are_relabeled_and_each_under_its_own_window(relabeled: dict[str, Any]) -> None:
    """The PARKED path, switched on: the exact ``position`` line, once per flagged row."""
    repaired = relabeled
    check_classes(repaired, relabel=repair.RELABEL_WINDOW)
    engine: ScriptedEngine = repaired["engine"]
    written = repaired["written"]
    flagged = sorted(
        key for key, cls in repaired["map"].items() if cls == repair.CLASS_RELABELED
    )
    assert flagged, "the stream has no repeat to re-label"
    want_lines = []
    for game_id, ply in flagged:
        history = corpus.history_for(repaired["truth"][game_id][ply])
        want_lines.append(f"position fen {history.root_fen} moves {' '.join(history.uci)}")
    assert engine.position_lines == want_lines
    assert engine.commands.count("ucinewgame") == len(flagged)
    assert engine.go_count == len(flagged)
    for key, row in written.items():
        if key in flagged:
            assert row["repair"]["label"] == repair.LABEL_RELABELED
            assert row["label_regime"] == repair.LABEL_REGIME_COLD_HISTORY
            block = row["phases"][0]["per_depth"][-1]
            assert block["depth"] == 3
            legal = {m.uci() for m in chess.Board(row["fen"]).legal_moves}
            assert {line[1] for line in block["lines"]} == legal
        else:
            assert row["repair"]["label"] == repair.LABEL_ORIGINAL
            assert row["label_regime"] == (
                repair.LABEL_REGIME_COLD_BLIND if key == COLD_RETRY_ROW
                else repair.LABEL_REGIME_CARRIED_BLIND
            )
            assert json.dumps(row["phases"], sort_keys=True) == json.dumps(
                fake_phases(chess.Board(row["fen"]), row["played_move"], nonce=row_nonce(*key)),
                sort_keys=True,
            )
    audit = [
        json.loads(line)
        for line in (repaired["out_dir"] / repair.RELABEL_AUDIT_NAME).read_text().splitlines()
    ]
    assert [(a["game_id"], a["ply"]) for a in audit] == flagged
    for entry in audit:
        assert entry["old_top1"] == written[(entry["game_id"], entry["ply"])]["played_move"]
        assert entry["new_top1"] == written[(entry["game_id"], entry["ply"])]["phases"][0]["per_depth"][-1]["lines"][0][1]


def test_the_relabel_modes_select_by_the_matching_tag(tmp_path: Path) -> None:
    in_dir = build_corpus(tmp_path)
    with pinned(tmp_path):
        off = repair.run(args_for(in_dir, None, audit_only=True, bench_encode_rows=0))
    tags = off["tags"]
    assert 0 < tags["rep_in_segment"] <= tags["rep_in_banked_window"]
    assert repair.CLASS_RELABELED not in off["classes"]
    with pinned(tmp_path):
        window = repair.run(args_for(
            in_dir, None, audit_only=True, bench_encode_rows=0, audit_mode=True, relabel="banked_window",
        ))
    assert window["classes"][repair.CLASS_RELABELED] == tags["rep_in_banked_window"]
    with pinned(tmp_path):
        segment = repair.run(args_for(
            in_dir, None, audit_only=True, bench_encode_rows=0, audit_mode=True, relabel="segment",
        ))
    assert segment["classes"][repair.CLASS_RELABELED] == tags["rep_in_segment"]


def test_the_output_is_a_corpus_the_deriver_reads(repaired: dict[str, Any]) -> None:
    """``summary.json`` + shards pass the deriver's own record and inventory checks."""
    out_dir = repaired["out_dir"]
    record = derive.read_corpus_record(out_dir)
    assert record.complete
    assert record.rows_claimed == repaired["manifest"]["rows_out"]
    assert {p.name for p in record.shards} == set(SHARDS)
    summary = json.loads((out_dir / corpus.SUMMARY_NAME).read_text())
    assert summary["row_schema"] == corpus.ROW_SCHEMA
    derived = derive.derive(
        corpus_dir=out_dir, out_dir=repaired["tmp_path"] / "derived",
        options=derive.DeriveOptions(
            scheme=derive.parse_scheme("uniform-d3"), temp=1.0, cp_slope=0.006,
            cp_draw_width=120.0, limit=0, seed=0, rows_per_shard=64, max_envelope_misses=0,
        ),
    )
    assert derived["realized"]["history_slots_nonzero_max"] == 8
    assert derived["realized"]["row_schema_counts"] == {str(corpus.ROW_SCHEMA): repaired["manifest"]["rows_out"]}
    assert derived["realized"]["input_key_verified"] == repaired["manifest"]["rows_out"]


def test_the_provenance_manifest_carries_the_input_sha_and_the_counts(repaired: dict[str, Any]) -> None:
    out_dir = repaired["out_dir"]
    manifest = json.loads((out_dir / repair.REPAIR_MANIFEST_NAME).read_text())
    assert manifest["input"]["manifest_sha256"] == repair.sha256_of(
        repaired["in_dir"] / corpus.MANIFEST_NAME,
    )
    assert manifest["classes"] == repaired["manifest"]["classes"]
    assert manifest["relabel"] == {**manifest["relabel"], "mode": repair.RELABEL_OFF, "rows": 0}
    book = Path(requested_config(repaired["tmp_path"])["book"])
    assert manifest["book"]["sha256"] == repair.sha256_of(book)
    assert manifest["book"]["size"] == book.stat().st_size
    assert manifest["book"]["expected_from"] == "production_pin"
    assert manifest["book"]["production_pin"]["sha256"] == repair.sha256_of(book)  # the seam
    assert manifest["production"] is True
    # Both unlisted shards were skipped BY NAME, counted, and reported --
    # worker 4's too, although no worker process ran for it.
    assert manifest["input"]["unlisted_shards_skipped"] == list(UNLISTED_SHARDS)
    assert manifest["input"]["workers"] == [WORKER]
    assert not (repaired["out_dir"] / UNLISTED_SHARD).exists()
    assert not (repaired["out_dir"] / UNLISTED_ONLY_SHARD).exists()
    assert manifest["input"]["unlisted_shards"] == [
        {
            "path": UNLISTED_SHARD, "decodable_rows": len(GAME_D.split()) + 4,
            "decodable_games": 2, "damage": None,
        },
        {
            "path": UNLISTED_ONLY_SHARD, "decodable_rows": len(GAME_D.split()),
            "decodable_games": 1, "damage": None,
        },
    ]
    # The inventory's claim per shard is recorded beside what was read.
    for shard in manifest["shards"]:
        assert shard["rows_claimed"] == shard["rows_in"]
    assert {s["path"] for s in manifest["shards"]} == set(SHARDS)
    # The input shards are untouched.
    for name in (*SHARDS, UNLISTED_SHARD):
        assert all("history_uci" not in r for r in read_rows(repaired["in_dir"] / name))


def test_audit_only_writes_nothing_and_runs_no_engine(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    in_dir = build_corpus(tmp_path)
    before = sorted(p.name for p in in_dir.iterdir())
    with pinned(tmp_path):
        assert repair.main(["--in", str(in_dir), "--audit-only", "--bench-encode-rows", "5"]) == 0
    assert sorted(p.name for p in in_dir.iterdir()) == before
    assert not (tmp_path / "out").exists()
    out = capsys.readouterr().out
    assert "rows in" in out
    assert "rows/s/core" in out
    assert "repaired" in out
    assert "rep_in_segment" in out


def test_a_different_engine_build_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    in_dir = build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(repair.RepairError, match="hashes to"):
        repair.run(args_for(in_dir, out_dir, audit_mode=True, engine="/bin/true", relabel="banked_window"))
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_a_manifest_without_an_engine_sha_cannot_relabel(tmp_path: Path) -> None:
    in_dir = build_corpus(tmp_path)
    manifest_path = in_dir / corpus.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    del manifest["engine"]["sha256"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(repair.RepairError, match="no engine sha256"):
        repair.run(args_for(in_dir, tmp_path / "out", audit_mode=True, engine="/bin/true", relabel="banked_window"))


def test_a_manifest_whose_stamp_does_not_hash_its_config_is_refused(tmp_path: Path) -> None:
    in_dir = build_corpus(tmp_path)
    manifest_path = in_dir / corpus.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["config_requested"]["seed"] = SEED + 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(repair.RepairError, match="inconsistent with itself"):
        repair.run(args_for(in_dir, None, audit_only=True, bench_encode_rows=0))


def test_a_different_book_is_refused_by_its_hash(tmp_path: Path) -> None:
    in_dir = build_corpus(tmp_path)
    with pytest.raises(repair.RepairError, match=r"hashes to .* not the expected 0{64} \(--book-sha256"):
        repair.run(args_for(
            in_dir, None, audit_only=True, bench_encode_rows=0, audit_mode=True, book_sha256="0" * 64,
        ))


def test_a_relative_book_path_resolves_against_the_repo_root_or_says_so(tmp_path: Path) -> None:
    requested = {**requested_config(tmp_path), "book": "data/opening_books/does_not_exist.pgn.zip"}
    with pytest.raises(repair.RepairError, match="repo root"):
        repair.opening_config_from_manifest(requested)
    cfg = repair.opening_config_from_manifest(requested, book_override=str(book_path(tmp_path)))
    assert cfg.opening_book_path == str(book_path(tmp_path))


def test_output_shards_are_always_zstd_named() -> None:
    assert repair.output_shard_name("w03-00100.jsonl.gz") == "w03-00100.jsonl.zst"
    assert repair.output_shard_name("w03-00100.jsonl.zst") == "w03-00100.jsonl.zst"
    with pytest.raises(repair.RepairError):
        repair.output_shard_name("notes.txt")


def test_a_populated_output_directory_is_refused(tmp_path: Path) -> None:
    in_dir = build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("x")
    with pinned(tmp_path), pytest.raises(FileExistsError):
        repair.run(args_for(in_dir, out_dir), searcher_factory=fake_factory(ScriptedEngine()))


def test_the_cli_process_hashes_input_keys_in_the_derivers_regime(tmp_path: Path) -> None:
    """⚑⚑ BLOCKING FINDING ROUND 2: ``row_key`` reads the process-global
    ``history_rep_fix``; a fresh CLI process must apply it before keying, or
    every far repetition (game F) is keyed under the wrong regime and the
    deriver refuses the corpus.  The whole output is derived, no --limit."""
    in_dir = build_corpus(tmp_path)
    out_dir = tmp_path / "out_cli"
    env = {**os.environ, "PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": ""}
    # --audit-mode only because the fixture's book is not the production
    # pin (the CLI has no seam for that, by design); the regime handling
    # under test is the same in both modes.
    proc = subprocess.run(
        [
            sys.executable, "scripts/repair_corpus_history.py", "--in", str(in_dir), "--out", str(out_dir),
            "--audit-mode", "--book-sha256", repair.sha256_of(book_path(tmp_path)),
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    manifest = json.loads((out_dir / repair.REPAIR_MANIFEST_NAME).read_text())
    assert manifest[corpus.KEY_HISTORY_REP_FIX] is True
    assert manifest["production"] is False
    rows = [r for shard in SHARDS for r in read_rows(out_dir / shard)]
    assert all(r["run"][corpus.KEY_HISTORY_REP_FIX] is True for r in rows)
    assert all(r["schema"] == corpus.ROW_SCHEMA == 3 for r in rows)
    # The record the deriver refuses BY NAME: present, and the deriver's own
    # regime.  A stamp of the other regime is refused before any row is read.
    summary_path = out_dir / corpus.SUMMARY_NAME
    summary = json.loads(summary_path.read_text())
    assert summary["row_schema"] == corpus.ROW_SCHEMA
    assert summary[corpus.KEY_HISTORY_REP_FIX] is True
    assert derive.read_corpus_record(out_dir).facts[corpus.KEY_HISTORY_REP_FIX] is True
    summary_path.write_text(json.dumps({**summary, corpus.KEY_HISTORY_REP_FIX: False}))
    with pytest.raises(derive.CorpusIntegrityError, match=corpus.KEY_HISTORY_REP_FIX):
        derive.read_corpus_record(out_dir)
    summary_path.write_text(json.dumps(summary))
    derived = derive.derive(
        corpus_dir=out_dir, out_dir=tmp_path / "derived_cli",
        options=derive.DeriveOptions(
            scheme=derive.parse_scheme("uniform-d3"), temp=1.0, cp_slope=0.006,
            cp_draw_width=120.0, limit=0, seed=0, rows_per_shard=64, max_envelope_misses=0,
        ),
    )
    assert derived["realized"]["input_key_verified"] == manifest["rows_out"] == len(rows)
    # ⚑ The take-effect gate itself, on the count it just published.
    derive.enforce_input_key_take_effect(derive.DeriveStats(
        row_schema_counts={corpus.ROW_SCHEMA: len(rows)}, input_key_verified=len(rows),
    ))
    with pytest.raises(derive.CorpusIntegrityError, match="input_key_verified"):
        derive.enforce_input_key_take_effect(derive.DeriveStats(
            row_schema_counts={corpus.ROW_SCHEMA: len(rows)}, input_key_verified=len(rows) - 1,
        ))


def test_game_f_is_keyed_differently_under_the_unfixed_regime(tmp_path: Path) -> None:
    """The fixture is not vacuous: with the flag OFF in a FRESH interpreter,
    the C play-path tensor of game F's rows with a repetition then an
    irreversible move inside their 8 frames hashes to different
    ``input_key``s -- the disagreement the CLI test would surface through the
    deriver.  Only the tensor key diverges; ``search_key`` (python zobrist
    over the segment) is regime-free.  And in that regime ``corpus.row_key``
    -- the only key the repair writes -- REFUSES rather than mis-hashes."""
    boards = true_boards(70, tmp_path)
    rep_fix.apply(True)
    fixed = [corpus.row_key(b) for b in boards[9:]]
    fixed_search = [corpus.search_key(b) for b in boards[9:]]
    code = (
        "import sys, json, chess\n"
        "from chess_anti_engine.encoding import rep_fix\n"
        "rep_fix.apply(False)\n"
        "from chess_anti_engine.encoding._lc0_ext import CBoard\n"
        "from chess_anti_engine.encoding.cboard_encode import encode_cboard\n"
        "from scripts import gen_sf_rooted_corpus as corpus\n"
        f"book = {BOOK_LINE!r}.split(); moves = {GAME_F!r}.split()\n"
        "b = chess.Board()\n"
        "for u in book: b.push_uci(u)\n"
        "keys = []\n"
        "def unfixed_key(board):\n"
        "    return corpus.input_tensor_key(encode_cboard(CBoard.from_board(board),"
        " input_history_encoding=corpus.INPUT_HISTORY_ENCODING,"
        " input_extra_features=corpus.INPUT_EXTRA_FEATURES))\n"
        "for i, u in enumerate(moves):\n"
        "    b.push_uci(u)\n"
        "    if i + 1 >= 9: keys.append([unfixed_key(b), corpus.search_key(b)])\n"
        "try:\n"
        "    corpus.row_key(b)\n"
        "except RuntimeError as exc:\n"
        "    refused = 'history_rep_fix' in str(exc)\n"
        "else:\n"
        "    refused = False\n"
        "print(json.dumps({'keys': keys, 'refused': refused}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": ".", "CUDA_VISIBLE_DEVICES": ""}, timeout=300, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["refused"] is True
    unfixed = payload["keys"]
    assert len(unfixed) == len(fixed)
    differing = [9 + i for i, (row, _search) in enumerate(unfixed) if row != fixed[i]]
    # Plies 17..23: the frames hold the repetition AND the d2d3 pawn move.
    assert differing == list(range(17, 24)), differing
    assert [search for _, search in unfixed] == fixed_search


def test_a_listed_shard_whose_rows_disagree_with_the_inventory_is_refused(tmp_path: Path) -> None:
    in_dir = build_corpus(tmp_path)
    rows = read_rows(in_dir / SHARDS[1])
    (in_dir / SHARDS[1]).unlink()
    write_shard(in_dir / SHARDS[1], rows[:-1])  # truncated at a line boundary
    with pinned(tmp_path), pytest.raises(repair.RepairError, match="claims"):
        repair.run(args_for(in_dir, None, audit_only=True, bench_encode_rows=0))


def test_unlisted_shards_are_recorded_and_never_repaired_in_audit_mode_either(tmp_path: Path) -> None:
    in_dir = build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    with pinned(tmp_path):
        manifest = repair.run(
            args_for(in_dir, out_dir, audit_mode=True, workers=WORKER),
            searcher_factory=fake_factory(ScriptedEngine()),
        )
    assert manifest["production"] is False
    assert manifest["input"]["workers"] == [WORKER]
    assert manifest["input"]["unlisted_shards_skipped"] == list(UNLISTED_SHARDS)
    assert [e["path"] for e in manifest["input"]["unlisted_shards"]] == list(UNLISTED_SHARDS)
    assert sorted(p.name for p in out_dir.glob("*.jsonl.zst")) == sorted(SHARDS)


def test_a_listed_shard_with_an_appended_row_is_refused(tmp_path: Path) -> None:
    """The inventory's claim is held as an EQUALITY: a shard that grew past
    its progress record is as unclaimed as one that shrank."""
    in_dir = build_corpus(tmp_path)
    rows = read_rows(in_dir / SHARDS[1])
    (in_dir / SHARDS[1]).unlink()
    write_shard(in_dir / SHARDS[1], [*rows, {**rows[-1], "ply": rows[-1]["ply"] + 1}])
    with pinned(tmp_path), pytest.raises(repair.RepairError, match="claims"):
        repair.run(args_for(in_dir, None, audit_only=True, bench_encode_rows=0))


def test_a_report_path_inside_a_corpus_or_an_existing_file_is_refused(tmp_path: Path) -> None:
    in_dir = build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    with pytest.raises(repair.RepairError, match="inside the output"):
        repair.run(args_for(in_dir, out_dir, audit_mode=True, report_json=str(out_dir / corpus.SUMMARY_NAME)))
    assert not out_dir.exists()  # refused before anything was written
    with pytest.raises(repair.RepairError, match="inside the input"):
        repair.run(args_for(in_dir, out_dir, audit_mode=True, report_json=str(in_dir / corpus.MANIFEST_NAME)))
    taken = tmp_path / "report.json"
    taken.write_text("{}")
    with pytest.raises(repair.RepairError, match="already exists"):
        repair.run(args_for(in_dir, out_dir, audit_mode=True, report_json=str(taken)))
    assert taken.read_text() == "{}"
    report = tmp_path / "fresh_report.json"
    with pinned(tmp_path):
        manifest = repair.run(
            args_for(in_dir, out_dir, audit_mode=True, report_json=str(report)),
            searcher_factory=fake_factory(ScriptedEngine()),
        )
    assert json.loads(report.read_text())["rows_out"] == manifest["rows_out"]


@pytest.mark.parametrize(
    ("before", "after", "plies", "count"),
    [
        # Game A's ambiguous gap: the b1 knight reaches c4 by a3 or d2.
        (14, 17, 3, 2),
        # Only the queen can play d1d2.
        (20, 21, 1, 1),
    ],
)
def test_the_bridge_counts_paths_and_never_picks_one(
    tmp_path: Path, before: int, after: int, plies: int, count: int,
) -> None:
    boards = true_boards(10, tmp_path)
    paths = repair.bridge_paths(boards[before], boards[after].fen(), plies, cap=8)
    assert len(paths) == count
    assert GAME_A.split()[before:after] in paths


def test_a_window_that_cannot_reproduce_its_row_is_refused_at_write_time() -> None:
    corpus.apply_history_rep_fix()  # `repaired_row` keys the board; the key requires the regime
    board = chess.Board()
    board.push_uci("e2e4")
    history = corpus.history_for(board)
    bad = repair.RowRepair(
        row_class=repair.CLASS_REPAIRED, board=board,
        history=corpus.RowHistory(
            fen=history.fen, root_fen=history.root_fen, uci=("d2d4",), reason=history.reason,
        ),
        history_kind=repair.HISTORY_CHAINED, tags=repair.RepeatTags(),
    )
    with pytest.raises(repair.RepairError, match="replaying"):
        repair.repaired_row(
            {"fen": board.fen(), "game_id": 0, "ply": 1, "run": {corpus.KEY_TT_CARRIED: True}},
            bad, phases=None,
        )


def test_could_have_double_stepped_reads_the_position_not_the_move() -> None:
    assert repair.could_have_double_stepped(chess.Board("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1"))
    assert not repair.could_have_double_stepped(chess.Board("4k3/8/8/8/4P3/4P3/8/4K3 b - - 0 1"))
    assert not repair.could_have_double_stepped(chess.Board())


def test_parse_shard_range() -> None:
    assert repair.parse_shard_range("100-102,110") == (100, 101, 102, 110)
    assert repair.parse_shard_range(None) is None
    with pytest.raises(ValueError, match="names no shard"):
        repair.parse_shard_range(",")
