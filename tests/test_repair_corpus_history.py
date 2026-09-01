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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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

GAMES: dict[int, tuple[str | None, str, frozenset[int]]] = {
    # game_id -> (start override, moves, dropped plies)
    10: (None, GAME_A, DROPPED_A),
    24: (None, GAME_B, DROPPED_B),
    38: (GAME_C_START, GAME_C, frozenset()),
    52: (None, GAME_D, frozenset()),
    66: (None, GAME_E, frozenset()),
}


@pytest.fixture(autouse=True)
def production_rep_fix() -> None:
    rep_fix.apply(True)


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


def true_boards(game_id: int, tmp_path: Path) -> list[chess.Board]:
    """Position ``i`` (before move ``i``) of the game, each with its live stack."""
    start_override, moves, _ = GAMES[game_id]
    if start_override is None:
        start = sample_starting_board(
            rng=corpus.book_rng(seed=SEED, worker_id=WORKER, game_id=game_id),
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


def fake_phases(board: chess.Board, played: str) -> list[dict[str, Any]]:
    """A schema-1 staircase block whose rank 1 is the played move."""
    legal = [m.uci() for m in board.legal_moves]
    ranked = [played, *[m for m in legal if m != played]]
    lines = [[rank, move, float(100 - rank), 1000] for rank, move in enumerate(ranked, start=1)]
    return [{
        "index": 0, "width_requested": "all", "width_realized": len(legal),
        "width_streamed": len(legal), "depth_requested": 3, "searchmoves": None,
        "per_depth": [{
            "depth": 3, "complete": True, "emissions": len(legal),
            "nodes_at_depth": 1000, "lines": lines,
        }],
        "nodes_at_depth": {"3": 1000},
        "anomalies": {},
    }]


def row_for(
    game_id: int, ply: int, board: chess.Board, played: str, *, config_sha: str,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "run": {"run_id": "test_repair", "config_sha256": config_sha, corpus.KEY_TT_CARRIED: True},
        "fen": board.fen(),
        "dedup_key": corpus.dedup_key(board),
        "worker_id": WORKER,
        "game_id": game_id,
        "ply": ply,
        "stm": "w" if board.turn else "b",
        "piece_count": chess.popcount(board.occupied),
        "game_phase": "opening",
        "played_move": played,
        "selection": {"temp": 1.0, "legal_moves": board.legal_moves.count()},
        "phases": fake_phases(board, played),
        "result": 0.0, "result_pgn": "1/2-1/2", "adjudication": None,
    }


def game_rows(game_id: int, tmp_path: Path) -> list[dict[str, Any]]:
    _, moves, dropped = GAMES[game_id]
    boards = true_boards(game_id, tmp_path)
    config_sha = corpus.stamp_sha256(requested_config(tmp_path))
    return [
        row_for(game_id, ply, boards[ply], uci, config_sha=config_sha)
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
        SHARDS[1]: [*tail_b, *game_rows(38, tmp_path), *game_rows(52, tmp_path), *game_rows(66, tmp_path)],
    }
    for name, rows in shard_rows.items():
        write_shard(in_dir / name, rows)
    # The unlisted shard holds a game the corpus does not claim.
    write_shard(in_dir / UNLISTED_SHARD, game_rows(52, tmp_path)[:2])
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
    }.get(game_id, {})
    for reason, plies in gaps.items():
        if set(span) & set(plies):
            return repair.QUARANTINE_PREFIX + reason, None
    if root < 0 and start_override is not None:
        return repair.QUARANTINE_PREFIX + repair.REASON_BOOK_MISMATCH, None
    bridged = bool(set(span) & set(dropped) - set(GAP_AMBIGUOUS_A) - set(GAP_UNBRIDGED_A))
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
        "rep_in_window": len(set(frames)) != len(frames),
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


def run_repair(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    in_dir = build_corpus(tmp_path)
    out_dir = tmp_path / "out"
    engine = ScriptedEngine(multipv=1)
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
        game_id: true_boards(game_id, tmp_path) for game_id in GAMES
    }
    return {
        "in_dir": in_dir, "out_dir": out_dir, "engine": engine, "manifest": manifest,
        "written": written, "inputs": inputs, "truth": truth, "map": row_map(out_dir),
        "tmp_path": tmp_path,
    }


@pytest.fixture
def repaired(tmp_path: Path) -> dict[str, Any]:
    """The DEFAULT run: labels preserved, rows tagged."""
    return run_repair(tmp_path)


@pytest.fixture
def relabeled(tmp_path: Path) -> dict[str, Any]:
    """The parked R2 path, switched on explicitly."""
    return run_repair(tmp_path, relabel=repair.RELABEL_WINDOW)


def banked_rows() -> list[tuple[int, int]]:
    return [
        (game_id, ply)
        for game_id, (_, moves, dropped) in GAMES.items()
        for ply in range(len(moves.split())) if ply not in dropped
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
    }
    if relabel != repair.RELABEL_OFF:
        want_classes.add(repair.CLASS_RELABELED)
    assert seen_classes == want_classes


def test_labels_are_copied_byte_for_byte_and_rows_are_tagged_and_keyed(repaired: dict[str, Any]) -> None:
    """⚑⚑ R2 IS PARKED: no original field changes; the additions are exactly the schema-2 set."""
    added = {
        "history_root_fen", "history_uci", "history_plies", "history_root_reason",
        "input_key", "search_key", "rep_in_window", "rep_in_segment",
        "cur_position_repeat_count", "label_regime", "repair",
    }
    tag_hits = {"rep_in_window": 0, "rep_in_segment": 0, "cur_position_repeat_count": 0}
    for key, row in repaired["written"].items():
        source = repaired["inputs"][key]
        label = f"game {key[0]} ply {key[1]}"
        assert json.dumps(row["phases"], sort_keys=True) == json.dumps(source["phases"], sort_keys=True), label
        for name, value in source.items():
            if name != "schema":
                assert row[name] == value, (label, name)
        assert set(row) - set(source) == added, label
        assert row["label_regime"] == repair.LABEL_REGIME_CARRIED_BLIND, label
        assert row["repair"]["label"] == repair.LABEL_ORIGINAL, label
        live = repaired["truth"][key[0]][key[1]]
        assert row["input_key"] == corpus.row_key(live), label
        assert row["search_key"] == corpus.search_key(live), label
        want = expected_tags(live)
        got = {name: row[name] for name in want}
        assert got == want, (label, got, want)
        tag_hits["rep_in_window"] += int(row["rep_in_window"])
        tag_hits["rep_in_segment"] += int(row["rep_in_segment"])
        tag_hits["cur_position_repeat_count"] += int(row["cur_position_repeat_count"] >= 2)
    # The tags are not vacuous: every one fires somewhere, and game E (the
    # clock-0 root recurring) is tagged on its shuffle rows.
    assert all(v > 0 for v in tag_hits.values()), tag_hits
    game_e = {ply: repaired["written"][(66, ply)] for ply in range(8, len(GAME_E.split()))}
    assert all(r["rep_in_segment"] and r["rep_in_window"] for r in game_e.values()), game_e
    assert max(r["cur_position_repeat_count"] for r in game_e.values()) >= 3
    manifest = repaired["manifest"]
    assert manifest["tags"]["rep_in_segment"] == tag_hits["rep_in_segment"]
    assert manifest["tags"]["rep_in_window"] == tag_hits["rep_in_window"]
    assert manifest["relabel"]["mode"] == repair.RELABEL_OFF
    assert manifest["classes"].get(repair.CLASS_RELABELED, 0) == 0
    assert manifest["label_regime"] == repair.LABEL_REGIME_CARRIED_BLIND


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
    # ambiguous gap) produced written rows in game A.
    manifest = repaired["manifest"]
    assert manifest["anchorless"] > 0
    assert manifest["book_games"] == {
        repair.BOOK_EXACT: 3, repair.BOOK_BRIDGED: 1, repair.BOOK_MISMATCH: 1,
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
            assert row["label_regime"] == repair.LABEL_REGIME_CARRIED_BLIND
            assert row["phases"] == fake_phases(chess.Board(row["fen"]), row["played_move"])
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
    off = repair.run(args_for(in_dir, None, audit_only=True, bench_encode_rows=0))
    tags = off["tags"]
    assert 0 < tags["rep_in_segment"] <= tags["rep_in_banked_window"]
    assert repair.CLASS_RELABELED not in off["classes"]
    window = repair.run(
        args_for(in_dir, None, audit_only=True, bench_encode_rows=0, relabel="window"),
    )
    assert window["classes"][repair.CLASS_RELABELED] == tags["rep_in_banked_window"]
    segment = repair.run(
        args_for(in_dir, None, audit_only=True, bench_encode_rows=0, relabel="segment"),
    )
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
    # The unlisted shard was skipped BY NAME and reported.
    assert manifest["input"]["unlisted_shards_skipped"] == [UNLISTED_SHARD]
    assert not (repaired["out_dir"] / UNLISTED_SHARD).exists()
    assert {s["path"] for s in manifest["shards"]} == set(SHARDS)
    # The input shards are untouched.
    for name in (*SHARDS, UNLISTED_SHARD):
        assert all("history_uci" not in r for r in read_rows(repaired["in_dir"] / name))


def test_audit_only_writes_nothing_and_runs_no_engine(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    in_dir = build_corpus(tmp_path)
    before = sorted(p.name for p in in_dir.iterdir())
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
        repair.run(args_for(in_dir, out_dir, engine="/bin/true", relabel="window"))
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_a_manifest_without_an_engine_sha_cannot_relabel(tmp_path: Path) -> None:
    in_dir = build_corpus(tmp_path)
    manifest_path = in_dir / corpus.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    del manifest["engine"]["sha256"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(repair.RepairError, match="no engine sha256"):
        repair.run(args_for(in_dir, tmp_path / "out", engine="/bin/true", relabel="window"))


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
    with pytest.raises(repair.RepairError, match="--book-sha256"):
        repair.run(args_for(in_dir, None, audit_only=True, bench_encode_rows=0, book_sha256="0" * 64))


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
    with pytest.raises(FileExistsError):
        repair.run(args_for(in_dir, out_dir), searcher_factory=fake_factory(ScriptedEngine()))


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
        repair.repaired_row({"fen": board.fen(), "game_id": 0, "ply": 1}, bad, phases=None)


def test_could_have_double_stepped_reads_the_position_not_the_move() -> None:
    assert repair.could_have_double_stepped(chess.Board("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1"))
    assert not repair.could_have_double_stepped(chess.Board("4k3/8/8/8/4P3/4P3/8/4K3 b - - 0 1"))
    assert not repair.could_have_double_stepped(chess.Board())


def test_parse_shard_range() -> None:
    assert repair.parse_shard_range("100-102,110") == (100, 101, 102, 110)
    assert repair.parse_shard_range(None) is None
    with pytest.raises(ValueError, match="names no shard"):
        repair.parse_shard_range(",")
