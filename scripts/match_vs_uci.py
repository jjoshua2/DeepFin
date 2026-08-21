#!/usr/bin/env python3
"""UCI-vs-UCI match driver.

Pits two UCI engines against each other. Used to benchmark our model's UCI
engine against external references like rofChade/Stockfish/lc0 to get an
absolute Elo number.

Each side plays White and Black equally. Per-move budget can be fixed time
(``--time-ms``), fixed nodes (``--nodes-a`` / ``--nodes-b``), or a game clock
(``--clock-base-ms`` + ``--clock-inc-ms``). Standard chess rules apply
(5-fold repetition, 50-move, insufficient material). Adjudicates after
``--max-plies``.

Result tabulation prints:
- A wins / draws / A losses
- A score (winrate including half-credit for draws)
- Crude Elo estimate from winrate (saturates near 0/1)

Usage::

    python scripts/match_vs_uci.py \\
        --engine-a "python -m chess_anti_engine.uci --checkpoint <path> --device cpu" \\
        --engine-b /path/to/rofChadeAVX2 \\
        --games 50 --time-ms 200

    python scripts/match_vs_uci.py \\
        --engine-a "python -m chess_anti_engine.uci --checkpoint <path>" \\
        --engine-b /path/to/rofChadeBMI2AVX2 \\
        --games 50 --nodes-a 1000 --nodes-b 640000 --enforce-nodes-b

    python scripts/match_vs_uci.py \\
        --engine-a "python -m chess_anti_engine.uci --checkpoint <path>" \\
        --engine-b /path/to/rofChadeBMI2AVX2 \\
        --games 20 --clock-base-ms 30000 --clock-inc-ms 500
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import chess.syzygy
import contextlib

from chess_anti_engine.utils.game_log import (
    GameLogWriter,
    default_game_log_path,
    fingerprint_differences,
    latest_rows_by_key,
    read_game_log,
    refuse_existing_log_message,
    refuse_settings_mismatch_message,
    settings_fingerprint,
    take_over_header_only_log,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# Per-run game logs, used when neither --games-out nor --pgn-out says where.
# The settings fingerprint is in the name so two runs that differ in any
# resume-critical setting cannot land in the same file.
DEFAULT_GAME_LOG_DIR = REPO_ROOT / "runs" / "match_games"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite value > 0")
    return parsed


def _side_limit(*, time_ms: int, nodes: int | None) -> chess.engine.Limit:
    if nodes is not None:
        return chess.engine.Limit(nodes=nodes)
    return chess.engine.Limit(time=time_ms / 1000.0)


def _budget_label(*, time_ms: int, nodes: int | None) -> str:
    if nodes is not None:
        return f"{nodes} nodes"
    return f"{time_ms}ms/move"


def _clock_label(*, base_ms: int | None, inc_ms: int) -> str | None:
    if base_ms is None:
        return None
    return f"{base_ms / 1000.0:g}+{inc_ms / 1000.0:g}s"


def _elo_from_score(score: float) -> float | None:
    if not 0.0 < score < 1.0:
        return None
    return -400.0 * math.log10(1.0 / score - 1.0)


def _score_ci(points: list[float], *, z: float = 1.96) -> tuple[float, float] | None:
    n = len(points)
    if n < 2:
        return None
    mean = sum(points) / n
    var = sum((p - mean) ** 2 for p in points) / (n - 1)
    half = z * math.sqrt(var / n)
    return max(0.0, mean - half), min(1.0, mean + half)


def _parse_opening_line(line: str) -> chess.Board | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    fields = stripped.split()
    if len(fields) >= 6 and fields[4].isdigit() and fields[5].isdigit():
        fen = " ".join(fields[:6])
    elif len(fields) >= 4:
        fen = " ".join([*fields[:4], "0", "1"])
    else:
        return None
    try:
        return chess.Board(fen)
    except ValueError:
        return None


def _load_openings(path: Path, *, limit: int | None = None, offset: int = 0) -> list[chess.Board]:
    boards: list[chess.Board] = []
    skipped = 0
    with path.open() as f:
        for line in f:
            board = _parse_opening_line(line)
            if board is None:
                continue
            if skipped < offset:
                skipped += 1
                continue
            boards.append(board)
            if limit is not None and len(boards) >= limit:
                break
    if not boards:
        raise ValueError(f"no valid opening FEN/EPD rows found in {path}")
    return boards


@dataclass(frozen=True)
class GameRecord:
    result: str
    plies: int
    start_board: chess.Board
    moves: tuple[chess.Move, ...]
    move_records: tuple[MoveRecord, ...]
    termination: str = "rules"


@dataclass(frozen=True)
class MoveRecord:
    ply: int
    color: str
    move: str
    elapsed_s: float
    nodes: int | None
    engine_time_s: float | None
    score_cp: int | None
    white_clock_before_s: float | None
    black_clock_before_s: float | None
    white_clock_after_s: float | None
    black_clock_after_s: float | None


@dataclass(frozen=True)
class MoveResult:
    move: chess.Move | None
    info: dict[str, Any]


@dataclass(frozen=True)
class EngineSide:
    label: str
    engine: chess.engine.SimpleEngine
    time_ms: int
    nodes: int | None
    clock_base_ms: int | None
    clock_inc_ms: int
    enforce_nodes: bool


def _play_label(
    *,
    time_ms: int,
    nodes: int | None,
    clock_base_ms: int | None,
    clock_inc_ms: int,
) -> str:
    clock = _clock_label(base_ms=clock_base_ms, inc_ms=clock_inc_ms)
    if clock is not None:
        return clock
    return _budget_label(time_ms=time_ms, nodes=nodes)


def _side_limit_for(side: EngineSide) -> chess.engine.Limit:
    return _side_limit(time_ms=side.time_ms, nodes=side.nodes)


def _game_sides(
    side_a: EngineSide,
    side_b: EngineSide,
    *,
    a_is_white: bool,
) -> tuple[EngineSide, EngineSide]:
    return (side_a, side_b) if a_is_white else (side_b, side_a)


def _score_for_a(result: str, *, a_is_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    a_win_result = "1-0" if a_is_white else "0-1"
    return 1.0 if result == a_win_result else 0.0


def _clock_limit(
    *,
    white_clock_s: float,
    black_clock_s: float,
    white_inc_s: float,
    black_inc_s: float,
) -> chess.engine.Limit:
    return chess.engine.Limit(
        white_clock=max(0.0, white_clock_s),
        black_clock=max(0.0, black_clock_s),
        white_inc=max(0.0, white_inc_s),
        black_inc=max(0.0, black_inc_s),
    )


def _node_timeout_s(nodes: int) -> float:
    # RofChade on this workstation is ~1M nps/thread, while DeepFin can be far
    # lower depending on device and contention. Keep the timeout generous enough
    # for slow neural searches, but finite so a non-reporting engine cannot hang.
    return max(30.0, min(600.0, nodes / 10_000.0))


# Off-clock warmup deadline: must comfortably exceed the one-time torch.compile
# (max-autotune) of a neural engine, which can take well over a minute cold.
_WARMUP_TIMEOUT_S = 300.0

# ⚑ python-chess's PER-COMMAND deadline, passed to ``SimpleEngine.popen_uci(timeout=)``.
# This is the deadline that actually kills a cold compile, NOT ``_WARMUP_TIMEOUT_S``:
# ``_WARMUP_TIMEOUT_S`` bounds the node-counting loop that runs AFTER ``eng.analysis()``
# has already returned, whereas the compile stalls INSIDE ``eng.analysis()``, which
# python-chess bounds with this value. Raising --warmup-timeout-s alone therefore could
# not extend a compile window (measured 2026-08-11: a 60-game Cheese block died at
# exactly 300s of warmup under --warmup-timeout-s 1800), so ``main`` lifts BOTH.
_ENGINE_COMMAND_TIMEOUT_S = 300.0


def _info_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _info_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _score_cp(value: Any) -> int | None:
    try:
        score_obj = value.white() if hasattr(value, "white") else value
        if hasattr(score_obj, "score"):
            return score_obj.score(mate_score=100_000)
        return None if score_obj is None else int(score_obj)
    except (AttributeError, TypeError, ValueError):
        return None


def _open_engine(
    spec: str,
    cwd: str | None = None,
    *,
    timeout_s: float = _ENGINE_COMMAND_TIMEOUT_S,
) -> chess.engine.SimpleEngine:
    """``spec`` is either a single binary path or a shell-style command.

    For single-binary specs, defaults to spawning with cwd set to the
    binary's directory — many engines (rofChade, Stockfish with NNUE
    sidecar) look for net/data files relative to cwd.

    ``timeout_s`` is python-chess's PER-COMMAND deadline, and it is the one that
    actually aborts a cold ``torch.compile``. See ``_warmup_engine``.
    """
    parts = shlex.split(spec)
    if not parts:
        raise ValueError(f"empty engine spec: {spec!r}")
    # Generous per-command timeout: a neural engine's first command may pay a cold
    # torch.compile (max-autotune, up to a few minutes). python-chess defaults to
    # 10s, which aborts mid-compile (asyncio TimeoutError in analysis/play setup),
    # so the off-clock warmup is meant to absorb it. The default is a finite backstop
    # that still catches a genuinely hung engine; --warmup-timeout-s raises it.
    if len(parts) == 1 and Path(parts[0]).is_file():
        binary = Path(parts[0]).resolve()
        return chess.engine.SimpleEngine.popen_uci(
            str(binary), cwd=cwd or str(binary.parent), timeout=timeout_s,
        )
    return chess.engine.SimpleEngine.popen_uci(parts, cwd=cwd, timeout=timeout_s)


def _open_warm_engine(
    spec: str,
    opts: dict[str, str],
    *,
    nodes: int | None,
    label: str,
    warmup_timeout_s: float,
) -> chess.engine.SimpleEngine:
    """Open an engine, warm it off-clock, and hand back a engine on the plain backstop.

    ⚑ The relaxed deadline MUST NOT outlive the warmup. ``SimpleEngine.timeout`` is set
    once at popen and re-read by EVERY later command, so leaving it raised would give the
    whole match -- every per-game ``analysis()``, every ``configure()``, and the ``quit()``
    calls in the teardown -- a multi-minute deadline instead of the finite backstop that
    exists to catch a wedged engine (a real failure mode on this box: WSL2 vmbus GPU
    wedges). So the timeout is raised ONLY across the warmup and restored immediately.

    An engine given no warmup never leaves the backstop at all.
    """
    command_timeout = (
        max(_ENGINE_COMMAND_TIMEOUT_S, warmup_timeout_s)
        if nodes is not None
        else _ENGINE_COMMAND_TIMEOUT_S
    )
    print(f"[match]   command timeout: {command_timeout:.0f}s during warmup, "
          f"{_ENGINE_COMMAND_TIMEOUT_S:.0f}s thereafter")
    eng = _open_engine(spec, timeout_s=command_timeout)
    _set_options(eng, opts)
    print(f"[match]   id={eng.id.get('name', '?')!r}")
    try:
        _warmup_engine(eng, nodes=nodes, label=label, timeout_s=warmup_timeout_s)
    finally:
        # Restore even if the warmup raised: a half-open engine still gets quit().
        eng.timeout = _ENGINE_COMMAND_TIMEOUT_S
    return eng


def _set_options(eng: chess.engine.SimpleEngine, opts: dict[str, str]) -> None:
    for k, v in opts.items():
        if k in eng.options:
            try:
                option = eng.options[k]
                if option.type == "spin":
                    eng.configure({k: int(v)})
                elif option.type == "check":
                    eng.configure({k: str(v).lower() in ("1", "true", "yes")})
                else:
                    eng.configure({k: v})
            except (chess.engine.EngineError, ValueError) as exc:
                print(f"[warn] could not set {k}={v}: {exc}")


def _open_syzygy_tablebase(path: str) -> chess.syzygy.Tablebase | None:
    dirs = [p.strip() for p in path.split(os.pathsep) if p.strip()]
    if not dirs:
        return None
    tablebase = chess.syzygy.Tablebase()
    added = 0
    for directory in dirs:
        try:
            tablebase.add_directory(directory)
            added += 1
        except OSError:
            continue
    if added == 0:
        tablebase.close()
        return None
    return tablebase


def _syzygy_table_counts(tablebase: Any | None) -> tuple[int, int]:
    if tablebase is None:
        return 0, 0
    # python-chess exposes loaded Syzygy files through these table dictionaries;
    # there is no public coverage API, so validation depends on this layout.
    return len(getattr(tablebase, "wdl", ())), len(getattr(tablebase, "dtz", ()))


def _syzygy_max_pieces(tablebase: Any | None, kind: str) -> int:
    if tablebase is None:
        return 0
    tables = getattr(tablebase, kind, {})
    return max((len(str(material).replace("v", "")) for material in tables), default=0)


def _validate_syzygy_adjudicator(
    path: str,
    tablebase: Any | None,
    *,
    max_pieces: int,
) -> tuple[int, int, int, int]:
    if path and tablebase is None:
        raise SystemExit(f"could not open any Syzygy directories from {path!r}")
    wdl_count, dtz_count = _syzygy_table_counts(tablebase)
    if path and wdl_count == 0:
        raise SystemExit(
            f"Syzygy adjudication path {path!r} opened {dtz_count} DTZ tables "
            "but 0 WDL tables; adjudication requires WDL"
        )
    wdl_max = _syzygy_max_pieces(tablebase, "wdl")
    dtz_max = _syzygy_max_pieces(tablebase, "dtz")
    if path and wdl_max < max_pieces:
        raise SystemExit(
            f"Syzygy adjudication path {path!r} only has WDL up to {wdl_max} pieces; "
            f"requested adjudication up to {max_pieces}"
        )
    return wdl_count, dtz_count, wdl_max, dtz_max


def _tb_adjudicate_result(
    board: chess.Board,
    tablebase: Any | None,
    *,
    max_pieces: int,
) -> str | None:
    """Return a WDL tablebase result for covered positions.

    This is theoretical WDL only. The caller checks normal claimable game rules
    before probing, but WDL does not account for DTZ-vs-halfmove-clock nuance in
    theoretically won positions close to the fifty-move counter.
    """
    if tablebase is None:
        return None
    if chess.popcount(board.occupied) > max_pieces:
        return None
    if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(chess.BLACK):
        return None
    try:
        wdl = int(tablebase.probe_wdl(board))
    except (KeyError, IndexError, chess.syzygy.MissingTableError):
        return None
    if wdl > 0:
        return "1-0" if board.turn == chess.WHITE else "0-1"
    if wdl < 0:
        return "0-1" if board.turn == chess.WHITE else "1-0"
    return "1/2-1/2"


def _warmup_engine(
    eng: chess.engine.SimpleEngine,
    *,
    nodes: int | None,
    label: str,
    timeout_s: float = _WARMUP_TIMEOUT_S,
) -> None:
    if nodes is None:
        return
    print(f"[match] warming {label}: {nodes} nodes (off-clock)", flush=True)
    started = time.monotonic()
    # Generous timeout: the warmup's first search pays the one-time torch.compile
    # (max-autotune); the 512x16 net compiles in >300s cold, so the deadline is a
    # flag — the node-derived deadline would cancel the compile and crash.
    result = _play_node_limited(eng, chess.Board(), nodes=nodes, timeout_s=timeout_s)
    elapsed = time.monotonic() - started
    info = result.info
    nodes_seen = _info_int(info.get("nodes"))
    print(
        f"[match] warmed {label}: move={result.move} elapsed={elapsed:.1f}s "
        f"nodes={nodes_seen if nodes_seen is not None else '?'}",
        flush=True,
    )


def _play_node_limited(
    eng: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    nodes: int,
    game: object = None,
    timeout_s: float | None = None,
) -> MoveResult:
    # timeout_s overrides the node-derived deadline — used by the off-clock warmup,
    # whose first search may pay a long one-time torch.compile that the default
    # _node_timeout_s(nodes) would prematurely cancel (then crash on the late readyok).
    eff_timeout = _node_timeout_s(nodes) if timeout_s is None else float(timeout_s)
    analysis = eng.analysis(board, chess.engine.Limit(), game=game, info=chess.engine.INFO_ALL)
    deadline = time.monotonic() + eff_timeout
    best_from_pv: chess.Move | None = None
    latest_info: dict[str, Any] = {}
    nodes_seen = 0
    try:
        while True:
            if not analysis.would_block():
                try:
                    info = analysis.get()
                except chess.engine.AnalysisComplete:
                    break
                latest_info.update(info)
                with contextlib.suppress(TypeError, ValueError):
                    nodes_seen = max(nodes_seen, int(info.get("nodes", 0)))
                pv = info.get("pv")
                if pv:
                    best_from_pv = pv[0]
                if nodes_seen >= nodes:
                    analysis.stop()
                    break
            elif time.monotonic() >= deadline:
                analysis.stop()
                raise TimeoutError(
                    f"engine did not report {nodes} nodes within {eff_timeout:.1f}s "
                    f"while externally enforcing nodes (last nodes={nodes_seen}); "
                    "use native go nodes or a smaller enforced budget for slow engines"
                )
            else:
                time.sleep(0.005)
        best: Any = analysis.wait()
    finally:
        with contextlib.suppress(chess.engine.EngineError):
            analysis.stop()
    if nodes_seen:
        latest_info["nodes"] = nodes_seen
    return MoveResult(best.move or best_from_pv, latest_info)


def _play_move(
    eng: chess.engine.SimpleEngine,
    board: chess.Board,
    limit: chess.engine.Limit,
    *,
    enforce_nodes: bool,
    game: object = None,
) -> MoveResult:
    if enforce_nodes and limit.nodes is not None:
        return _play_node_limited(eng, board, nodes=int(limit.nodes), game=game)
    result = eng.play(board, limit, game=game, info=chess.engine.INFO_ALL)
    return MoveResult(result.move, dict(result.info))


def play_one_game(
    eng_w: chess.engine.SimpleEngine,
    eng_b: chess.engine.SimpleEngine,
    *,
    limit_w: chess.engine.Limit,
    limit_b: chess.engine.Limit,
    enforce_nodes_w: bool,
    enforce_nodes_b: bool,
    clock_base_w_ms: int | None = None,
    clock_base_b_ms: int | None = None,
    clock_inc_w_ms: int = 0,
    clock_inc_b_ms: int = 0,
    clock_grace_ms: int = 0,
    max_plies: int,
    start_board: chess.Board | None = None,
    move_callback: Callable[[MoveRecord], None] | None = None,
    syzygy_adjudicator: Any | None = None,
    syzygy_adjudicate_max_pieces: int = 7,
    game: object = None,
) -> GameRecord:
    """Play one game. Result string is in {"1-0","0-1","1/2-1/2"}.

    ``game`` is the python-chess game identifier: a new value triggers
    ``ucinewgame`` on both engines, so hash/tree state doesn't leak between
    games of a match.
    """
    board = chess.Board() if start_board is None else start_board.copy(stack=False)
    initial = board.copy(stack=False)
    clock_mode = clock_base_w_ms is not None or clock_base_b_ms is not None
    if clock_mode and (clock_base_w_ms is None or clock_base_b_ms is None):
        raise ValueError("clock mode requires both white and black base times")
    white_clock_s = 0.0 if clock_base_w_ms is None else clock_base_w_ms / 1000.0
    black_clock_s = 0.0 if clock_base_b_ms is None else clock_base_b_ms / 1000.0
    white_inc_s = max(0, int(clock_inc_w_ms)) / 1000.0
    black_inc_s = max(0, int(clock_inc_b_ms)) / 1000.0
    # Grace buffer: tolerate a small per-move overrun (e.g. GPU/IPC jitter) instead
    # of forfeiting the instant the clock dips below zero. The overrun is NOT
    # forgiven — the clock is left negative and the increment is added on top, so
    # the debt carries into (is subtracted from) the next move. Forfeit only if a
    # move drives the clock past -grace.
    clock_grace_s = max(0, int(clock_grace_ms)) / 1000.0
    plies = 0
    moves: list[chess.Move] = []
    move_records: list[MoveRecord] = []
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        tb_result = _tb_adjudicate_result(
            board,
            syzygy_adjudicator,
            max_pieces=syzygy_adjudicate_max_pieces,
        )
        if tb_result is not None:
            return GameRecord(tb_result, plies, initial, tuple(moves), tuple(move_records), "syzygy")
        white_to_move = board.turn == chess.WHITE
        eng = eng_w if white_to_move else eng_b
        white_before = white_clock_s if clock_mode else None
        black_before = black_clock_s if clock_mode else None
        if clock_mode:
            limit = _clock_limit(
                white_clock_s=white_clock_s,
                black_clock_s=black_clock_s,
                white_inc_s=white_inc_s,
                black_inc_s=black_inc_s,
            )
            enforce_nodes = False
        else:
            limit = limit_w if white_to_move else limit_b
            enforce_nodes = enforce_nodes_w if white_to_move else enforce_nodes_b
        started = time.monotonic()
        move_result = _play_move(eng, board, limit, enforce_nodes=enforce_nodes, game=game)
        elapsed_s = time.monotonic() - started
        move = move_result.move
        info = move_result.info

        def append_move_record(
            move_uci: str,
            *,
            white_after_s: float | None,
            black_after_s: float | None,
            # loop values bound at definition time (called same-iteration only;
            # defaults make that explicit and B023-safe)
            _ply: int = plies + 1,
            _color: str = "W" if white_to_move else "B",
            _elapsed_s: float = elapsed_s,
            _info: dict = info,
            _white_before: float | None = white_before,
            _black_before: float | None = black_before,
        ) -> None:
            move_record = MoveRecord(
                ply=_ply,
                color=_color,
                move=move_uci,
                elapsed_s=_elapsed_s,
                nodes=_info_int(_info.get("nodes")),
                engine_time_s=_info_float(_info.get("time")),
                score_cp=_score_cp(_info.get("score")),
                white_clock_before_s=_white_before,
                black_clock_before_s=_black_before,
                white_clock_after_s=white_after_s,
                black_clock_after_s=black_after_s,
            )
            move_records.append(move_record)
            if move_callback is not None:
                move_callback(move_record)

        if move is None or move not in board.legal_moves:
            append_move_record(
                "0000" if move is None else move.uci(),
                white_after_s=white_clock_s if clock_mode else None,
                black_after_s=black_clock_s if clock_mode else None,
            )
            return GameRecord(
                result="0-1" if white_to_move else "1-0",
                plies=plies,
                start_board=initial,
                moves=tuple(moves),
                move_records=tuple(move_records),
                termination="illegal_or_null",
            )
        if clock_mode:
            if white_to_move:
                white_clock_s -= elapsed_s
                if white_clock_s < -clock_grace_s:
                    append_move_record(
                        move.uci(),
                        white_after_s=white_clock_s,
                        black_after_s=black_clock_s,
                    )
                    return GameRecord("0-1", plies, initial, tuple(moves), tuple(move_records), "time")
                white_clock_s += white_inc_s
            else:
                black_clock_s -= elapsed_s
                if black_clock_s < -clock_grace_s:
                    append_move_record(
                        move.uci(),
                        white_after_s=white_clock_s,
                        black_after_s=black_clock_s,
                    )
                    return GameRecord("1-0", plies, initial, tuple(moves), tuple(move_records), "time")
                black_clock_s += black_inc_s
        append_move_record(
            move.uci(),
            white_after_s=white_clock_s if clock_mode else None,
            black_after_s=black_clock_s if clock_mode else None,
        )
        moves.append(move)
        board.push(move)
        plies += 1
    if board.is_game_over(claim_draw=True):
        return GameRecord(board.result(claim_draw=True), plies, initial, tuple(moves), tuple(move_records))
    return GameRecord("1/2-1/2", plies, initial, tuple(moves), tuple(move_records), "max_plies")


def _play_one_pairing(
    *,
    white: EngineSide,
    black: EngineSide,
    max_plies: int,
    start_board: chess.Board,
    move_callback: Callable[[MoveRecord], None] | None,
    syzygy_adjudicator: Any | None,
    syzygy_adjudicate_max_pieces: int,
    clock_grace_ms: int = 0,
    game: object = None,
) -> GameRecord:
    return play_one_game(
        white.engine,
        black.engine,
        limit_w=_side_limit_for(white),
        limit_b=_side_limit_for(black),
        enforce_nodes_w=white.enforce_nodes,
        enforce_nodes_b=black.enforce_nodes,
        clock_base_w_ms=white.clock_base_ms,
        clock_base_b_ms=black.clock_base_ms,
        clock_inc_w_ms=white.clock_inc_ms,
        clock_inc_b_ms=black.clock_inc_ms,
        clock_grace_ms=clock_grace_ms,
        max_plies=max_plies,
        start_board=start_board,
        move_callback=move_callback,
        syzygy_adjudicator=syzygy_adjudicator,
        syzygy_adjudicate_max_pieces=syzygy_adjudicate_max_pieces,
        game=game,
    )


def _write_pgn_game(
    fh,
    *,
    record: GameRecord,
    event: str,
    white_name: str,
    black_name: str,
    round_name: str,
) -> None:
    game = chess.pgn.Game()
    game.headers["Event"] = event
    game.headers["Round"] = round_name
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = record.result
    game.headers["Termination"] = record.termination
    if record.start_board.fen() != chess.Board().fen():
        game.setup(record.start_board)
    node = game
    for move in record.moves:
        node = node.add_variation(move)
    print(game, file=fh, end="\n\n")


def _write_move_record(
    writer: csv.DictWriter,
    *,
    game_index: int,
    a_is_white: bool,
    label_a: str,
    label_b: str,
    opening_index: int,
    move: MoveRecord,
) -> None:
    engine = label_a if (move.color == "W") == a_is_white else label_b
    writer.writerow({
        "game": game_index,
        "opening": opening_index,
        "ply": move.ply,
        "color": move.color,
        "engine": engine,
        "move": move.move,
        "elapsed_s": f"{move.elapsed_s:.6f}",
        "nodes": "" if move.nodes is None else move.nodes,
        "engine_time_s": "" if move.engine_time_s is None else f"{move.engine_time_s:.6f}",
        "score_cp_white": "" if move.score_cp is None else move.score_cp,
        "white_clock_before_s": "" if move.white_clock_before_s is None else f"{move.white_clock_before_s:.6f}",
        "black_clock_before_s": "" if move.black_clock_before_s is None else f"{move.black_clock_before_s:.6f}",
        "white_clock_after_s": "" if move.white_clock_after_s is None else f"{move.white_clock_after_s:.6f}",
        "black_clock_after_s": "" if move.black_clock_after_s is None else f"{move.black_clock_after_s:.6f}",
    })


# ---------------------------------------------------------------------------
# Crash-resilient game log + resume
# ---------------------------------------------------------------------------
#
# One flushed JSONL record per FINISHED game, so a crash (CUDA OOM, an engine
# dying, `timeout -k`) loses at most the game in flight instead of the whole
# match, and --resume replays only what did not finish.
#
# The schedule needs no seed to be reproducible: game ``i`` is played by
# ``a_is_white = (i % 2 == 0)`` from ``openings[(i // 2) % len(openings)]``,
# both pure functions of ``i`` and the opening file. There is no RNG in this
# driver at all, so "same settings" is the whole of "same schedule" — and
# ``load_match_resume`` re-derives the colour and the opening for every
# recorded game and refuses if either disagrees, rather than trusting that.


def _game_schedule(index: int, openings: Sequence[chess.Board]) -> tuple[bool, int]:
    """``(a_is_white, opening_index)`` for game ``index`` — the whole schedule."""
    return (index % 2 == 0), (index // 2) % len(openings)


def match_game_log_settings(
    args: argparse.Namespace,
    *,
    time_ms_a: int,
    time_ms_b: int,
    nodes_a: int | None,
    nodes_b: int | None,
    clock_base_a: int | None,
    clock_base_b: int | None,
    clock_inc_a: int,
    clock_inc_b: int,
    enforce_nodes_a: bool,
    enforce_nodes_b: bool,
    warmup_nodes_a: int | None,
    warmup_nodes_b: int | None,
    opts_a: dict[str, str],
    opts_b: dict[str, str],
) -> dict[str, Any]:
    """The settings a --resume must MATCH: both engines and both budgets.

    Everything here changes what the games ARE. ``--pgn-out`` /
    ``--move-log-out`` / ``--games-out`` are output paths and are excluded.
    """
    return {
        "engine_a": args.engine_a,
        "engine_b": args.engine_b,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "games": int(args.games),
        "time_ms_a": int(time_ms_a),
        "time_ms_b": int(time_ms_b),
        "nodes_a": None if nodes_a is None else int(nodes_a),
        "nodes_b": None if nodes_b is None else int(nodes_b),
        "clock_base_ms_a": None if clock_base_a is None else int(clock_base_a),
        "clock_base_ms_b": None if clock_base_b is None else int(clock_base_b),
        "clock_inc_ms_a": int(clock_inc_a),
        "clock_inc_ms_b": int(clock_inc_b),
        "clock_grace_ms": int(args.clock_grace_ms),
        "enforce_nodes_a": bool(enforce_nodes_a),
        "enforce_nodes_b": bool(enforce_nodes_b),
        "warmup_nodes_a": None if warmup_nodes_a is None else int(warmup_nodes_a),
        "warmup_nodes_b": None if warmup_nodes_b is None else int(warmup_nodes_b),
        "max_plies": int(args.max_plies),
        "openings": "" if args.openings is None else str(args.openings),
        "openings_limit": (
            None if args.openings_limit is None else int(args.openings_limit)
        ),
        "openings_offset": int(args.openings_offset),
        "options_a": dict(opts_a),
        "options_b": dict(opts_b),
        "syzygy_adjudicate_path": args.syzygy_adjudicate_path,
        "syzygy_adjudicate_max_pieces": int(args.syzygy_adjudicate_max_pieces),
    }


def resolve_game_log_path(args: argparse.Namespace, *, fingerprint: str) -> Path:
    """Where this match's per-game log lives.

    ``--games-out`` wins. Otherwise it rides ``--pgn-out``, which is already a
    per-run path (it is opened ``"w"``, i.e. every caller gives each match its
    own). With neither, fall back to a fingerprinted name under
    ``runs/match_games/`` so two different matches cannot share a log.
    """
    if args.games_out is not None:
        return Path(args.games_out)
    if args.pgn_out is not None:
        pgn = Path(args.pgn_out)
        return pgn.with_name(pgn.stem + ".games.jsonl")
    return default_game_log_path(
        DEFAULT_GAME_LOG_DIR,
        label=f"{args.label_a}_vs_{args.label_b}",
        fingerprint=fingerprint,
    )


@dataclass(frozen=True)
class MatchResume:
    """Finished games reloaded from a game log."""

    rows: dict[int, dict[str, Any]]   # game index -> last recorded row
    truncated_tail: bool


def load_match_resume(
    path: Path, *, settings: Mapping[str, Any], openings: Sequence[chess.Board],
) -> MatchResume:
    """Reload finished games, refusing anything that is not the same match."""
    log = read_game_log(path)
    diffs = fingerprint_differences(log.settings, settings)
    if diffs:
        raise SystemExit(refuse_settings_mismatch_message(
            path, differences=diffs, resume_flag="--resume",
        ))
    rows = latest_rows_by_key(log.games, key=lambda r: int(r["game_index"]))
    out: dict[int, dict[str, Any]] = {}
    for index, row in rows.items():
        if not 0 <= index < int(settings["games"]):
            raise SystemExit(
                f"--resume: {path} records game_index {index}, outside the "
                f"{settings['games']} games this invocation plays"
            )
        a_is_white, opening_index = _game_schedule(index, openings)
        if bool(row["a_is_white"]) != a_is_white:
            raise SystemExit(
                f"--resume: {path} game {index} was played with "
                f"a_is_white={row['a_is_white']}, but the schedule gives "
                f"{a_is_white}. The log does not describe this match."
            )
        if int(row["opening_index"]) != opening_index or (
            str(row.get("opening_fen", "")) != openings[opening_index].fen()
        ):
            raise SystemExit(
                f"--resume: {path} game {index} was played from opening "
                f"{row['opening_index']} ({row.get('opening_fen', '')!r}), but "
                f"the schedule gives opening {opening_index} "
                f"({openings[opening_index].fen()!r}). The opening list has "
                "changed under the log; refusing to mix two populations."
            )
        out[index] = row
    return MatchResume(rows=out, truncated_tail=log.truncated_tail)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine-a", required=True, help="UCI engine A (binary path or shell command)")
    p.add_argument("--engine-b", required=True, help="UCI engine B (binary path or shell command)")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--games", type=_positive_int, default=20)
    p.add_argument("--time-ms", type=_positive_int, default=200, help="fallback per-move time budget (ms)")
    p.add_argument("--time-ms-a", type=_positive_int, default=None, help="A per-move time budget (ms)")
    p.add_argument("--time-ms-b", type=_positive_int, default=None, help="B per-move time budget (ms)")
    p.add_argument("--nodes", type=_positive_int, default=None, help="fixed nodes per move for both engines")
    p.add_argument("--nodes-a", type=_positive_int, default=None, help="fixed nodes per move for engine A")
    p.add_argument("--nodes-b", type=_positive_int, default=None, help="fixed nodes per move for engine B")
    p.add_argument("--clock-base-ms", type=_positive_int, default=None, help="base clock for both engines (ms)")
    p.add_argument("--clock-base-ms-a", type=_positive_int, default=None, help="base clock for engine A (ms)")
    p.add_argument("--clock-base-ms-b", type=_positive_int, default=None, help="base clock for engine B (ms)")
    p.add_argument("--clock-inc-ms", type=int, default=0, help="increment for both engines (ms)")
    p.add_argument("--clock-inc-ms-a", type=int, default=None, help="increment for engine A (ms)")
    p.add_argument("--clock-inc-ms-b", type=int, default=None, help="increment for engine B (ms)")
    p.add_argument("--clock-grace-ms", type=int, default=0,
                   help="clock mode: tolerate up to this per-move overrun instead of "
                        "forfeiting the instant the clock hits 0 (the overrun carries "
                        "as debt into the next move). Absorbs GPU/IPC jitter. Default 0 "
                        "(strict). E.g. 100 for a 100ms grace.")
    p.add_argument(
        "--enforce-nodes",
        action="store_true",
        help="externally stop both engines after their reported node budget; use for engines that ignore 'go nodes'",
    )
    p.add_argument("--enforce-nodes-a", action="store_true", help="externally stop engine A at --nodes-a/--nodes")
    p.add_argument("--enforce-nodes-b", action="store_true", help="externally stop engine B at --nodes-b/--nodes")
    p.add_argument("--max-plies", type=_positive_int, default=300)
    p.add_argument("--openings", type=Path, default=None, help="FEN/EPD opening file; games are paired by opening with swapped colors")
    p.add_argument("--openings-limit", type=_positive_int, default=None, help="maximum opening rows to load")
    p.add_argument("--openings-offset", type=int, default=0, help="skip this many valid opening rows before loading")
    p.add_argument("--pgn-out", type=Path, default=None, help="write completed games to a PGN file")
    p.add_argument("--games-out", type=Path, default=None,
                   help="per-game JSONL written AS EACH GAME FINISHES (and read "
                        "back by --resume). Default: <--pgn-out stem>.games.jsonl, "
                        f"or {DEFAULT_GAME_LOG_DIR}/<labels>.<settings-fingerprint>"
                        ".games.jsonl when no PGN is requested.")
    p.add_argument("--resume", action="store_true",
                   help="skip the games already recorded in the game log and "
                        "fold their results into the final tabulation, as if "
                        "the match had run once. REFUSES if the log's engines, "
                        "budgets, openings or any other recorded setting differ "
                        "from this invocation. Without this flag an existing "
                        "log is an error, so two matches can never be mixed. "
                        "A resumed match is statistically valid under those "
                        "settings but is NOT bit-identical to an uninterrupted "
                        "one: both engines are restarted, so hash tables, "
                        "learned time control state and any internal RNG begin "
                        "fresh at the resume point. The fingerprint also stores "
                        "engine COMMANDS and checkpoint paths, not content "
                        "hashes — replacing a binary or a weights file in place "
                        "between segments is undetectable here, so don't.")
    p.add_argument("--move-log-out", type=Path, default=None, help="write per-move timing/search stats to CSV")
    p.add_argument("--warmup-nodes", type=_positive_int, default=None, help="run an unrated warmup search for both engines")
    p.add_argument("--warmup-nodes-a", type=_positive_int, default=None, help="run an unrated warmup search for engine A")
    p.add_argument("--warmup-nodes-b", type=_positive_int, default=None, help="run an unrated warmup search for engine B")
    p.add_argument("--warmup-timeout-s", type=_positive_float, default=_WARMUP_TIMEOUT_S,
                   help=(
                       "off-clock warmup deadline; must exceed the one-time torch.compile of a "
                       f"neural engine. RAISES ONLY: the engine's per-command deadline is "
                       f"max(this, {_ENGINE_COMMAND_TIMEOUT_S:.0f}s), so a value below "
                       f"{_ENGINE_COMMAND_TIMEOUT_S:.0f} does NOT make the warmup fail faster than "
                       f"{_ENGINE_COMMAND_TIMEOUT_S:.0f}s -- the floor keeps the uciok/isready "
                       "handshake from timing out"
                   ))
    p.add_argument("--option-a", action="append", default=[], help="Set UCI option on A (Name=Value)")
    p.add_argument("--option-b", action="append", default=[], help="Set UCI option on B (Name=Value)")
    p.add_argument(
        "--syzygy-adjudicate-path",
        default="",
        help="adjudicate covered tablebase positions using theoretical WDL",
    )
    p.add_argument(
        "--syzygy-adjudicate-max-pieces",
        type=_positive_int,
        default=6,
        help="maximum material count to probe for --syzygy-adjudicate-path",
    )
    args = p.parse_args()

    def _parse_opts(specs: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for spec in specs:
            if "=" not in spec:
                raise SystemExit(f"--option must be Name=Value, got {spec!r}")
            k, v = spec.split("=", 1)
            out[k.strip()] = v.strip()
        return out

    opts_a = _parse_opts(args.option_a)
    opts_b = _parse_opts(args.option_b)
    time_ms_a = int(args.time_ms_a if args.time_ms_a is not None else args.time_ms)
    time_ms_b = int(args.time_ms_b if args.time_ms_b is not None else args.time_ms)
    nodes_a = args.nodes_a if args.nodes_a is not None else args.nodes
    nodes_b = args.nodes_b if args.nodes_b is not None else args.nodes
    clock_base_a = args.clock_base_ms_a if args.clock_base_ms_a is not None else args.clock_base_ms
    clock_base_b = args.clock_base_ms_b if args.clock_base_ms_b is not None else args.clock_base_ms
    clock_inc_a = args.clock_inc_ms_a if args.clock_inc_ms_a is not None else args.clock_inc_ms
    clock_inc_b = args.clock_inc_ms_b if args.clock_inc_ms_b is not None else args.clock_inc_ms
    warmup_nodes_a = args.warmup_nodes_a if args.warmup_nodes_a is not None else args.warmup_nodes
    warmup_nodes_b = args.warmup_nodes_b if args.warmup_nodes_b is not None else args.warmup_nodes
    if clock_inc_a < 0 or clock_inc_b < 0:
        raise SystemExit("--clock-inc-ms values must be >= 0")
    if clock_base_a is not None and nodes_a is not None:
        raise SystemExit("engine A cannot combine clock and fixed nodes")
    if clock_base_b is not None and nodes_b is not None:
        raise SystemExit("engine B cannot combine clock and fixed nodes")
    if (clock_base_a is None) != (clock_base_b is None):
        raise SystemExit("clock mode requires a base time for both engines")
    if args.openings_offset < 0:
        raise SystemExit("--openings-offset must be >= 0")
    tb_adjudicator = _open_syzygy_tablebase(args.syzygy_adjudicate_path)
    eng_a: chess.engine.SimpleEngine | None = None
    eng_b: chess.engine.SimpleEngine | None = None
    try:
        tb_wdl_count, tb_dtz_count, tb_wdl_max, tb_dtz_max = _validate_syzygy_adjudicator(
            args.syzygy_adjudicate_path,
            tb_adjudicator,
            max_pieces=args.syzygy_adjudicate_max_pieces,
        )

        enforce_nodes_a = bool(args.enforce_nodes or args.enforce_nodes_a)
        enforce_nodes_b = bool(args.enforce_nodes or args.enforce_nodes_b)
        openings = (
            _load_openings(args.openings, limit=args.openings_limit, offset=args.openings_offset)
            if args.openings is not None
            else [chess.Board()]
        )
        log_settings = match_game_log_settings(
            args,
            time_ms_a=time_ms_a, time_ms_b=time_ms_b,
            nodes_a=nodes_a, nodes_b=nodes_b,
            clock_base_a=clock_base_a, clock_base_b=clock_base_b,
            clock_inc_a=clock_inc_a, clock_inc_b=clock_inc_b,
            enforce_nodes_a=enforce_nodes_a, enforce_nodes_b=enforce_nodes_b,
            warmup_nodes_a=warmup_nodes_a, warmup_nodes_b=warmup_nodes_b,
            opts_a=opts_a, opts_b=opts_b,
        )
        fingerprint = settings_fingerprint(log_settings)
        game_log_path = resolve_game_log_path(args, fingerprint=fingerprint)
        had_log = game_log_path.exists() and game_log_path.stat().st_size > 0
        # Decided BEFORE the engines are started: a refusal must not leave two
        # UCI subprocesses (and a compiled net) behind.
        if had_log and not args.resume:
            # A crash between the header write and the first finished game
            # (engine warmup, a slow checkpoint load inside engine A) leaves a
            # log with zero games; refusing the retry then protects nothing.
            taken_over, note = take_over_header_only_log(
                game_log_path, fingerprint=fingerprint,
            )
            if taken_over:
                print(f"[match] {note}", flush=True)
                had_log = False
            else:
                message = refuse_existing_log_message(
                    game_log_path, resume_flag="--resume",
                    out_flag="--games-out",
                    # The default path carries the fingerprint; --games-out and
                    # --pgn-out are given by hand and compared against nothing.
                    fingerprint_keyed=(
                        args.games_out is None and args.pgn_out is None
                    ),
                )
                raise SystemExit(message + ("\n" + note if note else ""))
        resumed = (
            load_match_resume(
                game_log_path, settings=log_settings, openings=openings,
            )
            if had_log else None
        )
        done_games: dict[int, dict[str, Any]] = {} if resumed is None else resumed.rows
        if resumed is not None:
            print(
                f"[match] RESUMED {len(done_games)}/{args.games} finished games "
                f"from {game_log_path}",
                flush=True,
            )
            if resumed.truncated_tail:
                print(
                    "[match] resume: the log's last line was truncated mid-write; "
                    "that game is replayed",
                    flush=True,
                )
        elif args.resume:
            print(
                f"[match] WARNING --resume: no game log at {game_log_path} — "
                f"starting from scratch (settings fingerprint {fingerprint})",
                flush=True,
            )
        if args.pgn_out is not None:
            args.pgn_out.parent.mkdir(parents=True, exist_ok=True)
            if done_games and not (
                args.pgn_out.exists() and args.pgn_out.stat().st_size > 0
            ):
                # The PGN is opened in APPEND mode on a resume, so this process
                # writes only the games it plays. With the earlier segment's
                # PGN missing (or never requested), the file that appears is a
                # partial record of a match whose JSONL log is complete, and
                # nothing downstream can tell.
                print(
                    f"[match] WARNING: --pgn-out {args.pgn_out} is missing or "
                    f"empty, but {game_log_path} already holds "
                    f"{len(done_games)} finished game(s). The PGN will contain "
                    f"ONLY the games this process plays, not the whole match "
                    f"the summary reports. Point --pgn-out at the earlier "
                    f"segment's file, or treat this one as a partial record.",
                    file=sys.stderr, flush=True,
                )
        if args.move_log_out is not None:
            args.move_log_out.parent.mkdir(parents=True, exist_ok=True)

        print(f"[match] opening A: {args.engine_a}")
        eng_a = _open_warm_engine(
            args.engine_a, opts_a, nodes=warmup_nodes_a, label=args.label_a,
            warmup_timeout_s=float(args.warmup_timeout_s),
        )
        print(f"[match] opening B: {args.engine_b}")
        eng_b = _open_warm_engine(
            args.engine_b, opts_b, nodes=warmup_nodes_b, label=args.label_b,
            warmup_timeout_s=float(args.warmup_timeout_s),
        )
        side_a = EngineSide(
            label=args.label_a,
            engine=eng_a,
            time_ms=time_ms_a,
            nodes=nodes_a,
            clock_base_ms=clock_base_a,
            clock_inc_ms=clock_inc_a,
            enforce_nodes=enforce_nodes_a,
        )
        side_b = EngineSide(
            label=args.label_b,
            engine=eng_b,
            time_ms=time_ms_b,
            nodes=nodes_b,
            clock_base_ms=clock_base_b,
            clock_inc_ms=clock_inc_b,
            enforce_nodes=enforce_nodes_b,
        )

        # Seeded from the resumed games so every tabulated number below (W/D/L,
        # score, colour balance, plies, the CI over `points`) is what one
        # uninterrupted match would have produced. `points` ends up ordered
        # resumed-then-new rather than by game index; the mean and the CI are
        # order-free, and nothing else reads the list.
        a_wins = a_draws = a_losses = 0
        a_white_count = 0
        a_black_count = 0
        total_plies = 0
        points: list[float] = []
        for _idx in sorted(done_games):
            _row = done_games[_idx]
            _point = float(_row["score_a"])
            points.append(_point)
            if _point == 1.0:
                a_wins += 1
            elif _point == 0.0:
                a_losses += 1
            else:
                a_draws += 1
            if bool(_row["a_is_white"]):
                a_white_count += 1
            else:
                a_black_count += 1
            total_plies += int(_row["plies"])
        resumed_games = len(done_games)
        t0 = time.time()
        print(
            f"[match] playing {args.games} games, "
            f"A budget={_play_label(time_ms=time_ms_a, nodes=nodes_a, clock_base_ms=clock_base_a, clock_inc_ms=clock_inc_a)}, "
            f"B budget={_play_label(time_ms=time_ms_b, nodes=nodes_b, clock_base_ms=clock_base_b, clock_inc_ms=clock_inc_b)}, "
            f"max_plies={args.max_plies}, "
            f"enforce_nodes=A:{enforce_nodes_a} B:{enforce_nodes_b}, "
            f"openings={len(openings)}"
        )
        if tb_adjudicator is not None:
            print(
                f"[match] Syzygy adjudication: path={args.syzygy_adjudicate_path!r} "
                f"max_pieces={args.syzygy_adjudicate_max_pieces} "
                f"tables={tb_wdl_count} WDL/{tb_dtz_count} DTZ "
                f"coverage={tb_wdl_max} WDL/{tb_dtz_max} DTZ",
                flush=True,
            )

        # Append when resuming: the games already on file must not be erased by
        # the run that continues them. (Without --resume these keep truncating,
        # which is the behaviour every caller has today.)
        pgn_mode = "a" if resumed is not None else "w"
        move_mode = "a" if resumed is not None else "w"
        move_log_had_rows = (
            args.move_log_out is not None
            and resumed is not None
            and args.move_log_out.exists()
            and args.move_log_out.stat().st_size > 0
        )
        pgn_fh = args.pgn_out.open(pgn_mode) if args.pgn_out is not None else None
        try:
            move_fh = (
                args.move_log_out.open(move_mode, newline="")
                if args.move_log_out is not None else None
            )
        except Exception:
            if pgn_fh is not None:
                pgn_fh.close()
            raise
        move_writer: csv.DictWriter | None = None
        if move_fh is not None:
            move_writer = csv.DictWriter(
                move_fh,
                fieldnames=[
                    "game",
                    "opening",
                    "ply",
                    "color",
                    "engine",
                    "move",
                    "elapsed_s",
                    "nodes",
                    "engine_time_s",
                    "score_cp_white",
                    "white_clock_before_s",
                    "black_clock_before_s",
                    "white_clock_after_s",
                    "black_clock_after_s",
                ],
            )
            if not move_log_had_rows:
                # A resumed CSV already carries its header; a second one mid-file
                # would be parsed as a data row by every reader of it.
                move_writer.writeheader()

        def live_move_callback(
            *,
            game_index: int,
            a_is_white: bool,
            opening_index: int,
        ) -> Callable[[MoveRecord], None] | None:
            if move_writer is None or move_fh is None:
                return None
            writer = move_writer
            fh = move_fh

            def _write_live(move: MoveRecord) -> None:
                _write_move_record(
                    writer,
                    game_index=game_index,
                    a_is_white=a_is_white,
                    label_a=args.label_a,
                    label_b=args.label_b,
                    opening_index=opening_index,
                    move=move,
                )
                fh.flush()

            return _write_live

        game_log = GameLogWriter(
            game_log_path, driver="match_vs_uci", settings=log_settings,
            resuming=had_log,
        )
        print(f"[match] game log -> {game_log_path} (fingerprint {fingerprint})",
              flush=True)
        try:
            for i in range(args.games):
                a_white, opening_idx = _game_schedule(i, openings)
                if i in done_games:
                    continue
                white, black = _game_sides(side_a, side_b, a_is_white=a_white)
                start_board = openings[opening_idx]
                _g_t0 = time.time()
                if a_white:
                    a_white_count += 1
                else:
                    a_black_count += 1
                record = _play_one_pairing(
                    white=white,
                    black=black,
                    max_plies=args.max_plies,
                    start_board=start_board,
                    move_callback=live_move_callback(
                        game_index=i + 1,
                        a_is_white=a_white,
                        opening_index=opening_idx + 1,
                    ),
                    syzygy_adjudicator=tb_adjudicator,
                    syzygy_adjudicate_max_pieces=args.syzygy_adjudicate_max_pieces,
                    clock_grace_ms=args.clock_grace_ms,
                    game=i + 1,
                )
                result = record.result
                plies = record.plies
                point = _score_for_a(result, a_is_white=a_white)
                points.append(point)
                if point == 1.0:
                    a_wins += 1
                elif point == 0.0:
                    a_losses += 1
                else:
                    a_draws += 1
                if pgn_fh is not None:
                    _write_pgn_game(
                        pgn_fh,
                        record=record,
                        event=f"{args.label_a} vs {args.label_b}",
                        white_name=white.label,
                        black_name=black.label,
                        round_name=str(i + 1),
                    )
                if pgn_fh is not None:
                    pgn_fh.flush()
                if move_fh is not None:
                    move_fh.flush()
                # Durable BEFORE the next game starts: this is the record that
                # makes a crashed match resumable rather than lost. It is also
                # written LAST on purpose — JSONL is the commit record, and
                # anything not in it is replayed on resume. A crash between the
                # PGN write above and this one therefore costs a duplicated PGN
                # game, not a silently missing one.
                game_log.write_game({
                    "game_index": i,          # 0-based schedule index
                    "game_number": i + 1,     # what the console and the CSV show
                    "opening_index": opening_idx,
                    "opening_fen": openings[opening_idx].fen(),
                    "a_is_white": bool(a_white),
                    "white": white.label,
                    "black": black.label,
                    "result": result,
                    "score_a": point,
                    "plies": int(plies),
                    "termination": record.termination,
                    "duration_s": round(time.time() - _g_t0, 2),
                })
                total_plies += plies
                elapsed = time.time() - t0
                total = a_wins + a_draws + a_losses
                score = (a_wins + 0.5 * a_draws) / max(1, total)
                print(
                    f"[game {i+1:>3}/{args.games}] {args.label_a}={'W' if a_white else 'B'} "
                    f"opening={opening_idx + 1}/{len(openings)} result={result} "
                    f"termination={record.termination} plies={plies}  "
                    f"running A: {a_wins}W-{a_draws}D-{a_losses}L "
                    f"score={score:.3f}  ({elapsed:.0f}s)",
                    flush=True,
                )
        finally:
            game_log.close()
            if pgn_fh is not None:
                pgn_fh.close()
            if move_fh is not None:
                move_fh.close()
    finally:
        if tb_adjudicator is not None:
            tb_adjudicator.close()
        if eng_a is not None:
            eng_a.quit()
        if eng_b is not None:
            eng_b.quit()

    total = a_wins + a_draws + a_losses
    if total == 0:
        print("[match] no games completed.")
        return
    score = (a_wins + 0.5 * a_draws) / total
    dt = time.time() - t0
    played = total - resumed_games
    print()
    if resumed_games:
        print(
            f"[match] RESUMED run: {resumed_games} games reloaded from "
            f"{game_log_path}, {played} played in this process — the "
            f"tabulation below covers all {total}, the timing only this process"
        )
    per_game = dt / played if played else float("nan")
    print(f"[match] {total} games in {dt:.0f}s ({per_game:.1f}s/game, avg {total_plies/total:.0f} plies/game)")
    print(f"[match] {args.label_a} (A) vs {args.label_b} (B):")
    print(f"  A wins   : {a_wins}")
    print(f"  draws    : {a_draws}")
    print(f"  A losses : {a_losses}")
    print(f"  A score  : {score:.3f}  (incl. half for draws)")
    print(f"  A as W/B : {a_white_count}/{a_black_count}")
    ci = _score_ci(points)
    if ci is not None:
        lo, hi = ci
        elo_lo = _elo_from_score(lo)
        elo_hi = _elo_from_score(hi)
        print(f"  Score 95% CI: [{lo:.3f}, {hi:.3f}]")
        if elo_lo is not None and elo_hi is not None:
            print(f"  Elo 95% CI  : [{elo_lo:+.0f}, {elo_hi:+.0f}]")
    elo = _elo_from_score(score)
    if elo is not None:
        print(f"  Elo (A - B) ≈ {elo:+.0f}")


if __name__ == "__main__":
    main()
