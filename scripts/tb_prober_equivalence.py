#!/usr/bin/env python3
"""C Syzygy prober (``chess_anti_engine.tb._tb_ext``, vendored Fathom) vs python-chess.

Generates a deterministic sample of legal positions for every material class
the tables cover, probes each with python-chess (the oracle) and with the C
prober (through both the FEN entry and the CBoard entry), and banks every
observation plus a manifest under ``--out``. Optionally times both probers on
the same sample (single thread, CPU only).

Categories per material class (see docs/tablebase_prober.md for the mapping
and every expected difference):

* ``random``     uniformly placed legal positions, either side to move, halfmove
                 clock 0 (half), 99 (a fifth) or 1..98;
* ``ep``         a double pawn push just happened, en-passant square set;
* ``castling``   castling rights present -- neither prober may answer;
* ``bogus_ep``   a legal board given an arbitrary, often impossible, ep square
                 (see ``ep_must_be_refused`` for the one case the C prober
                 refuses where python-chess applies a nonsense capture);
* ``terminal``   checkmates and stalemates reached by one move from ``random``.
* ``ep_a1``      a FEN whose ep field is ``a1`` -- "no ep" for both readers;
* ``ep_origin_occupied``  a capturable ep square with an enemy pawn on ep-/+8
                 but ep+/-8 occupied: python-chess's status() calls it invalid,
                 both probers still capture the real pawn and must agree;
* ``castling_letters``  random ``KQkq`` letter sets on boards with and without
                 back-rank rooks/kings: the FEN reader must map them exactly as
                 python-chess's parser does (clean, stale and absent rights).

Root moves: for the first ``--root-per-class`` ``random`` positions of each
class, every legal move's DTZ (from the mover's view) and Fathom's
``root_probe_dtz`` rank are compared against the repo's python-chess root
logic (``tablebase._dtz_of_played_move``).

Usage (read-only tables; at most two worker processes)::

    PYTHONPATH=. nice -n 10 python3 -u scripts/tb_prober_equivalence.py \\
        --syzygy-path data/syzygy_3-4-5:data/syzygy_6 --out <dir> --workers 2
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import chess.syzygy

SEPARATOR = ":"
_PROBE_ERRORS = (KeyError, IndexError, chess.syzygy.MissingTableError)


def material_classes(syzygy_path: str) -> list[str]:
    """Every WDL table stem (e.g. ``KQRvKN``) across all components."""
    stems: set[str] = set()
    for part in syzygy_path.split(SEPARATOR):
        for name in os.listdir(part):
            stem, ext = os.path.splitext(name)
            if ext == ".rtbw":
                stems.add(stem)
    return sorted(stems, key=lambda s: (len(s), s))


def open_python_tablebase(syzygy_path: str) -> chess.syzygy.Tablebase:
    tb = chess.syzygy.Tablebase(max_fds=None)
    for part in syzygy_path.split(SEPARATOR):
        tb.add_directory(part)
    return tb


# ---------------------------------------------------------------- sampling

_PIECE = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP,
          "N": chess.KNIGHT, "P": chess.PAWN}
_PAWN_SQUARES = list(range(8, 56))


def _put(board: chess.Board, rng: random.Random, letters: str, color: chess.Color,
         free: list[int]) -> bool:
    for letter in letters:
        piece = _PIECE[letter]
        options = [sq for sq in free if piece != chess.PAWN or 8 <= sq < 56]
        if not options:
            return False
        sq = rng.choice(options)
        free.remove(sq)
        board.set_piece_at(sq, chess.Piece(piece, color))
    return True


def _clock(rng: random.Random) -> int:
    u = rng.random()
    if u < 0.5:
        return 0
    if u < 0.7:
        return 99
    return rng.randint(1, 98)


def _orient(rng: random.Random, cls: str) -> tuple[str, str]:
    """(white letters, black letters) for one random colour assignment."""
    strong, weak = cls.split("v")
    return (strong, weak) if rng.random() < 0.5 else (weak, strong)


def gen_random(rng: random.Random, cls: str) -> chess.Board | None:
    white, black = _orient(rng, cls)
    board = chess.Board(None)
    free = list(range(64))
    if not (_put(board, rng, white, chess.WHITE, free) and _put(board, rng, black, chess.BLACK, free)):
        return None
    board.turn = rng.random() < 0.5
    board.halfmove_clock = _clock(rng)
    return board if board.is_valid() else None


def gen_ep(rng: random.Random, cls: str) -> chess.Board | None:
    white, black = _orient(rng, cls)
    stm = rng.random() < 0.5
    us, them = (white, black) if stm else (black, white)
    if "P" not in us or "P" not in them:
        return None
    f = rng.randrange(8)
    df = rng.choice([d for d in (-1, 1) if 0 <= f + d < 8])
    if stm:  # black just played f7-f5
        pawn_sq, ep_sq, origin = 32 + f, 40 + f, 48 + f
        capturer = 32 + f + df
    else:    # white just played f2-f4
        pawn_sq, ep_sq, origin = 24 + f, 16 + f, 8 + f
        capturer = 24 + f + df
    board = chess.Board(None)
    board.set_piece_at(pawn_sq, chess.Piece(chess.PAWN, not stm))
    board.set_piece_at(capturer, chess.Piece(chess.PAWN, stm))
    free = [sq for sq in range(64) if sq not in (pawn_sq, ep_sq, origin, capturer)]
    us_rest, them_rest = us.replace("P", "", 1), them.replace("P", "", 1)
    if not (_put(board, rng, us_rest, stm, free) and _put(board, rng, them_rest, not stm, free)):
        return None
    board.turn = stm
    board.ep_square = ep_sq
    board.halfmove_clock = 0  # a double push resets the clock
    return board if board.is_valid() else None


def gen_castling(rng: random.Random, cls: str) -> chess.Board | None:
    white, black = _orient(rng, cls)
    color = rng.random() < 0.5
    letters = white if color else black
    if "R" not in letters:
        return None
    king_sq = chess.E1 if color else chess.E8
    rook_sq = rng.choice([chess.H1, chess.A1] if color else [chess.H8, chess.A8])
    board = chess.Board(None)
    board.set_piece_at(king_sq, chess.Piece(chess.KING, color))
    board.set_piece_at(rook_sq, chess.Piece(chess.ROOK, color))
    free = [sq for sq in range(64) if sq not in (king_sq, rook_sq)]
    own_rest = letters.replace("K", "", 1).replace("R", "", 1)
    other = black if color else white
    if not (_put(board, rng, own_rest, color, free) and _put(board, rng, other, not color, free)):
        return None
    board.turn = rng.random() < 0.5
    board.castling_rights = chess.BB_SQUARES[rook_sq]
    board.halfmove_clock = _clock(rng)
    return board if board.is_valid() and board.castling_rights else None


def gen_edge_king(rng: random.Random, cls: str) -> chess.Board | None:
    """Like gen_random, but one king (random colour) on the board edge.

    Feeds the checkmate/stalemate search: mates in one are rare among
    uniformly placed positions and common with a king on the edge.
    """
    white, black = _orient(rng, cls)
    edge = [sq for sq in range(64) if chess.square_rank(sq) in (0, 7) or chess.square_file(sq) in (0, 7)]
    color = rng.random() < 0.5
    board = chess.Board(None)
    king_sq = rng.choice(edge)
    board.set_piece_at(king_sq, chess.Piece(chess.KING, color))
    free = [sq for sq in range(64) if sq != king_sq]
    own = (white if color else black).replace("K", "", 1)
    other = black if color else white
    if not (_put(board, rng, own, color, free) and _put(board, rng, other, not color, free)):
        return None
    board.turn = not color  # the edge king's opponent moves (and may mate)
    board.halfmove_clock = _clock(rng)
    return board if board.is_valid() else None


def gen_bogus_ep(rng: random.Random, cls: str) -> chess.Board | None:
    """A legal board given an en-passant square that may be IMPOSSIBLE.

    Half the time any square; otherwise a "victim" square: a side-to-move pawn
    beside an opponent piece on the capture rank (often the king), with the ep
    square behind that piece. Fathom's ep capture deletes whatever stands on
    ep-8/ep+8, so this is the case that used to crash it.
    """
    board = gen_random(rng, cls)
    if board is None:
        return None
    stm = board.turn
    if rng.random() < 0.5:
        board.ep_square = rng.randrange(0, 64) or None  # python-chess: 0 is no ep
        return board
    capture_rank = 4 if stm else 3
    victims = [sq for sq in chess.SquareSet(board.occupied_co[not stm])
               if chess.square_rank(sq) == capture_rank]
    if not victims:
        board.ep_square = rng.randrange(0, 64) or None  # python-chess: 0 is no ep
        return board
    victim = rng.choice(victims)
    side = [f for f in (chess.square_file(victim) - 1, chess.square_file(victim) + 1) if 0 <= f < 8]
    capturer = chess.square(rng.choice(side), capture_rank)
    if board.piece_at(capturer) is None and board.pawns & board.occupied_co[stm] and rng.random() < 0.8:
        # Move one of the side-to-move's pawns next to the victim.
        mover = rng.choice(list(chess.SquareSet(board.pawns & board.occupied_co[stm])))
        board.remove_piece_at(mover)
        board.set_piece_at(capturer, chess.Piece(chess.PAWN, stm))
    board.ep_square = victim + (8 if stm else -8)
    plain = board.copy(stack=False)
    plain.ep_square = None
    return board if plain.is_valid() else None  # only the ep square may be wrong


def ep_must_be_refused(board: chess.Board) -> bool:
    """The C prober refuses (ValueError) exactly these ep positions.

    A side-to-move pawn can capture onto the ep square, but the square behind
    it (ep-8 / ep+8, which the capture removes) is not an enemy pawn.
    python-chess would apply that nonsense capture (removing a knight, or the
    king); the C prober treats the board as illegal instead, as python-chess's
    own ``status()`` does (STATUS_INVALID_EP_SQUARE). Every other ep square --
    including a real enemy pawn with something standing on ep+8/ep-8, which
    python-chess's status() also calls invalid -- gives python-chess's values.
    """
    if not board.ep_square or not any(board.generate_pseudo_legal_ep()):
        return False
    behind = board.ep_square - 8 if board.turn else board.ep_square + 8
    return not board.pawns & board.occupied_co[not board.turn] & chess.BB_SQUARES[behind]


def expected_answers(tb: chess.syzygy.Tablebase, board: chess.Board):
    """(answer for the FEN path, answer for the CBoard path), each (wdl, dtz).

    Both are python-chess's answer, except: an ep square ``ep_must_be_refused``
    flags is REFUSED on both paths; and CBoard keeps corner castling bits only,
    so a right on a non-corner rook (python-chess KeyError) is invisible to it
    -- its expected answer is the board with the corner rights alone.
    """
    if ep_must_be_refused(board):
        return (REFUSED, REFUSED), (REFUSED, REFUSED)
    want = py_probe(tb, board)
    cb_want = want
    if board.castling_rights & ~chess.BB_CORNERS:
        corner_only = board.copy(stack=False)
        corner_only.castling_rights = board.castling_rights & chess.BB_CORNERS
        cb_want = py_probe(tb, corner_only)
    return want, cb_want


def _with_field(fen: str, index: int, value: str) -> str:
    parts = fen.split(" ")
    parts[index] = value
    return " ".join(parts)


def gen_ep_a1(rng: random.Random, cls: str) -> str | None:
    """A FEN whose ep field is ``a1`` (square 0): "no ep" for python-chess."""
    board = gen_random(rng, cls)
    return None if board is None else _with_field(board.fen(), 3, "a1")


def gen_ep_origin_occupied(rng: random.Random, cls: str) -> chess.Board | None:
    """A real double push, then some piece placed on the square it came from.

    python-chess's status() calls this ep square invalid, but the capture only
    removes the enemy pawn, so both probers must give the same value.
    """
    board = gen_ep(rng, cls)
    if board is None or board.ep_square is None:
        return None
    origin = board.ep_square + (8 if board.turn else -8)
    movers = [sq for sq in chess.SquareSet(board.occupied)
              if board.piece_type_at(sq) not in (chess.KING, chess.PAWN)]
    if not movers or board.piece_at(origin) is not None:
        return None
    sq = rng.choice(movers)
    piece = board.remove_piece_at(sq)
    assert piece is not None
    board.set_piece_at(origin, piece)
    plain = board.copy(stack=False)
    plain.ep_square = None
    return board if plain.is_valid() else None


_CASTLING_LETTERS = ["K", "Q", "k", "q", "KQ", "kq", "Kk", "Qq", "KQkq", "Kq", "Qk"]


def gen_castling_letters(rng: random.Random, cls: str) -> str | None:
    """A FEN with castling letters the placement may not back (python-chess's
    parser keeps, drops or re-targets each letter by its own rules)."""
    board = gen_random(rng, cls)
    if board is None:
        return None
    if rng.random() < 0.5:  # sometimes put a king on its home square
        color = rng.random() < 0.5
        king = board.king(color)
        home = chess.E1 if color else chess.E8
        if king is not None and board.piece_at(home) is None:
            board.remove_piece_at(king)
            board.set_piece_at(home, chess.Piece(chess.KING, color))
            if not board.is_valid():
                return None
    return _with_field(board.fen(), 2, rng.choice(_CASTLING_LETTERS))


def terminal_children(board: chess.Board) -> list[chess.Board]:
    out = []
    for move in list(board.legal_moves):
        board.push(move)
        try:
            if board.is_checkmate() or board.is_stalemate():
                out.append(board.copy(stack=False))
        finally:
            board.pop()
    return out


# ---------------------------------------------------------------- probing

REFUSED = "ValueError"


def _c_call(fn, arg) -> int | str | None:
    try:
        return fn(arg)
    except ValueError:
        return REFUSED


def py_probe(tb: chess.syzygy.Tablebase, board: chess.Board) -> tuple[int | None, int | None]:
    try:
        wdl: int | None = int(tb.probe_wdl(board))
    except _PROBE_ERRORS:
        wdl = None
    try:
        dtz: int | None = int(tb.probe_dtz(board))
    except _PROBE_ERRORS:
        dtz = None
    return wdl, dtz


def fathom_rank(dtz: int, clock: int) -> int:
    """Fathom root_probe_dtz's rank for a move, hasRepeated=False (tbprobe.c)."""
    if dtz > 0:
        return 1000 if dtz + clock <= 99 else 1000 - (dtz + clock)
    if dtz < 0:
        return -1000 if -dtz * 2 + clock < 100 else -1000 + (-dtz + clock)
    return 0


def _dtz_ignoring_fifty(child: chess.Board, tb: chess.syzygy.Tablebase) -> int:
    """``_dtz_of_played_move`` without its fifty-move short-circuit."""
    if child.is_checkmate():
        return 1
    if child.is_stalemate() or child.is_insufficient_material() or child.is_repetition(3):
        return 0
    if child.halfmove_clock == 0:
        return {2: 1, 1: 101, 0: 0, -1: -101, -2: -1}[-int(tb.probe_wdl(child))]
    child_dtz = int(tb.probe_dtz(child))
    if child_dtz > 0:
        return -child_dtz - 1
    if child_dtz < 0:
        return -child_dtz + 1
    return 0


def py_root(board: chess.Board, tb: chess.syzygy.Tablebase) -> list[tuple[str, int, int, bool]] | None:
    from chess_anti_engine.tablebase import _dtz_of_played_move

    rows = []
    for move in list(board.legal_moves):
        board.push(move)
        try:
            dtz = _dtz_of_played_move(board, tb)
            raw = _dtz_ignoring_fifty(board, tb)
            fifty = board.is_fifty_moves() and not board.is_checkmate()
        except _PROBE_ERRORS:
            return None
        finally:
            board.pop()
        rows.append((move.uci(), dtz, raw, fifty))
    return rows


def compare_root(board: chess.Board, py_rows, c_rows) -> dict[str, Any]:
    from chess_anti_engine.tablebase import _rule50_dtz_key

    out: dict[str, Any] = {"ok": True, "fifty_moves": 0, "n_moves": 0, "why": ""}
    if py_rows is None or c_rows is None:
        out["ok"] = (py_rows is None) == (c_rows is None)
        out["why"] = "" if out["ok"] else f"availability py={py_rows is None} c={c_rows is None}"
        return out
    py = {u: (d, raw, fifty) for u, d, raw, fifty in py_rows}
    c = {u: (d, r) for u, d, r, _ in c_rows}
    out["n_moves"] = len(c)
    if set(py) != set(c):
        out.update(ok=False, why="move sets differ")
        return out
    clock = board.halfmove_clock
    for u, (d, raw, fifty) in py.items():
        cd, cr = c[u]
        expect = raw if fifty else d
        out["fifty_moves"] += int(fifty and raw != d)
        if cd != expect or cr != fathom_rank(expect, clock):
            out.update(ok=False, why=f"{u}: py dtz={d} raw={raw} fifty={fifty} c dtz={cd} rank={cr}")
            return out
    if py and out["fifty_moves"] == 0:
        # Outside the fifty-claim divergence the repo's root pick must be one
        # of Fathom's top-ranked moves (Fathom ties wins inside a bucket; the
        # repo breaks those ties by distance).
        best = max(py, key=lambda u: _rule50_dtz_key(py[u][0], clock))
        top = max(r for _, r in c.values())
        out["py_best_in_fathom_top"] = c[best][1] == top
        if c[best][1] != top:
            out.update(ok=False, why=f"repo root pick {best} not in Fathom's top rank")
    return out


def _flags(board: chess.Board) -> dict[str, Any]:
    def safe(fn, default):
        try:
            return fn()
        except Exception:  # an impossible ep square can confuse python-chess
            return default

    legal = safe(lambda: list(board.legal_moves), [])
    return {
        "stm": "w" if board.turn else "b",
        "clock": board.halfmove_clock,
        "ep_legal": safe(board.has_legal_en_passant, False),
        "has_capture": any(board.is_capture(m) for m in legal),
        "has_pawn_move": any(board.piece_type_at(m.from_square) == chess.PAWN for m in legal),
        "mate": safe(board.is_checkmate, False),
        "stalemate": safe(board.is_stalemate, False),
        "men": chess.popcount(board.occupied),
    }


def run_class(job: tuple[str, str, int, int, int, int]) -> dict[str, Any]:
    cls, syzygy_path, seed, per_class, root_per_class, terminal_cap = job
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.tb import _tb_ext

    # path() reports symlink-resolved absolute components; compare like with like
    # or every class job would re-init the tables.
    resolved = os.pathsep.join(os.path.realpath(c) for c in syzygy_path.split(os.pathsep))
    if _tb_ext.path() != resolved:
        _tb_ext.init(syzygy_path)
    tb = _PY_TB.get(syzygy_path)
    if tb is None:
        tb = _PY_TB[syzygy_path] = open_python_tablebase(syzygy_path)

    rng = random.Random(f"{seed}:{cls}")
    boards: list[tuple[str, chess.Board, str]] = []
    seen: set[str] = set()

    def add(cat: str, b: chess.Board | str | None) -> bool:
        """``b`` is a board, or a FEN string (classes whose FEN text matters)."""
        if b is None:
            return False
        if isinstance(b, str):
            fen, board = b, chess.Board(b)
        else:
            fen, board = b.fen(en_passant="fen"), b
        if fen in seen:
            return False
        seen.add(fen)
        boards.append((cat, board, fen))
        return True

    randoms: list[chess.Board] = []
    attempts = 0
    while len(randoms) < per_class and attempts < per_class * 200:
        attempts += 1
        b = gen_random(rng, cls)
        if add("random", b):
            assert b is not None
            randoms.append(b)
    for gen, cat, want in ((gen_ep, "ep", per_class // 8), (gen_castling, "castling", 8),
                           (gen_bogus_ep, "bogus_ep", per_class // 8),
                           (gen_ep_a1, "ep_a1", per_class // 40),
                           (gen_ep_origin_occupied, "ep_origin_occupied", per_class // 16),
                           (gen_castling_letters, "castling_letters", per_class // 16)):
        got = tries = 0
        while got < want and tries < want * 200:
            tries += 1
            got += add(cat, gen(rng, cls))
    term = tries = 0
    while term < terminal_cap and tries < terminal_cap * 100:
        tries += 1
        parent = gen_edge_king(rng, cls)
        if parent is None:
            continue
        for child in terminal_children(parent):
            if term >= terminal_cap:
                break
            term += add("terminal", child)

    rows = []
    mismatches = []
    for cat, b, fen in boards:
        (py_wdl, py_dtz), cb_want = expected_answers(tb, b)
        noncorner = bool(b.castling_rights & ~chess.BB_CORNERS)
        cb = CBoard.from_board(b)
        c_wdl = _c_call(_tb_ext.probe_wdl_fen, fen)
        c_dtz = _c_call(_tb_ext.probe_dtz_fen, fen)
        cb_wdl = _c_call(_tb_ext.probe_wdl, cb)
        cb_dtz = _c_call(_tb_ext.probe_dtz, cb)
        row = {"key": fen, "cls": cls, "cat": cat, **_flags(b),
               "py_wdl": py_wdl, "py_dtz": py_dtz, "c_wdl": c_wdl, "c_dtz": c_dtz,
               "cb_wdl": cb_wdl, "cb_dtz": cb_dtz, "cb_noncorner_castling": noncorner}
        rows.append(row)
        if not ((py_wdl, py_dtz) == (c_wdl, c_dtz) and cb_want == (cb_wdl, cb_dtz)):
            mismatches.append(row)

    root_rows = []
    for b in randoms[:root_per_class]:
        if b.is_checkmate() or b.is_stalemate():
            continue
        py_rows = py_root(b, tb)
        c_rows = _tb_ext.probe_root(CBoard.from_board(b))
        cmp = compare_root(b, py_rows, c_rows)
        root_rows.append({"key": b.fen(en_passant="fen"), "cls": cls, "clock": b.halfmove_clock,
                          "py": py_rows, "c": c_rows, **cmp})
    return {"cls": cls, "rows": rows, "mismatches": mismatches, "root": root_rows}


_PY_TB: dict[str, chess.syzygy.Tablebase] = {}


def _worker_init() -> None:
    try:
        os.nice(10)
    except OSError:
        pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def throughput(rows: list[dict[str, Any]], syzygy_path: str) -> dict[str, Any]:
    """Single-thread probes/second on the probeable part of the sample."""
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.tb import _tb_ext

    _tb_ext.init(syzygy_path)
    tb = open_python_tablebase(syzygy_path)
    probeable = [r["key"] for r in rows if isinstance(r["py_wdl"], int) and isinstance(r["py_dtz"], int)]
    boards = [chess.Board(fen) for fen in probeable]
    cbs = [CBoard.from_board(b) for b in boards]

    def timed(fn, items) -> float:
        t0 = time.perf_counter()
        for item in items:
            fn(item)
        return time.perf_counter() - t0

    plan = [
        ("python-chess probe_wdl", tb.probe_wdl, boards),
        ("C probe_wdl(CBoard)", _tb_ext.probe_wdl, cbs),
        ("C probe_wdl_fen", _tb_ext.probe_wdl_fen, probeable),
        ("python-chess probe_dtz", tb.probe_dtz, boards),
        ("C probe_dtz(CBoard)", _tb_ext.probe_dtz, cbs),
        ("C probe_dtz_fen", _tb_ext.probe_dtz_fen, probeable),
    ]
    out: dict[str, Any] = {"n_positions": len(probeable), "passes": []}
    for pass_idx in range(2):  # pass 0 also warms the page cache for both
        res = {}
        for name, fn, items in plan:
            secs = timed(fn, items)
            res[name] = {"seconds": round(secs, 3), "probes_per_s": round(len(items) / secs, 1)}
        out["passes"].append(res)
        print(f"throughput pass {pass_idx}: {json.dumps(res)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--syzygy-path", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--per-class", type=int, default=360)
    ap.add_argument("--root-per-class", type=int, default=10)
    ap.add_argument("--terminal-cap", type=int, default=30)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--classes", default="", help="comma list to restrict (pilot runs)")
    ap.add_argument("--throughput", action="store_true")
    args = ap.parse_args()
    if args.workers > 2:
        ap.error("at most 2 workers beside a live run")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    args.out.mkdir(parents=True, exist_ok=True)
    classes = material_classes(args.syzygy_path)
    if args.classes:
        wanted = set(args.classes.split(","))
        classes = [c for c in classes if c in wanted]
    jobs = [(c, args.syzygy_path, args.seed, args.per_class, args.root_per_class, args.terminal_cap)
            for c in classes]
    t0 = time.time()
    positions_path = args.out / "positions.jsonl.gz"
    root_path = args.out / "root_moves.jsonl.gz"
    all_rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    root_bad: list[dict[str, Any]] = []
    tally: Counter[str] = Counter()
    root_tally: Counter[str] = Counter()
    by_class: dict[str, Counter[str]] = {}
    with gzip.open(positions_path, "wt") as pf, gzip.open(root_path, "wt") as rf, \
            mp.get_context("spawn").Pool(args.workers, initializer=_worker_init) as pool:
        for i, res in enumerate(pool.imap(run_class, jobs)):
            cnt = by_class.setdefault(res["cls"], Counter())
            for row in res["rows"]:
                pf.write(json.dumps(row) + "\n")
                all_rows.append(row)
                cats = [row["cat"], f"stm_{row['stm']}",
                        "clock0" if row["clock"] == 0 else ("clock99+" if row["clock"] >= 99 else "clock1-98")]
                if row["ep_legal"]:
                    cats.append("ep_legal")
                if row["has_capture"]:
                    cats.append("has_capture")
                if row["has_pawn_move"]:
                    cats.append("has_pawn_move")
                if row["mate"]:
                    cats.append("checkmate")
                if row["stalemate"]:
                    cats.append("stalemate")
                if row["py_wdl"] in (1, -1):
                    cats.append("cursed_win" if row["py_wdl"] == 1 else "blessed_loss")
                if row["cb_noncorner_castling"]:
                    cats.append("cboard_noncorner_castling")
                if row["py_wdl"] == REFUSED:
                    cats.append("refused_impossible_ep")
                if row["py_wdl"] is None:
                    cats.append("not_probeable")
                for cat in cats:
                    tally[cat] += 1
                    cnt[cat] += 1
                tally["total"] += 1
                cnt["total"] += 1
            for row in res["mismatches"]:
                mismatches.append(row)
                cnt["mismatch"] += 1
            for row in res["root"]:
                rf.write(json.dumps(row) + "\n")
                root_tally["positions"] += 1
                root_tally["moves"] += row["n_moves"]
                root_tally["fifty_claim_moves"] += row["fifty_moves"]
                root_tally["clock99+"] += int(row["clock"] >= 99)
                if row.get("py_best_in_fathom_top") is False:
                    root_tally["py_best_not_in_fathom_top"] += 1
                if not row["ok"]:
                    root_tally["mismatch"] += 1
                    root_bad.append(row)
            if i % 25 == 0:
                print(f"[{time.time() - t0:7.0f}s] {i + 1}/{len(jobs)} {res['cls']} total={tally['total']} "
                      f"mismatch={len(mismatches)} root_mismatch={len(root_bad)}", flush=True)
    (args.out / "mismatches.json").write_text(json.dumps(mismatches[:5000], indent=1))
    (args.out / "root_mismatches.json").write_text(json.dumps(root_bad[:5000], indent=1))
    tp = throughput(all_rows, args.syzygy_path) if args.throughput else None
    from chess_anti_engine.tb import _tb_ext

    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                 cwd=Path(__file__).resolve().parents[1], check=False).stdout.strip()
    except OSError:
        git_sha = ""
    manifest = {
        "what": "C Syzygy prober (_tb_ext / Fathom) vs python-chess equivalence",
        "command": sys.argv,
        "git_sha": git_sha,
        "fathom_commit": _tb_ext.FATHOM_COMMIT,
        "python_chess": chess.__version__,
        "python": platform.python_version(),
        "syzygy_path": args.syzygy_path,
        "seed": args.seed,
        "per_class": args.per_class,
        "root_per_class": args.root_per_class,
        "n_classes": len(classes),
        "elapsed_s": round(time.time() - t0, 1),
        "totals": dict(tally),
        "root_totals": dict(root_tally),
        "position_mismatches": len(mismatches),
        "root_mismatches": len(root_bad),
        "by_class": {k: dict(v) for k, v in by_class.items()},
        "throughput": tp,
        "files": {p.name: _sha256(p) for p in (positions_path, root_path)},
        "row_schema": "positions: key=FEN(raw ep), py_*=python-chess, c_*=C via FEN, cb_*=C via "
                      "CBoard; None = not probeable. root_moves: py=[uci, dtz, dtz_ignoring_fifty, "
                      "fifty_claim], c=[uci, dtz, rank, score].",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("totals", "root_totals", "position_mismatches",
                                                  "root_mismatches", "elapsed_s")}, indent=1))
    return 1 if mismatches or root_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
