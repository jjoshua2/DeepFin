#!/usr/bin/env python3
"""Build/run the opt-in Bend legal-move/perft probe. No Torch is required."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from native.bend_engine.bitboard_probe.run_probe import check_compiler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODES = {"generic": [], "portable": ["-DBEND_U64_PORTABLE"],
         "native": ["-march=native"],
         "ubsan": ["-fsanitize=undefined", "-fno-sanitize-recover=all"]}
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Published orthodox perft fixtures, also used in tests/test_perft.py.
# Default new CI work is shallow; depth 5 is an explicit local command only.
CANONICAL = [
    ("startpos", START, [(0, 1), (1, 20), (2, 400), (3, 8902), (4, 197281)]),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [(3, 97862)]),
    ("rook_endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [(3, 2812)]),
    ("promotions", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [(3, 9467)]),
    ("promotion_race", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [(3, 62379)]),
    ("middlegame", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [(3, 89890)]),
]
EDGES = [
    ("castle_white", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
    ("castle_black", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
    ("promote_white", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
    ("promote_black", "4k3/8/8/8/8/8/p7/4K3 b - - 0 1"),
    ("promotion_capture", "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
    ("ep_white", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
    ("ep_file_pin", "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
    ("ep_rank_pin", "8/8/8/r4pPK/8/8/8/4k3 w - f6 0 1"),
    ("ep_check_evasion", "4k3/8/8/3pP3/4K3/8/8/8 w - d6 0 1"),
    ("ep_black", "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1"),
    ("ep_black_pin", "4k3/8/8/8/3Pp3/8/8/K3R3 b - d3 0 1"),
    ("castle_transit_attacked", "k4r2/8/8/8/8/8/8/4K2R w K - 0 1"),
    ("castle_destination_attacked", "k5r1/8/8/8/8/8/8/4K2R w K - 0 1"),
    ("queenside_b_attacked", "1r2k3/8/8/8/8/8/8/R3K3 w Q - 0 1"),
    ("double_check", "4r1k1/8/8/8/1b6/8/R7/4K3 w - - 0 1"),
    ("checkmate", "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"),
    ("stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
    ("draw_counter_ignored", START.replace("0 1", "150 100")),
    ("file_pinned_knight", "4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1"),
    ("pinned_pawn_capture", "4r1k1/8/8/8/8/3b4/4P3/4K3 w - - 0 1"),
    ("diagonal_pin", "6k1/8/7q/8/8/8/3B4/2K5 w - - 0 1"),
    ("second_blocker", "4r1k1/8/8/8/8/4N3/4P3/4K3 w - - 0 1"),
    ("blocker_without_pinner", "6k1/8/8/8/8/8/4N3/4K3 w - - 0 1"),
    ("black_file_pin", "4k3/4n3/8/8/8/8/8/4R1K1 b - - 0 1"),
]
Position = tuple[int, ...]  # 8 full-width bitboards, turn, rights, ep
Move = tuple[int, int, int, int]  # src, dst, promotion piece id, special flag


def fen_position(fen: str) -> Position:
    """Parse orthodox FEN at the harness boundary, not in the Bend rules core.

    Counters are syntax-checked but intentionally not used by standard perft.
    The native loader separately enforces structural board invariants.
    """
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError("FEN must have six fields")
    layout, side, castles, ep, half, full = fields
    if side not in ("w", "b") or not half.isascii() or not half.isdecimal() or not full.isascii() or not full.isdecimal() or int(full) < 1:
        raise ValueError("invalid FEN turn/counters")
    rows = layout.split("/")
    if len(rows) != 8:
        raise ValueError("FEN must have eight ranks")
    boards = [0] * 8
    for row_index, row in enumerate(rows):
        file = 0
        for symbol in row:
            if symbol in "12345678":
                file += int(symbol)
            elif symbol in "PNBRQKpnbrqk" and file < 8:
                bit = 1 << ((7 - row_index) * 8 + file)
                boards["pnbrqk".index(symbol.lower())] |= bit
                boards[6 if symbol.isupper() else 7] |= bit
                file += 1
            else:
                raise ValueError("invalid FEN piece/rank")
            if file > 8:
                raise ValueError("overfull FEN rank")
        if file != 8:
            raise ValueError("incomplete FEN rank")
    if castles != "-" and (not castles or any(c not in "KQkq" for c in castles) or len(set(castles)) != len(castles)):
        raise ValueError("invalid orthodox FEN castling rights")
    rights = sum(1 << "KQkq".index(c) for c in castles if c != "-")
    if ep != "-" and re.fullmatch("[a-h][36]", ep) is None:
        raise ValueError("invalid FEN en-passant square")
    ep_square = 64 if ep == "-" else (int(ep[1]) - 1) * 8 + ord(ep[0]) - ord("a")
    return (*boards, int(side == "w"), rights, ep_square)


def request(position: Position, depth: int = 0, mode: int = 0, seed: int = 1) -> str:
    if len(position) != 11 or not 0 <= seed < 1 << 32:
        raise ValueError("invalid request shape/seed")
    return " ".join(f"{n:x}" for n in (depth, mode, seed, *position)) + "\n"


def _numbers(tokens: list[str]) -> list[int]:
    if any(re.fullmatch(r"[0-9]+", token) is None for token in tokens):
        raise ValueError("nondecimal output word")
    values = list(map(int, tokens))
    if any(n >= 1 << 32 for n in values):
        raise ValueError("output word exceeds U32")
    return values


def _move(values: list[int]) -> Move:
    if len(values) != 4 or values[0] >= 64 or values[1] >= 64 or values[0] == values[1] or values[2] > 4 or values[3] > 2:
        raise ValueError("invalid move output")
    return values[0], values[1], values[2], values[3]


def _position(values: list[int]) -> Position:
    if len(values) != 19 or values[16] > 1 or values[17] > 15 or values[18] > 64:
        raise ValueError("invalid position output")
    return (*(values[i] << 32 | values[i+1] for i in range(0, 16, 2)), *values[16:])


def parse_moves(text: str, *, trace: bool = False) -> tuple[dict[Move, Position] | list[tuple[Move, Position]], str]:
    rows: list[tuple[Move, Position]] = []
    lines = text.splitlines()
    endings = ("checkmate", "stalemate", "ply_limit") if trace else ("end",)
    if not lines or lines[-1] not in endings:
        raise ValueError("missing legal/trace terminator")
    for line in lines[:-1]:
        tokens = line.split()
        if len(tokens) != 24 or tokens[0] != ("ply" if trace else "move"):
            raise ValueError(f"malformed move row: {line}")
        values = _numbers(tokens[1:])
        rows.append((_move(values[:4]), _position(values[4:])))
    if trace:
        return rows, lines[-1]
    if len({m[:3] for m, _ in rows}) != len(rows):
        raise ValueError("duplicate legal move")
    return dict(rows), lines[-1]


def parse_perft(text: str, depth: int) -> tuple[int, dict[Move, int]]:
    lines = text.splitlines()
    rows: dict[Move, int] = {}
    if not lines or len(lines[-1].split()) != 3 or not lines[-1].startswith("nodes "):
        raise ValueError("missing perft total")
    words = _numbers(lines[-1].split()[1:])
    total = words[0] << 32 | words[1]
    seen: set[tuple[int, int, int]] = set()
    for line in lines[:-1]:
        tokens = line.split()
        if len(tokens) != 7 or tokens[0] != "divide":
            raise ValueError("malformed divide row")
        values = _numbers(tokens[1:])
        move = _move(values[:4])
        if move[:3] in seen:
            raise ValueError("duplicate divide move")
        seen.add(move[:3])
        rows[move] = values[4] << 32 | values[5]
    if depth == 0:
        if total != 1 or rows:
            raise ValueError("perft(0) must be one without divide rows")
    elif sum(rows.values()) != total:
        raise ValueError("divide sum does not equal total")
    return total, rows


def command(args: list[str], *, input_text: str | None = None, timeout: int = 120) -> str:
    result = subprocess.run(args, cwd=ROOT, input=input_text, capture_output=True,
                            text=True, timeout=timeout, check=False,
                            env={**os.environ, "BEND_NO_TELEMETRY": "1"})
    if result.returncode:
        raise RuntimeError(f"{' '.join(args)}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def build(source: Path, directory: Path, bun: str, cc: str, modes: list[str]) -> dict[str, Path]:
    check_compiler(source)
    directory.mkdir(parents=True, exist_ok=True)
    generated = directory / "legal.generated.c"
    command([bun, str(source / "bend2/main.ts"), str(HERE / "main.bend"), "-o", str(generated)])
    binaries: dict[str, Path] = {}
    for mode in modes:
        obj = directory / f"support-{mode}.o"
        flags = MODES["ubsan"] if mode == "ubsan" else []
        command([cc, "-std=c11", "-O3", *flags, "-I", str(ROOT), "-c", str(HERE / "support.c"), "-o", str(obj)])
        binary = directory / f"bend-{mode}"
        command([cc, "-std=c11", "-O3", *MODES[mode], "-I", str(HERE), str(generated), str(obj), "-pthread", "-lm", "-o", str(binary)])
        binaries[mode] = binary
    oracle = directory / "cboard-reference"
    command([cc, "-std=c11", "-O3", "-DLEGAL_ORACLE", "-I", str(ROOT), str(HERE / "support.c"), "-pthread", "-lm", "-o", str(oracle)])
    binaries["reference"] = oracle
    return binaries


def run_binary(binary: Path, position: Position, depth: int = 0, mode: int = 0, seed: int = 1) -> str:
    # The C reference has no Bend runtime switches.
    args = [str(binary)] if binary.name == "cboard-reference" else [str(binary), "--threads", "1"]
    return command(args, input_text=request(position, depth, mode, seed))


def python_chess_check(expected: dict[Position, dict[Move, Position]], fixtures: list[tuple[str, str]]) -> int:
    """Optional second rules oracle. python-chess is a test-time dependency only."""
    import chess

    for label, fen in fixtures:
        board = chess.Board(fen)
        rows: dict[Move, Position] = {}
        for move in board.legal_moves:
            flag = 2 if board.is_castling(move) else int(board.is_en_passant(move))
            key = (move.from_square, move.to_square, move.promotion - 1 if move.promotion else 0, flag)
            child = board.copy(stack=False)
            child.push(move)
            rows[key] = fen_position(child.fen(en_passant="fen"))
        if rows != expected[fen_position(fen)]:
            raise ValueError(f"python-chess/CBoard oracle disagreement: {label}")
    return len(fixtures)


def feature_check(rows: dict[Position, dict[Move, Position]]) -> None:
    """Specific rule expectations, independent of the CBoard implementation."""
    probes = [
        ("castle_white", (4, 6, 0, 2), True), ("castle_white", (4, 2, 0, 2), True),
        ("castle_black", (60, 62, 0, 2), True), ("castle_black", (60, 58, 0, 2), True),
        ("castle_transit_attacked", (4, 6, 0, 2), False),
        ("castle_destination_attacked", (4, 6, 0, 2), False),
        ("queenside_b_attacked", (4, 2, 0, 2), True),
        ("ep_white", (36, 43, 0, 1), True), ("ep_file_pin", (36, 43, 0, 1), False),
        ("ep_rank_pin", (38, 45, 0, 1), False), ("ep_check_evasion", (36, 43, 0, 1), True),
        ("ep_black", (28, 19, 0, 1), True), ("ep_black_pin", (28, 19, 0, 1), False),
    ]
    probes.extend([
        ("file_pinned_knight", (12, 29, 0, 0), False),
        ("pinned_pawn_capture", (12, 19, 0, 0), False),
        ("pinned_pawn_capture", (12, 20, 0, 0), True),
        ("diagonal_pin", (11, 18, 0, 0), False),
        ("diagonal_pin", (11, 20, 0, 0), True),
        ("diagonal_pin", (11, 47, 0, 0), True),
        ("second_blocker", (20, 37, 0, 0), True),
        ("blocker_without_pinner", (12, 29, 0, 0), True),
        ("black_file_pin", (52, 35, 0, 0), False),
    ])
    positions = {label: fen_position(fen) for label, fen in EDGES}
    for label, move, present in probes:
        if (move in rows[positions[label]]) != present:
            raise ValueError(f"reference violates rule fixture: {label}/{move}")
    for promotion in range(1, 5):
        if (48, 57, promotion, 0) not in rows[positions["promotion_capture"]]:
            raise ValueError("missing promotion capture")
    if any(move[0] != 4 for move in rows[positions["double_check"]]):
        raise ValueError("non-king move from double check")


def verify(binaries: dict[str, Path], *, with_python_chess: bool = False) -> dict[str, object]:
    reference = binaries["reference"]
    fixtures = [(label, fen) for label, fen, _ in CANONICAL] + EDGES
    expected_moves: dict[Position, dict[Move, Position]] = {}
    expected_counts: dict[tuple[Position, int], tuple[int, dict[Move, int]]] = {}
    for _, fen in fixtures:
        board = fen_position(fen)
        rows, _ = parse_moves(run_binary(reference, board, mode=1))
        assert isinstance(rows, dict)
        expected_moves[board] = rows
    feature_check(expected_moves)
    third_oracle = python_chess_check(expected_moves, fixtures) if with_python_chess else 0
    for label, fen, depths in CANONICAL:
        board = fen_position(fen)
        for depth, expected in depths:
            count = parse_perft(run_binary(reference, board, depth), depth)
            if count[0] != expected:
                raise ValueError(f"CBoard disagrees with published {label}/{depth}: {count[0]} != {expected}")
            expected_counts[board, depth] = count
    for _, fen in EDGES:
        board = fen_position(fen)
        expected_counts[board, 2] = parse_perft(run_binary(reference, board, 2), 2)
    reports = []
    for mode, binary in binaries.items():
        if mode == "reference":
            continue
        start = time.perf_counter()
        for label, fen in fixtures:
            board = fen_position(fen)
            actual, _ = parse_moves(run_binary(binary, board, mode=1))
            if actual != expected_moves[board]:
                raise ValueError(f"{mode}/{label}: legal moves or child state mismatch")
        for (board, depth), expected in expected_counts.items():
            actual = parse_perft(run_binary(binary, board, depth), depth)
            if actual != expected:
                raise ValueError(f"{mode}/depth {depth}: perft/divide mismatch: {actual} != {expected}")
        # Every generated move and resulting position is checked against CBoard.
        trace_positions = [fen_position(START), fen_position(EDGES[0][1]),
                           fen_position(EDGES[5][1]), fen_position(EDGES[4][1])]
        plies = 0
        sampled = 0
        for seed, initial in zip((0, 1, 7, 4294967295), trace_positions, strict=True):
            output = run_binary(binary, initial, 32, 2, seed)
            if run_binary(binary, initial, 32, 2, seed) != output:
                raise ValueError("seeded trace was nondeterministic")
            trace, ending = parse_moves(output, trace=True)
            assert isinstance(trace, list)
            if len(trace) > 32 or (ending == "ply_limit" and len(trace) != 32):
                raise ValueError("incorrect trace length")
            board = initial
            for ply, (move, child) in enumerate(trace):
                rows, _ = parse_moves(run_binary(reference, board, mode=1))
                assert isinstance(rows, dict)
                if rows.get(move) != child:
                    raise ValueError(f"{mode}/seed {seed}: illegal trace move or wrong child: {move}")
                if ply % 8 == 0:
                    actual, _ = parse_moves(run_binary(binary, board, mode=1))
                    if actual != rows:
                        raise ValueError("incomplete move set in sampled random position")
                    if parse_perft(run_binary(binary, board, 2), 2) != parse_perft(run_binary(reference, board, 2), 2):
                        raise ValueError("sampled random perft/divide mismatch")
                    sampled += 1
                board = child
                plies += 1
            if ending != "ply_limit" and run_binary(reference, board, mode=2).strip() != ending:
                raise ValueError("incorrect random-play terminal classification")
        for label, fen in EDGES:
            if label in ("checkmate", "stalemate"):
                trace, ending = parse_moves(run_binary(binary, fen_position(fen), 1, 2), trace=True)
                if trace or ending != label:
                    raise ValueError("incorrect initial terminal classification")
        invalid = ["malformed\n", request((0,) * 8 + (1, 0, 64)),
                   request(fen_position(START), 6), request(fen_position(START), 0, 3),
                   request(fen_position("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")),
                   request(fen_position("4k3/8/8/8/8/8/8/4K3 w K - 0 1")),
                   request(fen_position("4k3/8/8/8/8/8/8/4K3 w - d6 0 1"))]
        for text in invalid:
            result = subprocess.run([str(binary), "--threads", "1"], input=text,
                                    text=True, capture_output=True, timeout=10, check=False)
            if result.returncode != 2 or "invalid" not in result.stderr:
                raise ValueError("invalid input was not rejected")
        reports.append({"mode": mode, "position_fixtures": len(fixtures),
                        "perft_divide_cases": len(expected_counts), "checked_random_plies": plies,
                        "sampled_random_move_sets_and_perft": sampled,
                        "invalid_inputs_rejected": len(invalid), "seconds_including_validation": time.perf_counter() - start})
    return {"results": reports, "python_chess_fixture_checks": third_oracle, "scope": "CPU rules/perft correctness; not a performance benchmark"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-root", type=Path, default=ROOT / "build/bend_u64_toolchain/source")
    parser.add_argument("--bun", default=os.environ.get("BUN", "bun"))
    parser.add_argument("--cc", default=os.environ.get("CC", "clang"))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--fen", help="run one position instead of the validation suite")
    parser.add_argument("--depth", type=int, choices=range(6))
    parser.add_argument("--random-plies", type=int, choices=range(1, 257))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--python-chess", action="store_true", help="add an optional independent rules oracle to validation")
    args = parser.parse_args()
    try:
        if args.fen is None and (args.depth is not None or args.random_plies is not None or args.seed != 1):
            raise ValueError("--fen is required for depth/random/seed options")
        if args.fen is not None and (args.report is not None or args.python_chess):
            raise ValueError("--report and --python-chess are validation-suite options")
        if args.depth is not None and args.random_plies is not None:
            raise ValueError("choose either --depth or --random-plies")
        if args.random_plies is None and args.seed != 1:
            raise ValueError("--seed requires --random-plies")
        depth = 3 if args.depth is None else args.depth
        for executable in (args.bun, args.cc):
            if shutil.which(executable) is None:
                raise ValueError(f"executable not found: {executable}")
        source = args.compiler_root.resolve()
        pin = check_compiler(source)
        with tempfile.TemporaryDirectory(prefix="deepfin-bend-legal-") as tmp:
            binaries = build(source, Path(tmp), args.bun, args.cc, args.modes)
            if args.fen is not None:
                board = fen_position(args.fen)
                for mode in args.modes:
                    output = run_binary(binaries[mode], board, args.random_plies or depth,
                                        2 if args.random_plies else 0, args.seed)
                    if args.random_plies:
                        parse_moves(output, trace=True)
                    else:
                        result = parse_perft(output, depth)
                        expected = parse_perft(run_binary(binaries["reference"], board, depth), depth)
                        if result != expected:
                            raise ValueError("CBoard perft/divide mismatch")
                    print(f"[{mode}]\n{output}", end="")
            else:
                report = {"compiler_revision": pin["revision"], "source_verified": True,
                          "clang": command([args.cc, "--version"]).splitlines()[0], **verify(binaries, with_python_chess=args.python_chess)}
                text = json.dumps(report, indent=2) + "\n"
                if args.report:
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    args.report.write_text(text)
                print(text, end="")
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.TimeoutExpired) as exc:
        parser.exit(1, f"Bend legal validation failed: {exc}\n")


if __name__ == "__main__":
    main()
