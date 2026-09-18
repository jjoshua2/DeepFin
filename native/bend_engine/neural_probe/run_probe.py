#!/usr/bin/env python3
"""Opt-in history/policy + CPU native AOTI qualification, not a trained engine."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import chess
import numpy as np
import torch

from chess_anti_engine.encoding._lc0_ext import CBoard, set_history_rep_fix
from chess_anti_engine.moves.encode import FULL_TO_COMPACT_POLICY, move_to_index
from native.bend_engine.legal_probe import run_probe as rules
from native.bend_engine.session_probe import run_probe as sessions
from native.bend_engine.neural_probe.adapter import (
    AOTI,
    Features,
    decode_key,
    key_for,
    mapping_hash,
    position,
    sha256,
)
from native.bend_engine.neural_probe.model import export, make_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def histories() -> list[tuple[str, str, list[chess.Move]]]:
    fixtures = [(label, fen, []) for label, fen, _ in rules.CANONICAL]
    fixtures += [(label, fen, []) for label, fen in rules.EDGES]
    for label, text in (
        ("repeated", "g1f3 g8f6 f3g1 f6g8 " * 3),
        ("black_history", "e2e4 e7e5 g1f3"),
        ("ep_history", "e2e4 a7a6 e4e5 d7d5"),
        ("captures", "e2e4 d7d5 e4d5 g8f6"),
    ):
        fixtures.append(
            (label, rules.START, [chess.Move.from_uci(u) for u in text.split()])
        )
    return fixtures


def replay(seed: chess.Board, history: list[chess.Move]) -> chess.Board:
    board = seed.copy(stack=False)
    for move in history:
        key_for(board, move)  # validate before unchecked python-chess push
        board.push(move)
    return board


def compare_features(
    features: Features,
    board: chess.Board,
    path: list[int],
    mode: int,
    extra: int,
    rep_fix: bool,
) -> tuple[np.ndarray, np.ndarray]:
    actions = [key_for(board, move) for move in board.legal_moves]
    encoded, full, compact = features.prepare(position(board), actions, path)
    set_history_rep_fix(rep_fix)
    reference = np.asarray(
        CBoard.from_board(board).encode_full(mode, extra), dtype=np.float32
    )
    if not np.array_equal(encoded.view(np.uint32), reference.view(np.uint32)):
        raise AssertionError(
            f"native history/feature mismatch at mode={mode}, extra={extra}, rep={rep_fix}: {board.fen()}"
        )
    indices = np.asarray(
        [move_to_index(move, board) for move in board.legal_moves], dtype=np.int32
    )
    np.testing.assert_array_equal(full, indices)
    np.testing.assert_array_equal(compact, FULL_TO_COMPACT_POLICY[indices])
    return encoded, compact


def verify_features(library: Path) -> dict[str, int | bool]:
    cases, child_cases = 0, 0
    for _label, fen, history in histories():
        seed = chess.Board(fen)
        board = replay(seed, history)
        for mode in (0, 1, 2):
            for extra in (34, 63, 67):
                for rep_fix in (False, True):
                    with Features(
                        library, seed, history, mode=mode, extra=extra, rep_fix=rep_fix
                    ) as features:
                        compare_features(features, board, [], mode, extra, rep_fix)
                        cases += 1
                        # Exercise history extension below the search root, including
                        # both colors, promotions/EP/castling when present.
                        special = [
                            m
                            for m in board.legal_moves
                            if m.promotion
                            or board.is_castling(m)
                            or board.is_en_passant(m)
                        ]
                        moves = special or list(board.legal_moves)[:1]
                        for move in moves:
                            child = board.copy(stack=True)
                            key = key_for(board, move)
                            child.push(move)
                            compare_features(
                                features, child, [key], mode, extra, rep_fix
                            )
                            child_cases += 1
    start = chess.Board()
    with Features(library, start, [], mode=1, extra=34, rep_fix=True) as features:
        p = position(start)
        actions = [key_for(start, m) for m in start.legal_moves]
        failures = [
            (p, actions[:-1], [], "incomplete legal"),
            (p, [actions[0]] * len(actions), [], "duplicate"),
            (p, [actions[0] | 1 << 15, *actions[1:]], [], "illegal"),
            (p, actions, [0], "illegal search"),
            (p, actions, [actions[0]], "leaf board/path"),
            (p, actions, [actions[0]] * 33, "invalid native"),
        ]
        for requested, moves, path, message in failures:
            try:
                features.prepare(requested, moves, path)
            except ValueError as ex:
                if message not in str(ex):
                    raise AssertionError(f"wrong boundary failure: {ex}") from ex
            else:
                raise AssertionError("invalid request accepted")
        a, f, c = features.prepare(p, actions, [])
        b, rf, rc = features.prepare(p, actions[::-1], [])
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(f, rf[::-1])
        np.testing.assert_array_equal(c, rc[::-1])
    # Same current position/clock with different prior states must not alias.
    history = histories()[-4][2]
    full_history = replay(start, history)
    no_history = chess.Board(full_history.fen(en_passant="fen"))
    with (
        Features(library, start, history, mode=1, extra=34, rep_fix=True) as a,
        Features(library, no_history, [], mode=1, extra=34, rep_fix=True) as b,
    ):
        actions = [key_for(full_history, m) for m in full_history.legal_moves]
        first = a.prepare(position(full_history), actions, [])[0]
        second = b.prepare(position(no_history), actions, [])[0]
        if np.array_equal(first, second):
            raise AssertionError("history erased despite identical current board")
    return {
        "roots_checked": cases,
        "search_children_checked": child_cases,
        "negative_cases": len(failures),
        "action_order_preserved": True,
        "history_changes_input": True,
    }


class CheckedEvaluator:
    def __init__(self, features: Features, native: AOTI, model: torch.nn.Module):
        self.features, self.native, self.model = features, native, model
        self.requests = 0
        self.max_error = 0.0
        self.distinct: set[bytes] = set()

    def __call__(
        self, board: rules.Position, actions: list[int], path: list[int]
    ) -> tuple[list[float], list[float]]:
        reference = self.features.root.copy(stack=True)
        for key in path:
            reference.push(decode_key(reference, key))
        if position(reference) != board:
            raise AssertionError("Bend supplied a path for a different position")
        enc, full, compact = self.features.prepare(board, actions, path)
        cfg = self.native.contract
        set_history_rep_fix(cfg["history_rep_fix"])
        py_enc = np.asarray(
            CBoard.from_board(reference).encode_full(
                cfg["history_mode"], cfg["planes"] - 112
            )
        )
        np.testing.assert_array_equal(enc.view(np.uint32), py_enc.view(np.uint32))
        py_full = np.asarray(
            [move_to_index(decode_key(reference, key), reference) for key in actions]
        )
        np.testing.assert_array_equal(full, py_full)
        np.testing.assert_array_equal(compact, FULL_TO_COMPACT_POLICY[py_full])
        v, p, rawp, rawv = self.native.evaluate(enc, compact)
        x = torch.from_numpy(py_enc)[None].to(dtype=getattr(torch, cfg["dtype"]))
        with torch.inference_mode():
            pol, wdl = self.model(x)
            # Independent dense-4672 expansion, then gather actual legal action
            # ids; native gathers compact ids directly, in Bend request order.
            from chess_anti_engine.moves.encode import COMPACT_TO_FULL_POLICY

            dense = torch.full((1, 4672), -1e9)
            dense[:, torch.from_numpy(COMPACT_TO_FULL_POLICY.astype(np.int64))] = (
                pol.float()
            )
            expected_p = dense[0, torch.from_numpy(py_full)].softmax(0).numpy()
            expected_v = wdl.float()[0].softmax(0).numpy()
        tolerance = 5e-5 if cfg["dtype"] == "float32" else 0.025
        for actual, expected in (
            (rawp, pol.float()[0].numpy()),
            (rawv, wdl.float()[0].numpy()),
            (p, expected_p),
            (v, expected_v),
        ):
            np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
            self.max_error = max(
                self.max_error, float(np.max(np.abs(actual - expected)))
            )
        self.distinct.add(rawp.tobytes())
        self.requests += 1
        # Only the native evaluator's probabilities, never the eager reference,
        # are returned to Bend and the diagnostic tree accounting oracle.
        return v.tolist(), p.tolist()


def verify_neural(
    library: Path,
    feature_library: Path,
    binaries: dict[str, Path],
    package: Path,
    model: torch.nn.Module,
) -> dict:
    if os.environ.get("AOTI_RUNTIME_CHECK_INPUTS") != "1":
        raise ValueError(
            "AOTI runtime input checks must be enabled before loading a model"
        )
    oracle = sessions.Oracle(binaries["reference"], with_python_chess=True)
    rows = []
    # Disabled generated-kernel input checks must never silently qualify.
    os.environ.pop("AOTI_RUNTIME_CHECK_INPUTS")
    try:
        try:
            with AOTI(library, package):
                raise AssertionError("unchecked native loader accepted")
        except ValueError as ex:
            if "AOTI_RUNTIME_CHECK_INPUTS" not in str(ex):
                raise
    finally:
        os.environ["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
    with AOTI(library, package) as native:
        cfg = native.contract
        selected = [histories()[0], *histories()[-4:]]
        # Fixed special rules supply all four promotions and castling/EP to the
        # policy boundary without forcing artificial search trajectories.
        selected += [
            item
            for item in histories()
            if item[0] in ("castle_white", "promote_black", "promotion_capture")
        ]
        for label, fen, history in selected:
            seed = chess.Board(fen)
            with Features(
                feature_library,
                seed,
                history,
                mode=cfg["history_mode"],
                extra=cfg["planes"] - 112,
                rep_fix=cfg["history_rep_fix"],
            ) as features:
                evaluator = CheckedEvaluator(features, native, model)
                board = position(features.root)
                peer = sessions.Peer(binaries["native"], board)
                try:
                    first = sessions.session(
                        peer, oracle, board, epoch=1, budget=12, evaluator=evaluator
                    )
                    repeat = sessions.session(
                        peer, oracle, board, epoch=2, budget=12, evaluator=evaluator
                    )
                    if repeat != first:
                        raise AssertionError(
                            "native evaluator reset changed deterministic results"
                        )
                    bad = sessions.session(
                        peer,
                        oracle,
                        board,
                        epoch=3,
                        budget=12,
                        fault="backend",
                        evaluator=evaluator,
                    )
                    if bad["completed"] != 2:
                        raise AssertionError("backend failure partially updated tree")
                    recovery = sessions.session(
                        peer, oracle, board, epoch=4, budget=12, evaluator=evaluator
                    )
                    if recovery != first:
                        raise AssertionError("native evaluator did not recover")
                    peer.finish()
                finally:
                    peer.close()
                if len(evaluator.distinct) < 2:
                    raise AssertionError(
                        "neural evaluator output is constant across positions"
                    )
                rows.append(
                    {
                        "fixture": label,
                        **first,
                        "native_evaluations": evaluator.requests,
                        "max_absolute_eager_error": evaluator.max_error,
                        "reset_and_recovery": True,
                    }
                )
        # C++ input guards fail before writing any output; a good call still
        # succeeds afterward on the same loaded package.
        with Features(
            feature_library,
            chess.Board(),
            [],
            mode=1,
            extra=cfg["planes"] - 112,
            rep_fix=True,
        ) as features:
            actions = [key_for(features.root, m) for m in features.root.legal_moves]
            enc, _, compact = features.prepare(position(features.root), actions, [])
            for inputs, ids, expected in (
                (np.full_like(enc, np.nan), compact, "nonfinite"),
                (enc, np.asarray([1858], dtype=np.uint32), "index out"),
                (enc[:-1], compact, "input buffers"),
            ):
                try:
                    native.evaluate(inputs, ids)
                except ValueError as ex:
                    if expected not in str(ex):
                        raise AssertionError(str(ex)) from ex
                else:
                    raise AssertionError("invalid AOTI request accepted")
            native.evaluate(enc, compact)
    wrong_input_rejected = False
    if cfg["planes"] < 179:
        # Deliberately inconsistent sidecar, same package hash/tuple schema.
        # Allocate MORE input storage, never a smaller unsafe buffer for this test.
        with tempfile.TemporaryDirectory(prefix="aoti-bad-contract-") as tmp:
            wrong = Path(tmp) / "wrong.pt2"
            shutil.copyfile(package, wrong)
            changed = dict(cfg, planes=179)
            wrong.with_suffix(".json").write_text(json.dumps(changed))
            with AOTI(library, wrong) as mismatched:
                try:
                    mismatched.evaluate(
                        np.zeros((179, 8, 8), dtype=np.float32),
                        np.asarray([0], dtype=np.uint32),
                    )
                except ValueError:
                    wrong_input_rejected = True
                else:
                    raise AssertionError("package input contract mismatch was accepted")
    return {
        "sessions": rows,
        "native_evaluations": sum(row["native_evaluations"] for row in rows),
        "loaded_search_model_instances": 1,
        "unchecked_loader_rejected": True,
        "actual_input_mismatch_rejected": wrong_input_rejected,
        "negative_aoti_inputs": 3,
    }


def build(
    work: Path, source: Path, bun: str, cc: str, cxx: str
) -> tuple[Path, Path, dict[str, Path]]:
    work.mkdir(parents=True, exist_ok=True)
    feature = work / "features.so"
    rules.command(
        [
            cc,
            "-std=c11",
            "-O2",
            "-fPIC",
            "-shared",
            "-I",
            str(ROOT),
            str(HERE / "features.c"),
            "-lm",
            "-o",
            str(feature),
        ]
    )
    cmake = work / "cmake"
    rules.command(
        [
            "cmake",
            "-S",
            str(HERE),
            "-B",
            str(cmake),
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
        ]
    )
    rules.command(["cmake", "--build", str(cmake), "--parallel", "1"])
    library = cmake / "libdeepfin_bend_aoti.so"
    dynamic = rules.command(["readelf", "-d", str(library)])
    if "libpython" in dynamic:
        raise AssertionError("native evaluator depends on libpython")
    return feature, library, sessions.build(source, work / "bend", bun, cc, ["native"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler-root", type=Path, default=ROOT / "build/bend_u64_toolchain/source"
    )
    parser.add_argument("--bun", default="bun")
    parser.add_argument("--cc", default="clang")
    parser.add_argument("--cxx", default="clang++")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--package",
        type=Path,
        help="existing private CPU tuple package plus matching .json contract",
    )
    parser.add_argument("--planes", type=int, choices=(146, 175, 179), default=146)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    # Isolated process/cache; no live package, checkpoint, or worker is changed.
    os.environ.setdefault("CXX", "/usr/bin/g++")
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    os.environ["AOTI_RUNTIME_CHECK_INPUTS"] = "1"
    torch.set_num_threads(2)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="bend-neural-") as tmp:
        work = Path(tmp)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(work / "inductor")
        feature, native, binaries = build(
            work, args.compiler_root.resolve(), args.bun, args.cc, args.cxx
        )
        encoding = verify_features(feature)
        if args.package is None:
            package = work / "chess_b1.pt2"
            model = export(package, args.planes, args.dtype)
        else:
            # The verifier can reload our reproducible seeded smoke only. A
            # trained checkpoint needs a matching eager reference, not a guess.
            package = args.package.resolve()
            from native.bend_engine.neural_probe.adapter import contract

            cfg = contract(package)
            if not str(cfg.get("model", "")).startswith("seed-1909 untrained"):
                raise ValueError(
                    "verification requires an explicit matching eager model"
                )
            model = make_model(cfg["planes"], cfg["dtype"])
        neural = verify_neural(native, feature, binaries, package, model)
        manifest = json.loads(package.with_suffix(".json").read_text())
        report = {
            "ok": True,
            "feature_contract": encoding,
            "neural_contract": neural,
            "package": manifest,
            "native_library_no_libpython": True,
            "policy_map_sha256": mapping_hash(),
            "native_source_sha256": {
                p.name: sha256(p) for p in (HERE / "features.c", HERE / "aoti.cpp")
            },
            "clang": rules.command([args.cc, "--version"]).splitlines()[0],
            "seconds_including_build_and_checks": time.monotonic() - start,
            "scope": "CPU untrained ChessNet AOTI + native history/policy + Bend PUCT; Python test relay; no CUDA/strength/performance claim",
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
