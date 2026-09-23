#!/usr/bin/env python3
"""Opt-in BT4 root-policy corpus worker (experimental, no replay consumer).

Every accepted row is a >=7-piece pre-move root. The actor samples the raw
BT4 root policy; no search or tablebase-guided move selection is performed.
Six-man Syzygy is used only to finish games under rule50_match_v1. One game is
buffered until its outcome is known, then published as one atomic NPZ file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import chess
import chess.syzygy
import numpy as np

from chess_anti_engine import tablebase
from chess_anti_engine.encoding import _lc0_ext, rep_fix
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from scripts import bt4_generation_evaluator
from scripts.bt4_policy_dump import file_sha256, open_session
from scripts.bt4_root_policy_stepper import (
    BT4DiscardedGame,
    BT4FinalizedGame,
    BT4RootPolicyStepper,
)
from scripts.gen_sf_rooted_corpus import INPUT_EXTRA_FEATURES, INPUT_HISTORY_ENCODING


SCHEMA = "bt4_root_policy_games_v1"
OUTCOME_MODE = "rule50_match_v1"
MAX_BUFFERED_ROWS = 4096
_SOURCE_FILES = (
    "scripts/bt4_root_policy_worker.py",
    "scripts/bt4_root_policy_stepper.py",
    "scripts/bt4_generation_evaluator.py",
    "scripts/bt4_policy_dump.py",
    "scripts/bt4_raw_corpus_sidecar.py",
    "chess_anti_engine/selfplay/bt4_outcome.py",
    "chess_anti_engine/tablebase.py",
    "chess_anti_engine/moves/leela_index.py",
    "chess_anti_engine/encoding/cboard_encode.py",
    "chess_anti_engine/encoding/lc0.py",
)


class RootEvaluator(Protocol):
    def evaluate_roots(
        self, boards: list[chess.Board], x_batch: np.ndarray,
    ) -> list[bt4_generation_evaluator.BT4RootOutput]: ...


@dataclass(frozen=True)
class WorkerSpec:
    out: Path
    games: int
    seed: int
    max_plies: int
    parallel_games: int
    temperature: float
    initial_fen: str
    syzygy_path: str
    model_sha256: str
    outcome_mode: str
    model_path: str
    providers: tuple[str, ...]

    def validate(self) -> None:
        if self.outcome_mode != OUTCOME_MODE:
            raise ValueError("BT4 worker requires explicit rule50_match_v1 outcome mode")
        if self.games < 1 or self.seed < 0 or self.max_plies < 1:
            raise ValueError("games/max_plies must be positive and seed nonnegative")
        if self.parallel_games < 1 or self.parallel_games * self.max_plies > MAX_BUFFERED_ROWS:
            raise ValueError(f"parallel_games * max_plies must be <= {MAX_BUFFERED_ROWS}")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if len(self.model_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.model_sha256
        ):
            raise ValueError("model_sha256 must be a lowercase SHA-256")
        if not self.syzygy_path or not self.providers or not self.model_path:
            raise ValueError("model, provider and Syzygy provenance are required")
        board = chess.Board(self.initial_fen)
        if not board.is_valid() or chess.popcount(board.occupied) < 7:
            raise ValueError("initial FEN must be legal and have at least seven pieces")


def table_file_inventory(path: str) -> dict[str, Any]:
    """Pin names/stat identities. Capacity/probe correctness is checked by TB.

    A stat inventory is deliberately not represented as a file-content digest.
    """
    directories = [Path(part).resolve() for part in path.split(os.pathsep)]
    files: list[dict[str, Any]] = []
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"missing Syzygy directory: {directory}")
        for entry in sorted(directory.iterdir()):
            if entry.suffix not in (".rtbw", ".rtbz") or not entry.is_file():
                continue
            stat = entry.stat()
            files.append({
                "path": str(entry.resolve()), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })
    if not files:
        raise ValueError("Syzygy inventory has no WDL/DTZ files")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "identity_basis": "resolved_path_size_mtime_ns; content_not_hashed",
        "directories": [str(directory) for directory in directories],
        "files": files,
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    writing = path.with_name(path.name + ".writing")
    try:
        with writing.open("xb") as handle:
            handle.write((json.dumps(document, sort_keys=True, indent=2) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(writing, path)
    finally:
        writing.unlink(missing_ok=True)


def _game_payload(
    game: BT4FinalizedGame, *, initial_fen: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if game.outcome_provenance.mode != OUTCOME_MODE:
        raise ValueError("game outcome mode differs from worker contract")
    if isinstance(game, BT4DiscardedGame):
        if game.discarded_rows != game.attempted_plies:
            raise ValueError("discard must account for every buffered row")
        meta = {
            "schema": SCHEMA, "game_id": game.slot_id, "initial_fen": initial_fen,
            "status": "discarded", "result": None,
            "termination": game.termination, "detail": game.detail,
            "attempted_plies": game.attempted_plies,
            "discarded_rows": game.discarded_rows, "rows": [],
            "outcome_provenance": vars(game.outcome_provenance),
        }
        return meta, {
            "x": np.empty((0, 0, 8, 8), dtype=np.float32),
            "policy_t1": np.empty((0, COMPACT_POLICY_SIZE), dtype=np.float32),
            "wdl_raw": np.empty((0, 3), dtype=np.float32),
        }
    rows: list[dict[str, Any]] = []
    for index, labeled in enumerate(game.records):
        played = labeled.played
        teacher = played.teacher
        if chess.popcount(chess.Board(teacher.fen).occupied) < 7:
            raise ValueError("BT4 worker refuses a below-seven-piece root row")
        if teacher.model_sha256 != game.records[0].played.teacher.model_sha256:
            raise ValueError("BT4 game mixes models")
        rows.append({
            "index": index, "fen": teacher.fen,
            "input_key": teacher.input_key, "source_key": teacher.source_key.hex(),
            "ply_index": played.ply_index, "pov_white": played.pov_white,
            "move_uci": played.move.uci(), "temperature": played.temperature,
            "wdl_target": labeled.wdl_target,
            "teacher": {
                "kind": "raw_root_inference", "policy_encoding": "lc0_1858_compact",
                "policy_output": teacher.policy_output,
                "wdl_output": teacher.wdl_output, "wdl_kind": teacher.wdl_kind,
                "wdl_pov": "side_to_move", "wdl_order": ["win", "draw", "loss"],
                "wdl_dtype": teacher.wdl_raw.dtype.name,
                "input_name": teacher.input_name, "input_dtype": teacher.input_dtype,
                "input_history_encoding": teacher.input_history_encoding,
                "input_extra_features": teacher.input_extra_features,
                "history_rep_fix": teacher.history_rep_fix,
                "model_sha256": teacher.model_sha256,
            },
        })
    meta = {
        "schema": SCHEMA, "game_id": game.slot_id, "initial_fen": initial_fen,
        "status": "completed", "result": game.result,
        "termination": game.termination, "detail": game.detail,
        "attempted_plies": len(rows), "discarded_rows": 0,
        "rows": rows, "outcome_provenance": vars(game.outcome_provenance),
    }
    if rows:
        arrays = {
            "x": np.stack([row.played.x for row in game.records]),
            "policy_t1": np.stack([row.played.teacher.policy_t1 for row in game.records]),
            "wdl_raw": np.stack([row.played.teacher.wdl_raw for row in game.records]),
        }
    else:
        arrays = {
            "x": np.empty((0, 0, 8, 8), dtype=np.float32),
            "policy_t1": np.empty((0, COMPACT_POLICY_SIZE), dtype=np.float32),
            "wdl_raw": np.empty((0, 3), dtype=np.float32),
        }
    return meta, arrays


def write_finalized_game(
    game: BT4FinalizedGame, directory: Path, *, initial_fen: str,
) -> dict[str, Any]:
    """Publish one finalized game as one file, after validating all its rows."""
    meta, arrays = _game_payload(game, initial_fen=initial_fen)
    target = directory / f"game_{game.slot_id:08d}.npz"
    writing = target.with_name(target.name + ".writing")
    if target.exists():
        raise FileExistsError(target)
    try:
        with writing.open("xb") as handle:
            np.savez_compressed(
                handle, **arrays,
                metadata=np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(writing, target)
    finally:
        writing.unlink(missing_ok=True)
    return {
        "path": target.name, "sha256": file_sha256(target),
        "game_id": game.slot_id, "status": meta["status"],
        "rows": len(meta["rows"]), "discarded_rows": meta["discarded_rows"],
    }


def run_worker(
    spec: WorkerSpec, evaluator: RootEvaluator,
    match_tablebase: chess.syzygy.Tablebase,
    table_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a finite seeded corpus. A missing strict probe aborts without summary."""
    spec.validate()
    if rep_fix.current() is not True:
        raise RuntimeError("history_rep_fix must be configured before BT4 boards")
    # Recheck the passed handle before creating output; do not trust its type.
    tablebase.SyzygyProbe(
        spec.syzygy_path, max_pieces=6, rule50_aware=True, tablebase=match_tablebase,
    )
    root = Path(__file__).resolve().parents[1]
    manifest: dict[str, Any] = {
        "schema": SCHEMA, "status": "launched", "actor": "bt4_root_policy_no_search",
        "outcome_mode": spec.outcome_mode, "minimum_emitted_root_pieces": 7,
        "terminal_wdl_target": "game_result_from_root_side_to_move",
        "teacher_observation": "raw_named_onnx_heads_unmodified_by_outcome",
        "seed": spec.seed, "games": spec.games, "max_plies": spec.max_plies,
        "parallel_games": spec.parallel_games, "max_buffered_rows": MAX_BUFFERED_ROWS,
        "temperature": spec.temperature, "initial_fen": spec.initial_fen,
        "model": {"path": spec.model_path, "sha256": spec.model_sha256,
                  "providers": list(spec.providers)},
        "input_history_encoding": INPUT_HISTORY_ENCODING,
        "input_extra_features": INPUT_EXTRA_FEATURES, "history_rep_fix": True,
        "syzygy": {
            **table_inventory,
            "path": spec.syzygy_path,
            "max_pieces": 6,
            "wdl_table_count": len(match_tablebase.wdl),
            "dtz_table_count": len(match_tablebase.dtz),
        },
        "source_sha256": {name: file_sha256(root / name) for name in _SOURCE_FILES},
        "native_encoder_sha256": file_sha256(Path(_lc0_ext.__file__)),
    }
    spec.out.mkdir(parents=True, exist_ok=False)
    games_dir = spec.out / "games"
    games_dir.mkdir()
    _atomic_json(spec.out / "launch.json", manifest)
    receipts: list[dict[str, Any]] = []
    discarded: Counter[str] = Counter()
    emitted = attempted = 0
    for first in range(0, spec.games, spec.parallel_games):
        ids = range(first, min(first + spec.parallel_games, spec.games))
        boards = {game_id: chess.Board(spec.initial_fen) for game_id in ids}
        rngs = {game_id: np.random.default_rng(np.random.SeedSequence([spec.seed, game_id]))
                for game_id in ids}
        stepper = BT4RootPolicyStepper(
            boards, rngs, max_plies=spec.max_plies, syzygy_path=spec.syzygy_path,
            input_history_encoding=INPUT_HISTORY_ENCODING,
            input_extra_features=INPUT_EXTRA_FEATURES, history_rep_fix=True,
            model_sha256=spec.model_sha256, outcome_mode=OUTCOME_MODE,
            match_tablebase=match_tablebase,
        )
        while stepper.counts.games_completed + stepper.counts.games_discarded < len(boards):
            batch, finalized = stepper.prepare_roots()
            for game in finalized:
                receipt = write_finalized_game(game, games_dir, initial_fen=spec.initial_fen)
                receipts.append(receipt)
                emitted += receipt["rows"]
                attempted += receipt["rows"] + receipt["discarded_rows"]
                if isinstance(game, BT4DiscardedGame):
                    discarded[game.termination] += 1
            if batch is None:
                continue
            inference_boards, inputs = batch.inference_inputs()
            outputs = evaluator.evaluate_roots(inference_boards, inputs)
            stepper.apply_root_outputs(
                batch, outputs,
                temperatures={root.slot_id: spec.temperature for root in batch.roots},
            )
    summary: dict[str, Any] = {
        "schema": SCHEMA, "status": "complete", "games": spec.games,
        "completed": spec.games - sum(discarded.values()),
        "discarded": dict(discarded), "rows_emitted": emitted,
        "rows_attempted": attempted, "game_files": receipts,
        "launch_sha256": file_sha256(spec.out / "launch.json"),
    }
    _atomic_json(spec.out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--syzygy-path", required=True)
    parser.add_argument("--outcome-mode", required=True, choices=[OUTCOME_MODE])
    parser.add_argument("--wdl-output", required=True)
    parser.add_argument("--wdl-kind", required=True, choices=["logits", "probabilities"])
    parser.add_argument("--policy-output")
    parser.add_argument("--games", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-plies", required=True, type=int)
    parser.add_argument("--parallel-games", type=int, default=4)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--initial-fen", default=chess.STARTING_FEN)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    model_path = args.onnx.resolve(strict=True)
    model_sha256 = file_sha256(model_path)
    spec = WorkerSpec(
        out=args.out.resolve(), games=args.games, seed=args.seed,
        max_plies=args.max_plies, parallel_games=args.parallel_games,
        temperature=args.temperature, initial_fen=args.initial_fen,
        syzygy_path=args.syzygy_path, model_sha256=model_sha256,
        outcome_mode=args.outcome_mode, model_path=str(model_path), providers=("pending",),
    )
    spec.validate()
    # No native board is constructed until the mode is installed. Validation
    # above only constructs python-chess boards.
    rep_fix.apply(True, boards_discarded=True)
    tb = tablebase.open_strict_match_tablebase(args.syzygy_path, max_pieces=6)
    try:
        inventory = table_file_inventory(args.syzygy_path)
        sess, input_name, input_dtype, providers = open_session(
            str(model_path), gpu_mem_gb=0, threads=args.threads,
        )
        evaluator = bt4_generation_evaluator.BT4OnnxEvaluator(
            sess, input_name=input_name, input_dtype=input_dtype,
            policy_output=args.policy_output, wdl_output=args.wdl_output,
            wdl_kind=args.wdl_kind, input_history_encoding=INPUT_HISTORY_ENCODING,
            input_extra_features=INPUT_EXTRA_FEATURES, model_sha256=model_sha256,
            history_rep_fix=True,
        )
        realized = replace(spec, providers=tuple(providers))
        print(json.dumps(run_worker(realized, evaluator, tb, inventory), sort_keys=True))
    finally:
        tb.close()


if __name__ == "__main__":
    main()
