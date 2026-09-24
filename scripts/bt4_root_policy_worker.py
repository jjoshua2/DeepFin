#!/usr/bin/env python3
"""Opt-in BT4 root-policy corpus worker (experimental, no replay consumer).

Every accepted row is a >=7-piece pre-move root. The actor samples the raw
BT4 root policy; no search or tablebase-guided move selection is performed.
Six-man Syzygy is used only to finish games under rule50_match_v1. One game is
buffered until its outcome is known, then published as one atomic NPZ file.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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
MAX_GAMES = 32
MAX_TOTAL_REQUESTED_PLIES = 4096
RESEARCH_GAMES = 128
RESEARCH_MAX_PLIES = 400
RESEARCH_PARALLEL_GAMES = frozenset({16, 32, 64})
RESEARCH_MAX_BUFFERED_ROWS = 64 * RESEARCH_MAX_PLIES
MAX_GPU_MEM_GB = 8.0
GPU_LOCK = Path("/home/josh/projects/chess/scratchpad/gpu0_experiment.lock")
_NEURAL_OPS = {"Conv", "FusedConv", "NhwcConv", "MatMul", "FusedMatMul", "Gemm", "FusedGemm"}
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
    "scripts/gen_sf_rooted_corpus.py",
    "chess_anti_engine/eval/rvg_surgery.py",
    "chess_anti_engine/mcts/sampling.py",
    "chess_anti_engine/selfplay/game.py",
)


class RootEvaluator(Protocol):
    def evaluate_roots(
        self, boards: list[chess.Board], x_batch: np.ndarray,
    ) -> list[bt4_generation_evaluator.BT4RootOutput]: ...


def acquire_gpu_lock() -> int:
    """Child owns the canonical GPU lease; a busy slot fails without waiting."""
    fd = os.open(GPU_LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError(f"canonical GPU lock is busy: {GPU_LOCK}") from exc
    except OSError:
        os.close(fd)
        raise
    return fd


def cuda_profile_proof(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Prove that first root inference ran neural work on CUDA, not just ORT setup."""
    cuda: Counter[str] = Counter()
    cpu: Counter[str] = Counter()
    for event in events:
        details = event.get("args")
        if not isinstance(details, dict):
            continue
        provider = details.get("provider")
        operation = details.get("op_name")
        if not isinstance(operation, str) or not operation:
            continue
        if provider == "CUDAExecutionProvider":
            cuda[operation] += 1
        elif provider == "CPUExecutionProvider":
            cpu[operation] += 1
        elif provider is not None:
            raise RuntimeError(f"unexpected profiled provider {provider!r}")
    neural = sum(cuda[op] for op in _NEURAL_OPS)
    if neural == 0:
        raise RuntimeError("CUDA profile has no neural model-compute node")
    return {
        "schema": 1, "scope": "first_root_inference_in_this_session",
        "cuda_neural_nodes": neural, "cuda_nodes": sum(cuda.values()),
        "cpu_nodes": sum(cpu.values()),
        "cuda_ops": dict(sorted(cuda.items())), "cpu_ops": dict(sorted(cpu.items())),
        "claim": "observed_cuda_neural_compute; not_all_ops_cuda",
    }


def validate_cuda_provider_controls(
    requested_provider: str, *, cudnn_conv_algo_search: str | None,
    cudnn_conv_use_max_workspace: int | None,
) -> None:
    if cudnn_conv_algo_search is not None and cudnn_conv_algo_search not in (
        "EXHAUSTIVE", "HEURISTIC", "DEFAULT",
    ):
        raise ValueError("invalid cuDNN convolution algorithm search mode")
    if (cudnn_conv_use_max_workspace is not None
            and (type(cudnn_conv_use_max_workspace) is not int
                 or cudnn_conv_use_max_workspace not in (0, 1))):
        raise ValueError("cuDNN maximum workspace must be 0 or 1")
    if requested_provider == "cpu" and (
        cudnn_conv_algo_search is not None or cudnn_conv_use_max_workspace is not None
    ):
        raise ValueError("CPU session does not accept CUDA settings")


def verify_cuda_session(
    session: Any, *, gpu_mem_gb: float,
    cudnn_conv_algo_search: str | None = None,
    cudnn_conv_use_max_workspace: int | None = None,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Read back realized EP order and cap before any inference or output."""
    providers = tuple(session.get_providers())
    if not providers or providers[0] != "CUDAExecutionProvider":
        raise RuntimeError("CUDA requested but ONNX Runtime fell back to CPU")
    options = session.get_provider_options()
    realized = {str(k): str(v) for k, v in options.get("CUDAExecutionProvider", {}).items()}
    if (realized.get("device_id") != "0"
            or realized.get("gpu_mem_limit") != str(int(gpu_mem_gb * 1024 ** 3))):
        raise RuntimeError("CUDA device or memory cap differs from request")
    if (cudnn_conv_algo_search is not None
            and realized.get("cudnn_conv_algo_search") != cudnn_conv_algo_search):
        raise RuntimeError("CUDA cuDNN algorithm search differs from request")
    if (cudnn_conv_use_max_workspace is not None
            and realized.get("cudnn_conv_use_max_workspace") != str(cudnn_conv_use_max_workspace)):
        raise RuntimeError("CUDA cuDNN maximum workspace differs from request")
    return providers, realized


def verify_cuda_model_schema(
    session: Any, *, input_name: str, input_dtype: str,
    policy_output: str, wdl_output: str, wdl_kind: str,
) -> None:
    """Pin this qualification to the preregistered BT4 input and named heads."""
    inputs = session.get_inputs()
    # The saved v2_threats tensor has more planes; the evaluator converts it
    # to this exact LC0 model input before inference.
    planes = 112
    if (len(inputs) != 1 or inputs[0].name != input_name
            or input_dtype != "float32" or inputs[0].type != "tensor(float)"
            or len(inputs[0].shape) != 4
            or list(inputs[0].shape[1:]) != [planes, 8, 8]):
        raise ValueError("CUDA BT4 input name/type/planes differ from qualification contract")
    outputs = session.get_outputs()
    if policy_output == wdl_output or wdl_kind != "probabilities":
        raise ValueError("CUDA BT4 requires distinct policy/WDL and probability WDL")
    for name, width in ((policy_output, COMPACT_POLICY_SIZE), (wdl_output, 3)):
        matches = [output for output in outputs if output.name == name]
        if (len(matches) != 1 or matches[0].type != "tensor(float)"
                or len(matches[0].shape) != 2 or matches[0].shape[1] != width):
            raise ValueError(f"CUDA BT4 named head {name!r} differs from float32 width {width}")


def open_worker_session(
    onnx: str, *, requested_provider: str, gpu_mem_gb: float,
    threads: int, profile_prefix: Path | None,
    cudnn_conv_algo_search: str | None = None,
    cudnn_conv_use_max_workspace: int | None = None,
) -> tuple[Any, str, np.dtype[Any], tuple[str, ...], dict[str, str]]:
    validate_cuda_provider_controls(
        requested_provider, cudnn_conv_algo_search=cudnn_conv_algo_search,
        cudnn_conv_use_max_workspace=cudnn_conv_use_max_workspace,
    )
    if requested_provider == "cpu":
        if gpu_mem_gb != 0 or profile_prefix is not None:
            raise ValueError("CPU session does not accept CUDA settings")
        session, name, dtype, providers = open_session(
            onnx, gpu_mem_gb=0, threads=threads,
        )
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError("CPU request realized another ONNX provider")
        return session, name, dtype, tuple(providers), {}
    if requested_provider != "cuda" or profile_prefix is None:
        raise ValueError("CUDA session requires a profile prefix")
    if not math.isfinite(gpu_mem_gb) or not 0 < gpu_mem_gb <= MAX_GPU_MEM_GB:
        raise ValueError("CUDA memory cap must be finite and within (0, 8] GiB")
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDA requested but ONNX Runtime has no CUDAExecutionProvider")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.enable_profiling = True
    options.profile_file_prefix = str(profile_prefix)
    cuda_options: dict[str, int | str] = {
        "device_id": 0, "gpu_mem_limit": int(gpu_mem_gb * 1024 ** 3),
    }
    if cudnn_conv_algo_search is not None:
        cuda_options["cudnn_conv_algo_search"] = cudnn_conv_algo_search
    if cudnn_conv_use_max_workspace is not None:
        cuda_options["cudnn_conv_use_max_workspace"] = str(cudnn_conv_use_max_workspace)
    session = ort.InferenceSession(
        onnx, sess_options=options,
        providers=[
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider",
        ],
        enable_fallback=False,
    )
    session.disable_fallback()
    providers, realized = verify_cuda_session(
        session, gpu_mem_gb=gpu_mem_gb,
        cudnn_conv_algo_search=cudnn_conv_algo_search,
        cudnn_conv_use_max_workspace=cudnn_conv_use_max_workspace,
    )
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].type not in ("tensor(float16)", "tensor(float)"):
        raise ValueError("BT4 CUDA model needs one float16/float32 input tensor")
    dtype = np.dtype(np.float16 if inputs[0].type == "tensor(float16)" else np.float32)
    return session, inputs[0].name, dtype, providers, realized


class CudaQualifiedEvaluator:
    """Profile the first root call before allowing a sampled move or game file."""

    def __init__(self, base: RootEvaluator, session: Any, out: Path) -> None:
        self.base = base
        self.session = session
        self.out = out
        self.proof: dict[str, Any] | None = None
        self.qualification_seconds = 0.0
        self.qualified_at: float | None = None

    def evaluate_roots(
        self, boards: list[chess.Board], x_batch: np.ndarray,
    ) -> list[bt4_generation_evaluator.BT4RootOutput]:
        if self.proof is not None:
            if self.session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError("CUDA provider changed after qualification")
            return self.base.evaluate_roots(boards, x_batch)
        started = time.monotonic()
        outputs = self.base.evaluate_roots(boards, x_batch)
        profile_path = Path(self.session.end_profiling())
        events = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(events, list):
            raise RuntimeError("ORT profile is not an event array")
        proof = cuda_profile_proof(events)
        raw_profile = profile_path.read_bytes()
        _atomic_bytes(self.out / "provider_profile.json", raw_profile)
        proof["profile_sha256"] = hashlib.sha256(raw_profile).hexdigest()
        proof["providers_after_first_call"] = list(self.session.get_providers())
        if not proof["providers_after_first_call"] or proof["providers_after_first_call"][0] != "CUDAExecutionProvider":
            raise RuntimeError("CUDA provider fell back during first inference")
        self.qualification_seconds = time.monotonic() - started
        proof["qualification_seconds"] = self.qualification_seconds
        _atomic_json(self.out / "provider_proof.json", proof)
        self.proof = proof
        self.qualified_at = time.monotonic()
        return outputs


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
    requested_provider: str = "cpu"
    gpu_mem_gb: float = 0.0
    cudnn_conv_algo_search: str | None = None
    cudnn_conv_use_max_workspace: int | None = None
    research_capacity_128x400: bool = False
    provider_options: Mapping[str, str] = field(default_factory=dict)

    def validate(self, *, check_realized_provider: bool = True) -> None:
        validate_cuda_provider_controls(
            self.requested_provider,
            cudnn_conv_algo_search=self.cudnn_conv_algo_search,
            cudnn_conv_use_max_workspace=self.cudnn_conv_use_max_workspace,
        )
        if self.outcome_mode != OUTCOME_MODE:
            raise ValueError("BT4 worker requires explicit rule50_match_v1 outcome mode")
        if self.games < 1 or self.seed < 0 or self.max_plies < 1:
            raise ValueError("games/max_plies must be positive and seed nonnegative")
        if type(self.research_capacity_128x400) is not bool:
            raise TypeError("research capacity profile must be an explicit bool")
        if self.research_capacity_128x400:
            if (self.games != RESEARCH_GAMES or self.max_plies != RESEARCH_MAX_PLIES
                    or self.parallel_games not in RESEARCH_PARALLEL_GAMES):
                raise ValueError("research capacity profile requires 128 games, 400 plies, "
                                 "and parallel_games in {16,32,64}")
        else:
            if self.games > MAX_GAMES or self.games * self.max_plies > MAX_TOTAL_REQUESTED_PLIES:
                raise ValueError(
                    f"experimental run requires games <= {MAX_GAMES} and "
                    f"games * max_plies <= {MAX_TOTAL_REQUESTED_PLIES}"
                )
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
        if self.requested_provider == "cpu":
            if self.gpu_mem_gb != 0 or (check_realized_provider and self.providers != ("CPUExecutionProvider",)):
                raise ValueError("CPU worker requires zero GPU memory and CPU provider")
        elif self.requested_provider == "cuda":
            if (not math.isfinite(self.gpu_mem_gb) or not 0 < self.gpu_mem_gb <= MAX_GPU_MEM_GB
                    or (check_realized_provider and self.providers[0] != "CUDAExecutionProvider")):
                raise ValueError("CUDA worker requires realized CUDA and 0 < GPU memory <= 8 GiB")
            limit = int(self.gpu_mem_gb * 1024 ** 3)
            if check_realized_provider and (self.provider_options.get("device_id") != "0"
                    or self.provider_options.get("gpu_mem_limit") != str(limit)):
                raise ValueError("CUDA provider options differ from device 0 and memory cap")
            if (check_realized_provider and self.cudnn_conv_algo_search is not None
                    and self.provider_options.get("cudnn_conv_algo_search")
                    != self.cudnn_conv_algo_search):
                raise ValueError("CUDA provider options differ from cuDNN algorithm search")
            if (check_realized_provider and self.cudnn_conv_use_max_workspace is not None
                    and self.provider_options.get("cudnn_conv_use_max_workspace")
                    != str(self.cudnn_conv_use_max_workspace)):
                raise ValueError("CUDA provider options differ from cuDNN maximum workspace")
        else:
            raise ValueError("worker provider must be cpu or cuda")
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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    writing = path.with_name(path.name + ".writing")
    try:
        with writing.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(writing, path)
    finally:
        writing.unlink(missing_ok=True)


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(document, sort_keys=True, indent=2) + "\n").encode())


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
                "kind": "root_inference_no_search", "policy_encoding": "lc0_1858_compact",
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
                handle,
                x=arrays["x"], policy_t1=arrays["policy_t1"],
                wdl_raw=arrays["wdl_raw"],
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
) -> dict[str, Any]:
    """Run a finite seeded corpus. A missing strict probe aborts without summary."""
    spec.validate()
    if spec.requested_provider == "cuda" and not isinstance(evaluator, CudaQualifiedEvaluator):
        raise TypeError("CUDA run requires the first-root profiled evaluator")
    if rep_fix.current() is not True:
        raise RuntimeError("history_rep_fix must be configured before BT4 boards")
    # Recheck the passed handle before creating output; do not trust its type.
    tablebase.SyzygyProbe(
        spec.syzygy_path, max_pieces=6, rule50_aware=True, tablebase=match_tablebase,
    )
    table_inventory = table_file_inventory(spec.syzygy_path)
    root = Path(__file__).resolve().parents[1]
    run_started = time.monotonic()
    manifest: dict[str, Any] = {
        "schema": SCHEMA, "status": "launched", "actor": "bt4_root_policy_no_search",
        "outcome_mode": spec.outcome_mode, "minimum_emitted_root_pieces": 7,
        "terminal_wdl_target": "game_result_from_root_side_to_move",
        "teacher_observation": "root_inference_compact_t1_policy_and_native_wdl_unmodified_by_outcome",
        "seed": spec.seed, "games": spec.games, "max_plies": spec.max_plies,
        "parallel_games": spec.parallel_games,
        "research_capacity_128x400": spec.research_capacity_128x400,
        "max_buffered_rows": (RESEARCH_MAX_BUFFERED_ROWS
                              if spec.research_capacity_128x400 else MAX_BUFFERED_ROWS),
        "temperature": spec.temperature, "initial_fen": spec.initial_fen,
        "model": {"path": spec.model_path, "sha256": spec.model_sha256,
                  "providers": list(spec.providers),
                  "requested_provider": spec.requested_provider,
                  "gpu_mem_gb": spec.gpu_mem_gb,
                  **({"cudnn_conv_algo_search": spec.cudnn_conv_algo_search}
                     if spec.cudnn_conv_algo_search is not None else {}),
                  **({"cudnn_conv_use_max_workspace": spec.cudnn_conv_use_max_workspace}
                     if spec.cudnn_conv_use_max_workspace is not None else {}),
                  "provider_options": dict(spec.provider_options),
                  "gpu_lock": str(GPU_LOCK) if spec.requested_provider == "cuda" else None},
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
    writer_seconds = 0.0
    gpu_proof_verified = False
    effective_batch_sizes: Counter[int] = Counter()
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
                if spec.requested_provider == "cuda" and not gpu_proof_verified:
                    raise RuntimeError("CUDA game cannot publish before root provider proof")
                writing_started = time.monotonic()
                receipt = write_finalized_game(game, games_dir, initial_fen=spec.initial_fen)
                writer_seconds += time.monotonic() - writing_started
                receipts.append(receipt)
                emitted += receipt["rows"]
                attempted += receipt["rows"] + receipt["discarded_rows"]
                if isinstance(game, BT4DiscardedGame):
                    discarded[game.termination] += 1
            if batch is None:
                continue
            inference_boards, inputs = batch.inference_inputs()
            effective_batch_sizes[len(inference_boards)] += 1
            outputs = evaluator.evaluate_roots(inference_boards, inputs)
            if spec.requested_provider == "cuda" and not gpu_proof_verified:
                proof_file = spec.out / "provider_proof.json"
                if not proof_file.exists():
                    raise RuntimeError("CUDA root inference has no provider proof")
                proof = json.loads(proof_file.read_text(encoding="utf-8"))
                providers_after = proof.get("providers_after_first_call")
                if (int(proof.get("cuda_neural_nodes", 0)) < 1
                        or not isinstance(providers_after, list) or not providers_after
                        or providers_after[0] != "CUDAExecutionProvider"
                        or proof.get("profile_sha256") != file_sha256(spec.out / "provider_profile.json")):
                    raise RuntimeError("CUDA provider proof is incomplete or changed")
                gpu_proof_verified = True
            stepper.apply_root_outputs(
                batch, outputs,
                temperatures={root.slot_id: spec.temperature for root in batch.roots},
            )
    if table_file_inventory(spec.syzygy_path) != table_inventory:
        raise RuntimeError("Syzygy file inventory changed during BT4 generation")
    if spec.requested_provider == "cuda":
        if not gpu_proof_verified or not isinstance(evaluator, CudaQualifiedEvaluator):
            raise RuntimeError("CUDA run had no profiled root inference")
        proof = json.loads((spec.out / "provider_proof.json").read_text(encoding="utf-8"))
        providers_after = proof.get("providers_after_first_call")
        if (int(proof.get("cuda_neural_nodes", 0)) < 1
                or not isinstance(providers_after, list) or not providers_after
                or providers_after[0] != "CUDAExecutionProvider"
                or proof.get("profile_sha256") != file_sha256(spec.out / "provider_profile.json")):
            raise RuntimeError("CUDA profile changed before completion")
    finished = time.monotonic()
    qualified_at = getattr(evaluator, "qualified_at", None)
    summary: dict[str, Any] = {
        "schema": SCHEMA, "status": "complete", "games": spec.games,
        "completed": spec.games - sum(discarded.values()),
        "discarded": dict(discarded), "rows_emitted": emitted,
        "rows_attempted": attempted, "game_files": receipts,
        "launch_sha256": file_sha256(spec.out / "launch.json"),
        "run_wall_seconds": finished - run_started,
        "writer_wall_seconds": writer_seconds,
        "requested_parallel_games": spec.parallel_games,
        "effective_batch_size_histogram": {
            str(size): count for size, count in sorted(effective_batch_sizes.items())
        },
        "inference_calls": sum(effective_batch_sizes.values()),
        "full_batch_calls": effective_batch_sizes[spec.parallel_games],
        "underfilled_calls": sum(count for size, count in effective_batch_sizes.items()
                                 if size < spec.parallel_games),
        "max_effective_batch_size": max(effective_batch_sizes, default=0),
        "post_qualification_wall_seconds": (
            finished - qualified_at if spec.requested_provider == "cuda" and qualified_at is not None
            else None
        ),
        "provider_proof_sha256": (
            file_sha256(spec.out / "provider_proof.json") if spec.requested_provider == "cuda"
            else None
        ),
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
    parser.add_argument("--research-capacity-128x400", action="store_true")
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--initial-fen", default=chess.STARTING_FEN)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--gpu-mem-gb", type=float, default=0.0)
    parser.add_argument("--cudnn-conv-algo-search", choices=["EXHAUSTIVE", "HEURISTIC", "DEFAULT"])
    parser.add_argument("--cudnn-conv-use-max-workspace", type=int, choices=[0, 1])
    parser.add_argument("--expected-onnx-sha256")
    parser.add_argument("--expected-input-name")
    parser.add_argument("--expected-input-dtype", choices=["float16", "float32"])
    args = parser.parse_args()
    if args.threads not in (1, 2):
        parser.error("experimental worker requires --threads 1 or 2")
    if args.provider == "cuda":
        if (not args.expected_onnx_sha256 or not args.expected_input_name
                or not args.expected_input_dtype or not args.policy_output):
            parser.error("CUDA requires explicit expected model/input and named policy output")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            parser.error("CUDA qualification requires CUDA_VISIBLE_DEVICES=0")
    model_path = args.onnx.resolve(strict=True)
    model_sha256 = file_sha256(model_path)
    if args.expected_onnx_sha256 is not None and model_sha256 != args.expected_onnx_sha256:
        raise ValueError("ONNX file SHA-256 differs from expected qualification model")
    spec = WorkerSpec(
        out=args.out.resolve(), games=args.games, seed=args.seed,
        max_plies=args.max_plies, parallel_games=args.parallel_games,
        temperature=args.temperature, initial_fen=args.initial_fen,
        syzygy_path=args.syzygy_path, model_sha256=model_sha256,
        outcome_mode=args.outcome_mode, model_path=str(model_path), providers=("pending",),
        requested_provider=args.provider, gpu_mem_gb=args.gpu_mem_gb,
        cudnn_conv_algo_search=args.cudnn_conv_algo_search,
        cudnn_conv_use_max_workspace=args.cudnn_conv_use_max_workspace,
        research_capacity_128x400=args.research_capacity_128x400,
    )
    spec.validate(check_realized_provider=False)
    # No native board is constructed until the mode is installed. Validation
    # above only constructs python-chess boards.
    rep_fix.apply(True, boards_discarded=True)
    tb = tablebase.open_strict_match_tablebase(args.syzygy_path, max_pieces=6)
    try:
        if args.provider == "cuda":
            # The CUDA session and its threads stay in this one process. The
            # descriptor is deliberately held until process exit, including
            # ORT teardown. An outer supervisor supplies the finite deadline.
            _gpu_fd = acquire_gpu_lock()
        with tempfile.TemporaryDirectory(prefix="bt4_root_ort_") as profile_dir:
            prefix = Path(profile_dir) / "first_root" if args.provider == "cuda" else None
            sess, input_name, input_dtype, providers, provider_options = open_worker_session(
                str(model_path), requested_provider=args.provider,
                gpu_mem_gb=args.gpu_mem_gb, threads=args.threads,
                profile_prefix=prefix,
                cudnn_conv_algo_search=args.cudnn_conv_algo_search,
                cudnn_conv_use_max_workspace=args.cudnn_conv_use_max_workspace,
            )
            if args.provider == "cuda":
                verify_cuda_model_schema(
                    sess, input_name=args.expected_input_name,
                    input_dtype=args.expected_input_dtype,
                    policy_output=args.policy_output, wdl_output=args.wdl_output,
                    wdl_kind=args.wdl_kind,
                )
            evaluator = bt4_generation_evaluator.BT4OnnxEvaluator(
                sess, input_name=input_name, input_dtype=input_dtype,
                policy_output=args.policy_output, wdl_output=args.wdl_output,
                wdl_kind=args.wdl_kind, input_history_encoding=INPUT_HISTORY_ENCODING,
                input_extra_features=INPUT_EXTRA_FEATURES, model_sha256=model_sha256,
                history_rep_fix=True,
            )
            realized = replace(
                spec, providers=providers, provider_options=provider_options,
            )
            realized.validate()
            actor: RootEvaluator = (
                CudaQualifiedEvaluator(evaluator, sess, realized.out)
                if args.provider == "cuda" else evaluator
            )
            print(json.dumps(run_worker(realized, actor, tb), sort_keys=True))
    finally:
        tb.close()


if __name__ == "__main__":
    main()
