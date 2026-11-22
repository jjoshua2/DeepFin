#!/usr/bin/env python3
"""Bounded, CPU-only post-search census; see the transposition direction plan.

This measures cached-board occupancy, not evaluation events or saved NN work.
Each root/budget starts a fresh tree. Run in a dedicated process: native TT
counters are process-wide, and census requires a quiescent tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding import _lc0_ext
from chess_anti_engine.encoding.features import extra_feature_plane_count
from chess_anti_engine.encoding.lc0 import c_input_history_mode
from chess_anti_engine.eval.model_provenance import resolve_checkpoint, sha256_file
from chess_anti_engine.inference import BatchEvaluator
from chess_anti_engine.mcts import _mcts_tree
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.selfplay.network_turn import SelfplaySearchShape, build_selfplay_search_shape
from chess_anti_engine.simple_rl.collect import CollectSettings, build_configs
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


# Production's literal at the selfplay call site (chess_anti_engine/selfplay/network_turn.py,
# the run_gumbel_root_many_c call). tests/test_profile_transposition_reuse.py compares this
# value with that call site's keyword, so a divergence from production fails a test.
ALLOW_TERMINAL_ROOT_SHORTCUTS = True


class SyntheticEvaluator:
    """Deterministic row-pure fixture, never a strength or throughput model."""

    def __init__(self) -> None:
        rng = np.random.default_rng(20261122)
        self.policy = rng.standard_normal((64, 1858)).astype(np.float32)
        self.wdl = rng.standard_normal((64, 3)).astype(np.float32)

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        indices = [hashlib.blake2b(row.tobytes(), digest_size=1).digest()[0] % 64 for row in x]
        return self.policy[indices], self.wdl[indices]


def load_roots(path: Path, max_roots: int) -> list[tuple[str, chess.Board]]:
    """JSONL {id, moves:[UCI,...]}, replayed from the standard initial board.

    FEN-only / truncated-history records are intentionally refused in v1.
    Read the first max_roots records in file order; the report pins that order.
    """
    if not 1 <= max_roots <= 128:
        raise ValueError("max_roots must be in [1,128]")
    roots: list[tuple[str, chess.Board]] = []
    seen: set[str] = set()
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"id", "moves"}:
                raise ValueError(f"line {line_no}: expected exactly id and moves; FEN-only roots are unsupported")
            root_id, moves = row["id"], row["moves"]
            if not isinstance(root_id, str) or not root_id or root_id in seen:
                raise ValueError(f"line {line_no}: root id must be a unique nonempty string")
            if not isinstance(moves, list) or len(moves) > 600:
                raise ValueError(f"line {line_no}: moves must be a list of at most 600 UCI moves")
            board = chess.Board()
            for move in moves:
                if not isinstance(move, str) or board.is_game_over(claim_draw=False):
                    raise ValueError(f"line {line_no}: invalid move or history continues after game end")
                parsed = chess.Move.from_uci(move)
                if parsed not in board.legal_moves:
                    raise ValueError(f"line {line_no}: illegal or null move {move!r}")
                board.push(parsed)
            if board.is_game_over(claim_draw=True):
                raise ValueError(f"line {line_no}: terminal/claimable-draw root is outside this probe")
            roots.append((root_id, board))
            seen.add(root_id)
            if len(roots) == max_roots:
                break
    if not roots:
        raise ValueError("root corpus is empty")
    return roots


def counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Do not reset shared counters or turn a reset into plausible evidence."""
    if before.keys() != after.keys():
        raise RuntimeError("TT counter schema changed during search")
    delta = {key: after[key] - value for key, value in before.items()}
    if any(value < 0 for value in delta.values()):
        raise RuntimeError("TT counters decreased; another caller reset them")
    return delta


def profile_root(
    board: chess.Board, *, shape: SelfplaySearchShape,
    model: torch.nn.Module | None, evaluator: BatchEvaluator | None,
    seed: int, noise_scale: float, max_cached_boards: int,
) -> dict[str, Any]:
    """Run the existing C runner unchanged, then inspect its completed tree."""
    if not hasattr(_mcts_tree.MCTSTree, "transposition_census"):
        raise RuntimeError("native census unavailable; rebuild C extensions from this source")
    before = _mcts_tree.tt_stats()
    started = time.perf_counter()
    policies, actions, values, masks, tree, root_ids = run_gumbel_root_many_c(
        model, [board], device="cpu", rng=np.random.default_rng(seed),
        evaluator=evaluator, per_game_add_noise=[noise_scale > 0],
        per_game_gumbel_scale=[noise_scale],
        allow_terminal_root_shortcuts=ALLOW_TERMINAL_ROOT_SHORTCUTS,
        **shape.runner_kwargs(),
    )
    search_seconds = time.perf_counter() - started
    after = _mcts_tree.tt_stats()
    started = time.perf_counter()
    census = tree.transposition_census(
        input_history_mode=c_input_history_mode(shape.cfg.input_history_encoding),
        n_extra=extra_feature_plane_count(shape.cfg.input_extra_features),
        max_cached_boards=max_cached_boards,
    )
    census_seconds = time.perf_counter() - started
    if _mcts_tree.tt_stats() != after:
        raise RuntimeError("TT counters changed during census; refuse contaminated report")
    # Full output, not just the move: equal moves can conceal altered targets.
    output = {
        "action": int(actions[0]), "policy": policies[0].tolist(),
        "root_value": float(values[0]), "legal_mask": masks[0].tolist(),
    }
    digest = hashlib.sha256(json.dumps(output, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {
        "simulations_requested": shape.cfg.simulations,
        "tree_nodes": tree.node_count(), "root_id": root_ids[0],
        "tt_counts": counter_delta(before, after), "census": census,
        "search_seconds": search_seconds, "census_seconds": census_seconds,
        "search_output_sha256": digest, "search_output": output,
        "realized_shape": {**asdict(shape), "allow_terminal_root_shortcuts": ALLOW_TERMINAL_ROOT_SHORTCUTS},
    }


def _git_state() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)),
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--roots", type=Path, help="JSONL {id,moves:[UCI,...]} from standard initial position")
    parser.add_argument("--collect-settings", type=Path, help="CollectSettings JSON; required with a checkpoint")
    parser.add_argument("--simulations", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--max-roots", type=int, default=32)
    parser.add_argument("--max-cached-boards", type=int, default=8192)
    parser.add_argument("--noise-scale", type=float, default=1.0, help="constant per-root scale; no ply schedule")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    if not 1 <= args.max_roots <= 128 or not all(1 <= n <= 4096 for n in args.simulations):
        parser.error("probe bounds: 1..128 roots and 1..4096 simulations per root/budget")
    if len(args.simulations) > 4 or len(set(args.simulations)) != len(args.simulations):
        parser.error("use at most four distinct simulation budgets")
    if not 1 <= args.max_cached_boards <= 65536 or not 0 <= args.noise_scale <= 10:
        parser.error("probe bounds: 1..65536 cached boards and finite noise scale in [0,10]")
    if args.checkpoint is not None and (args.roots is None or args.collect_settings is None):
        parser.error("--checkpoint requires --roots and --collect-settings; do not infer the live profile")
    if args.synthetic_smoke and args.collect_settings is not None:
        parser.error("synthetic smoke has a fixed fixture profile; --collect-settings requires --checkpoint")
    torch.set_num_threads(2)
    model: torch.nn.Module | None = None
    evaluator: BatchEvaluator | None = None
    if args.synthetic_smoke:
        arch: dict[str, Any] = {
            "input_history_encoding": "lc0_root", "input_extra_features": "v2_threats",
            "policy_encoding": "lc0_1858", "history_rep_fix": False,
        }
        settings = CollectSettings(games=1, concurrency=1, device="cpu", tree_reuse=False)
        evaluator = SyntheticEvaluator()
        model_provenance: dict[str, Any] = {"kind": "synthetic_fixture", "production_evidence": False}
    else:
        assert args.checkpoint is not None
        assert args.collect_settings is not None
        checkpoint = resolve_checkpoint(args.checkpoint)
        checkpoint_digest = sha256_file(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        arch = payload.get("arch", {}) if isinstance(payload, dict) else {}
        if not isinstance(arch, dict) or not {"input_history_encoding", "input_extra_features", "policy_encoding", "history_rep_fix"} <= arch.keys():
            parser.error("checkpoint must embed complete arch/encoding/history_rep_fix metadata")
        if arch.get("use_dynamic_relations") or arch.get("policy_dynamic_relations"):
            parser.error("relation-enabled checkpoints are outside this probe; refusing to drop their auxiliary inputs")
        del payload
        settings = CollectSettings(**json.loads(args.collect_settings.read_text()))
        if settings.syzygy is not None:
            parser.error("tablebase-backed collection is outside this CPU probe; use an explicit no-tablebase profile")
        model = load_model_from_checkpoint(checkpoint, device="cpu", require_complete=True)
        if sha256_file(checkpoint) != checkpoint_digest:
            raise RuntimeError("checkpoint changed while loading")
        model_provenance = {"kind": "checkpoint", "path": str(checkpoint.resolve()), "sha256": checkpoint_digest}
    # Must precede CBoard construction. No mid-process regime flip is allowed.
    rep_fix.apply(bool(arch["history_rep_fix"]))
    if args.roots is None:
        roots = [("initial", chess.Board())]
        roots_digest = None
    else:
        roots_digest = sha256_file(args.roots)
        roots = load_roots(args.roots, args.max_roots)
        if sha256_file(args.roots) != roots_digest:
            raise RuntimeError("root corpus changed while reading")
    configs = build_configs(settings, arch, Path("unused-in-single-root-probe"))
    rows = []
    for root_id, board in roots:
        for simulations in args.simulations:
            shape = build_selfplay_search_shape(
                search=replace(configs["search"], simulations=simulations),
                game=configs["game"], simulations=simulations,
            )
            row = profile_root(
                board, shape=shape, model=model, evaluator=evaluator, seed=args.seed,
                noise_scale=args.noise_scale, max_cached_boards=args.max_cached_boards,
            )
            rows.append({"id": root_id, "fen": board.fen(), "moves": [move.uci() for move in board.move_stack], **row})
    native = {}
    for extension in (_mcts_tree, _lc0_ext):
        assert extension.__file__ is not None
        native_path = Path(extension.__file__)
        native[extension.__name__] = {"path": str(native_path), "sha256": sha256_file(native_path)}
    report = {
        "schema_version": 1, "scope": "fresh_tree_single_root_cpu_cached_board_occupancy",
        "git": _git_state(), "model": model_provenance,
        "native": native, "mcts_abi": _mcts_tree.ABI_VERSION,
        "roots": {"path": str(args.roots) if args.roots else None, "sha256": roots_digest,
                  "selection": "first_max_roots_in_file_order", "selected": len(roots), "max_roots": args.max_roots,
                  "history_scope": "complete_from_standard_initial_position"},
        "requested_collect_settings": asdict(settings), "seed": args.seed,
        "runtime": {"device": "cpu", "torch_threads": 2, "tree_reuse": False, "eval_cache": False,
                    "tablebases": False, "batch_roots": 1, "constant_noise_scale": args.noise_scale,
                    "history_rep_fix": rep_fix.current(), "python": platform.python_version(),
                    "torch": torch.__version__, "numpy": np.__version__},
        "limitations": [
            "Cached nodes include roots and TT-hit recipients; census is not an evaluation event log.",
            "Input groups reconstruct float32 planes, not actual BF16 transport bytes or relation inputs.",
            "TT counters are counts, not hit rates; isolation is required, concurrent searches cannot be detected reliably.",
            "Fresh single-root CPU search does not reproduce batched GPU collection, warm trees or ply noise schedules.",
            "Relation-enabled checkpoints are refused; the census does not classify auxiliary inputs.",
            "Timing is descriptive only; no paired throughput, target-quality or playing-strength claim.",
        ],
        "results": rows,
    }
    _write_report(args.output, report)
    print(f"Wrote {len(rows)} complete censuses to {args.output}; no NN-savings or strength verdict.")


if __name__ == "__main__":
    main()
