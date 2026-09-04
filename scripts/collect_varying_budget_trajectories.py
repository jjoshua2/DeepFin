#!/usr/bin/env python3
"""Collect eight-chunk accumulating trajectories for the budget screen.

The search is always two-walker PUCT with 2,048 simulations per chunk and eight
chunks.  Shorter total horizons are counterfactual labels applied later; they
are never passed into search, so all compared horizons share identical prefix
states.  Only complete trajectories are appended.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.eval.audit import AuditPosition, legal_full_indices, load_audit_set
from chess_anti_engine.eval.audit_history import default_matched_rows_path
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.moves import ActionDecodeError, index_to_move_strict
from chess_anti_engine.tablebase import SyzygyProbe
from chess_anti_engine.uci.engine import EngineOptions
from chess_anti_engine.uci.time_manager import Deadline
from chess_anti_engine.utils.syzygy import default_syzygy_path, require_tablebases

SCHEMA = "deepfin.varying_budget_trajectory.v1"
CHUNK_SIMS = 2048
MAX_CHUNKS = 8
WALKERS = 2


def _score(cp: float) -> float:
    exponent = float(cp) * math.log(10.0) / 300.0
    if exponent >= 0.0:
        return 1.0 / (1.0 + math.exp(-exponent))
    value = math.exp(exponent)
    return value / (1.0 + value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": int(stat.st_size), "sha256": _sha256(resolved)}


def _git_sha() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _stratified(rows: list[AuditPosition], seed: int) -> list[AuditPosition]:
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[int, int], list[AuditPosition]] = defaultdict(list)
    for row in rows:
        buckets[(int(row.phase), int(row.source))].append(row)
    for key, values in buckets.items():
        order = rng.permutation(len(values))
        buckets[key] = [values[int(index)] for index in order]
    keys = sorted(buckets)
    result: list[AuditPosition] = []
    cursor = 0
    while any(buckets.values()):
        key = keys[cursor % len(keys)]
        if buckets[key]:
            result.append(buckets[key].pop())
        cursor += 1
    return result


class GroupIndex:
    """Accept only explicit, unambiguous source-game identities."""

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            keys = [str(value) for value in data["key"]]
            found = np.asarray(data["found"], dtype=bool)
            game_id = np.asarray(data["game_id"], dtype=np.int64)
            self.canonical = all(name in data for name in ("has_game_id", "source_cluster_ambiguous", "src_shard"))
            if not self.canonical:
                self._groups: dict[str, str] = {}
                return
            has_id = np.asarray(data["has_game_id"], dtype=bool)
            ambiguous = np.asarray(data["source_cluster_ambiguous"], dtype=bool)
            shards = [str(value).strip() for value in data["src_shard"]]
            snapshot = str(np.asarray(data["snapshot"]).reshape(-1)[0]) if "snapshot" in data else str(path.resolve())
        self._groups = {
            key: "\0".join((snapshot, shards[index], str(int(game_id[index]))))
            for index, key in enumerate(keys)
            if found[index] and has_id[index] and not ambiguous[index]
            and game_id[index] >= 0 and shards[index]
        }

    def lookup(self, key: str) -> str | None:
        return self._groups.get(str(key))


def group_for(position: AuditPosition, index: GroupIndex | None) -> tuple[str, str]:
    source_game = index.lookup(position.key) if index is not None else None
    return (source_game, "source_game") if source_game is not None else (f"position:{position.key}", "position")


def repair_resume_bank(path: Path) -> set[str]:
    """Atomically retain only complete, nonduplicated eight-row trajectories."""
    if not path.exists():
        return set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, text in enumerate(lines, 1):
        if not text.strip():
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            if any(line.strip() for line in lines[line_number:]):
                raise SystemExit(f"invalid JSON before bank tail at line {line_number}") from exc
            break
        if not isinstance(row, dict) or row.get("schema") != SCHEMA or not row.get("key"):
            raise SystemExit(f"unexpected resume row at line {line_number}")
        key = str(row["key"])
        if key not in grouped:
            order.append(key)
        grouped[key].append(row)
    kept: list[dict[str, Any]] = []
    complete: set[str] = set()
    for key in order:
        rows = grouped[key]
        chunks: list[int] = []
        for row in rows:
            chunk = row.get("chunk")
            if isinstance(chunk, bool) or not isinstance(chunk, int):
                chunks = []
                break
            chunks.append(chunk)
        if len(rows) == MAX_CHUNKS and sorted(chunks) == list(range(1, MAX_CHUNKS + 1)):
            kept.extend(sorted(rows, key=lambda row: int(row["chunk"])))
            complete.add(key)
    temp = path.with_name(f".{path.name}.repair-{os.getpid()}")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()
    return complete


def _build_worker(checkpoint: str, device: str, syzygy_path: str):
    from chess_anti_engine.uci.__main__ import _make_evaluator_factory
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint
    from chess_anti_engine.uci.search import SearchWorker

    options = EngineOptions()
    compile_mode = "max-autotune" if device.startswith("cuda") else None
    if compile_mode:
        from chess_anti_engine.worker import _configure_shared_compile_cache

        _configure_shared_compile_cache(cache_dir=Path("~/.cache/deepfin/worker_cache").expanduser())
    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    history = str(getattr(model, "input_history_encoding", "legacy"))
    extras = str(getattr(model, "input_extra_features", "v1"))
    policy = str(getattr(model, "policy_encoding", "lc0_1858"))
    relations = bool(getattr(model, "use_dynamic_relations", False))
    factory = _make_evaluator_factory([model], (device,), True, WALKERS, int(options.leaf_gather), compile_mode=compile_mode)
    evaluator = factory(int(options.max_batch))
    cfg = GumbelConfig(
        simulations=CHUNK_SIMS, add_noise=False, temperature=0.0,
        input_history_encoding=history, input_extra_features=extras,
        policy_encoding=policy, compute_relations=relations,
        c_scale=float(options.c_scale), c_visit=float(options.c_visit),
        c_visit_root=float(options.c_visit_root), c_scale_root=float(options.c_scale_root),
        q_visit_exp_root=float(options.q_visit_exp_root), topk=int(options.topk),
        policy_temp=float(options.policy_temp), halving_div=int(options.halving_div),
        c_puct=float(options.cpuct), cpuct_factor=float(options.cpuct_factor),
        cpuct_base=float(options.cpuct_base), fpu_reduction=float(options.fpu_reduction),
    )
    worker = SearchWorker(evaluator, device=device, gumbel_cfg=cfg, chunk_sims=CHUNK_SIMS, n_walkers=WALKERS, vloss_weight=int(options.vloss_weight), walker_gather=int(options.leaf_gather))
    worker.set_max_tree_mb(int(options.hash_mb))
    worker.set_root_noise_scale(float(options.root_noise_scale))
    if syzygy_path:
        worker.set_tb_probe(SyzygyProbe(syzygy_path))
    info = {
        "input_history_encoding": history,
        "input_extra_features": extras,
        "policy_encoding": policy,
        "compute_relations": relations,
        "realized_search_path": worker.realized_search_path(),
        "realized_search_values": worker.realized_search_values(),
    }
    if info["realized_search_path"] != "walker":
        worker.close()
        raise SystemExit("required two-walker PUCT path was not realized")
    return worker, info, compile_mode


def _stability(last: int, count: int, action: int, gap: float, n: int) -> tuple[int, int]:
    if gap <= 0.0 and n != 1:
        return last, 0
    return (last, count + 1) if action == last else (action, 0)


def _complexity(stable: int, gap: float, n: int) -> bool:
    return not (stable >= 2 and (n == 1 or gap >= 0.25))


def _callback(worker: Any, position: AuditPosition, board: chess.Board, legal_ucis: list[str], legal_actions: np.ndarray, group_id: str, group_kind: str, snapshots: list[dict[str, Any]], started: float):
    last_action = -1
    stable = 0
    previous_q: float | None = None
    previous_shares: dict[int, float] | None = None

    def on_chunk(total_nodes: int) -> None:
        nonlocal last_action, stable, previous_q, previous_shares
        actions, visits = worker._filtered_root_visits(None)
        if actions.size == 0:
            return
        action = int(worker._emitted_action(actions, visits, None))
        try:
            move = index_to_move_strict(action, board)
        except ActionDecodeError as exc:
            raise RuntimeError(f"{position.key}: undecodable emitted action") from exc
        uci = move.uci()
        if move not in board.legal_moves or uci not in legal_ucis:
            raise RuntimeError(f"{position.key}: emitted action is not a legal audit move")
        action_list = [int(value) for value in actions]
        if len(set(action_list)) != len(action_list) or set(action_list) != {int(value) for value in legal_actions}:
            raise RuntimeError(f"{position.key}: root support differs from legal support")
        total = int(visits.sum())
        if total <= 0 and len(action_list) != 1:
            raise RuntimeError("multi-action root has no visits")
        shares = visits.astype(np.float64) / total if total > 0 else np.ones(1)
        share_map = dict(zip(action_list, [float(value) for value in shares], strict=True))
        action_index = action_list.index(action)
        gap = shares[action_index] - max(np.delete(shares, action_index), default=0.0)
        entropy = -sum(float(value) * math.log(float(value)) for value in shares if value > 0.0)
        tree, root_id = worker._tree, worker._root_id
        if tree is None or root_id is None:
            raise RuntimeError("chunk callback has no tree")
        root_q = float(tree.node_q(root_id))
        child_actions, child_visits, child_q = tree.get_children_q(root_id, root_q)
        visit_map = {int(a): int(v) for a, v in zip(child_actions, child_visits, strict=True)}
        q_map = {int(a): float(q) for a, q in zip(child_actions, child_q, strict=True)}
        if any(a not in q_map or visit_map.get(a) != int(visits[index]) for index, a in enumerate(action_list)):
            raise RuntimeError("root visit/Q readbacks disagree")
        q_values = [q_map[a] for a in action_list]
        if not math.isfinite(root_q) or not all(math.isfinite(value) for value in q_values):
            raise RuntimeError("non-finite Q observation")
        other_q = [q for index, (visit, q) in enumerate(zip(visits, q_values, strict=True)) if index != action_index and int(visit) > 0]
        q_gap = q_values[action_index] - max(other_q) if int(visits[action_index]) > 0 and other_q else None
        flip = bool(snapshots and snapshots[-1]["emitted_action"] != action)
        last_action, stable = _stability(last_action, stable, action, float(gap), len(action_list))
        q_drift = None if previous_q is None else abs(root_q - previous_q)
        churn = None if previous_shares is None else 0.5 * sum(abs(share_map.get(a, 0.0) - previous_shares.get(a, 0.0)) for a in set(share_map) | set(previous_shares))
        chosen_cp = float(position.move_cp.get(uci, min(position.move_cp.values())))
        snapshots.append({
            "schema": SCHEMA, "key": position.key, "group_id": group_id,
            "group_kind": group_kind, "fen": position.fen,
            "phase": int(position.phase), "source": int(position.source),
            "chunk": len(snapshots) + 1, "nodes": int(total_nodes),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "piece_count": chess.popcount(board.occupied),
            "legal_move_count": board.legal_moves.count(),
            "emitted_action": action, "chosen_uci": uci,
            "reference_best_cp": float(position.best_cp),
            "chosen_reference_cp": chosen_cp,
            "chosen_reference_listed": uci in position.move_cp,
            "regret_score": _score(position.best_cp) - _score(chosen_cp),
            "visit_gap": float(gap), "visit_entropy": entropy,
            "q_gap": q_gap, "root_q": root_q, "bestmove_flip": flip,
            "stable_chunks": stable, "q_drift": q_drift,
            "visit_churn": churn,
            "complexity_continue": _complexity(stable, float(gap), len(action_list)),
            "root_actions": action_list,
            "root_visits": [int(value) for value in visits],
            "root_child_q": q_values,
        })
        previous_q, previous_shares = root_q, share_map

    return on_chunk


def main() -> None:
    parser = argparse.ArgumentParser(prog="collect_varying_budget_trajectories")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    parser.add_argument("--matched-rows", type=Path)
    parser.add_argument("--require-game-groups", action="store_true")
    parser.add_argument("--max-positions", type=int, default=512)
    parser.add_argument("--syzygy-path", default=default_syzygy_path())
    parser.add_argument("--no-syzygy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/backtest/varying_budget_trajectories.jsonl"))
    args = parser.parse_args()
    if args.max_positions <= 0:
        raise SystemExit("--max-positions must be positive")
    syzygy_path = "" if args.no_syzygy else str(args.syzygy_path)
    if syzygy_path:
        try:
            require_tablebases(syzygy_path, what="varying-budget --syzygy-path")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    matched = args.matched_rows or default_matched_rows_path(args.audit_set)
    group_index = GroupIndex(matched) if matched.exists() else None
    if args.require_game_groups and (group_index is None or not group_index.canonical):
        raise SystemExit("--require-game-groups needs an enriched canonical matched index")
    positions = _stratified(load_audit_set(args.audit_set), args.seed)
    if args.require_game_groups:
        positions = [position for position in positions if group_for(position, group_index)[1] == "source_game"]
    if len(positions) < args.max_positions:
        raise SystemExit(f"only {len(positions)} eligible positions; requested {args.max_positions}")

    meta = Path(str(args.out) + ".meta.json")
    if (args.out.exists() or meta.exists()) and not args.resume:
        raise SystemExit("output exists; pass --resume or choose another path")
    complete = repair_resume_bank(args.out) if args.resume else set()
    previous = json.loads(meta.read_text()) if args.resume and meta.exists() else {}
    excluded = list(previous.get("excluded", []))
    excluded_keys = {str(row.get("key")) for row in excluded if isinstance(row, dict)}
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    checkpoint_file = checkpoint / "trainer.pt" if checkpoint.is_dir() else checkpoint
    if not checkpoint_file.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint_file}")
    if {args.out.resolve(), meta.resolve()} & {checkpoint_file, args.audit_set.resolve(), matched.resolve()}:
        raise SystemExit("output aliases a consumed input")

    options = EngineOptions()
    compile_mode = "max-autotune" if str(args.device).startswith("cuda") else None
    config = {
        "schema": SCHEMA, "git_sha": _git_sha(), "checkpoint": _stamp(checkpoint_file),
        "audit_set": _stamp(args.audit_set), "matched_rows": _stamp(matched) if matched.exists() else None,
        "device": str(args.device), "walkers": WALKERS,
        "leaf_gather": int(options.leaf_gather), "max_batch": int(options.max_batch),
        "hash_mb": int(options.hash_mb), "compile_mode": compile_mode,
        "production_shape": bool(str(args.device).startswith("cuda") and compile_mode == "max-autotune"),
        "syzygy_path": syzygy_path, "chunk_sims": CHUNK_SIMS,
        "max_chunks": MAX_CHUNKS, "requested_positions": int(args.max_positions),
        "seed": int(args.seed),
    }
    old = previous.get("config")
    if (
        isinstance(old, dict)
        and (
            {key: value for key, value in old.items() if key != "requested_positions"}
            != {key: value for key, value in config.items() if key != "requested_positions"}
            or int(args.max_positions) < int(old.get("requested_positions", 0))
        )
    ):
        raise SystemExit("resume configuration differs from the existing bank")
    progress = {"config": config, "model": previous.get("model"), "complete": False, "completed_positions": len(complete), "excluded": excluded}
    _atomic_json(meta, progress)
    if len(complete) >= args.max_positions:
        progress.update({"complete": True, "output": _stamp(args.out)})
        _atomic_json(meta, progress)
        return

    worker, model_info, _ = _build_worker(args.checkpoint, str(args.device), syzygy_path)
    progress["model"] = model_info
    _atomic_json(meta, progress)
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as output:
            for scanned, position in enumerate(positions, 1):
                if len(complete) >= args.max_positions:
                    break
                if position.key in complete or position.key in excluded_keys:
                    continue
                board = chess.Board(position.fen)
                legal_ucis, legal_actions = legal_full_indices(board)
                group_id, group_kind = group_for(position, group_index)
                worker.reset_tree()
                snapshots: list[dict[str, Any]] = []
                result = worker.run(
                    board, stop_event=threading.Event(), deadline=Deadline(deadline_ms=None),
                    max_nodes=MAX_CHUNKS * CHUNK_SIMS, optimum_ms=None,
                    allow_terminal_shortcuts=True,
                    on_chunk=_callback(worker, position, board, legal_ucis, legal_actions, group_id, group_kind, snapshots, time.perf_counter()),
                )
                expected = [CHUNK_SIMS * chunk for chunk in range(1, MAX_CHUNKS + 1)]
                if len(snapshots) != MAX_CHUNKS or [row["nodes"] for row in snapshots] != expected:
                    excluded.append({"key": position.key, "reason": "production_terminal_shortcut" if not snapshots and int(result.nodes) <= 1 else "incomplete_fixed_node_trajectory", "chunks_observed": len(snapshots)})
                    excluded_keys.add(position.key)
                else:
                    output.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in snapshots))
                    output.flush()
                    complete.add(position.key)
                progress.update({"completed_positions": len(complete), "excluded": excluded})
                if scanned % 10 == 0:
                    _atomic_json(meta, progress)
                    print(f"[varying-budget] {scanned} scanned; {len(complete)} complete", flush=True)
    finally:
        worker.close()
    progress.update({"complete": len(complete) >= args.max_positions, "completed_positions": len(complete), "excluded": excluded, "output": _stamp(args.out)})
    _atomic_json(meta, progress)
    if not progress["complete"]:
        raise SystemExit(f"only {len(complete)} complete trajectories were available")
    print(f"[varying-budget] bank -> {args.out}")
    print(f"[varying-budget] manifest -> {meta}")


if __name__ == "__main__":
    main()
