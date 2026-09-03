#!/usr/bin/env python3
"""Per-chunk search trajectory on the REAL accumulating tree, for the time-management
predictor: at chunk k, can we predict whether more chunks change/improve the move?

Runs the production SearchWorker path (walker PUCT at the shipped Threads=2 default;
classic Gumbel only when explicitly requested with --walkers 1) one position at a time
with abort disabled. It snapshots root state after every chunk via the run() on_chunk
hook — the node-horizon states the clock-free complexity predicate sees. Each row carries
the chosen move + its
deep-SF regret, plus the STATIC settledness (visit-gap, entropy, q-gap) and the DYNAMIC
"is the search still moving" signals (bestmove_flip / q_drift / visit_churn vs the
previous chunk). Downstream: label each chunk by whether the move changes / regret drops
in later chunks, and learn which features predict it — the abort/extend rule.

Unlike backtest_time_value.py (independent one-shot Gumbel per budget), this preserves
one accumulating tree. It does not contain a real clock or the production controller's
provable visit-lead branch, so a positive screen authorizes only a real-clock bank—not a
deployed controller. Slower (single board), so run a few hundred positions.

Usage:
  PYTHONPATH=. python3 scripts/backtest_chunk_trajectory.py --checkpoint <ckpt> \\
    --device cuda --chunk-sims 2048 --max-chunks 8 --max-positions 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.eval.audit import legal_full_indices, load_audit_set, move_regrets
from chess_anti_engine.eval.audit_history import MatchedAuditRows, default_matched_rows_path
from chess_anti_engine.mcts.gumbel import (
    PLAY_PUCT_DEFAULTS,
    PLAY_SEARCH_DEFAULTS,
    PLAY_SEARCH_TARGET_BATCH,
    PLAY_SEARCH_VLOSS_WEIGHT,
)
from chess_anti_engine.moves import index_to_move
from chess_anti_engine.uci.engine import EngineOptions
from chess_anti_engine.uci.search import (
    _ABORT_MIN_STABLE_CHUNKS,
    _ABORT_VISIT_GAP_MARGIN,
    _visit_gap,
)
from chess_anti_engine.utils.syzygy import default_syzygy_path, require_tablebases
from scripts.analyze_chunk_controller import _update_stability
from scripts.backtest_time_value import _score, _stratified

_SCHEMA = "deepfin.chunk_trajectory.v2"


def _entropy(shares: np.ndarray) -> float:
    p = shares[shares > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            [
                "git", "-C", str(repo_root), "status", "--porcelain",
                "--untracked-files=normal",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return sha, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _artifact(path: Path, *, require_file: bool) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if require_file and not resolved.is_file():
        raise SystemExit(f"decision-grade provenance requires a regular file: {resolved}")
    if not resolved.exists():
        raise SystemExit(f"artifact does not exist: {resolved}")
    stat = resolved.stat()
    out: dict[str, Any] = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if resolved.is_file():
        out["sha256"] = _sha256(resolved)
    return out


def _tablebase_inventory(path_value: str) -> dict[str, Any]:
    """Cheap, durable identity for the large production tablebase directories."""
    directories: list[dict[str, Any]] = []
    for raw_directory in path_value.split(os.pathsep):
        directory = Path(raw_directory.strip()).expanduser().resolve()
        digest = hashlib.sha256()
        counts = {"rtbw": 0, "rtbz": 0}
        total_bytes = 0
        files = sorted(
            entry for entry in directory.iterdir()
            if entry.is_file() and entry.suffix in (".rtbw", ".rtbz")
        )
        for entry in files:
            stat = entry.stat()
            counts[entry.suffix[1:]] += 1
            total_bytes += int(stat.st_size)
            digest.update(entry.name.encode())
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode())
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode())
            digest.update(b"\n")
        directories.append({
            "path": str(directory),
            "rtbw_count": counts["rtbw"],
            "rtbz_count": counts["rtbz"],
            "total_bytes": total_bytes,
            "inventory_sha256": digest.hexdigest(),
        })
    return {"path": path_value, "directories": directories}


def _checkpoint_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    candidate = resolved / "trainer.pt" if resolved.is_dir() else resolved
    if not candidate.is_file():
        raise SystemExit(f"checkpoint has no trainer.pt: {resolved}")
    return candidate


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp_path.open("x") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _require_search_take_effect(
    *,
    expected_mode: str,
    expected_workers: int,
    active_parameters: dict[str, float | int | bool],
    realized: dict[str, float | int | bool | str],
) -> None:
    """Fail closed if the worker did not install the requested live search shape."""
    failures: list[str] = []
    if realized.get("concurrency_mode") != expected_mode:
        failures.append(
            f"mode requested={expected_mode!r} realized="
            f"{realized.get('concurrency_mode')!r}"
        )
    if realized.get("concurrency_workers") != expected_workers:
        failures.append(
            f"workers requested={expected_workers!r} realized="
            f"{realized.get('concurrency_workers')!r}"
        )
    for name, requested in active_parameters.items():
        actual = realized.get(name)
        if actual != requested:
            failures.append(f"{name} requested={requested!r} realized={actual!r}")
    if failures:
        raise RuntimeError("search-path take-effect check failed: " + "; ".join(failures))


def _evaluator_stack_name(evaluator: Any) -> str:
    """Read the wrapper chain from the objects that will actually evaluate leaves."""
    name = type(evaluator).__name__
    if name == "BatchCoalescingDispatcher":
        return f"{name}({_evaluator_stack_name(evaluator._inner)})"
    if name == "ThreadSafeGPUDispatcher":
        return f"{name}({_evaluator_stack_name(evaluator._eval)})"
    return name


def main() -> None:
    ap = argparse.ArgumentParser(prog="backtest_chunk_trajectory")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    ap.add_argument("--chunk-sims", type=int, default=2048)
    ap.add_argument("--max-chunks", type=int, default=8, help="chunks per position (8 -> up to 16k sims)")
    ap.add_argument("--max-positions", type=int, default=200)
    ap.add_argument(
        "--c-scale", type=float, default=PLAY_SEARCH_DEFAULTS["c_scale"],
        help="classic-Gumbel only; rejected as inert when --walkers >1",
    )
    ap.add_argument(
        "--c-visit", type=float, default=PLAY_SEARCH_DEFAULTS["c_visit"],
        help="classic-Gumbel only; rejected as inert when --walkers >1",
    )
    ap.add_argument(
        "--c-visit-root", type=float, default=PLAY_SEARCH_DEFAULTS["c_visit_root"],
        help="classic-Gumbel only; rejected as inert when --walkers >1",
    )
    ap.add_argument("--c-scale-root", type=float, default=PLAY_SEARCH_DEFAULTS["c_scale_root"],
                    help="ROOT-ONLY c_scale (descent keeps --c-scale). Default matches the "
                         "production play search (root-log); <0 = legacy linear root.")
    ap.add_argument("--q-visit-exp-root", type=float, default=PLAY_SEARCH_DEFAULTS["q_visit_exp_root"],
                    help="ROOT-ONLY value-transform exponent. Default matches the production play "
                         "search (root-log); >=90 = legacy linear root.")
    ap.add_argument(
        "--gumbel-topk", type=int, default=PLAY_SEARCH_DEFAULTS["topk"],
        help="classic-Gumbel only; rejected as inert when --walkers >1",
    )
    ap.add_argument(
        "--walkers", type=int, default=EngineOptions().threads,
        help="walker threads; default is the production UCI Threads default",
    )
    ap.add_argument(
        "--matched-rows", type=Path, default=None,
        help="audit-to-game index (default: <audit-set>.matched_rows.npz)",
    )
    ap.add_argument(
        "--syzygy-path", default=default_syzygy_path(),
        help="production Syzygy directory pair; required for decision-grade banks",
    )
    ap.add_argument(
        "--methodology-smoke", action="store_true",
        help="allow missing game groups; output is stamped non-decision-grade",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/backtest/chunk_trajectory.jsonl"))
    args = ap.parse_args()

    if args.chunk_sims <= 0 or args.max_chunks < 2 or args.max_positions < 0:
        raise SystemExit("--chunk-sims must be >0, --max-chunks >=2, --max-positions >=0")
    if args.walkers <= 0:
        raise SystemExit("--walkers must be >0")
    if not args.methodology_smoke and not str(args.device).startswith("cuda"):
        raise SystemExit("decision-grade trajectory banks require the production CUDA path")
    if args.syzygy_path or not args.methodology_smoke:
        try:
            require_tablebases(args.syzygy_path, what="trajectory --syzygy-path")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    gumbel_defaults = {
        "c_scale": float(PLAY_SEARCH_DEFAULTS["c_scale"]),
        "c_visit": float(PLAY_SEARCH_DEFAULTS["c_visit"]),
        "c_visit_root": float(PLAY_SEARCH_DEFAULTS["c_visit_root"]),
        "c_scale_root": float(PLAY_SEARCH_DEFAULTS["c_scale_root"]),
        "q_visit_exp_root": float(PLAY_SEARCH_DEFAULTS["q_visit_exp_root"]),
        "gumbel_topk": int(PLAY_SEARCH_DEFAULTS["topk"]),
    }
    inert_overrides = [
        name for name, default in gumbel_defaults.items()
        if getattr(args, name) != default
    ]
    if int(args.walkers) > 1 and inert_overrides:
        raise SystemExit(
            "--walkers >1 selects production walker PUCT; these classic-Gumbel "
            "overrides would be inert: " + ", ".join(inert_overrides)
        )
    meta_path = Path(str(args.out) + ".meta.json")
    if not args.overwrite and (args.out.exists() or meta_path.exists()):
        raise SystemExit(f"refusing to overwrite {args.out} or {meta_path}; pass --overwrite")

    positions = _stratified(load_audit_set(args.audit_set), int(args.max_positions))
    matched_path = args.matched_rows or default_matched_rows_path(args.audit_set)
    matched: MatchedAuditRows | None = None
    if matched_path.exists():
        matched = MatchedAuditRows(matched_path)
        matched.require_index_layout()
    elif not args.methodology_smoke:
        raise SystemExit(
            f"decision-grade trajectory banks require game groups from {matched_path}; "
            "build it with scripts/match_audit_rows.py, or pass --methodology-smoke "
            "for a non-decision-grade pipeline check"
        )
    game_ids: dict[str, int | None] = {}
    source_dirs: dict[str, str | None] = {}
    source_shards: dict[str, str | None] = {}
    group_ids: dict[str, str | None] = {}
    for pos in positions:
        gid = matched.game_id(pos.key) if matched is not None and pos.key in matched else None
        source_dir = matched.snapshot if matched is not None and pos.key in matched else None
        source_shard = (
            matched.source_shard(pos.key)
            if matched is not None and pos.key in matched else None
        )
        if (
            (gid is None or gid < 0 or not source_dir or not source_shard)
            and not args.methodology_smoke
        ):
            raise SystemExit(
                "decision-grade trajectory bank lacks the full "
                f"(source_dir, shard, game_id) cluster for audit key {pos.key!r}"
            )
        game_ids[pos.key] = gid
        source_dirs[pos.key] = source_dir
        source_shards[pos.key] = source_shard
        group_ids[pos.key] = (
            None
            if gid is None or not source_dir or not source_shard
            else "\0".join((source_dir, source_shard, str(gid)))
        )

    checkpoint_path = _checkpoint_file(Path(args.checkpoint))
    producer_git_sha, producer_git_dirty = _git_state()
    if producer_git_dirty and not args.methodology_smoke:
        raise SystemExit(
            "decision-grade trajectory banks require a clean producer checkout; "
            "commit or stash changes, or pass --methodology-smoke"
        )
    provenance = {
        "schema": _SCHEMA,
        "decision_grade": not args.methodology_smoke,
        "analysis_scope": "fixed_node_horizons_only",
        "clock_conditioning_available": False,
        "root_position_history": "fen_only_from_audit_fen",
        "game_group_kind": "source_dir:shard:game_id",
        "complexity_predicate": {
            "kind": "clock_free_visit_gap_and_stability",
            "minimum_stable_chunks": int(_ABORT_MIN_STABLE_CHUNKS),
            "minimum_visit_gap": float(_ABORT_VISIT_GAP_MARGIN),
        },
        "producer_git_sha": producer_git_sha,
        "producer_git_dirty": producer_git_dirty,
        "producer_script": _artifact(Path(__file__), require_file=True),
        "checkpoint": _artifact(checkpoint_path, require_file=True),
        "audit_set": _artifact(args.audit_set, require_file=True),
        "matched_rows": (
            _artifact(matched_path, require_file=True) if matched_path.exists() else None
        ),
        "syzygy": (
            _tablebase_inventory(str(args.syzygy_path)) if args.syzygy_path else None
        ),
    }

    import torch

    from chess_anti_engine.inference import DirectGPUEvaluator
    from chess_anti_engine.inference_dispatcher import (
        BatchCoalescingDispatcher,
        ThreadSafeGPUDispatcher,
    )
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.mcts import _mcts_tree as mcts_extension
    from chess_anti_engine.mcts.gumbel_c import _REQUIRED_MCTS_ABI
    from chess_anti_engine.tablebase import SyzygyProbe
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint
    from chess_anti_engine.uci.search import SearchWorker
    from chess_anti_engine.uci.time_manager import Deadline

    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    pol_enc = str(getattr(model, "policy_encoding", "lc0_1858"))
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    engine_options = EngineOptions()
    cfg = GumbelConfig(
        simulations=int(args.chunk_sims), add_noise=False, temperature=0.0,
        input_history_encoding=hist, input_extra_features=extra, policy_encoding=pol_enc,
        compute_relations=use_rel, topk=int(args.gumbel_topk),
        c_scale=float(args.c_scale), c_visit=float(args.c_visit),
        c_visit_root=float(args.c_visit_root),
        c_scale_root=float(args.c_scale_root), q_visit_exp_root=float(args.q_visit_exp_root),
        policy_temp=float(PLAY_SEARCH_DEFAULTS["policy_temp"]),
        halving_div=int(engine_options.halving_div),
        c_puct=float(PLAY_PUCT_DEFAULTS["c_puct"]),
        cpuct_factor=float(PLAY_PUCT_DEFAULTS["cpuct_factor"]),
        cpuct_base=float(PLAY_PUCT_DEFAULTS["cpuct_base"]),
        fpu_reduction=float(PLAY_PUCT_DEFAULTS["fpu_reduction"]),
    )
    compact_bf16 = os.environ.get("CAE_UCI_COMPACT_BF16", "0") == "1"
    direct = DirectGPUEvaluator(
        model,
        device=args.device,
        max_batch=int(engine_options.max_batch),
        n_slots=2,
        input_bf16=compact_bf16,
        legal_bf16=compact_bf16,
    )
    thread_safe = ThreadSafeGPUDispatcher(direct)
    evaluator = (
        BatchCoalescingDispatcher(thread_safe, max_batch=int(engine_options.max_batch))
        if int(args.walkers) > 1 else thread_safe
    )
    worker = SearchWorker(
        evaluator, device=args.device,
        gumbel_cfg=cfg, chunk_sims=int(args.chunk_sims), n_walkers=int(args.walkers),
        vloss_weight=int(PLAY_SEARCH_VLOSS_WEIGHT),
        walker_gather=int(engine_options.leaf_gather),
    )
    worker.set_max_tree_mb(int(engine_options.hash_mb))
    worker.set_minibatch_size(int(PLAY_SEARCH_TARGET_BATCH))
    worker.set_root_noise_scale(float(engine_options.root_noise_scale))
    tb_probe = SyzygyProbe(str(args.syzygy_path)) if args.syzygy_path else None
    worker.set_tb_probe(tb_probe)
    concurrency_mode, concurrency_workers = worker.concurrency_profile()
    expected_mode = "walker_puct" if int(args.walkers) > 1 else "gumbel"
    active_parameters: dict[str, float | int | bool] = (
        {
            **{name: float(value) for name, value in PLAY_PUCT_DEFAULTS.items()},
            "vloss_weight": int(PLAY_SEARCH_VLOSS_WEIGHT),
            "walker_gather": int(engine_options.leaf_gather),
            "policy_temp": float(PLAY_SEARCH_DEFAULTS["policy_temp"]),
        }
        if concurrency_mode == "walker_puct"
        else {
            "c_scale": float(args.c_scale),
            "c_visit": float(args.c_visit),
            "c_visit_root": float(args.c_visit_root),
            "c_scale_root": float(args.c_scale_root),
            "q_visit_exp_root": float(args.q_visit_exp_root),
            "topk": int(args.gumbel_topk),
            "policy_temp": float(PLAY_SEARCH_DEFAULTS["policy_temp"]),
            "halving_div": int(engine_options.halving_div),
            "root_noise_scale": float(engine_options.root_noise_scale),
            "vloss_weight": int(PLAY_SEARCH_VLOSS_WEIGHT),
            "minibatch_size": int(PLAY_SEARCH_TARGET_BATCH),
        }
    )
    realized_search: dict[str, float | int | bool | str] = {
        **worker.realized_search_values(),
        "concurrency_mode": concurrency_mode,
        "concurrency_workers": concurrency_workers,
    }
    if concurrency_mode == "walker_puct":
        walker_pool = getattr(worker, "_walker_pool", None)
        pool_config = getattr(walker_pool, "_cfg", None)
        realized_search["walker_gather"] = int(getattr(pool_config, "gather", 0))
    expected_evaluator_stack = (
        "BatchCoalescingDispatcher(ThreadSafeGPUDispatcher(DirectGPUEvaluator))"
        if int(args.walkers) > 1
        else "ThreadSafeGPUDispatcher(DirectGPUEvaluator)"
    )
    realized_evaluator = {
        "stack": _evaluator_stack_name(evaluator),
        "direct_max_batch": int(direct._max_batch),
        "outer_max_batch": int(getattr(evaluator, "_max_batch", thread_safe.max_batch)),
        "n_slots": int(direct.n_slots),
        "input_bf16": bool(direct.supports_input_bf16_bits),
        "legal_bf16": bool(direct.supports_legal_bf16),
    }
    expected_evaluator = {
        "stack": expected_evaluator_stack,
        "direct_max_batch": int(engine_options.max_batch),
        "outer_max_batch": int(engine_options.max_batch),
        "n_slots": 2,
        "input_bf16": compact_bf16,
        "legal_bf16": compact_bf16,
    }
    if realized_evaluator != expected_evaluator:
        worker.close()
        raise RuntimeError(
            "evaluator take-effect check failed: requested "
            f"{expected_evaluator!r}, realized {realized_evaluator!r}"
        )
    try:
        _require_search_take_effect(
            expected_mode=expected_mode,
            expected_workers=int(args.walkers),
            active_parameters=active_parameters,
            realized=realized_search,
        )
    except RuntimeError:
        worker.close()
        raise
    print(f"[traj] {len(positions)} positions x {args.max_chunks} chunks of {args.chunk_sims} "
          f"(path={concurrency_mode} workers={concurrency_workers})",
          flush=True)
    import threading

    def action_to_uci(act: int, board: chess.Board, ucis: list[str]) -> tuple[str | None, int]:
        # The worker's own action->move decode (handles policy encoding + orientation);
        # then locate it in the position's legal-move order for the regret lookup.
        try:
            uci = index_to_move(int(act), board).uci()
        except Exception:
            return None, -1
        return (uci, ucis.index(uci)) if uci in ucis else (uci, -1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.out.with_name(f".{args.out.name}.tmp-{os.getpid()}")
    n_rows = 0
    completed_positions = 0
    excluded_positions: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        with tmp_path.open("x") as fh:
            for pi, pos in enumerate(positions):
                board = chess.Board(pos.fen)
                ucis, _ = legal_full_indices(board)
                regrets = move_regrets(pos, ucis)
                worker.reset_tree()
                snaps: list[dict[str, Any]] = []
                position_started = time.perf_counter()

            # Default-arg binding captures this iteration's loop vars (avoids
            # cell-var-from-loop); the list defaults are intentional, not a bug.
                def on_chunk(
                    total_nodes: int, _b=board, _u=ucis, _r=regrets, _s=snaps,
                    _t0=position_started, _best_cp=pos.best_cp,
                ) -> None:
                    actions, visits = worker._filtered_root_visits(None)
                    if actions.size == 0:
                        return
                    best = worker._emitted_action(actions, visits, None)
                    uci, li = action_to_uci(int(best), _b, _u)
                    tot = float(visits.sum())
                    shares = (
                        visits.astype(np.float64) / tot
                        if tot > 0 else visits.astype(np.float64)
                    )
                    ngap = _visit_gap(actions, visits, int(best))
                    rq, qg = 0.0, float("nan")
                    tree, rid = worker._tree, worker._root_id
                    if tree is not None and rid is not None:
                        rq = float(tree.node_q(rid))
                        ca, _cv, cq = tree.get_children_q(rid, rq)
                        if ca.size >= 2 and best in ca.tolist():
                            bm = ca == best
                            oth = cq[~bm]
                            qg = float(cq[bm].max() - oth.max()) if oth.size else float("inf")
                    regret_cp = float(_r[li]) if li >= 0 else float(_r.max())
                    _s.append({
                        "nodes": total_nodes, "elapsed_ms": (time.perf_counter() - _t0) * 1000.0,
                        "emitted_action": int(best), "uci": uci,
                        "regret_cp": regret_cp,
                        "regret_score": _score(_best_cp) - _score(_best_cp - regret_cp),
                        "visit_gap": ngap, "visit_entropy": _entropy(shares),
                        "q_gap": qg, "root_q": rq,
                        "actions": [int(a) for a in actions],
                        "shares": {int(a): float(s) for a, s in zip(actions, shares)},
                    })

                worker.run(
                    board, stop_event=threading.Event(), deadline=Deadline(deadline_ms=None),
                    max_nodes=int(args.max_chunks) * int(args.chunk_sims), optimum_ms=None,
                    allow_terminal_shortcuts=False, on_chunk=on_chunk,
                )
                if len(snaps) != int(args.max_chunks):
                    excluded_positions.append({
                        "key": pos.key,
                        "chunks_observed": len(snaps),
                        "chunks_required": int(args.max_chunks),
                        "reason": "incomplete_or_terminal_search",
                    })
                    continue
                final_uci = snaps[-1]["uci"]
                final_regret = float(snaps[-1]["regret_cp"])
                abort_last_best = -1
                stable_chunks = 0
                for k, s in enumerate(snaps):
                    prev = snaps[k - 1] if k > 0 else None
                    flip = bool(prev is not None and s["uci"] != prev["uci"])
                    abort_last_best, stable_chunks = _update_stability(
                        abort_last_best,
                        stable_chunks,
                        emitted_action=int(s["emitted_action"]),
                        visit_gap=float(s["visit_gap"]),
                        action_count=len(s["actions"]),
                    )
                    qdrift = abs(float(s["root_q"]) - float(prev["root_q"])) if prev else None
                    churn = None
                    if prev is not None:
                        keys = set(s["shares"]) | set(prev["shares"])
                        churn = 0.5 * sum(
                            abs(s["shares"].get(a, 0.0) - prev["shares"].get(a, 0.0))
                            for a in keys
                        )
                    row = {
                        "schema": _SCHEMA,
                        "key": pos.key,
                        "source_dir": source_dirs[pos.key],
                        "shard": source_shards[pos.key],
                        "game_id": game_ids[pos.key],
                        "group_id": group_ids[pos.key],
                        "phase": pos.phase, "source": pos.source,
                        "piece_count": chess.popcount(board.occupied),
                        "legal_move_count": board.legal_moves.count(), "chunk": k + 1,
                        "nodes": s["nodes"], "elapsed_ms": s["elapsed_ms"],
                        "emitted_action": s["emitted_action"], "uci": s["uci"],
                        "regret_cp": s["regret_cp"], "regret_score": s["regret_score"],
                        "visit_gap": s["visit_gap"], "visit_entropy": s["visit_entropy"],
                        "q_gap": (
                            None if isinstance(s["q_gap"], float) and math.isnan(s["q_gap"])
                            else s["q_gap"]
                        ),
                        "root_q": s["root_q"], "root_actions": s["actions"],
                        "root_visit_shares": [s["shares"][a] for a in s["actions"]],
                        "bestmove_flip": flip, "stable_chunks": stable_chunks,
                        "complexity_predicate_continue": not (
                            stable_chunks >= _ABORT_MIN_STABLE_CHUNKS
                            and float(s["visit_gap"]) >= _ABORT_VISIT_GAP_MARGIN
                        ),
                        "q_drift": qdrift, "visit_churn": churn,
                        "changes_to_final": bool(s["uci"] != final_uci),
                        "regret_vs_final_cp": float(s["regret_cp"]) - final_regret,
                    }
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
                    n_rows += 1
                completed_positions += 1
                if (pi + 1) % 25 == 0:
                    print(f"[traj] {pi + 1}/{len(positions)}", flush=True)
                    if str(args.device).startswith("cuda"):
                        torch.cuda.empty_cache()
        os.replace(tmp_path, args.out)
    finally:
        worker.close()
        if tmp_path.exists():
            tmp_path.unlink()

    manifest = {
        **provenance,
        "complete": True,
        "row_count": n_rows,
        "position_count": completed_positions,
        "requested_position_count": len(positions),
        "excluded_position_count": len(excluded_positions),
        "excluded_positions": excluded_positions,
        "chunk_count": int(args.max_chunks),
        "runtime_seconds": time.perf_counter() - started,
        "output": _artifact(args.out, require_file=True),
        "requested_search": {
            "device": str(args.device), "chunk_sims": int(args.chunk_sims),
            "max_chunks": int(args.max_chunks), "walkers": int(args.walkers),
            "active_path": concurrency_mode,
            "active_parameters": active_parameters,
        },
        "realized_search": realized_search,
        "requested_evaluator": expected_evaluator,
        "realized_evaluator": realized_evaluator,
        "realized_tablebase": {
            "installed": tb_probe is not None,
            "cursed_as_draw": True,
            "n_wdl": int(tb_probe.n_wdl) if tb_probe is not None else 0,
            "n_dtz": int(tb_probe.n_dtz) if tb_probe is not None else 0,
            "max_pieces": int(tb_probe.max_pieces) if tb_probe is not None else 0,
        },
        "mcts_extension": {
            **_artifact(Path(mcts_extension.__file__), require_file=True),
            "abi_version": int(getattr(mcts_extension, "ABI_VERSION", 0)),
            "required_abi_version": int(_REQUIRED_MCTS_ABI),
        },
    }
    _write_json_atomic(meta_path, manifest)

    print(f"[traj] wrote {n_rows} rows -> {args.out}")
    print(f"[traj] provenance -> {meta_path}")


if __name__ == "__main__":
    main()
