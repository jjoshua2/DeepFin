#!/usr/bin/env python3
"""Benchmark Stockfish pool sizes through the production-shaped slot pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_anti_engine.utils.engine_discovery import default_stockfish

# The publish/AOT defaults below name paths INSIDE the checkout, so they are
# derived from this file rather than written absolute — an absolute default
# names one machine's copy and silently misses on any other.
#
# ⚑ THE ENGINE IS NOT ONE OF THEM. `e2e_server/publish/` is UNTRACKED runtime
# output, so a checkout-relative engine path resolves in the one tree that
# published it and nowhere else — including the `git worktree` CLAUDE.md
# mandates for branch work. PR #441 converted this default from absolute to
# `_REPO / "e2e_server/..."` and reintroduced exactly the regression it had just
# fixed for `tests/stockfish_binary.py`. It goes through the shared discovery
# in `chess_anti_engine.utils.engine_discovery` instead, which is the ONE
# definition; a private copy here is how the copies drift.
_REPO = Path(__file__).resolve().parents[1]


class _MeasuredEvaluator:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self._lock = threading.Lock()
        self.positions = 0
        self.requests = 0

    @property
    def supports_input_bf16_bits(self) -> bool:
        return bool(getattr(self.inner, "supports_input_bf16_bits", False))

    @property
    def supports_legal_bf16(self) -> bool:
        return bool(getattr(self.inner, "supports_legal_bf16", True))

    @property
    def supports_compact_root_policy(self) -> bool:
        return bool(getattr(self.inner, "supports_compact_root_policy", False))

    def _record(self, rows: int) -> None:
        with self._lock:
            self.positions += rows
            self.requests += 1

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        out = self.inner.evaluate_encoded(x, relations=relations)
        self._record(int(x.shape[0]))
        return out

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        out = self.inner.evaluate_legal_bf16(x, legal_flat, legal_counts)
        self._record(int(x.shape[0]))
        return out

    def reset(self) -> None:
        with self._lock:
            self.positions = 0
            self.requests = 0


def _selfplay_configs(reco: dict[str, Any], max_plies: int) -> tuple[dict[str, Any], tuple[Any, ...]]:
    from chess_anti_engine.worker import WorkerSession

    session = WorkerSession.__new__(WorkerSession)
    session.args = SimpleNamespace()
    session.opening_book_path = None
    session.opening_book_path_2 = None
    session.opening_fen_list_path = None
    cfgs, sf_args = session._build_selfplay_configs(reco)
    cfgs["game"] = replace(
        cfgs["game"], max_plies=max_plies, blindspot_harvest_out_path="",
    )
    return cfgs, sf_args


def _worker(
    *, worker_index: int, slot_names: list[str], max_batch: int,
    reco: dict[str, Any], stockfish_path: str, sf_nice: int, threads: int,
    games_per_thread: int, max_plies: int, command_queue: Any,
    result_queue: Any,
) -> None:
    from chess_anti_engine.inference import MultiSlotInferenceClient
    from chess_anti_engine.selfplay import play_batch
    from chess_anti_engine.stockfish import StockfishPool

    logging.getLogger("chess_anti_engine.selfplay").setLevel(logging.ERROR)

    client = MultiSlotInferenceClient(
        slot_names=slot_names,
        max_batch=max_batch,
        input_planes=175,
        request_timeout_s=180.0,
    )
    evaluator = _MeasuredEvaluator(client)
    cfgs, sf_args = _selfplay_configs(reco, max_plies)
    sf_nodes, sf_multipv, sf_hash_mb, sf_syzygy_path = sf_args
    pools: dict[int, Any] = {}

    try:
        while True:
            command = command_queue.get()
            action = str(command["action"])
            if action == "close":
                break
            sf_workers = int(command["sf_workers"])
            if action == "prepare":
                if sf_workers not in pools:
                    pools[sf_workers] = StockfishPool(
                        path=stockfish_path,
                        nodes=int(sf_nodes),
                        num_workers=sf_workers,
                        multipv=int(sf_multipv),
                        hash_mb=int(sf_hash_mb),
                        syzygy_path=sf_syzygy_path,
                        nice=sf_nice,
                    )
                result_queue.put({
                    "kind": "prepared", "worker": worker_index,
                    "sf_workers": sf_workers,
                })
                continue
            if action != "run":
                raise ValueError(f"unknown action: {action}")

            start_at = float(command["start_at"])
            while time.monotonic() < start_at:
                time.sleep(0.001)
            evaluator.reset()
            seeds = [worker_index * 1_000_000 + i for i in range(threads)]

            def _run_thread(
                thread_index: int,
                *,
                current_pool: Any = pools[sf_workers],
                current_seeds: list[int] = seeds,
            ) -> Any:
                completed_games = 0

                def _completed(batch: Any) -> None:
                    nonlocal completed_games
                    completed_games += int(batch.games)

                return play_batch(
                    None,
                    device="cuda",
                    rng=np.random.default_rng(current_seeds[thread_index]),
                    stockfish=current_pool,
                    evaluator=evaluator,
                    games=games_per_thread,
                    on_game_complete=_completed,
                    stop_fn=lambda: completed_games >= games_per_thread,
                    **cfgs,
                )[1]

            started = time.monotonic()
            stats: list[Any] = []

            def _collect(index: int, current_stats: list[Any] = stats) -> None:
                current_stats.append(_run_thread(index))

            workers = [threading.Thread(target=_collect, args=(index,)) for index in range(threads)]
            for thread in workers:
                thread.start()
            for thread in workers:
                thread.join()
            elapsed = time.monotonic() - started
            result_queue.put({
                "kind": "result",
                "worker": worker_index,
                "sf_workers": sf_workers,
                "elapsed": elapsed,
                "games": sum(int(stat.games) for stat in stats),
                "inference_positions": evaluator.positions,
                "inference_requests": evaluator.requests,
            })
    finally:
        for pool in pools.values():
            pool.close()
        client.close()


def _prepare_publish(source: Path) -> Path:
    target = Path(tempfile.mkdtemp(prefix="cae-sf-pool-publish-"))
    shutil.copy2(source / "manifest.json", target / "manifest.json")
    os.symlink(source / "latest_model.pt", target / "latest_model.pt")
    return target


def _wait_for(
    queue: Any, *, kind: str, count: int, timeout_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout_s
    while len(rows) < count:
        try:
            row = queue.get(timeout=max(0.01, min(5.0, deadline - time.monotonic())))
        except Empty:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {kind}") from None
            continue
        if row.get("kind") != kind:
            raise RuntimeError(f"expected {kind}, received {row}")
        rows.append(row)
    return rows


def _parse_orders(raw: str) -> list[list[int]]:
    return [[int(value) for value in round_.split(",")] for round_ in raw.split(":")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--publish-dir",
        default=str(_REPO / "runs/pbt2_small/server/trials/4c17c_00000/publish"),
    )
    parser.add_argument("--aot-dir", default=str(_REPO / "data/aot_models_512"))
    parser.add_argument("--stockfish-path", default=default_stockfish())
    parser.add_argument("--orders", default="4,6,8:8,6,4")
    parser.add_argument("--max-plies", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--games-per-thread", type=int, default=2)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--max-batch", type=int, default=1190)
    parser.add_argument("--batch-wait-ms", type=float, default=20.0)
    parser.add_argument("--adaptive-idle-ms", type=float, default=4.0)
    parser.add_argument("--sf-nodes", type=int, default=0)
    parser.add_argument("--sf-multipv", type=int, default=0)
    parser.add_argument("--sf-nice", type=int, default=19)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)

    source = Path(args.publish_dir).resolve()
    manifest_data = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    reco = dict(manifest_data["recommended_worker"])
    if args.sf_nodes:
        reco["sf_nodes"] = int(args.sf_nodes)
    if args.sf_multipv:
        reco["sf_multipv"] = int(args.sf_multipv)
    publish = _prepare_publish(source)
    prefix = f"cae-sf-pool-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    broker_cmd = [
        sys.executable, "-m", "chess_anti_engine.inference",
        "--publish-dir", str(publish),
        "--device", "cuda",
        "--batch-wait-ms", str(args.batch_wait_ms),
        "--adaptive-idle-ms", str(args.adaptive_idle_ms),
        "--num-slots", str(int(args.workers) * int(args.slots)),
        "--max-batch-per-slot", str(args.max_batch),
        "--slot-prefix", prefix,
        "--input-planes", "175",
        "--aot-dir", str(args.aot_dir),
    ]
    broker = subprocess.Popen(broker_cmd, start_new_session=True)
    processes: list[mp.Process] = []
    command_queues: list[Any] = []
    result_queue: Any = mp.Queue()

    try:
        broker_manifest = publish / "broker_slots.json"
        deadline = time.monotonic() + 240.0
        while not broker_manifest.exists():
            if broker.poll() is not None:
                raise RuntimeError(f"broker exited with {broker.returncode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("broker startup timed out")
            time.sleep(0.05)

        for worker_index in range(int(args.workers)):
            command_queue: Any = mp.Queue()
            command_queues.append(command_queue)
            first = worker_index * int(args.slots)
            slot_names = [f"{prefix}-{i}" for i in range(first, first + int(args.slots))]
            process = mp.Process(
                target=_worker,
                kwargs={
                    "worker_index": worker_index,
                    "slot_names": slot_names,
                    "max_batch": int(args.max_batch),
                    "reco": reco,
                    "stockfish_path": str(args.stockfish_path),
                    "sf_nice": int(args.sf_nice),
                    "threads": int(args.threads),
                    "games_per_thread": int(args.games_per_thread),
                    "max_plies": int(args.max_plies),
                    "command_queue": command_queue,
                    "result_queue": result_queue,
                },
            )
            process.start()
            processes.append(process)

        prepared: set[int] = set()
        all_results: list[dict[str, Any]] = []
        for round_index, order in enumerate(_parse_orders(str(args.orders)), start=1):
            for sf_workers in order:
                if sf_workers not in prepared:
                    print(f"preparing sf={sf_workers}/worker ...", flush=True)
                    for queue in command_queues:
                        queue.put({"action": "prepare", "sf_workers": sf_workers})
                    _wait_for(
                        result_queue, kind="prepared", count=int(args.workers),
                        timeout_s=float(args.timeout_s),
                    )
                    prepared.add(sf_workers)

                start_at = time.monotonic() + 2.0
                for queue in command_queues:
                    queue.put({
                        "action": "run", "sf_workers": sf_workers,
                        "start_at": start_at,
                    })
                rows = _wait_for(
                    result_queue, kind="result", count=int(args.workers),
                    timeout_s=float(args.timeout_s),
                )
                elapsed = max(float(row["elapsed"]) for row in rows)
                result = {
                    "round": round_index,
                    "sf_workers": sf_workers,
                    "elapsed": elapsed,
                    "games_per_s": sum(int(row["games"]) for row in rows) / elapsed,
                    "inference_positions_per_s": (
                        sum(int(row["inference_positions"]) for row in rows) / elapsed
                    ),
                    "inference_requests_per_s": (
                        sum(int(row["inference_requests"]) for row in rows) / elapsed
                    ),
                    "games": sum(int(row["games"]) for row in rows),
                }
                all_results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)

        print(json.dumps(all_results, indent=2), flush=True)
        print("\nmedians:", flush=True)
        for sf_workers in sorted({int(row["sf_workers"]) for row in all_results}):
            selected = [row for row in all_results if int(row["sf_workers"]) == sf_workers]
            print(
                f"sf={sf_workers}/worker "
                f"games/s={statistics.median(float(row['games_per_s']) for row in selected):.3f} "
                f"inference_positions/s="
                f"{statistics.median(float(row['inference_positions_per_s']) for row in selected):.0f}",
                flush=True,
            )
    finally:
        for queue in command_queues:
            queue.put({"action": "close"})
        for process in processes:
            process.join(timeout=60.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10.0)
        if broker.poll() is None:
            os.killpg(broker.pid, signal.SIGTERM)
        try:
            broker.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            os.killpg(broker.pid, signal.SIGKILL)
            broker.wait(timeout=5.0)
        shutil.rmtree(publish, ignore_errors=True)


if __name__ == "__main__":
    main()
