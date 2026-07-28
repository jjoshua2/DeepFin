"""Profile production-shaped Python-to-native conversion candidates.

This benchmark deliberately removes model, Stockfish, and filesystem latency so
the remaining CPU orchestration is visible. It compares the production Gumbel C
fast path's Python glue with disk-replay batch sampling and reports project-Python
self time from cProfile.
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import logging
import pstats
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import chess
import numpy as np
import chess_anti_engine.selfplay.finalize as finalize_module

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.lc0 import LC0_HISTORY_ROOT_LEGACY_META
from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.moves import (
    COMPACT_POLICY_SIZE,
    COMPACT_TO_FULL_POLICY,
    POLICY_SIZE,
    POLICY_ENCODING_LC0_1858,
)
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.selfplay.finalize import _build_replay_samples
from chess_anti_engine.selfplay.state import _NetRecord
from chess_anti_engine.train.targets import hlgauss_target


class _ZeroEvaluator:
    """Allocation-free evaluator exposing search-side CPU overhead."""

    supports_input_bf16_bits = True

    def __init__(self, max_batch: int) -> None:
        self._policy = np.zeros((max_batch, COMPACT_POLICY_SIZE), dtype=np.float32)
        self._legal_policy_bf16 = np.zeros((max_batch * 64,), dtype=np.uint16)
        self._wdl = np.zeros((max_batch, 3), dtype=np.float32)

    def evaluate_encoded(
        self,
        x: np.ndarray,
        relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        n = int(x.shape[0])
        if n > self._policy.shape[0]:
            raise ValueError(f"benchmark evaluator capacity {self._policy.shape[0]} < {n}")
        return self._policy[:n], self._wdl[:n]

    def evaluate_legal_bf16(
        self,
        x: np.ndarray,
        legal_flat: np.ndarray,
        legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(x.shape[0])
        n_legal = int(legal_flat.shape[0])
        if n > self._wdl.shape[0] or n_legal > self._legal_policy_bf16.shape[0]:
            raise ValueError("benchmark evaluator legal-output capacity exceeded")
        if int(np.asarray(legal_counts).sum()) != n_legal:
            raise ValueError("legal_counts do not sum to legal_flat length")
        return self._legal_policy_bf16[:n_legal], self._wdl[:n]


class _ProfileCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("gumbel profile"):
            self.messages.append(message)


def _make_boards(n: int) -> tuple[list[chess.Board], list[CBoard]]:
    rng = np.random.default_rng(20260710)
    boards: list[chess.Board] = []
    for i in range(n):
        board = chess.Board()
        target_plies = 8 + (i % 57)
        for _ in range(target_plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[int(rng.integers(0, len(moves)))])
            if board.is_game_over():
                break
        if board.is_game_over():
            board = chess.Board()
        boards.append(board)
    return boards, [CBoard.from_board(board) for board in boards]


def _hash_value(value: Any) -> str:
    digest = hashlib.sha256()

    def _add(item: Any) -> None:
        if isinstance(item, np.ndarray):
            digest.update(str(item.dtype).encode())
            digest.update(str(item.shape).encode())
            digest.update(item.tobytes(order="C"))
        elif isinstance(item, (list, tuple)):
            for child in item:
                _add(child)
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode())
                _add(item[key])
        elif isinstance(item, (int, float, bool)):
            digest.update(repr(item).encode())

    _add(value)
    return digest.hexdigest()[:16]


def _profile_project_python(fn: Callable[[], Any]) -> dict[str, Any]:
    profiler = cProfile.Profile()
    profiler.enable()
    value = fn()
    profiler.disable()
    stats = pstats.Stats(profiler)
    raw_stats = cast(Any, stats).stats
    total_self = float(sum(row[2] for row in raw_stats.values()))
    project_rows: list[tuple[float, str, int, str]] = []
    for (filename, line, name), row in raw_stats.items():
        own = float(row[2])
        if own > 0.0 and "/chess_anti_engine/" in filename and filename.endswith(".py"):
            project_rows.append((own, filename, line, name))
    project_rows.sort(reverse=True)
    project_self = float(sum(row[0] for row in project_rows))
    del value
    gc.collect()
    return {
        "profile_total_s": total_self,
        "project_python_self_s": project_self,
        "project_python_fraction": project_self / max(total_self, 1e-12),
        "top_project_python": [
            {
                "self_ms": own * 1000.0,
                "function": f"{Path(filename).name}:{line}:{name}",
            }
            for own, filename, line, name in project_rows[:8]
        ],
    }


def _timed_repeats(
    fn: Callable[[], Any], repeats: int,
) -> tuple[float, list[str]]:
    times: list[float] = []
    hashes: list[str] = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - start)
        hashes.append(_hash_value(value))
        del value
        gc.collect()
    return float(np.median(np.asarray(times))), hashes


def _bench_gumbel(boards: int, simulations: int, repeats: int) -> dict[str, Any]:
    py_boards, cboards = _make_boards(boards)
    evaluator = _ZeroEvaluator(max_batch=max(512, boards * 32))
    cfg = GumbelConfig(
        simulations=simulations,
        topk=16,
        temperature=0.0,
        input_history_encoding=LC0_HISTORY_ROOT_LEGACY_META,
        input_extra_features="v2_threats",
        policy_encoding=POLICY_ENCODING_LC0_1858,
        gumbel_scale=0.75,
    )

    logger = logging.getLogger("chess_anti_engine.mcts.gumbel_c")
    capture = _ProfileCapture()
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(capture)

    def _run() -> Any:
        result = run_gumbel_root_many_c(
            None,
            py_boards,
            device="cpu",
            rng=np.random.default_rng(42),
            cfg=cfg,
            evaluator=evaluator,
            cboards=cboards,
            allow_terminal_root_shortcuts=False,
        )
        return result[:4]

    try:
        _run()
        capture.messages.clear()
        median_s, hashes = _timed_repeats(_run, repeats)
        profile = _profile_project_python(_run)
    finally:
        logger.removeHandler(capture)
        logger.setLevel(old_level)

    return {
        "median_s": median_s,
        "hash": hashes[0],
        "hashes_stable": len(set(hashes)) == 1,
        "last_phase_profile": capture.messages[-1] if capture.messages else "",
        **profile,
    }


def _synthetic_replay_chunk(
    n: int,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    policy = np.zeros((n, COMPACT_POLICY_SIZE), dtype=np.float16)
    cols = rng.integers(0, COMPACT_POLICY_SIZE, size=(n, 16))
    rows = np.arange(n, dtype=np.int64)[:, None]
    policy[rows, cols] = np.float16(1.0 / 16.0)
    return {
        "x": np.zeros((n, 175, 8, 8), dtype=np.float16),
        "policy_target": policy,
        "wdl_target": rng.integers(0, 3, size=n, dtype=np.int8),
        "priority": rng.random(n, dtype=np.float32) + np.float32(0.01),
        "has_policy": np.ones(n, dtype=np.uint8),
    }


def _bench_replay(rows: int, batch_size: int, repeats: int) -> dict[str, Any]:
    chunk_rows = min(500, rows)
    with tempfile.TemporaryDirectory(prefix="cae-native-profile-") as tmp:
        buf = DiskReplayBuffer(
            capacity=rows,
            shard_dir=Path(tmp),
            rng=np.random.default_rng(1),
            # Writes synthetic chunks (`add_many_arrays` below) into a
            # TemporaryDirectory this function owns.
            read_only=False,
            shuffle_cap=rows,
            shard_size=rows + 1,
            refresh_interval=0,
            input_planes=175,
        )
        try:
            remaining = rows
            chunk_id = 0
            while remaining > 0:
                n = min(chunk_rows, remaining)
                buf.add_many_arrays(_synthetic_replay_chunk(n, seed=100 + chunk_id))
                remaining -= n
                chunk_id += 1

            def _run() -> Any:
                buf.rng = np.random.default_rng(42)
                return buf.sample_batch_arrays(batch_size)

            _run()
            median_s, hashes = _timed_repeats(_run, repeats)
            profile = _profile_project_python(_run)
        finally:
            buf.close()

    return {
        "median_s": median_s,
        "hash": hashes[0],
        "hashes_stable": len(set(hashes)) == 1,
        **profile,
    }


def _bench_finalize(records_n: int, repeats: int) -> dict[str, Any]:
    full_moves = np.asarray(COMPACT_TO_FULL_POLICY[:40], dtype=np.int64)
    policy = np.zeros((POLICY_SIZE,), dtype=np.float32)
    policy[full_moves] = np.float32(1.0 / float(full_moves.size))
    legal_mask = np.zeros((POLICY_SIZE,), dtype=np.uint8)
    legal_mask[full_moves] = 1
    x = np.zeros((175, 8, 8), dtype=np.float32)
    wdl = np.asarray([0.45, 0.25, 0.30], dtype=np.float32)
    raw = np.zeros((40, 5), dtype=np.int16)
    raw[:, 0] = full_moves.astype(np.int16)
    raw[:, 1] = np.arange(200, 160, -1, dtype=np.int16)
    raw[:, 3] = np.arange(600, 560, -1, dtype=np.int16)
    raw[:, 4] = np.int16(200)

    records: list[_NetRecord] = []
    for ply in range(records_n):
        rec = _NetRecord(
            x=x,
            policy_probs=policy,
            net_wdl_est=wdl,
            search_wdl_est=wdl,
            pov_color=chess.WHITE if ply % 2 == 0 else chess.BLACK,
            ply_index=ply,
            has_policy=True,
            priority=1.0,
            sample_weight=1.0,
            keep_prob=1.0,
            legal_mask=legal_mask,
            sf_policy_target=policy,
            sf_move_index=int(full_moves[0]),
            sf_wdl=wdl,
            x_lc0_root=x,
        )
        rec.sf_multipv_raw = raw
        rec.sf_label_meta = np.asarray([700_000, 25, 100, 0, 600, 200], dtype=np.int32)
        rec.sf_played_move_index = int(full_moves[1])
        rec.sf_played_rank = 2
        rec.sf_played_regret = 0.05
        rec.sf_legal_mask = legal_mask
        records.append(rec)

    state = cast(Any, SimpleNamespace(
        selfplay_arr=[True],
        starting_boards=[chess.Board()],
        opening_source_arr=["profile"],
        move_idx_history=[full_moves[np.arange(records_n) % full_moves.size].tolist()],
        rng=np.random.default_rng(1),
        game=SimpleNamespace(
            categorical_bins=32,
            hlgauss_sigma=0.04,
            categorical_blend_frac=0.0,
            categorical_search_blend_frac=0.0,
            max_plies=450,
            policy_encoding=POLICY_ENCODING_LC0_1858,
            soft_policy_temp=3.0,
            input_history_encoding=LC0_HISTORY_ROOT_LEGACY_META,
            history_rep_fix=True,
            record_fast_ply_value=False,
            record_dense_sf_policy=False,
            record_sf_p0_policy=True,
            record_sf_p0_regret=True,
        ),
    ))
    ply_to_index = {int(rec.ply_index): idx for idx, rec in enumerate(records)}
    vol: list[np.ndarray | None] = [
        np.asarray([0.01, 0.02, 0.01], dtype=np.float32) for _ in records
    ]

    def _run() -> Any:
        state.rng = np.random.default_rng(1)
        samples = _build_replay_samples(
            state,
            0,
            records,
            result="1-0",
            tb_policy_overrides={},
            vol_targets=vol,
            sf_vol_targets=vol,
            total_plies_played=records_n,
            ply_to_index=ply_to_index,
        )
        return (
            len(samples),
            [sample.game_id for sample in samples],
            [sample.policy_target for sample in samples],
            [sample.categorical_target for sample in samples],
            [sample.sf_p0_regret for sample in samples],
        )

    _run()
    median_s, hashes = _timed_repeats(_run, repeats)
    profile = _profile_project_python(_run)
    return {
        "median_s": median_s,
        "hash": hashes[0],
        "hashes_stable": len(set(hashes)) == 1,
        **profile,
    }


def _print_result(name: str, result: dict[str, Any]) -> None:
    median_s = float(result["median_s"])
    fraction = float(result["project_python_fraction"])
    python_upper_s = median_s * fraction
    qualifies = fraction >= 0.10 and python_upper_s >= 0.010
    print(f"\n{name}")
    print(f"  median wall: {median_s * 1000.0:.3f} ms")
    print(f"  project Python self: {fraction * 100.0:.1f}%")
    print(f"  Python wall upper bound: {python_upper_s * 1000.0:.3f} ms")
    print(f"  stable hash: {result['hashes_stable']} ({result['hash']})")
    print(f"  conversion threshold: {'PASS' if qualifies else 'FAIL'}")
    phase = str(result.get("last_phase_profile", ""))
    if phase:
        print(f"  {phase}")
    print("  top project Python self-time:")
    for row in result["top_project_python"]:
        print(f"    {float(row['self_ms']):9.3f} ms  {row['function']}")


def _print_finalize_result(result: dict[str, Any], records_n: int) -> None:
    median_s = float(result["median_s"])
    fraction = float(result["project_python_fraction"])
    search_proxy_s = records_n * 1.210975 / 384.0
    share = median_s / search_proxy_s
    qualifies = median_s >= 0.010 and fraction >= 0.50 and share >= 0.03
    print("\nPer-game replay finalization")
    print(f"  median wall: {median_s * 1000.0:.3f} ms ({records_n} records)")
    print(f"  project Python self: {fraction * 100.0:.1f}%")
    print(f"  CPU-only search-proxy share: {share * 100.0:.2f}%")
    print(f"  stable hash: {result['hashes_stable']} ({result['hash']})")
    print(f"  conversion threshold: {'PASS' if qualifies else 'FAIL'}")
    print("  top project Python self-time:")
    for row in result["top_project_python"]:
        print(f"    {float(row['self_ms']):9.3f} ms  {row['function']}")


def _uncached_finalization_hlgauss_target(
    value: float, *, num_bins: int, sigma: float,
) -> np.ndarray:
    return hlgauss_target(value, num_bins=num_bins, sigma=sigma)


def _compare_finalize_cache(records_n: int, repeats: int, paired_rounds: int) -> None:
    cached_builder = finalize_module._finalization_hlgauss_target
    uncached: list[float] = []
    cached: list[float] = []
    hashes: set[str] = set()
    stable = True
    try:
        for round_idx in range(paired_rounds):
            order = (False, True) if round_idx % 2 == 0 else (True, False)
            for cache_enabled in order:
                finalize_module._finalization_hlgauss_target = (
                    cached_builder if cache_enabled else _uncached_finalization_hlgauss_target
                )
                result = _bench_finalize(records_n, repeats)
                (cached if cache_enabled else uncached).append(float(result["median_s"]))
                hashes.add(str(result["hash"]))
                stable = stable and bool(result["hashes_stable"])
    finally:
        finalize_module._finalization_hlgauss_target = cached_builder

    uncached_median = float(np.median(np.asarray(uncached, dtype=np.float64)))
    cached_median = float(np.median(np.asarray(cached, dtype=np.float64)))
    speedup = uncached_median / max(cached_median, 1e-12)
    print("\nPaired ternary HL-Gauss cache comparison")
    print(f"  uncached median: {uncached_median * 1000.0:.3f} ms")
    print(f"  cached median: {cached_median * 1000.0:.3f} ms")
    print(f"  cached speedup: {speedup:.6f}x")
    print(f"  hashes stable/equal: {stable and len(hashes) == 1} ({', '.join(sorted(hashes))})")


def _compare_native_finalize(records_n: int, repeats: int, paired_rounds: int) -> None:
    native_builder = finalize_module._prepare_sf_multipv_native
    python_builder = finalize_module._prepare_sf_multipv_python
    python_times: list[float] = []
    native_times: list[float] = []
    hashes: set[str] = set()
    stable = True
    try:
        for round_idx in range(paired_rounds):
            order = (False, True) if round_idx % 2 == 0 else (True, False)
            for native_enabled in order:
                finalize_module._prepare_sf_multipv_for_finalize = (
                    native_builder if native_enabled else python_builder
                )
                result = _bench_finalize(records_n, repeats)
                (native_times if native_enabled else python_times).append(
                    float(result["median_s"]),
                )
                hashes.add(str(result["hash"]))
                stable = stable and bool(result["hashes_stable"])
    finally:
        finalize_module._prepare_sf_multipv_for_finalize = native_builder

    python_median = float(np.median(np.asarray(python_times, dtype=np.float64)))
    native_median = float(np.median(np.asarray(native_times, dtype=np.float64)))
    speedup = python_median / max(native_median, 1e-12)
    print("\nPaired native SF-finalization comparison")
    print(f"  Python median: {python_median * 1000.0:.3f} ms")
    print(f"  native median: {native_median * 1000.0:.3f} ms")
    print(f"  native speedup: {speedup:.6f}x")
    print(f"  hashes stable/equal: {stable and len(hashes) == 1} ({', '.join(sorted(hashes))})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boards", type=int, default=384)
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--replay-rows", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--finalize-records", type=int, default=0)
    parser.add_argument("--only-finalize", action="store_true")
    parser.add_argument("--compare-finalize-cache", action="store_true")
    parser.add_argument("--compare-native-finalize", action="store_true")
    parser.add_argument("--paired-rounds", type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.paired_rounds <= 0:
        raise SystemExit("--paired-rounds must be positive")
    if args.only_finalize and args.finalize_records <= 0:
        raise SystemExit("--only-finalize requires --finalize-records > 0")
    if args.compare_finalize_cache and args.compare_native_finalize:
        raise SystemExit("choose only one finalization comparison")
    if not args.only_finalize and min(args.boards, args.simulations, args.replay_rows, args.batch_size) <= 0:
        raise SystemExit("all numeric arguments must be positive")

    if not args.only_finalize:
        gumbel = _bench_gumbel(args.boards, args.simulations, args.repeats)
        replay = _bench_replay(args.replay_rows, args.batch_size, args.repeats)
        _print_result("Gumbel C search Python boundary", gumbel)
        _print_result("Disk replay sampling", replay)
    if args.finalize_records > 0:
        if args.compare_native_finalize:
            _compare_native_finalize(args.finalize_records, args.repeats, args.paired_rounds)
        elif args.compare_finalize_cache:
            _compare_finalize_cache(args.finalize_records, args.repeats, args.paired_rounds)
        else:
            finalize = _bench_finalize(args.finalize_records, args.repeats)
            _print_finalize_result(finalize, args.finalize_records)


if __name__ == "__main__":
    main()
