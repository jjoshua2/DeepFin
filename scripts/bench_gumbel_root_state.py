"""Benchmark native Gumbel root-state initialization."""
from __future__ import annotations

import argparse
import hashlib
import statistics
import time

import chess
import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.mcts._mcts_tree import MCTSTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boards", type=int, default=384)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()

    cb = CBoard.from_board(chess.Board())
    legal = cb.legal_move_indices().astype(np.int32, copy=False)
    topk = min(args.topk, int(legal.size))
    candidates = [int(action) for action in legal[:topk]]
    prior = np.linspace(0.1, 1.0, 4672, dtype=np.float64)
    gumbel = np.linspace(-1.0, 1.0, 4672, dtype=np.float64)
    tree = MCTSTree()
    root_id = tree.add_root(1, 0.0)
    tree.expand(
        root_id,
        legal,
        np.full(legal.size, 1.0 / legal.size, dtype=np.float64),
    )

    cboards = [cb] * args.boards
    root_ids = np.full(args.boards, root_id, dtype=np.int32)
    remaining = [candidates] * args.boards
    gumbels = [gumbel] * args.boards
    priors = [prior] * args.boards
    budgets = np.zeros(args.boards, dtype=np.int32)
    root_qs = np.zeros(args.boards, dtype=np.float64)
    enc_buf = np.empty((1, 175, 8, 8), dtype=np.float32)

    samples: list[float] = []
    digests: set[str] = set()
    for round_idx in range(args.rounds + 2):
        start = time.perf_counter()
        for _ in range(args.iterations):
            result = tree.start_gumbel_sims(
                cboards, root_ids, remaining, gumbels, priors, budgets, root_qs,
                0.1, 50.0, 2.5, 1.2, True, enc_buf, 0, 0, 0,
            )
            if result is not None:
                raise AssertionError(f"zero-budget search returned {result}")
        elapsed = time.perf_counter() - start
        digest = hashlib.sha256(repr(tree.get_gumbel_remaining()).encode()).hexdigest()
        digests.add(digest)
        if round_idx >= 2:
            samples.append(elapsed)

    if len(digests) != 1:
        raise AssertionError(f"unstable candidate state: {digests}")
    median = statistics.median(samples)
    dense_bytes = args.boards * 4672 * 8 * 2
    candidate_bytes = args.boards * topk * 8 * 2
    print(f"median_seconds={median:.9f}")
    print(f"iterations_per_second={args.iterations / median:.3f}")
    print(f"dense_prior_gumbel_bytes={dense_bytes}")
    print(f"candidate_prior_gumbel_bytes={candidate_bytes}")
    print(f"candidate_ratio={candidate_bytes / dense_bytes:.9f}")
    print(f"checksum={next(iter(digests))}")


if __name__ == "__main__":
    main()
