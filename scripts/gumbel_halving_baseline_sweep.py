"""Measure whether a change to the C root-halving rule moves the search at all.

Banked instrument for the ``gss_score_and_halve`` root-baseline fix (the C used
to eliminate against the running ``W[root]/N[root]`` instead of the fresh
``root_qs``). It runs the C Gumbel path ONLY -- no Python-reference comparison,
because the two build different trees on positions where the C's proven-node
short-circuit fires -- and dumps every board's played action, root value and a
digest of its improved policy. Run it once per build and diff the dumps:

    # build A (e.g. origin/main), then:
    PYTHONPATH=. python3 scripts/gumbel_halving_baseline_sweep.py dump before.json
    # rebuild with the change, then:
    PYTHONPATH=. python3 scripts/gumbel_halving_baseline_sweep.py dump after.json
    PYTHONPATH=. python3 scripts/gumbel_halving_baseline_sweep.py compare before.json after.json

⚑ The evaluator is a deterministic hash of the encoded planes, NOT a net. That
makes the sweep reproducible and build-independent, and it means a null here is
"this rule change does not move the search shape", not "this rule change is
worth zero Elo". It is evidence about the SIZE of the intervention, not a
substitute for an arena readout.

Why a null is the expected result for a baseline change: ``raw_value`` enters
only through ``mixed_value = (raw + N*weighted_q) / (N + 1)``, so a change of
``d`` in the root value moves the mix by ``d / (N + 1)`` -- and it can only
change a RANKING if the mix is the min or max of the completed-Q vector. At
production's first halving round ``N`` is already 64 (topk 16 x 4 visits), so
the shift is ``d/65``, far inside the spread of the children's own Q values.
The measured 4,320-board-run null on that fix is exactly that arithmetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c

POLICY_SIZE = 4672

POSITION_SEEDS = (7, 13, 29)
EVALUATOR_SEEDS = (11, 22, 33)
# (simulations, topk, target_batch). Explicit tuples, not a cross product: the
# pairs are chosen to span the halving schedules that actually occur (vpa 0->1
# clamp at low sims, several rounds at production topk) plus both batching
# regimes -- target_batch=1 flushes per rep, 0 is production's cross-rep batch.
SHAPES = (
    (32, 16, 1),
    (64, 16, 1),
    (100, 16, 1),
    (200, 16, 1),
    (48, 8, 1),
    (24, 4, 1),
    (100, 16, 0),
    (256, 16, 0),
)
N_POSITIONS = 60


class HashEvaluator:
    """Deterministic in the encoded position, and identical across builds."""

    def __init__(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._pol = rng.standard_normal((4096, POLICY_SIZE)).astype(np.float32)
        self._wdl = rng.standard_normal((4096, 3)).astype(np.float32)

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations  # BatchEvaluator conformance
        n = int(x.shape[0])
        flat = np.ascontiguousarray(x, dtype=np.float32).reshape(n, -1)
        idx = np.array(
            [int.from_bytes(flat[i].tobytes()[:8], "little") % 4096 for i in range(n)],
            dtype=np.int64,
        )
        return self._pol[idx], self._wdl[idx]


def random_positions(n: int, *, plies: int, seed: int) -> list[chess.Board]:
    """Random-walk middlegames: varied, seeded, and free of a curated bias."""
    rng = np.random.default_rng(seed)
    out: list[chess.Board] = []
    while len(out) < n:
        board = chess.Board()
        ok = True
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                ok = False
                break
            board.push(moves[int(rng.integers(len(moves)))])
        if ok and not board.is_game_over() and board.legal_moves.count() >= 12:
            out.append(board)
    return out


def run_dump(out_path: Path, *, n_positions: int) -> int:
    rows: list[dict[str, object]] = []
    for pos_seed in POSITION_SEEDS:
        boards = random_positions(n_positions, plies=16, seed=pos_seed)
        for eval_seed in EVALUATOR_SEEDS:
            for sims, topk, target_batch in SHAPES:
                cfg = GumbelConfig(
                    simulations=sims, topk=topk, temperature=0.0,
                    add_noise=False, c_scale=0.1,
                )
                result = run_gumbel_root_many_c(
                    None, [b.copy() for b in boards], device="cpu",
                    rng=np.random.default_rng(0), cfg=cfg,
                    evaluator=HashEvaluator(eval_seed),
                    target_batch=target_batch, vloss_weight=0,
                )
                probs, actions, values = result[0], result[1], result[2]
                digest = hashlib.sha256(
                    b"".join(
                        np.asarray(p, dtype=np.float32).tobytes() for p in probs
                    ),
                ).hexdigest()
                rows.append({
                    "pos_seed": pos_seed, "eval_seed": eval_seed, "sims": sims,
                    "topk": topk, "target_batch": target_batch,
                    "actions": [int(a) for a in actions],
                    "values": [round(float(v), 12) for v in values],
                    "policy_sha256": digest,
                })
                print(
                    f"pos{pos_seed} eval{eval_seed} sims{sims} topk{topk} "
                    f"tb{target_batch}: {len(boards)} boards",
                    flush=True,
                )
    out_path.write_text(json.dumps(rows), encoding="utf-8")
    print(
        f"wrote {len(rows)} configs x {n_positions} positions = "
        f"{len(rows) * n_positions} board-runs -> {out_path}",
    )
    return 0


def run_compare(a_path: Path, b_path: Path) -> int:
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    if len(a) != len(b):
        print(f"MISMATCHED DUMPS: {len(a)} configs vs {len(b)}")
        return 2
    board_runs = act_diff = val_diff = pol_diff = 0
    for x, y in zip(a, b):
        board_runs += len(x["actions"])
        act_diff += sum(
            1 for p, q in zip(x["actions"], y["actions"]) if p != q
        )
        val_diff += sum(1 for p, q in zip(x["values"], y["values"]) if p != q)
        if x["policy_sha256"] != y["policy_sha256"]:
            pol_diff += 1
            print(
                "  differs: pos{pos_seed} eval{eval_seed} sims{sims} "
                "topk{topk} tb{target_batch}".format(**x),
            )
    print(
        f"configs={len(a)} board_runs={board_runs} action_diffs={act_diff} "
        f"value_diffs={val_diff} policy_digest_differing_configs={pol_diff}",
    )
    return 1 if (act_diff or val_diff or pol_diff) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dump = sub.add_parser("dump", help="run the sweep and write a JSON dump")
    dump.add_argument("out", type=Path)
    dump.add_argument("--positions", type=int, default=N_POSITIONS)
    cmp_ = sub.add_parser("compare", help="diff two dumps; exit 1 if they differ")
    cmp_.add_argument("before", type=Path)
    cmp_.add_argument("after", type=Path)
    args = parser.parse_args()
    if args.command == "dump":
        return run_dump(args.out, n_positions=int(args.positions))
    return run_compare(args.before, args.after)


if __name__ == "__main__":
    raise SystemExit(main())
