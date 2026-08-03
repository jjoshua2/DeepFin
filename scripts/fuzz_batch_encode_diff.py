"""Differential fuzzer: _mcts_tree batch encoders vs the single-board oracle.

The production inference path encodes positions through the batch encoders in
``chess_anti_engine/mcts/_mcts_tree.c`` (``batch_encode_146*`` for selfplay
network turns, ``batch_compute_relations`` for the dynamic-relations bet).
Those share the pure-C encode implementation with ``_lc0_ext``, but each .so
carries its own copies of the buffer-layout loops and module globals (e.g. the
``history_rep_fix`` flag), so batch/single parity is an invariant worth
fuzzing, not an identity.

Plays random games and, every ``--check-every`` plies, snapshots a batch of
boards with deliberately mixed provenance (push-built, ``from_board``
reconstructions, copies of earlier positions) and asserts:

  - each ``batch_encode_146{,_lc0_root,_lc0_root_legacy_meta}`` row is
    bit-identical to ``encode_cboard`` for that board (the single-board path,
    itself verified bit-identical to python-chess by fuzz_cboard_diff.py);
  - each bf16 variant equals the fp32 batch output converted with the same
    round-to-nearest-even rule the C uses;
  - each ``batch_compute_relations`` row equals the python
    ``relation_matrices`` oracle.

Both ``history_rep_fix`` phases are exercised (the flag is applied before any
board in the phase is constructed, per the rep_fix ordering contract), at the
production plane count (``--extra-features``, default ``v2_threats`` = 175).

Output buffers are poisoned (NaN / sentinel bytes) before every batch call, so
a row or plane the C side fails to write reads as a divergence instead of
accidentally matching a zero init (zero is the correct value for most cells).

Residual gap: the tree-internal leaf encodes (``start_gumbel_sims``,
``batch_process_ply``) reach the shared kernel via ``cboard_encode_planes_into``
(``feat_prezeroed=0``), whose feature-plane memset branch the public batch
wrappers (``feat_prezeroed=1``) skip. That single branch is the only encode
code this fuzzer cannot reach; everything downstream of it is covered.

Run under the sanitized extension build for memory/UB coverage via
``scripts/fuzz/run_fuzz.sh batch [games]``. Exits non-zero with the UCI move
list needed to reproduce the first divergence.
"""
from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass

import chess
import numpy as np

from chess_anti_engine.encoding import input_plane_count, rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.encoding.features import relation_matrices
from chess_anti_engine.mcts._mcts_tree import (
    batch_compute_relations,
    batch_encode_146,
    batch_encode_146_bf16,
    batch_encode_146_lc0_root,
    batch_encode_146_lc0_root_bf16,
    batch_encode_146_lc0_root_legacy_meta,
    batch_encode_146_lc0_root_legacy_meta_bf16,
)
from chess_anti_engine.moves import move_to_index

# Production feature version, from configs/pbt2_small.yaml:71 -> 175 planes.
# This fuzzer used to pin "v1" (146) with hardcoded 146-wide buffers, so the 29
# v2_threats planes had never been through the batch differential at all
# (encoding audit E1). Overridable via --extra-features.
PROD_EXTRA_FEATURES = "v2_threats"

# (history mode for the single-board oracle, fp32 batch fn, bf16 batch fn).
# The deprecated "legacy"/None mode is the plain batch_encode_146 pair.
_MODES = (
    (None, batch_encode_146, batch_encode_146_bf16),
    ("lc0_root", batch_encode_146_lc0_root, batch_encode_146_lc0_root_bf16),
    (
        "lc0_root_legacy_meta",
        batch_encode_146_lc0_root_legacy_meta,
        batch_encode_146_lc0_root_legacy_meta_bf16,
    ),
)


def _f32_to_bf16_bits(a: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even fp32 -> bf16 bits, matching float_to_bf16_bits."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    u = u + np.uint32(0x7FFF) + lsb
    return (u >> np.uint32(16)).astype(np.uint16)


@dataclass
class Failure:
    context: str
    detail: str
    moves: list[str]

    def __str__(self) -> str:
        repro = " ".join(self.moves)
        return f"{self.context}: {self.detail}\n  repro (UCI from startpos): {repro}"


@dataclass
class _PoolEntry:
    cb: CBoard
    board: chess.Board
    label: str  # provenance, for failure messages
    want: dict[str | None, np.ndarray]  # per-mode single-board oracle planes
    want_rel: np.ndarray


def _make_entry(
    cb: CBoard, board: chess.Board, label: str, *, input_extra_features: str,
) -> _PoolEntry:
    # Oracle outputs are computed once per entry: the snapshots are immutable,
    # so re-encoding them at every later checkpoint would only re-verify the
    # oracle against itself. Caching the creation-time truth also catches a
    # batch encoder that mutates its input CBoard — later batch rows would
    # drift from it, where a freshly recomputed oracle would drift along.
    want = {
        mode: encode_cboard(
            cb, input_history_encoding=mode,
            input_extra_features=input_extra_features,
        )
        for mode, _, _ in _MODES
    }
    return _PoolEntry(cb, board, label, want, relation_matrices(board))


def _check_batch(
    entries: list[_PoolEntry], ctx: str, moves: list[str], *, n_planes: int,
) -> Failure | None:
    cbs = [e.cb for e in entries]
    n = len(cbs)

    for mode, fn32, fn16 in _MODES:
        out = np.full((n, n_planes, 8, 8), np.nan, dtype=np.float32)
        fn32(cbs, out)
        for i, e in enumerate(entries):
            want = e.want[mode]
            if not np.array_equal(out[i], want):
                bad = np.argwhere(out[i] != want)
                return Failure(ctx, (
                    f"batch row diverges from encode_cboard mode={mode!r} "
                    f"board#{i} ({e.label}) fen={e.board.fen()} "
                    f"first diff plane={int(bad[0][0])} ({len(bad)} cells)"
                ), moves)

        out16 = np.full((n, n_planes, 8, 8), 0xFFFF, dtype=np.uint16)
        fn16(cbs, out16)
        want16 = _f32_to_bf16_bits(out)
        if not np.array_equal(out16, want16):
            bad = np.argwhere(out16 != want16)
            i = int(bad[0][0])
            return Failure(ctx, (
                f"bf16 batch diverges from RNE(fp32 batch) mode={mode!r} "
                f"board#{i} ({entries[i].label}) fen={entries[i].board.fen()} "
                f"({len(bad)} cells)"
            ), moves)

    rel = np.full((n, 5, 64, 64), 0xAA, dtype=np.uint8)
    batch_compute_relations(cbs, rel)
    for i, e in enumerate(entries):
        if not np.array_equal(rel[i], e.want_rel):
            return Failure(ctx, (
                f"batch_compute_relations diverges from relation_matrices "
                f"board#{i} ({e.label}) fen={e.board.fen()}"
            ), moves)
    return None


def run(
    *, games: int, seed: int, max_plies: int, check_every: int,
    batch_cap: int, history_rep_fix: bool,
    input_extra_features: str = PROD_EXTRA_FEATURES,
) -> Failure | None:
    """Play random games; batch-encode mixed-provenance snapshots and compare."""
    # boards_discarded: every board this run touches is created below.
    rep_fix.apply(bool(history_rep_fix), boards_discarded=True)
    n_planes = int(input_plane_count(input_extra_features))
    rng = random.Random(seed)
    tag = f"repfix={int(history_rep_fix)} v={input_extra_features}"
    for g in range(games):
        b = chess.Board()
        cb = CBoard.from_board(b)
        moves: list[str] = []
        pool: list[_PoolEntry] = [
            _make_entry(
                cb.copy(), b.copy(), "startpos",
                input_extra_features=input_extra_features,
            )
        ]
        for ply in range(1, rng.randrange(2, max_plies + 1)):
            legal = list(b.legal_moves)
            if not legal:
                break
            m = rng.choice(legal)
            cb.push_index(move_to_index(m, b))
            b.push(m)
            moves.append(m.uci())
            if ply % check_every:
                continue
            # Same position, three history-construction paths: the live
            # push-built board, a copy of it, and a from_board rebuild whose
            # history comes from python's _stack instead of C pushes.
            pool.append(_make_entry(
                cb.copy(), b.copy(), f"push@{ply}",
                input_extra_features=input_extra_features,
            ))
            pool.append(_make_entry(
                CBoard.from_board(b), b.copy(), f"from_board@{ply}",
                input_extra_features=input_extra_features,
            ))
            if len(pool) > batch_cap:
                del pool[: len(pool) - batch_cap]
            entries = list(pool)
            rng.shuffle(entries)
            fail = _check_batch(
                entries, f"{tag} game{g} ply{ply}", moves, n_planes=n_planes,
            )
            if fail is not None:
                return fail
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0xBA7C4)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--check-every", type=int, default=8)
    parser.add_argument("--batch-cap", type=int, default=48)
    parser.add_argument(
        "--extra-features", type=str, default=PROD_EXTRA_FEATURES,
        help=(
            "input_extra_features for both the batch buffers and the oracle "
            f"(default {PROD_EXTRA_FEATURES} = 175 planes, production)"
        ),
    )
    args = parser.parse_args()
    try:
        for flag in (False, True):
            fail = run(
                games=args.games, seed=args.seed, max_plies=args.max_plies,
                check_every=args.check_every, batch_cap=args.batch_cap,
                history_rep_fix=flag,
                input_extra_features=str(args.extra_features),
            )
            if fail is not None:
                print(f"DIVERGENCE FOUND\n{fail}", file=sys.stderr)
                return 1
    finally:
        # Restore the process-wide flag even on a divergence return or an
        # exception — in-process callers (the pytest smoke) must not inherit
        # a stale history_rep_fix=True.
        rep_fix.apply(False, boards_discarded=True)
    print(
        f"OK: {args.games} games x both history_rep_fix phases "
        f"(seed={args.seed:#x}, batch cap {args.batch_cap}, "
        f"v={args.extra_features}) — "
        "batch encoders match the single-board oracle"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
