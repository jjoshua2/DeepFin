"""Is an ARENA blind to the two target-only knobs? MEASURE, do not trust the comment.

Backs the severity claim in ``docs/experiment_ledger.md`` (2026-08-15, "INSTRUMENT
DEFECT: ``--search-shape training`` measured a search production does not run").

``gumbel_target_max_visit_cap`` and ``gumbel_target_untempered_prior`` were
promoted into production on 2026-08-10 23:28 and the arena's training shape did
not carry them until this fix. Whether that CORRUPTED the six banked rows or only
MISLABELLED them turns entirely on one question: can either knob change the move
an arena plays? Both `mcts/gumbel.py` and `mcts/gumbel_c.py` say no -- the played
move comes from ``imp_all`` and only the STORED row comes from ``imp_store`` --
and `selfplay/match.pick_moves_for_boards` discards the policy entirely. That is
a reading of the code, so it is a hypothesis, not a result.

Three arms, because a bare "the actions matched" is consistent with a probe that
cannot see anything at all:

  POSITIVE CONTROL  the STORED policy must MOVE (else the knobs are inert and the
                    premise of the whole entry is wrong).
  NEGATIVE CONTROL  the PLAYED action must NOT move.
  INSTRUMENT CHECK  a knob the arena DOES carry (``c_scale``) must move the
                    played action, or the negative control proves nothing.

CPU only, no GPU, no training. Run with ``PROBE_SIMS=32`` (the banked rows' budget)
and ``PROBE_SIMS=256``; the cap bites harder at higher sims, so a single budget
would not bound the claim.

    PYTHONPATH=. CUDA_VISIBLE_DEVICES= PROBE_SIMS=32  python3 scratchpad/arenashape_blindness_probe.py
    PYTHONPATH=. CUDA_VISIBLE_DEVICES= PROBE_SIMS=256 python3 scratchpad/arenashape_blindness_probe.py

Measured 2026-08-15 (16 positions, both budgets): played actions differing 0/16,
root values identical, stored policy changed 16/16 (L1 mean 0.031 at 32 sims,
0.713 at 256); ``c_scale`` 0.1->0.9 moved 14/16 actions at 32 sims and 10/16 at
256. => an arena is blind to both knobs; the banked rows are mislabelled, not
invalidated.
"""
from __future__ import annotations

import dataclasses
import os

import chess
import numpy as np
import torch

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.selfplay import match as match_mod

SIMS = int(os.environ.get("PROBE_SIMS", "32"))

torch.manual_seed(0)
# A small net on purpose: the question is about the SEARCH's arithmetic, which
# does not depend on the evaluator's strength, and this keeps the probe on CPU.
model = build_model(
    ModelConfig(
        embed_dim=64, num_layers=2, num_heads=4, ffn_mult=1.5,
        input_history_encoding="lc0_root_legacy_meta",
    )
).eval()
model.input_history_encoding = "lc0_root_legacy_meta"
model.input_extra_features = "v2_threats"
model.policy_encoding = "lc0_1858"

boards: list[chess.Board] = []
board = chess.Board()
for uci in [
    "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6",
    "b1c3", "a7a6", "c1e3", "e7e5", "d4b3", "c8e6", "f2f3", "f8e7",
]:
    boards.append(board.copy())
    board.push_uci(uci)

# The arena's training shape as it stood BEFORE this fix, at production's values
# for the knobs it did carry.
arena_cfg = GumbelConfig(
    simulations=SIMS, temperature=0.1, add_noise=True,
    input_history_encoding="lc0_root_legacy_meta",
    input_extra_features="v2_threats", policy_encoding="lc0_1858",
    topk=16, c_scale=0.1, policy_temp=1.5,
)
# ...and what production actually searches with.
prod_cfg = dataclasses.replace(
    arena_cfg, target_max_visit_cap=5, target_untempered_prior=True,
)


def run(cfg: GumbelConfig) -> tuple[np.ndarray, list[int], list[float]]:
    """One C-path search over every position, at a FIXED seed."""
    result = run_gumbel_root_many_c(
        model, [b.copy() for b in boards], device="cpu",
        rng=np.random.default_rng(1234), cfg=cfg,
        allow_terminal_root_shortcuts=True,
        vloss_weight=1, target_batch=0,  # production's C-path controls
    )
    return np.asarray(result[0]), list(result[1]), list(result[2])


def main() -> None:
    print(f"sims={SIMS}  positions={len(boards)}  C path={match_mod._HAS_GUMBEL_C}")
    if not match_mod._HAS_GUMBEL_C:
        raise SystemExit("the C path is what production and the arena both run; abort")

    probs_a, act_a, val_a = run(arena_cfg)
    probs_p, act_p, val_p = run(prod_cfg)

    differing = sum(x != y for x, y in zip(act_a, act_p, strict=True))
    l1 = np.abs(probs_a - probs_p).sum(axis=1)

    print(f"NEGATIVE CONTROL  played actions differing: {differing}/{len(boards)}")
    print(f"                  root values identical:    {np.allclose(val_a, val_p)}")
    print(
        "POSITIVE CONTROL  stored policy L1  min %.6f  max %.6f  mean %.6f"
        % (l1.min(), l1.max(), l1.mean())
    )
    print(
        f"                  stored policy changed on {int((l1 > 1e-6).sum())}"
        f"/{len(boards)} positions"
    )

    _, act_sharp, _ = run(dataclasses.replace(arena_cfg, c_scale=0.9))
    moved = sum(x != y for x, y in zip(act_a, act_sharp, strict=True))
    print(f"INSTRUMENT CHECK  c_scale 0.1->0.9 moved {moved}/{len(boards)} actions")

    # ⚑ All THREE arms are ENFORCED, not just printed. The conclusion below is
    # the evidence the ledger cites for "the banked rows are mislabelled, not
    # invalidated", so the script must refuse to print it whenever any arm fails
    # -- above all the NEGATIVE control. A probe that prints "never affects the
    # played move" while its own measurement says otherwise would launder a
    # regression into a citation.
    if differing != 0:
        raise SystemExit(
            f"NEGATIVE CONTROL FAILED: {differing}/{len(boards)} played actions "
            "changed. The two knobs are NOT target-only on this code, so an "
            "arena is NOT blind to them and the ledger's claim that the "
            "2026-08-10..15 rows keep their Elo does NOT hold. Re-derive it."
        )
    if moved == 0:
        raise SystemExit("the probe cannot see ANY action change; it proves nothing")
    if int((l1 > 1e-6).sum()) == 0:
        raise SystemExit("the two knobs changed nothing at all; premise is wrong")
    print(
        "\n=> the two target-only knobs move the STORED target and never the "
        "PLAYED move: an arena is blind to them."
    )


if __name__ == "__main__":
    main()
