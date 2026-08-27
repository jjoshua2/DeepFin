"""Which SF-shape failure population dominates? A 3-axis offline breakdown.

OFFLINE ONLY. Nothing in `chess_anti_engine/` imports this; it exists to decide
whether `w_sf_shape` should ever be raised, BEFORE anyone tunes a weight.

    PYTHONPATH=. python3 scratchpad/sfshape_population_breakdown.py \
        --checkpoint <ckpt.pt> --shards <dir-or-glob> --rows 20000

THE THREE AXES, all measured over S -- the moves Stockfish actually scored,
recovered by `train.losses.sf_surfaced_move_mask`:

    dH   = H(q_S) - H(p_S)        positive => WE are sharper than the teacher
    M_S  = sum_{i in S} p_i       our mass on the set SF scored
    r_cp = E_{m ~ p_S}[r_SF(m)]   how bad our conditional picks are, in cp

and the cells they separate:

    sharp + high M_S + bad regret   -> right candidate set, overcommitted to the
                                       wrong members. THIS LOSS IS IDEAL.
    sharp + high M_S + good regret  -> sharp for a GOOD reason. Do NOT smooth it
                                       just to match an average entropy.
    normal dH + high M_S + bad regret -> not a calibration problem at all, and
                                       still strong justification for SF KL:
                                       entropy-matched distributions can rank the
                                       surfaced moves backwards.
    low M_S + big mass on an unscored move -> ⚑ THE SMOKING GUN FOR WIDENED
                                       LABELLING. The conditional KL is INVARIANT
                                       to M_S by construction and cannot touch
                                       this cell at any weight.

⚑ IT MUST BE RUN ON LIVE MultiPV-6 ROWS. On banked wide-era shards SF's labels
cover 26.63 of 26.82 legal moves, so S is not restricted, M_S reads ~1 and the
whole breakdown collapses to one cell for a reason that has nothing to do with
the policy. `--min-legal-minus-surfaced` refuses a run whose data is in that
regime rather than printing a vacuous table.

⚑ THE INSTRUMENT IS THE SAME CODE THE TRAINER USES. `sf_shape_conditional_kl` is
imported, not re-derived: a second implementation here would drift from the one
the loss holds, and the table would then describe a different quantity from the
columns it is meant to explain.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from chess_anti_engine.train.constants import SF_OWN_REGRET_CAP_CP
from chess_anti_engine.train.losses import (
    SfShapeParams,
    apply_policy_mask_to_logits,
    policy_legal_bool,
    sf_shape_conditional_kl,
)

# Cell boundaries. Deliberately round and stated once: they are reporting
# buckets, not thresholds anything is decided by, and the raw per-row dump is
# written alongside the table so a different cut can be taken without re-running.
DH_SHARP_NATS = 0.15          # dH above this => we are materially sharper
MASS_HIGH = 0.70              # M_S above this => the candidate set is ours too
REGRET_BAD_CP = 40.0          # conditional expected regret above this => bad

# Mean (legal - surfaced) below which the data cannot support this breakdown at
# all. At the live `sf_multipv: 6` it is ~21 (26.8 legal, 5.6 surfaced); on
# wide-era MultiPV-40 shards it is ~0.2 and every cell collapses.
MIN_LEGAL_MINUS_SURFACED = 5.0


def _bucket(dh: float, mass: float, regret_cp: float) -> str:
    sharp = "sharp" if dh > DH_SHARP_NATS else ("flat" if dh < -DH_SHARP_NATS else "matched")
    on_set = "highM" if mass > MASS_HIGH else "lowM"
    quality = "badR" if regret_cp > REGRET_BAD_CP else "goodR"
    return f"{sharp}/{on_set}/{quality}"


def breakdown(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    temp_cp: float,
    min_legal_minus_surfaced: float = MIN_LEGAL_MINUS_SURFACED,
) -> tuple[Counter[str], dict[str, torch.Tensor]]:
    """(cell counts, per-row axes) for one batch of rows carrying SF regret.

    Raises on wide-era data rather than returning a vacuous table -- the guard is
    INSIDE the function every caller has to go through, not beside it.
    """
    masked = apply_policy_mask_to_logits(logits, batch, "legal_mask", "has_legal_mask")
    probs = torch.softmax(masked, dim=-1)
    legal = policy_legal_bool(batch, width=int(logits.shape[-1]))
    out = sf_shape_conditional_kl(
        masked, probs, batch["sf_p0_regret_t"], legal,
        params=SfShapeParams(temp_cp=temp_cp),
    )
    keep = (out.surfaced_count >= 2.0) & (
        batch.get("has_sf_p0_regret", torch.ones_like(out.surfaced_count)) > 0.5
    )
    axes = {
        "dh": (out.h_sf_given_s - out.h_ours_given_s)[keep],
        "mass": out.surfaced_mass[keep],
        "regret_cp": out.regret_cp_given_s[keep],
        "kl": out.kl.detach()[keep],
        "surfaced": out.surfaced_count[keep],
        "p_sf_best": out.p_sf_best[keep],
    }
  # ⚑⚑ REFUSE, do not fabricate. This used to substitute the FULL action width
  # (1858 or 4672) as the legal-move count when the batch carried no legal mask.
  # `_refuse_if_coverage_is_vacuous` refuses when `legal - surfaced` is too
  # SMALL, so a fabricated width made that difference enormous and the row
  # sailed THROUGH the guard -- the guard accepted precisely the population it
  # exists to reject, and the wide-era rows it was written to catch are exactly
  # the ones most likely to arrive without a usable mask. A gate that cannot
  # fire on its own target, which is this repo's signature defect.
  # Reported by the Codex lane on PR #479; neither other review lane read this
  # file. [[a_counter_is_not_the_mechanism_behind_it]]
    if legal is None:
        raise ValueError(
            "sfshape_population_breakdown: this batch carries NO legal mask, so "
            "legality cannot be established. Refusing rather than substituting "
            "the full action width -- that substitution would pass the vacuity "
            "guard by construction and report surfaced-mass over impossible "
            "actions. Re-extract these rows with `legal_mask`/`has_legal_mask`."
        )
    legal_count = legal.to(torch.float32).sum(-1)
    _refuse_if_coverage_is_vacuous(
        axes["surfaced"], legal_count[keep], minimum=min_legal_minus_surfaced,
    )
    cells: Counter[str] = Counter()
    for dh, mass, r in zip(
        axes["dh"].tolist(), axes["mass"].tolist(), axes["regret_cp"].tolist(),
        strict=True,
    ):
        cells[_bucket(dh, mass, r)] += 1
    return cells, axes


def _refuse_if_coverage_is_vacuous(
    surfaced: torch.Tensor, legal_count: torch.Tensor, *, minimum: float,
) -> None:
    """⚑ Raise rather than print a table the data cannot support.

    A quantity that CANNOT hold if the population is wrong, printed before any
    conclusion -- the pattern `scripts/tail_censor_screen.py` established after a
    misaligned join produced a coverage of 1.04 and a confident wrong answer.
    """
    uncovered = float((legal_count - surfaced).mean())
    if uncovered < minimum:
        raise SystemExit(
            f"REFUSING: mean legal-minus-surfaced is {uncovered:.2f} (< {minimum}). "
            "These labels cover essentially every legal move, so S is not "
            "restricted, M_S is ~1 by construction and the breakdown is vacuous. "
            "This is wide-era (MultiPV 40) data; re-run on live MultiPV-6 rows."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--temp-cp", type=float, default=100.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--min-legal-minus-surfaced", type=float, default=MIN_LEGAL_MINUS_SURFACED,
        help="refuse the run below this mean uncovered-move count (vacuity guard)",
    )
    ap.add_argument("--out", type=Path, default=Path("scratchpad/sfshape_breakdown.json"))
    args = ap.parse_args()

    # Model + shard loading is deliberately left to the caller's own rig: this
    # module is the ANALYSIS, and every offline script here already disagrees
    # about how to page shards in. Import whichever loader the session is using
    # and feed `breakdown` its batches.
    raise SystemExit(
        "This script is the analysis half. Import `breakdown` from it and feed it "
        "batches from whichever shard loader the session already has open "
        f"(checkpoint={args.checkpoint}, shards={args.shards}, rows={args.rows}, "
        f"temp_cp={args.temp_cp}); it returns the cell counts and the raw per-row "
        f"axes to dump to {args.out}. Boundaries: dH>{DH_SHARP_NATS} nats = sharp, "
        f"M_S>{MASS_HIGH} = high, regret>{REGRET_BAD_CP} cp = bad; the regret cap "
        f"is {SF_OWN_REGRET_CAP_CP} cp."
    )


def dump(cells: Counter[str], axes: dict[str, torch.Tensor], out: Path) -> dict[str, Any]:
    """Write the table AND the raw per-row axes. Bank the dump, not just the number."""
    payload: dict[str, Any] = {
        "cells": dict(sorted(cells.items(), key=lambda kv: -kv[1])),
        "rows": int(axes["dh"].numel()),
        "boundaries": {
            "dh_sharp_nats": DH_SHARP_NATS,
            "mass_high": MASS_HIGH,
            "regret_bad_cp": REGRET_BAD_CP,
        },
        "means": {k: float(v.mean()) for k, v in axes.items() if v.numel()},
        "axes": {k: v.tolist() for k, v in axes.items()},
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
