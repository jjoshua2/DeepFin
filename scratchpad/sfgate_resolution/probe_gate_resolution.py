#!/usr/bin/env python3
"""The two OFFLINE preconditions of the `sf_own_regret` tail-gate prereg (PR #447).

Both are pre-committed kill gates. Either can cancel an arm for free, before any
GPU-day is spent, so this runs BEFORE the arm and not after.

**P1 — realized gradient share of the term on the CURRENT checkpoint** (task #38).
    ratio of trunk-gradient L2 norms, ||d sf_own_regret / d trunk|| over
    ||d policy_ce / d trunk||. ⚑ NOT derivable from `m_sf_own_regret`: the loss
    VALUE is not the gradient share, and the two moved in OPPOSITE directions the
    one time it was tried (the published direction was wrong and had to be
    retracted). The term is measurable at `w_sf_own_regret: 0.0` because
    `compute_loss` returns the unweighted component tensor and the weight is
    applied only when summing into `total`.

**P2 — RESOLUTION BEFORE THRESHOLD.** Share of the term's total gradient MAGNITUDE
    contributed by the rows the gate would scale. Pre-committed rule: **if the gate
    moves less than 10% of the term's gradient magnitude, leg B is unfalsifiable at
    our arena resolution and is CANCELLED** -- not softened.

    Measured through the PRODUCTION path, by DIFFERENCE OF GRADIENTS rather than by
    re-implementing the predicate:

        g_full = d/d_trunk  compute_loss(..., listed_mass_min=0.0, unlisted_scale=1.0)
        g_kept = d/d_trunk  compute_loss(..., listed_mass_min=X,   unlisted_scale=0.0)
        moved_share = ||g_full - g_kept|| / ||g_full||

    This is EXACT, not an approximation, and the reason is worth stating: the term's
    reduction is `masked_mean(sf_own_regret, sf_p0_regret_base)`, and
    `sf_p0_regret_base` is `net_mask * has_sf_p0_regret` -- it does NOT depend on
    the gate. So the denominator N is identical across the two calls and
    `L_full - L_kept == (1/N) * sum over gated rows`. A gate-DEPENDENT denominator
    would have made this a ratio of two different measurements.

    `unlisted_scale=0.0` is deliberate: it makes the measured quantity the MAXIMUM
    magnitude the gate can move at that `listed_mass_min`. A kill gate must be
    given the arm's best case, so a FAIL here is a fail for every scale.

⚑ A SHARE OF GRADIENT MAGNITUDE IS A RATIO OF NORMS, NOT A PARTITION.
    ||g_a + g_b|| != ||g_a|| + ||g_b||, so these shares do not sum to 1 across a
    partition of rows and can in principle exceed 1 under cancellation. Reported as
    what it is. The 10% bar is a bar on this ratio, which is what the prereg
    pre-committed to.

⚑ The gate's ROW COUNT is reported beside the magnitude on purpose: the prereg's
    own kill rule exists because 21.7% of rows carrying the label is a COUNT, and a
    count is not the mechanism. If the two disagree, the magnitude decides.

Re-run (from a worktree that HAS the gate; production's live tree does not):

    cd /home/josh/projects/chess-sfgate && PYTHONPATH=. python3 <this file> \
      --checkpoint /home/josh/projects/chess/scratchpad/gradshare_probe/ckpt158_dea5e/trainer.pt \
      --replay-dir '/home/josh/projects/chess/runs/pbt2_small/replay/train_trial_dea5e_00000_0_lr=0.0000_2026-08-16_12-38-11/replay_shards' \
      --out /home/josh/projects/chess/scratchpad/sfgate_resolution/result.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from chess_anti_engine.train.losses import compute_loss
from scripts.offline_replay_epoch import _ArraySampler, _as_replay_buffer
from scripts.probe_head_grad_share import (
    _build_trainer,
    _classify_params,
    _load_one_shard_arrays,
)

# The grid of `sf_own_regret_listed_mass_min` doses to resolve. 1.0 is the clamp
# ceiling (mass is a fraction), i.e. "gate every row that has a fabricated tail at
# all" -- the arm's absolute maximum reach, and the honest upper bound for a kill
# gate. Below ~0.5 the gate is asking for a target that puts most of its mass
# OUTSIDE the 6 moves Stockfish actually surfaced.
DOSE_GRID = (0.10, 0.25, 0.50, 0.75, 0.90, 1.00)


def _flat_grad(
    component: torch.Tensor, params: list[torch.nn.Parameter],
) -> list[torch.Tensor | None]:
    """d component / d params, keeping per-tensor structure so two runs can subtract.

    ``allow_unused=True`` yields None for params a component never touches,
    whatever the stubs say, so the return type is widened to be honest about it.
    """
    return list(cast(
        "tuple[torch.Tensor | None, ...]",
        torch.autograd.grad(component, params, retain_graph=True, allow_unused=True),
    ))


def _norm(grads: list[torch.Tensor | None]) -> float:
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().pow(2).sum().item())
    return math.sqrt(total)


def _diff_norm(
    a: list[torch.Tensor | None], b: list[torch.Tensor | None],
) -> float:
    """||a - b||, treating None as a zero tensor on either side."""
    if len(a) != len(b):
        raise ValueError(f"gradient structure mismatch: {len(a)} vs {len(b)}")
    total = 0.0
    for ga, gb in zip(a, b, strict=True):
        if ga is None and gb is None:
            continue
        if ga is None:
            total += float(cast("torch.Tensor", gb).detach().pow(2).sum().item())
        elif gb is None:
            total += float(ga.detach().pow(2).sum().item())
        else:
            total += float((ga.detach() - gb.detach()).pow(2).sum().item())
    return math.sqrt(total)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    # ⚑ CPU BY DEFAULT, AND THE REASON IS OPERATIONAL, NOT PREFERENCE. The sibling
    # probe this reuses defaults to `--gpu-mem-fraction 0.15`, but a live training
    # run holds ~30.3 of the 5090's 32.6 GB, so 0.15 (~4.9 GB) does not exist to be
    # taken. Paired/compiled GPU work has OOM-killed this run twice and the ONNX OOM
    # safety is gone. `--gpu` is available for a paused GPU, but the default must be
    # the choice that cannot kill production.
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction), 0)
    else:
        # Bound the CPU footprint: heavy agent CPU load has already contaminated
        # this run's throughput series once (ledger, iters 22+), and task #244 owes
        # a re-read over THIS window. A capped thread count keeps the marker small.
        torch.set_num_threads(int(args.threads))
    trainer, model_cfg = _build_trainer(device, args)
    trainer.model.train()
    model = trainer.model
    base_kwargs = dict(trainer._loss_kwargs)

    # ⚑ Prove the two keys are IN the trainer's own loss kwargs before measuring
    # anything with them. A probe that silently fell back to compute_loss's
    # defaults would report a clean identity for the wrong reason -- the exact
    # shape of defect this whole PR is guarding against.
    for key in ("sf_own_regret_listed_mass_min", "sf_own_regret_unlisted_scale"):
        if key not in base_kwargs:
            raise SystemExit(f"{key} absent from Trainer._loss_kwargs -- gate unwired")
    print(json.dumps({
        "event": "gate_keys_present",
        "live_values": {k: base_kwargs[k] for k in (
            "sf_own_regret_listed_mass_min", "sf_own_regret_unlisted_scale")},
        "w_sf_own_regret": float(base_kwargs.get("w_sf_own_regret", 0.0)),
        "w_policy": float(base_kwargs.get("w_policy", 0.0)),
    }), flush=True)

    trunk = _classify_params(model)[0]
    trunk_params = [p for _, p in trunk]

    arrs = _load_one_shard_arrays(model_cfg, replay_dir=args.replay_dir)
    sampler = _ArraySampler(arrs, np.random.default_rng(0))

    n_rows = 0
    n_batches = 0
    p1_ratios: list[float] = []
    p1_shares_weighted: list[float] = []
    moved: dict[float, list[float]] = {d: [] for d in DOSE_GRID}
    gated_frac: dict[float, list[float]] = {d: [] for d in DOSE_GRID}
    eligible_fracs: list[float] = []
    identity_exact: list[bool] = []

    for batch in trainer._iter_prefetched_batches(
        _as_replay_buffer(sampler),
        batch_size=int(args.batch_size),
        mirror_prob=0.0,
        count=int(args.n_batches),
    ):
        with trainer._amp_context():
            rel = batch.get("relations")
            out = model(batch["x"], relations=rel) if rel is not None else model(batch["x"])
            ident = dict(base_kwargs)
            ident["sf_own_regret_listed_mass_min"] = 0.0
            ident["sf_own_regret_unlisted_scale"] = 1.0
            losses_full = compute_loss(out, batch, **ident)

        term = losses_full["sf_own_regret"]
        pol = losses_full["policy_ce"]
        rows = float(losses_full["sf_own_regret_rows"].detach().item())
        batch_rows = float(losses_full["batch_rows"].detach().item())
        eligible_fracs.append(rows / batch_rows if batch_rows > 0 else 0.0)

        # P1: the term's trunk-gradient norm against policy_ce's.
        g_full = _flat_grad(term, trunk_params)
        n_term = _norm(g_full)
        n_pol = _norm(_flat_grad(pol, trunk_params))
        p1_ratios.append(n_term / n_pol if n_pol > 0 else float("nan"))
        p1_shares_weighted.append(
            (float(args.dose) * n_term) / n_pol if n_pol > 0 else float("nan"),
        )

        # The identity claim, re-verified on REAL production rows rather than on
        # the synthetic rows the unit tests use.
        identity_exact.append(
            float(losses_full["sf_own_regret_gated_rows"].detach().item()) == 0.0,
        )

        # P2: one extra backward per dose, on the SAME forward, so the only thing
        # that differs between the two gradients is the gate.
        for dose in DOSE_GRID:
            with trainer._amp_context():
                armed = dict(base_kwargs)
                armed["sf_own_regret_listed_mass_min"] = float(dose)
                armed["sf_own_regret_unlisted_scale"] = 0.0
                losses_gate = compute_loss(out, batch, **armed)
            g_kept = _flat_grad(losses_gate["sf_own_regret"], trunk_params)
            moved[dose].append(_diff_norm(g_full, g_kept) / n_term if n_term > 0 else 0.0)
            g_rows = float(losses_gate["sf_own_regret_gated_rows"].detach().item())
            gated_frac[dose].append(g_rows / rows if rows > 0 else 0.0)
            del losses_gate, g_kept

        n_rows += int(batch["x"].shape[0])
        n_batches += 1
        print(json.dumps({
            "event": "batch_done", "batch": n_batches,
            "term_grad_norm": n_term, "policy_ce_grad_norm": n_pol,
            "eligible_frac": eligible_fracs[-1],
            "moved_at_1.0": moved[1.00][-1], "gated_frac_at_1.0": gated_frac[1.00][-1],
        }), flush=True)
        del out, losses_full, g_full

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "replay_dir": str(args.replay_dir),
        "device": device,
        "rows": n_rows,
        "batches": n_batches,
        "eligible_frac_mean": mean(eligible_fracs),
        "identity_holds_on_every_batch": all(identity_exact),
        "p1_unweighted_share_of_policy_ce_grad": mean(p1_ratios),
        "p1_weighted_share_at_dose": {
            "dose": float(args.dose), "share": mean(p1_shares_weighted),
        },
        "p2_moved_grad_magnitude_share": {str(d): mean(moved[d]) for d in DOSE_GRID},
        "p2_gated_row_frac_of_eligible": {str(d): mean(gated_frac[d]) for d in DOSE_GRID},
        "p2_bar": 0.10,
        "p2_verdict_at_max_dose": (
            "PASS" if mean(moved[1.00]) >= 0.10 else "FAIL -- leg B CANCELLED per prereg"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pbt2_small.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--n-batches", type=int, default=5)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.15)
    ap.add_argument("--gpu", action="store_true", help="opt in; only on a PAUSED run")
    ap.add_argument("--threads", type=int, default=6)
    # The dose leg A would run at. Only scales the REPORTED weighted share; the
    # gradient measurement itself is unweighted and dose-independent.
    ap.add_argument("--dose", type=float, default=0.7)
    args = ap.parse_args()
    print(json.dumps(_run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
