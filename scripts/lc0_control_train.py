#!/usr/bin/env python3
"""Supervised training driver for the lc0 positive control.

lc0-derived shards -> DiskReplayBuffer -> Trainer.train_steps -> checkpoint.
No selfplay, no MCTS, no PID, no curriculum, no server, no Ray. The model, the
optimizer and every loss weight come from `configs/lc0_positive_control.yaml`,
which is the production config with the value blend re-pointed (see that
file's header for the full measured diff).

⚑⚑ THIS SCRIPT'S REASON TO EXIST IS THE GUARD, NOT THE LOOP.

lc0 shards carry no `sf_wdl`. `train/losses.py` falls the SF component of the
value blend back to the RAW ONE-HOT GAME OUTCOME for any row without one — no
error, no warning, no metric named for it. ⚑ The worked example that used to sit
here quoted `sf_wdl_frac: 0.50` / `0.80` / `0.30` off **`main`'s** copy of
`configs/pbt2_small.yaml`, which is stale by construction — the live branch is
the only writer. Re-derived against the LIVE file: production runs
`sf_wdl_frac 0.69 + search_wdl_frac 0.31 = 1.00`, so production's intended raw
game-outcome share is **0.00**, and on a corpus with no `sf_wdl` column at all
this arm would train **1.00** of its value target on the deep game outcome.
Every number it produced would be about an experiment nobody chose.

Six checks, at four different levels, because each can be satisfied by a run
that still does the wrong thing:

  0. LAUNCH, architecture. `assert_control_matches_live_architecture` — the
     arm's premise is "test THE STACK WE RUN", and the in-tree
     `configs/pbt2_small.yaml` is not the file the live run reads. This one
     judges against the LIVE yaml when `$CHESS_LIVE_PRODUCTION_CONFIG` names
     it, and against a recorded pin otherwise.
  0c. LAUNCH, trainer. `assert_control_matches_live_trainer` — the same
     question about the OTHER half of "our exact net and trainer". Until
     2026-08-16 only the net half was checked and the recipe was `main`'s,
     thirteen kwargs behind live. ⚑ `--allow-leak` DOES downgrade this one to a
     banner (`preflight_trainer(..., allow_leak=...)`); the line that used to sit
     here claimed there was no downgrade flag, which overstated the guarantee to
     exactly the operator most likely to rely on it.
  0d. LAUNCH, replay buffer. `assert_control_matches_live_replay` — the THIRD
     half of "our exact stack", and the one neither pin above can see.
     `DiskReplayBuffer` is constructed from `TrialConfig` fields that
     `trainer_kwargs_from_config` does not read, so until 2026-08-17 the driver
     hand-wrote three buffer kwargs, took `disk_buffer.py`'s constructor
     defaults for the rest, and BOTH other pins passed while the buffer sat
     SEVEN axes off production — `shuffle_cap` 20000 against 100000 among them.
     A plateau produced by 5x less hot-pool diversity is a plateau of the RIG,
     and in the held-out slope it is indistinguishable from the plateau
     H_stack is about.
  1. LAUNCH, config-level. `assert_pid_cannot_reassert_sf_wdl` — an override
     the difficulty controller can undo is not an override.
  2. LAUNCH, corpus-level. The converter's own `run_config_problems` is REUSED
     (not restated) against the actual shard directories, driven by the
     MEASURED coverage of every directory — of BOTH value labels, `sf_wdl` and
     `search_wdl` — rather than `any()`.
  3. ⚑ REALIZED, per-step. `compute_loss` is wrapped for the duration of the
     run and the fracs it is ACTUALLY CALLED WITH — together with the batch's
     own EFFECTIVE label mass for both components — are fed to
     `value_blend_guard`. A configured value is not an applied value; this is
     the only one that measures the applied one. The same wrap feeds
     `assert_categorical_rebuild_is_inert`, which asks the identical question
     about the CATEGORICAL value head: production runs
     `rebuild_categorical_target: true` with `blend_frac 0.69`, and
     `targets.normalize_categorical_blend_fracs` drops an unavailable
     component's weight onto the raw outcome exactly as `losses.py` does. It is
     a no-op on today's corpus only because converted rows carry no
     `categorical_target` column — a property of the CONVERTER, so it is
     measured per batch rather than argued.

Check 3 fails loudly (non-zero exit, no checkpoint written).

⚑ `--allow-leak` DOWNGRADES ONLY THE LAUNCH GUARDS (0c, 0d, 1 and 2). It used to skip
check 3 as well, which made check 3 — the headline of this script —
UNREACHABLE: guard 1 refuses every `sf_wdl_frac > 0` config outright, nothing
downstream can raise the frac, and the only way past guard 1 also skipped the
assert. So `--allow-leak` on a production-blend config now runs the steps,
prints the realized leak, and then EXITS 1 WITHOUT A CHECKPOINT. That is the
demonstration; `scratchpad/lc0_positive_control/RIG_VERIFICATION_20260816.md`
records it.

⚑ AND UNDER `--allow-leak` THE PARTIAL-CORPUS REFUSAL IS A PRINT, SO A MIXED
CORPUS *IS* REACHABLE — check 3 is the only thing left, and it is what actually
holds (measured 14/14 seeds, PR #438's re-review). Do not read "the launch
guards make the states below unreachable" as a proof about the flag path; it is
a proof about the default path only.

`--allow-arch-drift` does the same for check 0, and exists because the arm's
plumbing has to be smoke-testable on a deliberately tiny model — the driver
tests run 1 step on a 4-row batch, and a run that had to build production's
61M-param net to exercise its wiring would not be a unit test. (Until
2026-08-16 it had a second reason: this branch could not build the live
architecture at all. PR #439 removed that one; the smoke-test reason stands.)
Both flags stamp `valid_control: false` into summary.json,
as does omitting `--purity-receipt` — with the reason named in
`validity_problems`.

⚑⚑ AND THE RUN EMITS TWO CHECKPOINTS, NOT ONE. The prereg's PRIMARY yardstick is
the JOINT reading of `Δ_heldout` and `Δ_train`, "both measured LAST vs
MID-BUDGET" — so a driver with one `train_steps` call and one `trainer.save`
could not produce the deciding statistic at all. `--mid-checkpoint-frac`
(default 0.5) writes `checkpoint_mid.pt` from INSIDE the single `train_steps`
call, because two calls of N/2 would run the `sqrt_release` ramp twice and put
MID and LAST on different LR trajectories. A run with no mid checkpoint records
that in `validity_problems` and is not a valid control — and so does one whose
`--mid-checkpoint-frac` is not the preregistered 0.5, because a
LAST-vs-99%-budget contrast is a different statistic wearing the deciding
statistic's name.

⚑ TWO MORE THINGS ARE RECORDED AND ONE IS REFUSED (independent review of #438,
2026-08-17), all of them the same shape as the guards above — a value accepted
after every preflight, deviating from the production one, with the run still
stamped `valid_control: true`.

⚑⚑ "RECORDED" IS NOT "TOLERATED": every `validity_problems` entry sets
`valid_control: false`, and `lc0_control_eval compare` refuses that with NO
waiver.

⚑⚑ AND SINCE 2026-08-17 THE RUN REFUSES TO START RATHER THAN RECORDING IT LATER.
Every entry is knowable from the arguments, the config and the window plan, and
twice a full budget ran to completion and was only then stamped
`valid_control: false` — an artifact nothing downstream will quote, for a reason
its own command line already contained. `control_validity_problems` is therefore
evaluated BEFORE the first optimizer step (with the PLANNED mid step) and again
after the last one (with the REALIZED one, the only input that can differ), and
`main` exits unless `--allow-invalid-control` is passed. That flag is what a
plumbing smoke or a deliberate leak demonstration passes; it does not make the run
valid, so the choice it offers is CHECKPOINTS vs NOTHING, never ARTIFACT vs
VERDICT.

  * `--batch-size` differing from the config's changes the examples per step and
    the gradient-noise regime. `batch_size` is not a trainer kwarg, so guard 0c
    is structurally blind to it -> `validity_problems`.
  * a `--shards` directory named TWICE is staged twice under distinct symlink
    names, so the buffer oversamples that hour; the coverage preflight sums over
    the list and `purity_receipt_problems` compares SETS, so neither could see
    it. That one is REFUSED at launch rather than recorded: de-duplicating
    silently would train on a corpus other than the one named.
  * the run banks a content-derived `run_id` and a per-checkpoint sha256/role,
    so `summary.json` can no longer vouch for a checkpoint it never saw and
    `compare` cannot pair two trajectories' LAST checkpoints as LAST-vs-MID.

Usage
-----
    PYTHONPATH=. python3 scripts/lc0_control_train.py \\
        --config configs/lc0_positive_control.yaml \\
        --shards <converted-dir> [<converted-dir> ...] \\
        --steps 20 --out-dir <run-dir>
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import numpy as np
import torch
import zarr

from chess_anti_engine.eval.lc0_control_arch import (
    LIVE_FILE_UNREAD as _LIVE_FILE_UNREAD,
    ControlArchitectureDrift,
    assert_control_matches_live_architecture,
    live_production_config_path,
    unique_storage_param_count,
)
from chess_anti_engine.eval.lc0_control_rows import (
    duplicate_resolved_dirs,
    sha256_file,
)
from chess_anti_engine.model import build_model, model_config_from_flat_config
from chess_anti_engine.replay.buffer import ReplayBuffer
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import iter_shard_paths
from chess_anti_engine.train import trainer as trainer_module
from chess_anti_engine.train.losses import normalize_value_blend_fracs
from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config
from chess_anti_engine.eval.lc0_control_replay import (
    ControlReplayDrift,
    apply_control_deviations,
    assert_control_matches_live_replay,
    replay_kwargs_signature,
)
from chess_anti_engine.eval.lc0_control_trainer import (
    ControlTrainerDrift,
    assert_control_matches_live_trainer,
    live_production_game_frac,
)
from chess_anti_engine.train.value_blend_guard import (
    PRODUCTION_GAME_FRAC,
    CategoricalRebuildReadout,
    ValueBlendMisconfigured,
    ValueBlendReadout,
    assert_categorical_rebuild_is_inert,
    assert_no_silent_outcome_fallback,
    assert_pid_cannot_reassert_sf_wdl,
    value_blend_readout,
)
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

# Reused, not restated: the converter owns the "these shards have no SF label"
# question and already ships the config check for it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lc0_data_to_rows import (
    run_config_problems,
    shard_dir_search_wdl_coverage,
    shard_dir_sf_wdl_coverage,
)


# ⚑ THE PREREG'S OWN FRACTION, NAMED. `docs/lc0_positive_control_prereg.md`
# AMENDMENT 4 reads the primary yardstick as two slopes "both measured LAST vs
# MID-BUDGET", and `--mid-checkpoint-frac` is what places the MID point. Any
# other value produces a LAST-vs-something contrast, which is a different
# statistic wearing the deciding statistic's name — so the number lives here,
# next to the check that reads it, rather than only in the argparse default a
# caller can override without leaving a trace.
PREREG_MID_CHECKPOINT_FRAC = 0.5

# ⚑⚑ AND THE COMPARISON IS TOLERANT, BECAUSE `int(0.5 * steps)` CANNOT BE EXACT ON
# AN ODD BUDGET. The first version of this guard used exact `!=` on the realized
# ratio, which is exact only for EVEN budgets — so every odd `--steps` was
# recorded invalid, unwaivably (`valid_control: false` has no waiver by design),
# and a day of GPU at `--steps 20001` became permanently unquotable over a
# 0.0025% discrepancy. Measured by the independent review; it is the previous
# round's fix over-correcting into a new failure.
#
# ⚑ THE BOUND, AND WHY THIS ONE. Truncation can move the realized fraction by at
# most `0.5 / steps` (odd `steps = 2k+1` gives `k/(2k+1)`, i.e. exactly
# `0.5/steps` below 0.5), so:
#   * 0.01 admits truncation for every budget >= 50 steps, and the arm's budget is
#     thousands — `--steps 20001` deviates by 0.000025, `1001` by 0.0005, `87` by
#     0.0058;
#   * it still catches every deliberate misplacement: `--mid-checkpoint-frac 0.99`
#     (0.25 off at steps=4), `0.6` (0.1 off), `--steps 3 --frac 0.5` (0.167 off);
#   * and a budget UNDER 50 steps — the only regime where the bound is tighter
#     than truncation — is already disqualified by the `warmup_steps` entry
#     (production's warmup is 1000), so this entry can never be the SOLE reason a
#     run is refused. That last clause is the one that makes 0.01 safe rather than
#     merely convenient, and `test_the_tolerance_can_never_be_the_only_reason_a_run_is_invalid`
#     is what keeps it true.
MID_CHECKPOINT_FRAC_TOLERANCE = 0.01

# ⚑⚑ PRODUCTION'S TRAIN-WINDOW CADENCE, AND WHY IT IS A NUMBER HERE RATHER THAN A
# CONFIG KEY. With `lr_schedule: sqrt_release` and `lr_release_cycle_steps: 0` —
# what this arm and production both run — `train_steps` derives the release cycle
# from ITS OWN `steps` argument (`effective_cycle_steps = max(1,
# requested_steps)`) and feeds `local_step = train_steps_done`, which restarts at
# 0 on every call. Production calls it once per ITERATION at ~88 steps, so
# production's LR is a SAWTOOTH: flat for `lr_release_start_frac` (0.80) of each
# window, annealed to `lr_release_min_scale` (0.1x) by the window's last step,
# then flat again.
#
# ⇒ ONE call of N steps is NOT the same trajectory. It holds LR flat for 80% of
# the entire experiment and anneals ONCE, so a MID checkpoint at 50% of budget
# sits at FULL base LR while LAST sits at 0.1x after a full anneal — and the
# deciding MID->LAST slope then contains an anneal that production never places
# between two of its own checkpoints. An annealed endpoint scores better on
# held-out top-1 for reasons that are not the trainer learning, in the flattering
# direction for H_stack. The control config DOCUMENTED this (lines 395-402);
# documenting a first-order confound in the primary yardstick is not controlling
# it. Independent review of #438, and Codex's #1 of 2026-08-17.
#
# ⚑ 88 is a REALIZED number, not a configured one: step budget comes from
# views-targeting (`train_views_per_position`), so production's steps/iteration is
# a function of ingest volume (CLAUDE.md: "~88 steps/iter"). It is therefore a CLI
# argument with this default, banked in the artifact, and NOT read from the yaml —
# there is no key to read.
PRODUCTION_TRAIN_WINDOW_STEPS = 88

def mid_fraction_deviates(*, saved_at_step: int, steps: int) -> bool:
    """Whether the REALIZED mid point is outside the preregistered tolerance.

    ⚑ A FUNCTION so the test can call THIS and not a copy of it. The first
    version of the odd-budget test reimplemented `abs(realized - 0.5) > tol` in
    the test file, which meant the arithmetic table could not see the driver's
    comparison operator change at all — only the (slow, small-budget) driver test
    could, and the reviewer's failing budgets are 1001 and 20001 steps. A guard
    whose test restates its criterion measures the restatement.
    """
    if steps <= 0:
        return False
    realized = saved_at_step / steps
    return abs(realized - PREREG_MID_CHECKPOINT_FRAC) > MID_CHECKPOINT_FRAC_TOLERANCE


def min_budget_for_mid_tolerance(window: int) -> int:
    """The smallest budget whose nearest window boundary is inside the tolerance.

    ⚑ `mid_step_on_window_boundary` SNAPS the mid point to a window boundary, so
    the realized fraction can miss 0.5 by up to half a window — a bound that
    shrinks with the budget. Below this many steps the strict tolerance is
    unreachable NO MATTER WHAT `--mid-checkpoint-frac` says, so the refusal has to
    name the number: an operator told only "0.4840 is more than 0.01 from 0.5" has
    no way to know which argument to change. The tolerance is NOT widened for
    short budgets — that would trade the prereg's mid point for the convenience of
    a run nobody needs (at the default window this floor is 4400 steps, and the
    arm's budget is 17600-20000).
    """
    window = max(1, int(window))
    return int(-(-(window / 2.0) // MID_CHECKPOINT_FRAC_TOLERANCE))


# The artifacts a completed (or half-completed) run leaves behind. `--out-dir`
# reuse is refused when any of these is present -- see `existing_run_artifacts`.
RUN_ARTIFACTS = ("checkpoint.pt", "checkpoint_mid.pt", "summary.json")


def existing_run_artifacts(out_dir: Path) -> list[str]:
    """Which of ``RUN_ARTIFACTS`` already exist in ``out_dir``.

    ⚑⚑ THE DEFECT THIS CLOSES DESTROYED DATA. `--out-dir` was created with
    `mkdir(exist_ok=True)`, so a rerun into a populated directory shared it with
    the previous run — and when the rerun then FAILED its realized value-blend
    guard, its cleanup (`mid_ckpt.unlink`, correct in intent and scoped only by
    "did THIS run save a mid checkpoint", which was true because run 2 had just
    written over run 1's file) IRREVERSIBLY DELETED THE PREVIOUS COMPLETED RUN'S
    `checkpoint_mid.pt` — a day of GPU in production — while that run's
    `summary.json` went on banking the file's path, step and sha256, and its
    surviving `checkpoint.pt` still verified as `role: last`. The directory was
    left reading as a scorable success with half the deciding statistic gone, and
    the emitted message "no checkpoint written (including the mid-budget one)"
    was FALSE about the directory. Reproduced end-to-end by the independent
    review.

    ⚑ The remedy is a REFUSAL, and `--move-existing-aside` RENAMES. There is
    deliberately no `--overwrite`: this file adds no new delete path, and a
    rename is reversible by hand while a delete is not.
    """
    return [name for name in RUN_ARTIFACTS if (Path(out_dir) / name).exists()]


def train_window_plan(*, steps: int, window: int) -> tuple[int, int, str | None]:
    """``(window, n_windows, problem)`` for running ``steps`` at production cadence.

    ⚑⚑ THE WINDOW COUNT IS A CEILING, SO NO STEP IS DISCARDED AND NO BUDGET IS
    REJECTED FOR BEING INDIVISIBLE. The first version required
    ``steps % (2 * window) == 0`` — which at the default 88 admits only multiples
    of **176**, i.e. it refused 175 of every 176 budgets INCLUDING 20000, and it
    silently trained ``steps // window * window`` steps while ``summary["steps"]``
    reported the requested number. Both were found by the third independent
    review, the first as the direct descendant of N1: the mid-fraction tolerance
    was widened to admit odd budgets and the cadence rule then refused them
    anyway, so N1's failure ("a day of GPU ends `valid_control: false`, no
    recovery") survived its own fix with a trigger set ~88x WIDER.
    ⇒ the divisibility requirement is GONE. A short final window is harmless:
    ``_scale_for_window_step`` returns ``min_scale`` at ``local_step ==
    cycle_steps - 1`` whatever the cycle length, so LAST still sits at the BOTTOM
    of a release cycle — and `mid_step_on_window_boundary` puts MID at a boundary
    too, which is the property that actually matters and the one the code's
    comments always claimed.

    The only remaining cadence problem is a budget shorter than TWO windows,
    which has no interior boundary and so cannot place MID at LAST's LR phase.
    """
    window = max(1, int(window))
    steps = int(steps)
    if steps <= 0:
        return window, 0, None
  # ⚑ CEILING, not floor: the loop trains a SHORT final window rather than
  # discarding the remainder, so the realized budget equals the requested one.
    n_windows = -(-steps // window)
  # ⚑ TWO windows, not one. The property that matters is that MID and LAST both
  # sit on a window boundary, and the only INTERIOR boundary of a budget shorter
  # than 2 x window is none at all: `mid_step_on_window_boundary` then has nowhere
  # to put MID. `steps == 2 * window` is the smallest budget that works, and it
  # puts MID at exactly 0.5.
    if steps < 2 * window:
        return min(window, steps), n_windows, (
            f"--steps {steps} is less than two --train-window-steps windows of "
            f"{window}, so there is no INTERIOR window boundary: its LR anneals "
            f"over the whole budget instead of every {window} steps as "
            "production's does, and MID cannot sit at the same phase of the "
            "release cycle as LAST"
        )
    return window, n_windows, None


def mid_step_on_window_boundary(*, steps: int, window: int, frac: float) -> int:
    """The mid-budget step, SNAPPED to the nearest train-window boundary.

    ⚑⚑ THIS IS THE PROPERTY THE CADENCE FIX IS FOR, AND NOTHING WAS ENFORCING IT.
    Two guards bounded proxies — the realized FRACTION (±0.01) and the budget's
    DIVISIBILITY — and neither bounded "MID lands on a window boundary". At the
    arm's budget ±0.01 is ±2 full 88-step windows, so the review executed
    ``--mid-checkpoint-frac 0.5028`` at 17600 steps: MID at LR scale **1.0**
    against LAST at **0.1**, both guards silent, `valid_control: true` — the exact
    anneal-in-the-slope confound the cadence fix closed, reachable through a
    PASSING knob.

    Snapping fixes it at the source: whatever fraction is asked for, MID lands at
    the END of a window, i.e. at the same LR phase (the release-cycle bottom) as
    LAST. The residual deviation from the requested fraction is at most half a
    window, which `mid_fraction_deviates` then judges — so on a budget of at least
    ~50 windows every fraction near 0.5 is admitted, and on a budget too short for
    that the run is refused for a reason that is TRUE rather than arithmetic.
    """
    steps, window = int(steps), max(1, int(window))
    if float(frac) <= 0.0 or steps <= 0:
        return 0
  # ⚑ A budget shorter than TWO windows has no interior boundary, so there is
  # nothing to snap to. Such a run is ALREADY disqualified by the cadence entry
  # (`train_window_plan` returns a problem for it) and it is exactly the shape of
  # every plumbing smoke in the test suite — which must still exercise the
  # mid-checkpoint path, the role assignment and the identity binding. So it falls
  # back to the plain interior clamp and the artifact records BOTH reasons it is
  # not the prereg's mid point, rather than silently emitting one checkpoint.
    if steps < 2 * window:
        return min(max(int(float(frac) * steps), 1), max(steps - 1, 0))
    target = float(frac) * steps
    boundary = round(target / window) * window
  # Interior only: 0 would save before any step ran, and `steps` would save LAST
  # under a second name (a pair discordant on zero rows).
    return min(max(boundary, window), steps - window)


def checkpoint_identities(
    *, last: Path, mid: Path | None,
) -> tuple[str, list[dict[str, Any]]]:
    """``(run_id, [{role, path, sha256}])`` — THIS trajectory's identity.

    ⚑⚑ WITHOUT THIS, `summary.json` VOUCHED FOR ANY CHECKPOINT. `valid_control`
    is a statement about a RUN, and the artifact carried nothing that tied it to
    a FILE, so `lc0_control_eval score --summary <valid run>/summary.json
    --checkpoint <some other run>/checkpoint.pt` banked `valid_control: true`
    for a checkpoint the summary had never seen — including one from a run the
    driver disqualified. And with no run identity, `compare` could pair two
    LAST checkpoints from two independently initialised trajectories and report
    it as the prereg's LAST-vs-MID-BUDGET slope.

    ⚑ The run id is CONTENT-DERIVED, not a uuid: it is a digest over the
    checkpoint hashes themselves, so it is exactly "which trajectory are these
    weights from" and cannot drift from the files it names. MID and LAST of one
    run therefore share it by construction — they are digested together and the
    result is written once per ``summary.json``.

    ⚑ AND THE CONVERSE IS ABOUT BYTES, NOT ABOUT ARGUMENTS. An earlier revision
    of this line claimed two runs "even two with identical arguments" cannot
    share a run id "because their weights differ". That is FALSE and the driver's
    own test says so: this arm seeds `torch` and the buffer RNG from `--seed`, so
    two runs at the same seed, config and corpus are BIT-IDENTICAL and do share
    the id. Which is correct — identical weights are the same trajectory in every
    sense the comparison uses — but a reader must not be told the id separates
    invocations. It separates WEIGHTS: a different seed, corpus, budget or code
    path gives a different id, and nothing else does.
    """
    entries = [{"role": "last", "path": str(Path(last).resolve()),
                "sha256": sha256_file(last)}]
    if mid is not None:
        entries.insert(0, {"role": "mid", "path": str(Path(mid).resolve()),
                           "sha256": sha256_file(mid)})
    fingerprint = "|".join(
        f"{entry['role']}:{entry['sha256']}"
        for entry in sorted(entries, key=lambda e: str(e["role"]))
    )
    run_id = hashlib.sha256(
        f"lc0-control-trajectory|{fingerprint}".encode(),
    ).hexdigest()
    return run_id, entries


class _LossCapture:
    """Records what ``compute_loss`` was ACTUALLY called with, every call.

    ⚑ Wrapping the call is the point. Reading `trainer.sf_wdl_frac` before the
    step would report the attribute, and the attribute is exactly what a live
    reload, a PID push or a PB2 mutation can change between the read and the
    step. The kwargs recorded here are the ones the trained objective used.
    """

    def __init__(self, *, rebuild_categorical: bool, categorical_params: Any) -> None:
        self.calls = 0
        self.kwargs: dict[str, Any] = {}
        self.readouts: list[ValueBlendReadout] = []
  # ⚑ The categorical rebuild happens in `Trainer._prepare_arrays`, BEFORE the
  # batch exists, so its two config inputs cannot be read off the call the way
  # the blend fracs can. They are taken from the trainer kwargs and the BATCH
  # supplies what actually varies: whether the corpus carries a
  # `categorical_target` at all, and how much of it is labelled.
        self._rebuild_categorical = bool(rebuild_categorical)
        self._categorical_params = categorical_params
        self.categorical_readouts: list[CategoricalRebuildReadout] = []

    def _observe_categorical(self, batch: dict[str, Any], rows: float) -> None:
        def labelled_frac(mask_key: str, column: str) -> float:
  # Mirrors `rebuild_categorical_target_in_arrays`: a component is available
  # only when BOTH its column and its mask say so, and an absent mask reads
  # 0.0 (`_get_mask`'s default), never "everything is labelled".
            column_value = batch.get(column)
            mask = batch.get(mask_key)
            if column_value is None or mask is None or rows <= 0.0:
                return 0.0
            return float(torch.as_tensor(mask).float().mean().item())

        target = batch.get("categorical_t")
        self.categorical_readouts.append(CategoricalRebuildReadout(
            rebuild_enabled=self._rebuild_categorical,
            blend_frac=float(getattr(self._categorical_params, "blend_frac", 0.0)),
            search_blend_frac=float(
                getattr(self._categorical_params, "search_blend_frac", 0.0),
            ),
            target_present=target is not None,
            sf_labelled_frac=labelled_frac("has_sf_wdl", "sf_wdl"),
            search_labelled_frac=labelled_frac("has_search_wdl", "search_wdl"),
            batch_rows=rows,
        ))

    def observe(
        self,
        batch: dict[str, Any],
        kwargs: dict[str, Any],
        result: dict[str, torch.Tensor],
    ) -> None:
        self.calls += 1
        self.kwargs = dict(kwargs)
        self._observe_categorical(batch, float(result["batch_rows"].detach().item()))
        self.readouts.append(value_blend_readout(
            sf_wdl_frac=float(kwargs.get("sf_wdl_frac", 0.0)),
            search_wdl_frac=float(kwargs.get("search_wdl_frac", 0.0)),
  # ⚑ EFFECTIVE, not `sf_wdl_rows`. `sf_wdl_rows` is the LABEL count: it
  # ignores the `sf_search_dampen_*` shortfall and reads non-zero on a batch
  # with no `sf_wdl` column at all. And the search count is not optional — it
  # is 1.00 of this arm's value target (the shipped control config runs
  # search_wdl_frac 1.0; the 0.70 this line quoted until 2026-08-16 came off
  # main's stale yaml) and had no reader until review F1.
            sf_effective_rows=float(result["sf_wdl_effective_rows"].detach().item()),
            search_effective_rows=float(
                result["search_wdl_effective_rows"].detach().item(),
            ),
            batch_rows=float(result["batch_rows"].detach().item()),
        ))

    @property
    def worst(self) -> ValueBlendReadout:
        """The call with the largest leak — the one a guard must judge.

        Not the first and not the mean: the failure this guards against can
        begin partway through a run (a live edit, a controller push), and a
        mean over 800 steps would dilute a leak that started at step 700 into
        something under any tolerance.
        """
        return max(self.readouts, key=lambda r: r.leaked_to_outcome)

    @property
    def worst_categorical(self) -> CategoricalRebuildReadout:
        """The categorical call with the largest outcome share — same rule as
        ``worst``, and for the same reason: a corpus whose ``categorical_target``
        appears partway through a run must not be averaged away."""
        return max(self.categorical_readouts, key=lambda r: r.outcome_borne_frac)


class CaptureRealizedLosses:
    """Wrap ``trainer.compute_loss`` for the duration of the block.

    A class rather than ``@contextmanager`` so the captured object has a real
    type at the call site: the whole point of this block is that a reviewer can
    see what the guard is judging.
    """

    def __init__(self, *, rebuild_categorical: bool, categorical_params: Any) -> None:
        self.capture = _LossCapture(
            rebuild_categorical=rebuild_categorical,
            categorical_params=categorical_params,
        )
        self._original = trainer_module.compute_loss

    def __enter__(self) -> _LossCapture:
        capture, original = self.capture, self._original

        def wrapped(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            result = original(*args, **kwargs)
  # ⚑ `compute_loss(outputs, batch, **loss_kwargs)` — the batch is the SECOND
  # POSITIONAL argument at both of `trainer.py`'s call sites. Reading it out of
  # `kwargs` would silently observe an empty batch on every step.
            batch = args[1] if len(args) > 1 else kwargs.get("batch", {})
            capture.observe(batch, kwargs, result)
            return result

        trainer_module.compute_loss = wrapped
        return capture

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        trainer_module.compute_loss = self._original


def _as_replay_buffer(sampler: Any) -> ReplayBuffer:
    """Duck-typed stand-in, exactly as ``scripts/offline_replay_epoch.py`` does.

    ``train_steps`` needs only ``sample_batch_arrays`` + ``rng``, which
    ``DiskReplayBuffer`` has, but the two classes share no declared base.
    """
    return cast(ReplayBuffer, sampler)


def stage_shards(shard_dirs: list[Path], staging: Path) -> int:
    """Symlink every shard into ONE flat directory with unique indices.

    ``DiskReplayBuffer`` reads a single directory and ``iter_shard_paths``
    does not recurse, while each conversion run numbers its shards from
    ``shard_000000`` — so N converted hours cannot be pointed at directly
    without colliding. Symlinks keep the buffer on the production ingest path
    (rather than a bespoke loader that would no longer be the stack under
    test) and copy no data.

    ⚑ The buffer is opened ``read_only=True`` downstream regardless. That
    class DELETES SHARDS from a writable directory whose capacity differs from
    the live one; through a symlink farm it would delete the CONVERTED
    ORIGINALS, which cost ~3.5 minutes per 400 games to rebuild.
    """
    staging.mkdir(parents=True, exist_ok=True)
  # ⚑ SYMLINKS are cleared; anything else is REFUSED rather than deleted. A real
  # `.zarr` copied in by hand, or a partial write, would otherwise survive into
  # this run's pool — `iter_shard_paths` would read it and the trained corpus
  # would not be the one `summary.json`'s `corpus.shard_dirs` names. Deleting it
  # instead is not an option on this path: nothing here removes data it did not
  # create. (Normally unreachable now, because a populated --out-dir is refused
  # one frame up; this is the backstop for a staging dir that is populated
  # without the run artifacts being present.)
    intruders = sorted(
        entry.name for entry in staging.iterdir() if not entry.is_symlink()
    )
    if intruders:
        raise ValueError(
            f"{staging} holds entries that are not symlinks: "
            + ", ".join(intruders)
            + ". They would be sampled as part of this run's corpus while "
            "summary.json named only --shards. Move them aside; this path "
            "deletes nothing.",
        )
    for stale in staging.iterdir():
        if stale.is_symlink():
            stale.unlink()
    index = 0
    for shard_dir in shard_dirs:
        paths = iter_shard_paths(shard_dir)
        if not paths:
            raise ValueError(f"no shards under {shard_dir}")
        for path in paths:
            (staging / f"shard_{index:06d}.zarr").symlink_to(path.resolve())
            index += 1
    if index == 0:
        raise ValueError("no shards found in any --shards directory")
    return index


def purity_receipt_problems(
    receipt_path: Path, shard_dirs: list[Path],
) -> tuple[dict[str, Any], list[str]]:
    """The banked purity check, and every reason it does not cover THIS corpus.

    ⚑ REVIEW F5. ``purity --train-shards`` and ``--shards`` here were two
    unconnected CLI arguments: nothing in ``summary.json`` recorded which
    directories were trained on, so the arm's held-out slope had no
    machine-checkable evidence that the corpus it trained on is the corpus
    purity cleared. Set equality against the RESOLVED directories is the check
    — not a count, not a "looks right", and not a flag somebody sets by hand.
    """
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
  # ⚑ BOTH SIDES RESOLVED (PR #438 review, finding 2). `used` was resolved and
  # `covered` was not, so the set difference compared a normalised path against
  # whatever spelling the receipt happened to carry. It agrees today only
  # because THIS repo's writer resolves before banking; any other producer, a
  # relative path, or a symlinked mount makes the comparison answer a question
  # about spelling instead of about identity — in both directions, so it can
  # pass wrongly as easily as it can fail wrongly.
    covered = {str(Path(d).resolve()) for d in receipt.get("train_shards", ())}
    used = {str(Path(d).resolve()) for d in shard_dirs}
    problems: list[str] = []
    if not receipt.get("pure", False):
        problems.append(
            f"the purity receipt {receipt_path} records pure=False "
            f"({receipt.get('exposed_inputs')} exposed inputs). The held-out "
            "set measures exposure recency, not generalisation.",
        )
    if used - covered:
        problems.append(
            "these --shards directories are NOT covered by the purity receipt: "
            + ", ".join(sorted(used - covered))
            + ". The receipt cleared: " + (", ".join(sorted(covered)) or "<none>"),
        )
  # ⚑⚑ THE BANKED HASH IS NOW APPLIED, NOT JUST CARRIED. `frozen_sha256` was
  # written by `lc0_control_heldout`, copied into `summary.json`, and printed —
  # and never once compared to the artifact it names, which made it a receipt
  # field rather than a check (the accepted-then-ignored shape). The directory
  # sets above tie the corpus by NAME; this is the only tie by CONTENT, and it
  # is what catches a held-out set re-frozen (new sample, new seed, repaired
  # split) after the receipt was written, which renames nothing.
  #
  # ⚑ A MISSING ARTIFACT IS A PROBLEM, NOT A SKIP. "verify when convenient" is
  # a gate that cannot fail; the receipt is meaningless without the set it
  # describes, and the fix is to re-freeze or re-run purity, not to launch.
    banked_sha = str(receipt.get("frozen_sha256") or "")
    frozen_named = str(receipt.get("frozen") or "")
    if not banked_sha or not frozen_named:
        problems.append(
            f"the purity receipt {receipt_path} names no frozen artifact "
            f"(frozen={frozen_named or '<absent>'}, "
            f"frozen_sha256={banked_sha or '<absent>'}), so nothing ties it to "
            "the held-out set by CONTENT. Re-run `lc0_control_heldout purity "
            "--receipt` with a build that banks both.",
        )
    elif not Path(frozen_named).exists():
        problems.append(
            f"the frozen held-out set the purity receipt describes is GONE: "
            f"{frozen_named}. Its sha256 {banked_sha} cannot be verified, so "
            "the receipt clears a corpus against a set nobody can produce. "
            "Re-freeze and re-run purity.",
        )
    else:
        digest = sha256_file(Path(frozen_named))
        if digest != banked_sha:
            problems.append(
                f"the frozen held-out set {frozen_named} has CHANGED since the "
                f"purity receipt was written: sha256 {digest} on disk, "
                f"{banked_sha} banked. The receipt cleared a different set — "
                "re-run purity against the set this run will actually be "
                "scored on.",
            )
    return receipt, problems


def preflight(
    cfg: dict[str, Any], shard_dirs: list[Path], *, allow_leak: bool,
    allow_mixed_history: bool = False,
) -> dict[str, dict[str, int]]:
    """The two LAUNCH-level value-blend guards. Returns the measured coverage.

    The coverage is RETURNED rather than only printed so ``summary.json`` can
    record what the run actually read — a corpus identity nobody can reconstruct
    from a log line that scrolled past (review F5).

    ``allow_leak`` downgrades them to a banner instead of skipping them. The
    point of the flag is to let the run REACH the realized (per-step) check so
    the leak can be observed as a number — and, since that check now also
    runs under the flag, so it can be observed RAISING.
    """
    def fail(message: str) -> None:
        if allow_leak:
            print(f"⚑ --allow-leak: IGNORING launch guard — {message}")
            return
        raise SystemExit(f"REFUSING TO LAUNCH — {message}")

    try:
        assert_pid_cannot_reassert_sf_wdl(
            sf_wdl_frac=float(cfg.get("sf_wdl_frac", 0.0)),
            sf_wdl_frac_floor=float(cfg.get("sf_wdl_frac_floor", 0.0)),
            context="launch config",
        )
    except ValueBlendMisconfigured as exc:
        fail(str(exc))
    else:
        print("[preflight] PID sf_wdl ramp disabled at source (sf_wdl_frac <= 0)")

  # ⚑ COVERAGE, not `any()`. A single SF-labelled row anywhere in a mixed
  # --shards list used to set have_sf=True, which made `run_config_problems`
  # return [] and waved the whole corpus through (Codex #2). The question the
  # gate needs answered is "do ALL these rows carry a label", so measure it.
  #
  # ⚑⚑ AND BOTH LABELS, not just the SF one (review F1). `search_wdl_frac` is
  # 1.00 of this arm's value target (search_wdl_frac 1.0 as shipped; the 0.70
  # here was main's stale number) and `compute_loss` falls the search
  # component back to the raw one-hot on an unlabelled row exactly as it falls
  # the SF component back. Measuring one and assuming the other is how the
  # larger term went unchecked.
    coverage: dict[str, tuple[int, int]] = {}
    for flag, measure in (
        ("sf_wdl", shard_dir_sf_wdl_coverage),
        ("search_wdl", shard_dir_search_wdl_coverage),
    ):
        labelled = rows = 0
        for shard_dir in shard_dirs:
            dir_labelled, dir_rows = measure(Path(shard_dir))
            labelled += dir_labelled
            rows += dir_rows
            print(f"[preflight] {Path(shard_dir).name}: {flag} label coverage "
                  f"{dir_labelled}/{dir_rows}")
        coverage[flag] = (labelled, rows)

    complete: dict[str, bool] = {}
    for flag, (labelled, rows) in coverage.items():
        complete[flag] = rows > 0 and labelled == rows
        print(f"[preflight] {flag} coverage over all shards: {labelled}/{rows} "
              f"-> treating the corpus as {flag}-labelled: {complete[flag]}")
        if 0 < labelled < rows:
            fail(
                f"the corpus is PARTIALLY {flag}-labelled ({labelled}/{rows} "
                "rows). Every value-blend decision downstream assumes one "
                "regime or the other; split the run rather than averaging two "
                "corpora.",
            )
    have_sf = complete["sf_wdl"]
    problems = run_config_problems(
        cfg,
        shards_have_sf_wdl=have_sf,
        shards_have_search_wdl=complete["search_wdl"],
    )
    if problems:
        fail("the config is wrong for these shards:\n  " + "\n  ".join(problems))
    else:
        print("[preflight] converter run-config gate: no problems")

  # ⚑ TWO SEPARATE VALUE-TARGET GATES, and the order matters for the operator
  # rather than for correctness: the identity gate runs UNCONDITIONALLY (a mixed
  # corpus is wrong under every blend, including the shipped one), and the
  # blend gate is about what THIS config would add on top. Folding them into one
  # function is what hid the mixing hazard behind the blend's early return.
    for message in value_scheme_identity_problems(shard_dirs):
        fail(message)
    # ⚑ NOT downgraded by --allow-leak: mixing input histories is not a value
    # leak, and it has its own explicit opt-in.
    for message in history_identity_problems(
        shard_dirs, allow_mixed_history=allow_mixed_history,
    ):
        raise SystemExit(f"REFUSING TO LAUNCH — {message}")
    for message in baked_value_blend_problems(cfg, shard_dirs):
        fail(message)
    return {
        flag: {"labelled_rows": labelled, "rows": rows}
        for flag, (labelled, rows) in coverage.items()
    }


#: The one ``derive_value_scheme`` stamp under which ``search_wdl`` is the bare
#: searched root value.  Every other value scheme
#: (``scripts/derive_corpus_targets.py``'s value round) has already blended some
#: share of the GAME OUTCOME into that column.
UNBAKED_VALUE_SCHEME = "search"

#: The ``derive_schema`` numbers THIS code knows how to read: 1 is the original
#: contract (``search_wdl`` is the searched root value) and 2 is the value
#: round's baked target.
#:
#: ⚑⚑ THE POINT OF READING IT AT ALL.  MEASURED on this tree: `derive_schema` had
#: no reader anywhere in the repo -- the deriver stamped it, one write-side test
#: asserted it, and `load_shard_arrays` copied the whole attrs dict through
#: without inspecting a single version field.  A schema number nobody reads is
#: this repo's signature defect in its purest form, and bumping such a number
#: protects nothing.  So the bump comes with the reader that makes it mean
#: something on the one path the value round actually trains through.
#:
#: ⚑ It FAILS CLOSED on a number from the future: a schema 3 written by a later
#: deriver is refused here rather than trained on under schema-2 assumptions,
#: which is the whole reason a version field exists.  The duplication with
#: ``derive_corpus_targets.DERIVE_SCHEMA*`` is deliberate -- importing that
#: module pulls the corpus generator's import chain into a training launcher --
#: and ``tests/test_derive_corpus_targets.py`` pins the two together, so a
#: divergence is a failing test rather than a silently narrowed gate.
KNOWN_DERIVE_SCHEMAS = (1, 2)

#: What ``derive_value_source`` reads as when a shard does not carry it.
#:
#: ⚑ THE PRE-``--value-depth`` CONTRACT: every corpus written before that flag
#: existed read its value off the scheme's own rung through the deepest phase
#: covering each move, and stamped nothing.  So this default is what those
#: shards HOLD, which is what keeps a legacy corpus and a freshly derived
#: depth-scheme corpus ONE identity and every existing arm launching.
#:
#: ⚑ It is wrong about exactly one legacy shape -- a ``nodes-<N>`` corpus
#: derived before the stamp, whose value really is ``phase0_only`` -- and that
#: is accepted knowingly: no round in the ledger has ever derived one, a nodes
#: corpus differs in its POLICY target too (a mixing hazard this gate never
#: claimed to cover), and the alternative is refusing every legacy corpus for
#: lack of a stamp it could not have written.
#:
#: ⚑ Duplicated from ``derive_corpus_targets.VALUE_SOURCE_DEEPEST`` for the same
#: reason ``KNOWN_DERIVE_SCHEMAS`` is -- importing that module pulls the corpus
#: generator's import chain into a training launcher -- and pinned to it by
#: ``tests/test_derive_corpus_targets.py``, so a divergence is a failing test
#: rather than a silently widened gate.
UNSTAMPED_VALUE_SOURCE = "deepest_phase_covering"


@dataclasses.dataclass(frozen=True)
class ShardValueStamps:
    """What ``--shards`` says about itself, read off the shards themselves."""

    #: ``derive_value_scheme`` -> one example ``dir/shard`` carrying it.  An
    #: absent stamp is recorded as :data:`UNBAKED_VALUE_SCHEME`, because that is
    #: what every pre-value-round corpus holds.
    schemes: dict[str, str]
    #: ``derive_schema`` -> one example ``dir/shard`` carrying it.
    schemas: dict[int, str]
    #: ``derive_value_source`` -> one example ``dir/shard`` carrying it.  An
    #: absent stamp is recorded as :data:`UNSTAMPED_VALUE_SOURCE`.
    sources: dict[str, str]


def read_value_stamps(shard_dirs: Sequence[Path]) -> ShardValueStamps:
    """Read every shard's value-identity attrs.  No policy, just the reading."""
    schemes: dict[str, str] = {}
    schemas: dict[int, str] = {}
    sources: dict[str, str] = {}
    for shard_dir in shard_dirs:
        for path in iter_shard_paths(Path(shard_dir)):
            attrs = zarr.open_group(str(path), mode="r").attrs
            where = f"{Path(shard_dir).name}/{path.name}"
            scheme = str(attrs.get("derive_value_scheme", UNBAKED_VALUE_SCHEME))
            schemes.setdefault(scheme, where)
            # ⚑ ABSENT MEANS 1, and it has to: every shard written before the
            # deriver stamped anything, and every `lc0_data_to_rows` corpus,
            # holds the original contract and carries no attr at all.
            schemas.setdefault(int(attrs.get("derive_schema", 1)), where)
            sources.setdefault(
                str(attrs.get("derive_value_source", UNSTAMPED_VALUE_SOURCE)),
                where,
            )
    return ShardValueStamps(schemes=schemes, schemas=schemas, sources=sources)


#: What a derived shard written before the history stamps existed holds: the
#: deriver of that era reconstructed a bare FEN per row, so its planes are the
#: zero-history distribution.  ⚑ Absent reads as THIS, deliberately: every
#: pre-#497 corpus is that shape, and the gate must keep them launchable while
#: still refusing to mix them with a history-aware one.
UNSTAMPED_CORPUS_ROW_SCHEMA = 1
UNSTAMPED_ZERO_HISTORY = True


@dataclasses.dataclass(frozen=True)
class ShardHistoryStamps:
    """What ``--shards`` says about the HISTORY its planes carry."""

    #: ``derive_corpus_row_schema`` (an int, or ``"mixed"``) -> example shard.
    row_schemas: dict[str, str]
    #: ``zero_history`` -> example shard.
    zero_history: dict[bool, str]

    @property
    def mixed(self) -> bool:
        return len(self.row_schemas) > 1 or len(self.zero_history) > 1


def read_history_stamps(shard_dirs: Sequence[Path]) -> ShardHistoryStamps:
    """Read every shard's history-identity attrs.  No policy, just the reading."""
    row_schemas: dict[str, str] = {}
    zero_history: dict[bool, str] = {}
    for shard_dir in shard_dirs:
        for path in iter_shard_paths(Path(shard_dir)):
            attrs = zarr.open_group(str(path), mode="r").attrs
            where = f"{Path(shard_dir).name}/{path.name}"
            row_schemas.setdefault(
                str(attrs.get("derive_corpus_row_schema", UNSTAMPED_CORPUS_ROW_SCHEMA)),
                where,
            )
            zero_history.setdefault(
                bool(attrs.get("zero_history", UNSTAMPED_ZERO_HISTORY)), where,
            )
    return ShardHistoryStamps(row_schemas=row_schemas, zero_history=zero_history)


def history_identity_problems(
    shard_dirs: Sequence[Path], *, allow_mixed_history: bool,
) -> list[str]:
    """⚑⚑ ONE input-history identity across ``--shards`` (Grok D3, round 2).

    ``derive_corpus_targets.py`` stamps ``derive_corpus_row_schema`` and
    ``zero_history`` on every shard, and until this reader existed nothing READ
    them: a directory derived from schema-1 bare-FEN rows (slots 1..7 and every
    repetition plane zero) and one derived from schema-3 windows carry the same
    ``input_history_encoding`` and the same ``history_rep_fix``, so the replay
    buffer's own identity merge lets them mix and the net trains on two input
    distributions no summary names.  ``--allow-mixed-history`` is the explicit
    opt-in, and the run's summary then records the mix.
    """
    stamps = read_history_stamps(shard_dirs)
    if not stamps.mixed or allow_mixed_history:
        return []
    schemas = ", ".join(f"{s} (e.g. {w})" for s, w in sorted(stamps.row_schemas.items()))
    zero = ", ".join(f"{z} (e.g. {w})" for z, w in sorted(stamps.zero_history.items()))
    return [
        f"--shards mixes input-history identities: derive_corpus_row_schema "
        f"{{{schemas}}}, zero_history {{{zero}}}. A zero-history shard fills "
        "history slot 0 only; a history-aware one fills all 8 frames and the "
        "repetition planes, and the champion flips 46.5% of its top-1 moves "
        "between the two. One replay buffer would sample both. Train one "
        "identity at a time, or pass --allow-mixed-history to record the mix "
        f"deliberately. (An absent stamp reads as row schema "
        f"{UNSTAMPED_CORPUS_ROW_SCHEMA}, zero_history {UNSTAMPED_ZERO_HISTORY}: "
        "corpora derived before the stamps existed are the bare-FEN shape.)",
    ]


def history_identity_record(
    stamps: ShardHistoryStamps, *, allow_mixed_history: bool,
) -> dict[str, Any]:
    """The summary's record of what the reader saw, mixed or not."""
    return {
        "row_schemas": sorted(stamps.row_schemas),
        "zero_history": sorted(stamps.zero_history),
        "mixed": stamps.mixed,
        "allow_mixed_history": bool(allow_mixed_history),
    }


def value_scheme_identity_problems(shard_dirs: Sequence[Path]) -> list[str]:
    """⚑⚑ ONE value-target identity across ``--shards``, whatever the blend is.

    This is NOT the ``game_frac`` guard below and must not be folded into it.
    That one asks "would training add the outcome to shards that already have
    it", and under the SHIPPED blend (``sf_wdl_frac 0.0`` / ``search_wdl_frac
    1.0``) its answer is legitimately no -- so it returns at its first branch
    and never looks at a stamp.  MEASURED: with the shipped blend, a
    ``--shards`` list naming a ``qz50`` directory, a ``qzsegment`` directory and
    a legacy one passed every launch guard, and the run trained on a per-row
    mixture of three different value targets -- a corpus that is none of the
    preregistered arms and that no summary anywhere would describe (Codex review
    of PR #491).

    The value round's entire method is that V0/A/B/C hold the SAME positions and
    differ only in ``search_wdl``.  A run whose buffer mixes two of them cannot
    be attributed to either, and nothing downstream would ever notice: the rows
    are shape-compatible, the encoding identity matches, and the buffer merges
    them without complaint.

    ⚑ An ABSENT stamp is ``search``, so a legacy corpus and a V0 corpus are ONE
    identity and mixing them is allowed -- they hold the same thing under the
    same contract.  That is the only default under which existing arms keep
    launching.

    ⚑⚑ THE IDENTITY IS THE PAIR ``(value scheme, value SOURCE)``, and the second
    half is not optional.  ``derive_corpus_targets.py --value-depth`` reads the
    searched value at a different rung from the scheme's own, so
    ``uniform-d7`` and ``uniform-d7 --value-depth 9`` derive from ONE bank into
    two corpora that agree on ``derive_value_scheme`` (both ``search``), on
    ``derive_schema`` (both 1) and even on their POLICY target -- while their
    ``search_wdl`` columns differ on every row.  Keyed on the scheme alone this
    gate returned ``[]`` for that pair: the value round's own mixing hazard, in
    the value round's own gate, invisible (Codex review of PR #494).  The
    source string spells the realized depth, so the pair separates them.
    """
    stamps = read_value_stamps(shard_dirs)
    problems: list[str] = []
    unknown = sorted(s for s in stamps.schemas if s not in KNOWN_DERIVE_SCHEMAS)
    if unknown:
        named = ", ".join(f"{s} (e.g. {stamps.schemas[s]})" for s in unknown)
        problems.append(
            f"--shards carries derive_schema {named}, which this code does not "
            f"know how to read (it understands {list(KNOWN_DERIVE_SCHEMAS)}). "
            "The schema number changes when the MEANING of an emitted column "
            "changes, so a newer one means search_wdl may hold something this "
            "trainer would misread as a value it is not. Update this launcher "
            "to the deriver that wrote these shards.",
        )
    if len(stamps.schemes) > 1:
        named = ", ".join(
            f"{scheme} (e.g. {where})" for scheme, where in sorted(stamps.schemes.items())
        )
        problems.append(
            f"--shards mixes {len(stamps.schemes)} different value-target "
            f"identities: {named}. Every row is sampled into ONE replay buffer, "
            "so the trained value target would be a per-row mixture of these "
            "arms rather than any one of them -- none of the preregistered "
            "V0/A/B/C cells, and nothing that could be compared against them. "
            "Train one --value-scheme at a time. (An absent stamp reads as "
            f"{UNBAKED_VALUE_SCHEME!r}: legacy and V0 corpora are one identity "
            "and may be mixed.)",
        )
    if len(stamps.sources) > 1:
        named = ", ".join(
            f"{source} (e.g. {where})" for source, where in sorted(stamps.sources.items())
        )
        problems.append(
            f"--shards mixes {len(stamps.sources)} different value SOURCES: "
            f"{named}. These corpora agree on their value scheme -- and may "
            "agree on their policy target too -- but their search_wdl columns "
            "were read at different depths, so every row sampled into the one "
            "replay buffer would carry whichever teacher depth its shard "
            "happened to come from. That is a per-row mixture of two "
            "value-depth arms and is none of them. Train one --value-depth at "
            f"a time. (An absent stamp reads as {UNSTAMPED_VALUE_SOURCE!r}: "
            "corpora predating the flag are one identity with the schemes' own "
            "read and may be mixed.)",
        )
    return problems


def baked_value_blend_problems(
    cfg: Mapping[str, Any], shard_dirs: Sequence[Path],
) -> list[str]:
    """⚑⚑ ``game_frac > 0`` against shards that already baked the outcome in.

    ``derive_corpus_targets.py --value-scheme {qz50,qzphase,qzsegment}`` writes a
    blend of the searched value and the game's outcome into ``search_wdl``, while
    ``wdl_target`` still carries the RAW outcome (it is a required shard field
    with no has-flag, so there is no way not to).  A run with ``game_frac > 0``
    therefore mixes the outcome in a SECOND time, on top of the share the
    derivation already chose -- producing a value target that is no arm of the
    value round, including the one whose name is stamped on the shards it came
    from.  Nothing else can see it: the shards are well formed, the coverage
    gates pass, the loss converges.

    ⚑ READ OFF THE SHARDS, not off a pin.  ``PRODUCTION_GAME_FRAC`` above
    already refuses a drift in production's own blend, and that guard is the
    reason this hazard is closed TODAY -- but it is closed by a hand-written
    constant that is correct for production's current recipe, so the day
    production legitimately moves to a positive ``game_frac`` and the pin is
    refreshed, this hazard silently opens.  The shard's own
    ``derive_value_scheme`` attr is a property of the DATA and cannot drift
    away from it.

    ⚑ Shards that carry no such attr are the pre-value-round corpora (and every
    ``lc0_data_to_rows`` corpus), which bake nothing: absent means unbaked, and
    that is the only default under which existing arms keep launching.

    ⚑⚑ THE OUTCOME SHARE IS DERIVED, NEVER READ BY NAME.  ``game_frac`` IS NOT A
    CONFIG KEY: ``configs/lc0_positive_control.yaml`` mentions it only in
    comments, and the value is the COMPLEMENT that
    ``normalize_value_blend_fracs`` computes from ``sf_wdl_frac`` and
    ``search_wdl_frac``.  The first cut of this gate read ``cfg["game_frac"]``,
    which is absent from every real flattened config -- so it took the ``0.0``
    default, returned ``[]`` at its first branch, and could not fire on ANY
    config that will ever be handed to it.  MEASURED: ``sf_wdl_frac 0.0 /
    search_wdl_frac 0.5`` leaves 0.5 of the value target on the raw outcome and
    the gate said ALLOWED.  A gate keyed to a name that never exists is this
    repo's signature defect wearing the uniform of the check written to stop it,
    and the tests could not see it because they passed literal
    ``{"game_frac": ...}`` dicts -- a shape no config file produces.

    So the share comes through the SAME function the loss uses (one
    implementation, two callers -- that function's own docstring), which is the
    same reason ``lc0_control_trainer.live_production_game_frac`` exists rather
    than a constant.  ⚑ From THIS RUN's ``cfg``, not from the live pin: the
    question here is what the arm being launched will put on the raw outcome,
    which is exactly what a deviating arm gets wrong.
    """
    _, _, game_frac = normalize_value_blend_fracs(
        float(cfg.get("sf_wdl_frac", 0.0) or 0.0),
        float(cfg.get("search_wdl_frac", 0.0) or 0.0),
    )
    if game_frac <= 0.0:
        return []
    baked = {
        scheme: where
        for scheme, where in read_value_stamps(shard_dirs).schemes.items()
        if scheme != UNBAKED_VALUE_SCHEME
    }
    if not baked:
        return []
    named = ", ".join(f"{scheme} (e.g. {where})" for scheme, where in sorted(baked.items()))
    return [
        f"game_frac={game_frac:.4f} would put that share of the RAW game outcome "
        f"on top of shards whose search_wdl ALREADY blended it in: {named}. "
        "Those shards were derived by derive_corpus_targets.py with a "
        f"--value-scheme other than {UNBAKED_VALUE_SCHEME!r}, whose manifest "
        "requires the outcome share to be 0 "
        "(required_training_overrides.game_frac). "
        # ⚑⚑ NAME THE KEYS THAT EXIST. An earlier version of this line told the
        # operator to "set game_frac=0.0" -- and `game_frac` is not a config
        # key, it is the complement this function has just DERIVED. Following
        # that advice puts an unknown key in the yaml, which
        # `flatten_run_config_defaults` raises on before the main parser is even
        # built, so the next launch dies at startup with an error about a key
        # this message invented (CLAUDE.md's live-yaml category (a)). A refusal
        # that hands the operator a remedy which cannot work is worse than one
        # that hands them none (Codex review of PR #491).
        "FIX: there is no game_frac key to set -- raise sf_wdl_frac + "
        "search_wdl_frac until they sum to at least 1.0, which leaves nothing "
        "for the outcome complement; the shipped control's sf_wdl_frac: 0.0 "
        "with search_wdl_frac: 1.0 already does. Or train "
        f"on a --value-scheme {UNBAKED_VALUE_SCHEME} corpus.",
    ]


def preflight_architecture(
    config_path: Path, *, allow_drift: bool, model: torch.nn.Module | None = None,
) -> str:
    """LAUNCH guard 0 — the arm must build the architecture PRODUCTION RUNS.

    Called TWICE, and the second call is the one that means something new.
    Before the model exists this can only compare ``model:`` sections; once it
    exists, ``model=`` puts the pinned 61,444,448 against the net this tree's
    code actually built from those keys. Two different failures — a config that
    says the wrong thing, and code that builds the wrong thing from the right
    config — and only the second is visible after ``build_model``.

    ⚑ The env var is resolved HERE and passed in, not read inside the assert:
    a library function whose verdict depends on the caller's shell fails
    differently in the operator's terminal than in the test suite (review F6).
    """
    raw = load_yaml_file(str(config_path))
    stage = "built model" if model is not None else "config"
    try:
        provenance = assert_control_matches_live_architecture(
            raw, model=model, live_config=live_production_config_path(),
            context=f"lc0 control launch ({stage})",
        )
    except ControlArchitectureDrift as exc:
        if not allow_drift:
            raise SystemExit(f"REFUSING TO LAUNCH — {exc}") from exc
        print(f"⚑⚑ --allow-arch-drift: THIS IS NOT A VALID POSITIVE CONTROL — {exc}")
        return "DRIFTED (--allow-arch-drift)"
    print(f"[preflight] architecture ({stage}) matches {provenance}")
    return provenance


def preflight_trainer(cfg: dict[str, Any], *, allow_leak: bool) -> str:
    """LAUNCH guard 0c — the arm must train with PRODUCTION'S RECIPE.

    ⚑ ``--allow-arch-drift`` does NOT downgrade this one, and ``--allow-leak``
    does. That split is not arbitrary. ``--allow-arch-drift`` exists because the
    plumbing has to be smoke-testable on a deliberately tiny model, and trainer
    kwargs are width-independent — every smoke run in
    ``tests/test_lc0_control_drivers.py`` shrinks ``model:`` and ``batch_size``
    while keeping the recipe verbatim, and ``batch_size`` is not a trainer
    kwarg — so a drifted architecture says nothing about the recipe.
    ``--allow-leak``, by contrast, works by putting the value blend BACK to
    production's, which is precisely the pair of kwargs
    ``LC0_TRAINER_DEVIATIONS`` says must differ. A leak demonstration is a
    trainer deviation by construction; refusing it here would make the headline
    realized guard unreachable again, which is exactly the defect review F2
    closed.

    ⚑ The env var is resolved HERE and passed in, so the library function's
    verdict does not depend on the caller's shell (review F6, same as guard 0).
    """
    try:
        provenance = assert_control_matches_live_trainer(
            cfg, live_config=live_production_config_path(),
            context="lc0 control launch (trainer)",
        )
    except ControlTrainerDrift as exc:
        if not allow_leak:
            raise SystemExit(f"REFUSING TO LAUNCH — {exc}") from exc
        print(f"⚑ --allow-leak: IGNORING launch guard — {exc}")
        return "DRIFTED (--allow-leak)"
    print(f"[preflight] trainer recipe matches {provenance}")
    return provenance


def preflight_replay(cfg: dict[str, Any], *, allow_leak: bool) -> str:
    """LAUNCH guard 0d — the arm must SAMPLE from production's buffer.

    ⚑⚑ THE AXIS NEITHER EXISTING PIN COULD SEE. Guard 0 compares the ``model:``
    section, guard 0c compares ``trainer_kwargs_from_config``. ``DiskReplayBuffer``
    is built by ``tune/trainable_init.py`` from ``TrialConfig`` fields that
    function does not read, so a driver that omitted them took the constructor
    defaults and BOTH pins still passed — the review's "a value accepted and
    then silently ignored", one construction site over.

    ⚑ ``--allow-leak`` downgrades this to a banner, on the same reasoning as
    guard 0c: the flag's job is to let a deliberately-wrong run REACH the
    realized per-step guard, and a smoke run that has to match production's
    100,000-row shuffle pool is not a smoke run. ``--allow-arch-drift`` does NOT
    downgrade it — buffer kwargs are width-independent, exactly as trainer
    kwargs are, so a toy model says nothing about the corpus draw.
    """
    try:
        provenance = assert_control_matches_live_replay(
            cfg, live_config=live_production_config_path(),
            context="lc0 control launch (replay)",
        )
    except ControlReplayDrift as exc:
        if not allow_leak:
            raise SystemExit(f"REFUSING TO LAUNCH — {exc}") from exc
        print(f"⚑ --allow-leak: IGNORING launch guard — {exc}")
        return "DRIFTED (--allow-leak)"
    print(f"[preflight] replay buffer matches {provenance}")
    return provenance


class SaveMidBudgetCheckpoint:
    """Save a MID-BUDGET checkpoint from inside the ONE ``train_steps`` call.

    ⚑⚑ THE PREREG'S DECIDING STATISTIC WAS NOT PRODUCIBLE BY THIS DRIVER.
    `docs/lc0_positive_control_prereg.md` reads the arm as the JOINT reading of
    two slopes, "both measured LAST vs MID-BUDGET" — and the driver had one
    ``train_steps`` call and one ``trainer.save``, so it emitted exactly one
    checkpoint. There was nothing to pair the last one against.

    ⚑⚑ AND THE OBVIOUS FIX IS WRONG. Splitting an N-step budget into
    ``train_steps(N/2)`` twice does NOT give one trajectory: with
    ``lr_schedule: sqrt_release`` and ``lr_release_cycle_steps: 0`` — which is
    what this arm and production both run — ``train_steps`` derives the release
    cycle from ITS OWN argument (``effective_cycle_steps = max(1,
    requested_steps)``) and feeds it ``local_step=train_steps_done``, which
    restarts at 0 on every call. Two calls of N/2 therefore run the release ramp
    TWICE, so MID and LAST would come off two different LR trajectories and the
    slope between them would be partly a schedule artifact. Measured, not
    argued: see ``test_the_mid_budget_split_would_have_re_ramped_the_lr``.

    So the checkpoint is taken by wrapping ``_run_optimizer_step`` — the same
    "wrap the real callee" idiom ``CaptureRealizedLosses`` uses one frame up —
    and fires AFTER the optimizer has applied step ``at_step``, with the run
    continuing into the same call. One trajectory, one LR ramp, two checkpoints.

    ⚑ Counted on RETURN, so a step that raised and is being retried by
    ``train_steps``'s CUDA-error loop is not counted twice.
    """

    def __init__(self, trainer: Trainer, *, at_step: int, path: Path) -> None:
        self.trainer = trainer
        self.at_step = int(at_step)
        self.path = path
        self.steps_done = 0
        self.saved_at_step: int | None = None
        self._original = trainer._run_optimizer_step

    def __enter__(self) -> SaveMidBudgetCheckpoint:
        if self.at_step <= 0:
            return self
        original = self._original

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self.steps_done += 1
            if self.steps_done == self.at_step:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.trainer.save(self.path)
                self.saved_at_step = self.steps_done
                print(f"\n[checkpoint] MID-BUDGET at step {self.steps_done}: "
                      f"{self.path}")
            return result

        self.trainer._run_optimizer_step = wrapped
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
  # ⚑ DELETE the instance attribute rather than rebinding to the original.
  # Rebinding leaves a permanent per-instance slot holding a bound method --
  # a reference cycle, and a trainer that no longer resolves the method through
  # its CLASS, so a later monkeypatch of the class would not be seen. `del`
  # restores the object to the state it was in before the wrapper.
  #
  # ⚑ `pop`, not `del`: `__enter__` returns EARLY without wrapping when
  # `at_step <= 0` (the `--mid-checkpoint-frac 0` path that disables the
  # feature), so the attribute may never have been set. A bare `del` there
  # raises AttributeError out of `__exit__` and takes down the whole run --
  # measured: 18 driver tests went red on exactly that. A cleanup path that
  # can fail on the DISABLED configuration is worse than the cycle it fixes.
        self.trainer.__dict__.pop("_run_optimizer_step", None)


def _metric_fields(metrics: Any, predicate: Any) -> list[tuple[str, Any]]:
    return [
        (field.name, getattr(metrics, field.name))
        for field in sorted(dataclasses.fields(type(metrics)), key=lambda f: f.name)
        if predicate(field.name)
    ]


def print_realized(capture: _LossCapture, metrics: Any) -> None:
    """The realized-vs-configured table, head losses, and head denominators."""
    readout = capture.worst
    print("\n=== REALIZED VALUE BLEND (read off compute_loss, worst of "
          f"{capture.calls} calls) ===")
    for name, value in readout.as_table():
        print(f"  {name:38s} {value:.6f}")
    print("\n=== REALIZED CATEGORICAL TARGET (worst of "
          f"{capture.calls} calls) ===")
    for name, value in capture.worst_categorical.as_table():
        print(f"  {name:38s} {value:.6f}")
    print("\n=== REALIZED LOSS WEIGHTS (the kwargs compute_loss received) ===")
    for key in sorted(capture.kwargs):
        if key.startswith(("w_", "sf_wdl_", "search_wdl_", "policy_target_temp")):
            print(f"  {key:38s} {capture.kwargs[key]!r}")
    print("\n=== HEAD-BY-HEAD LOSSES ===")
    for name, value in _metric_fields(
        metrics, lambda n: ("loss" in n or n.endswith(("_ce", "_acc"))) and "rows" not in n,
    ):
        if isinstance(value, (int, float)):
            print(f"  {name:38s} {value:.6f}")
    print("\n=== HEAD DENOMINATORS (rows that actually trained each head) ===")
    print("  ⚑ A weight on a head whose target the shards do not carry is "
          "INERT, not applied.\n     These counts are the proof, summed over "
          "the run's microbatches.")
    for name, value in _metric_fields(
        metrics, lambda n: n.endswith(("_rows", "_frac")) or n.startswith("has_"),
    ):
        print(f"  {name:38s} {value!r}")


def control_validity_problems(
    *,
    allow_leak: bool,
    allow_arch_drift: bool,
    has_purity_receipt: bool,
    steps: int,
    warmup_steps: int,
    mid_saved_at_step: int | None,
    mid_checkpoint_frac: float,
    window_steps: int,
    cadence_problem: str | None,
    device: str,
    configured_device: str,
    batch_size: int,
    configured_batch_size: int,
    live_config_unread: bool,
    live_game_frac: float,
) -> list[str]:
    """Every reason this run is not a valid control — ONE implementation.

    ⚑⚑ CALLED TWICE: ONCE BEFORE THE FIRST STEP, ONCE AFTER THE LAST. Every entry
    below is knowable BEFORE training — the flags, the corpus, the budget, the
    window plan, the mid point, the device, the batch size — and until now they
    were only computed AFTERWARDS. That is what made N1 and its successor cost a
    day of GPU each: `--steps 20001` (mid-fraction, wave 4) and `--steps 20000`
    (LR cadence, wave 5) both ran to completion and only then stamped
    `valid_control: false`, which `compare` refuses with no waiver. `main` now
    evaluates this at launch and REFUSES unless the run has declared itself a
    smoke, so the class "a day of compute produces an unquotable artifact for a
    reason that was knowable at launch" is closed rather than relocated.

    ⚑ `mid_saved_at_step` is the PREDICTED step at launch and the REALIZED one
    afterwards, which is the only input that differs between the two calls;
    `test_every_disqualifying_reason_was_already_knowable_at_launch` pins that
    they agree.
    """
  # ⚑ EVERY reason the run is not a valid control, NAMED. A bare
  # `valid_control: false` says a run is disqualified without saying what for,
  # which is the same "a flag instead of a measurement" shape as the rest of
  # this review's findings.
  #
  # ⚑⚑ AND "RECORDED" MEANS RECORDED AT LAUNCH-TIME, NOT SOFT. Every entry below
  # sets `valid_control: false` (`not validity_problems`), and
  # `lc0_control_eval compare` refuses `valid_control: false` with NO WAIVER AT
  # ALL. So the distinction earlier revisions of this block drew — "recorded
  # rather than refused ... every plumbing smoke here runs at batch 4 by design"
  # — was not a distinction between a hard block and a soft note. It was the
  # difference between REFUSING TO LAUNCH and LETTING A DAY OF GPU PRODUCE AN
  # UNQUOTABLE ARTIFACT, and the second is what happened twice. `main` now calls
  # this BEFORE the first step and refuses unless `--allow-invalid-control` is
  # passed, so a plumbing smoke still runs (it says so) and the arm cannot spend
  # a day to learn something the arguments already said.
  #
  # ⚑ The LR-warmup entry is not bookkeeping. Production's `warmup_steps` is
  # 1000 and this arm carries it verbatim, so a run whose whole budget is
  # shorter than the warmup NEVER REACHES THE BASE LR — its held-out slope is a
  # property of the warmup ramp, not of the trainer.
    return [
        message for flag, message in (
            (allow_leak, "--allow-leak: the LAUNCH value-blend guards were "
                         "downgraded to banners"),
            (allow_arch_drift, "--allow-arch-drift: this is NOT production's "
                               "architecture"),
            (not has_purity_receipt, "no --purity-receipt: the trained corpus is "
                                     "not tied to any held-out purity check"),
            (steps <= warmup_steps,
             f"--steps {steps} does not exceed warmup_steps "
             f"{int(warmup_steps)}: the LR never reached the base "
             "value, so no slope from this run describes the trainer"),
  # ⚑ The prereg's PRIMARY yardstick is two slopes "both measured LAST vs
  # MID-BUDGET" on ONE trajectory. Without the mid checkpoint the run has
  # produced half of the deciding statistic, and nothing downstream would say
  # so — `lc0_control_eval compare` would happily pair this run's LAST against
  # some OTHER run's LAST and report a number.
            (mid_saved_at_step is None,
             f"no mid-budget checkpoint (--mid-checkpoint-frac "
             f"{mid_checkpoint_frac}, --steps {steps}): "
             "the prereg's primary yardstick is LAST vs MID-BUDGET on ONE "
             "trajectory, and this run emitted only LAST"),
  # ⚑ AND A MID CHECKPOINT AT THE WRONG PLACE IS NOT THE MID-BUDGET ONE. Any
  # positive fraction is clamped to an interior step, so `--mid-checkpoint-frac
  # 0.99` (or 5.0) wrote a checkpoint, satisfied the entry above, and left the
  # run reading VALID — while `compare` would present a LAST-vs-99%-budget
  # contrast, which measures the final 1% of training, as the preregistered
  # LAST-vs-MID-BUDGET slope.
  #
  # ⚑⚑ AND THE PREDICATE IS THE REALIZED FRACTION, NOT THE KNOB. Judging
  # `args.mid_checkpoint_frac` broke this file's own rule ("a configured value is
  # not an applied value") inside the guard added to enforce a preregistered
  # value: the requested fraction is snapped to a window boundary and clamped to
  # the interior, so `--steps 3 --mid-checkpoint-frac 0.5` puts MID at step 1 of
  # 3 — 33.3% of the budget — and recorded NOTHING. Measured by the independent
  # review.
            (mid_saved_at_step is not None
             and mid_fraction_deviates(
                 saved_at_step=mid_saved_at_step, steps=steps,
             ),
             f"the mid checkpoint was taken at step {mid_saved_at_step} of "
             f"{steps} = "
             f"{(mid_saved_at_step or 0) / max(steps, 1):.4f} of the "
             f"budget, more than {MID_CHECKPOINT_FRAC_TOLERANCE} from the "
             f"preregistered {PREREG_MID_CHECKPOINT_FRAC} "
             f"(--mid-checkpoint-frac {mid_checkpoint_frac}, "
             "snapped to a train-window boundary and clamped to the interior): "
             "the LAST-vs-MID slope this run supports is not the prereg's LAST "
             f"vs MID-BUDGET. ⚑ A boundary can miss 0.5 by up to half a window, "
             f"so with --train-window-steps {window_steps} this tolerance is "
             f"unreachable below "
             f"{min_budget_for_mid_tolerance(window_steps)} steps — raise "
             "--steps rather than the tolerance"),
  # ⚑⚑ REVIEW F2 — THE PROPERTY THE CADENCE FIX IS ACTUALLY FOR, AND NOTHING WAS
  # ENFORCING IT. The two guards above bound PROXIES: the realized fraction
  # (±0.01) and, in the previous revision, the budget's divisibility. At 17600
  # steps ±0.01 is ±2 whole 88-step windows, so `--mid-checkpoint-frac 0.5028`
  # put MID at LR scale 1.0 against LAST at 0.1 with both guards silent and
  # `valid_control: true` — the anneal-in-the-slope confound, reachable through a
  # PASSING knob. `mid_step_on_window_boundary` now snaps MID to a boundary at
  # the source; this entry is what says so if any future path does not.
            (mid_saved_at_step is not None and window_steps > 0
             and mid_saved_at_step % window_steps != 0,
             f"the mid checkpoint sits at step {mid_saved_at_step}, which is NOT "
             f"a multiple of the {window_steps}-step train window: with "
             "`lr_release_cycle_steps: 0` the LR release cycle is derived from "
             "each call's step count, so MID is at a different phase of the "
             "anneal from LAST and part of the MID->LAST difference is the LR "
             "moving, not the trainer learning"),
  # ⚑⚑ THE LR CADENCE — the confound the control config DOCUMENTED and nothing
  # measured. See `PRODUCTION_TRAIN_WINDOW_STEPS`. Now that the driver runs
  # windows and the window count is a CEILING, this is EMPTY for every budget
  # that holds at least two windows; it fires only for a smoke shorter than one
  # window, which cannot place MID and LAST at distinct boundaries at all.
            (cadence_problem is not None,
             f"LR cadence: {cadence_problem}. The MID->LAST slope then contains "
             "an anneal production never places between two of its own "
             "checkpoints, and an annealed endpoint scores better on held-out "
             "top-1 for reasons that are not the trainer learning"),
  # ⚑ `--device` IS DISQUALIFYING, and the comment that used to defer this
  # ("plausibly objective-neutral", "an entry here would disqualify every smoke")
  # was measurably wrong on the second clause: a clean 4-step CPU smoke already
  # banks FOUR entries (--allow-arch-drift, no purity receipt, steps <=
  # warmup_steps, no live config), so the entry costs nothing and its absence
  # made `realized_after_guard` a banked-and-unread field — this repo's signature
  # defect. Guard 0c certifies the recipe with LIVE_TRAINER_PIN's "cuda" and the
  # driver then overwrites it, so a CPU run has different bf16 kernels and
  # execution semantics from the stack the arm claims to be testing.
            (str(device) != str(configured_device),
             f"--device {device} is not the configured "
             f"{configured_device}: guard 0c certified the trainer recipe with "
             f"device={configured_device!r} and this run then overwrote it, so "
             "the kernels and execution semantics are not the production "
             "stack's"),
  # ⚑ THE GRADIENT-NOISE REGIME IS A TRAINING SETTING, AND `--batch-size` SAT
  # AFTER EVERY PREFLIGHT. Guard 0c certifies the trainer recipe against live's,
  # but `batch_size` is not a trainer kwarg — it is an argument to
  # `train_steps` — so a `--batch-size 32` run against production's configured
  # 512 changed the examples per step, and therefore the gradient noise, with
  # every guard still passing and `valid_control` still true.
            (batch_size != configured_batch_size,
             f"--batch-size {batch_size} is not the configured "
             f"{configured_batch_size}: the examples per step and hence the "
             "gradient-noise regime are not production's, so no slope from this "
             "run describes the production training stack"),
  # ⚑ REVIEW F2. Judging the arm's WHOLE PREMISE — "production's net and
  # production's trainer" — against a committed pin is strictly weaker
  # evidence than judging it against the live file, because the pin is a
  # snapshot and `configs/pbt2_small.yaml` is written only on the live branch.
  # The tests that catch a stale pin SKIP when no live file is named, and that
  # is correct for CI (making CI read an absolute host path was rejected as
  # review F6). So the check has to live HERE, in the run artifact: the same
  # rule the receipt already gets one entry up, for the same reason.
            (live_config_unread,
             "no live config was given (CHESS_LIVE_PRODUCTION_CONFIG unset or "
             "--live-config omitted): the architecture and/or trainer premise "
             "was judged against a COMMITTED PIN, which cannot detect that the "
             "live file has moved since the pin was recorded"),
  # ⚑⚑ THE OUTCOME BAR IS DERIVED FROM THE LIVE RECIPE, NOT FROM A CONSTANT IN
  # THIS TREE. `PRODUCTION_GAME_FRAC` is a hand-written 0.0, correct for live's
  # 0.69/0.31 — and both blend fracs are `LC0_TRAINER_DEVIATIONS` entries, so
  # guard 0c is required to IGNORE them. If production moves to, say, 0.50/0.20,
  # live leaves 0.30 of the value target on the raw game outcome while this arm
  # holds 0.00, guard 0c stays silent by construction, and the arm would be
  # stamped valid while no longer testing production's objective. Reported by
  # the third independent review (thread 3795733617). `live_game_frac` is
  # derived from the SAME reference signature the preflight judged against.
            (abs(float(live_game_frac) - PRODUCTION_GAME_FRAC) > 1e-9,
             f"production's value blend now leaves game_frac="
             f"{float(live_game_frac):.4f} on the RAW GAME OUTCOME, but this arm "
             f"is built to hold it at {PRODUCTION_GAME_FRAC:.4f} and "
             "`LC0_TRAINER_DEVIATIONS` makes guard 0c ignore both blend fracs, "
             "so nothing else can see the divergence: the arm is no longer "
             "training production's value objective. Refresh LIVE_TRAINER_PIN, "
             "`PRODUCTION_GAME_FRAC` and the two deviation reasons together"),
        ) if flag
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/lc0_positive_control.yaml"))
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="0 = config batch_size. ⚑ Any other value changes the examples per "
             "step and the gradient-noise regime, and is recorded in "
             "`validity_problems`: the run still finishes and writes its "
             "checkpoints, and `lc0_control_eval compare` then refuses it with no "
             "waiver.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-compile", action="store_true",
        help="skip torch.compile; changes throughput, not the objective",
    )
    parser.add_argument(
        "--allow-mixed-history", action="store_true",
        help="let --shards mix zero-history (schema-1-derived) and "
             "history-aware (schema-3-derived) directories; the mix is then "
             "recorded in summary.json under corpus.history_identity.",
    )
    parser.add_argument(
        "--allow-leak", action="store_true",
        help="⚑ downgrade the LAUNCH guards (0c, 0d, 1 and 2) to a banner so the "
             "run reaches the REALIZED per-step guard and that guard can be "
             "observed raising. It does NOT skip the realized guard. The run "
             "is NOT a valid control.",
    )
    parser.add_argument(
        "--allow-arch-drift", action="store_true",
        help="⚑⚑ proceed even though the config does not build production's "
             "live architecture. PLUMBING SMOKE RUNS ONLY — no number from such "
             "a run may be quoted as 'our stack'.",
    )
    parser.add_argument(
        "--mid-checkpoint-frac", type=float, default=PREREG_MID_CHECKPOINT_FRAC,
        help="⚑ where in the budget to write `checkpoint_mid.pt`. The prereg's "
             "PRIMARY yardstick is two slopes 'both measured LAST vs "
             "MID-BUDGET', so a run that emits one checkpoint cannot produce "
             "the deciding statistic at all. 0 disables it; the run then "
             f"records that it cannot answer the prereg. ⚑ The REALIZED step "
             f"must land on {PREREG_MID_CHECKPOINT_FRAC} of the budget — the "
             "knob is clamped to the interior, so `--steps 3 --frac 0.5` gives "
             "33%% — and anything else is recorded in `validity_problems`, which "
             "`compare` then refuses with no waiver.",
    )
    parser.add_argument(
        "--allow-invalid-control", action="store_true",
        help="⚑⚑ launch even though the arguments already guarantee "
             "`valid_control: false`. WITHOUT this flag the driver REFUSES before "
             "the first step and prints every reason, because `lc0_control_eval "
             "compare` rejects such an artifact with NO WAIVER — twice now a full "
             "budget ran to completion and was then stamped invalid for something "
             "its own command line already said. The flag does not make the run "
             "valid; it says the operator wants the checkpoints anyway (a "
             "plumbing smoke, or a deliberate leak demonstration).",
    )
    parser.add_argument(
        "--move-existing-aside", action="store_true",
        help="when --out-dir already holds a run's artifacts, RENAME it to "
             "<out-dir>.superseded_<UTC timestamp> and start clean. ⚑ There is no "
             "--overwrite: a rerun sharing a directory could delete the previous "
             "run's mid-budget checkpoint (a day of GPU) while its summary went on "
             "banking that file's sha256. A rename is reversible; a delete is not.",
    )
    parser.add_argument(
        "--train-window-steps", type=int, default=PRODUCTION_TRAIN_WINDOW_STEPS,
        help="⚑ run the budget in windows of this many steps, because with "
             "`lr_release_cycle_steps: 0` the LR release cycle is derived from each "
             "`train_steps` CALL: production calls it at ~88 steps per iteration, "
             "so its LR is a sawtooth, while ONE call of --steps holds LR flat for "
             "80%% of the experiment and anneals once — putting an anneal between "
             "MID and LAST that production never has. --steps must be an EVEN "
             "number of windows or the deviation is recorded.",
    )
    parser.add_argument(
        "--purity-receipt", type=Path, default=None,
        help="the JSON written by `lc0_control_heldout.py purity --receipt`. "
             "Refuses to launch unless it covers every --shards directory, and "
             "banks its frozen sha256 in summary.json. WITHOUT IT the run is "
             "recorded as not a valid control: nothing then ties the trained "
             "corpus to the held-out purity check.",
    )
    args = parser.parse_args(argv)

    arch_provenance = preflight_architecture(
        Path(args.config), allow_drift=bool(args.allow_arch_drift),
    )
    cfg = flatten_run_config_defaults(load_yaml_file(str(args.config)))
    trainer_provenance = preflight_trainer(cfg, allow_leak=bool(args.allow_leak))
    replay_provenance = preflight_replay(cfg, allow_leak=bool(args.allow_leak))
    shard_dirs = [Path(d) for d in args.shards]
  # ⚑ BEFORE the coverage preflight and before staging, and a REFUSAL rather
  # than a silent de-duplication: a de-dupe would train on a corpus different
  # from the one the operator named, which is the same "accepted and then
  # quietly altered" shape as the finding itself. There is no reading of
  # `--shards a a` under which the duplicate is wanted — the buffer would draw
  # that hour twice as often — so the run stops and says which directory.
    duplicates = duplicate_resolved_dirs(shard_dirs)
    if duplicates:
        raise SystemExit(
            "REFUSING TO LAUNCH — these --shards directories are named more "
            "than once (resolved): " + ", ".join(duplicates)
            + ". Each occurrence is staged under its own shard_NNNNNN.zarr "
            "symlink, so the replay buffer would OVERSAMPLE that source, and "
            "neither the coverage preflight (which sums over the list) nor the "
            "purity receipt (which compares sets) can see it. Name each "
            "directory once.",
        )
    coverage = preflight(
        cfg, shard_dirs, allow_leak=bool(args.allow_leak),
        allow_mixed_history=bool(args.allow_mixed_history),
    )
    history_identity = history_identity_record(
        read_history_stamps(shard_dirs),
        allow_mixed_history=bool(args.allow_mixed_history),
    )

    receipt: dict[str, Any] | None = None
    receipt_path = Path(args.purity_receipt) if args.purity_receipt else None
    if receipt_path is not None:
        receipt, receipt_problems = purity_receipt_problems(
            receipt_path, shard_dirs,
        )
        if receipt_problems:
            raise SystemExit(
                "REFUSING TO LAUNCH — the purity receipt does not cover this "
                "corpus:\n  " + "\n  ".join(receipt_problems),
            )
        print(f"[preflight] purity receipt covers every --shards directory "
              f"(frozen sha256 {receipt.get('frozen_sha256')})")
    else:
        print("⚑ no --purity-receipt: NOTHING ties this corpus to a held-out "
              "purity check, and summary.json will record valid_control: false")

    torch.manual_seed(int(args.seed))
    out_dir = Path(args.out_dir)
  # ⚑⚑ BEFORE ANYTHING IS WRITTEN. See `existing_run_artifacts`: sharing an
  # --out-dir let a FAILING rerun's cleanup irreversibly delete a COMPLETED run's
  # mid-budget checkpoint while that run's summary went on banking its sha256.
  # Refused, and the escape RENAMES rather than deletes — nothing on this path may
  # remove a checkpoint it did not create.
    existing = existing_run_artifacts(out_dir)
    if existing and args.move_existing_aside:
        moved = out_dir.with_name(
            f"{out_dir.name}.superseded_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        )
        out_dir.rename(moved)
        print(f"⚑ --move-existing-aside: {out_dir} held {', '.join(existing)}; "
              f"RENAMED to {moved} (nothing deleted — recover by renaming back)")
    elif existing:
        raise SystemExit(
            f"REFUSING TO LAUNCH — {out_dir} already holds "
            + ", ".join(existing)
            + ". Reusing it shares the directory with a previous run: a rerun "
            "that fails its realized guard deletes the mid-budget checkpoint it "
            "found there, while the OLD summary.json goes on banking that file's "
            "path and sha256 and the old checkpoint.pt still verifies as "
            "role=last — a half-destroyed run that reads as a scorable success. "
            "Point --out-dir at a new directory, or pass --move-existing-aside "
            "to RENAME this one (there is deliberately no --overwrite: nothing "
            "here deletes a checkpoint).",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    staged = stage_shards(shard_dirs, out_dir / "staged_shards")
    print(f"[data] staged {staged} shard(s) from {len(shard_dirs)} directory(ies)")

    kwargs = trainer_kwargs_from_config(cfg)
  # ⚑ Read BEFORE the override, or the check below compares the override with
  # itself -- a gate that cannot fail.
    configured_device = str(kwargs["device"])
    if args.device:
        kwargs["device"] = args.device
    if args.no_compile:
        kwargs["use_compile"] = False
    configured_batch_size = int(cfg.get("batch_size", 512))
    batch_size = int(args.batch_size) or configured_batch_size

    model_cfg = model_config_from_flat_config(cfg)
    model = build_model(model_cfg)
  # ⚑ model_config is not decoration: the trainer derives its input-history
  # encoding from it, and without it `select_input_history_arrays` refuses
  # every LC0-root row in the corpus. Same construction as tune/trainable.py.
    trainer = Trainer(model, model_config=model_cfg, **kwargs)
  # ⚑ Unique STORAGE, not sum(numel) over the state_dict: the 16
  # `layer_smolgens.N.gen_weight.weight` keys are one shared tensor (CLAUDE.md).
    params = unique_storage_param_count(model)
    print(f"[model] {params} trainable params on {kwargs['device']}")
  # ⚑ GUARD 0b. The pinned count gated NOTHING until 2026-08-16: no caller ever
  # passed `model=`, so the arch fix's own headline number (61,444,448) was a
  # decoration inside the module that fixed a decoration (review F4). This is
  # the only check that can see this tree's code building a different net from
  # the same `model:` keys — a schema key the builder ignores, a normalisation
  # that differs, a head this tree wires up differently. The `model:` mapping
  # agreeing is not the net agreeing.
    preflight_architecture(
        Path(args.config), allow_drift=bool(args.allow_arch_drift), model=model,
    )

  # ⚑⚑ EVERY buffer kwarg comes from ONE dict, and it is the dict guard 0d
  # judged. Until 2026-08-17 this call site hand-wrote three kwargs and passed
  # nothing else, so `disk_buffer.py`'s CONSTRUCTOR DEFAULTS silently supplied
  # the rest: SEVEN axes off production (shuffle_cap 20000 vs 100000, shard_size
  # 1000 vs 2000, refresh_interval 5 vs 4, refresh_shards 3 vs 5, diff_focus
  # 0.0/0.0 vs 3.5/6.0, input_planes None vs 175) while the comment here claimed
  # two and both existing pins passed — `trainer_kwargs_from_config` does not
  # read a single one of them, so the trainer pin is structurally blind.
  #
  # ⚑ The two DECLARED deviations are applied by `apply_control_deviations`,
  # whose overrides are keys of `LC0_REPLAY_DEVIATIONS` — the same mapping the
  # guard reads. A hand-written override next to a check of different values is
  # review F6's defect (certified, then overwritten, with nothing in the
  # artifact saying so); the realized values are banked in summary.json.
    replay_kwargs = apply_control_deviations(replay_kwargs_signature(cfg))
    buf = DiskReplayBuffer(
        capacity=10**9,
        shard_dir=out_dir / "staged_shards",
        rng=np.random.default_rng(int(args.seed)),
  # ⚑ NEVER False here. See stage_shards: a writable buffer enforces its
  # window in __init__ and would delete the converted shards through the
  # symlinks.
        read_only=True,
        **replay_kwargs,
    )
    print(f"[data] replay buffer: {len(buf)} rows in the hot shuffle pool "
          f"over {staged} shard(s)")
    print("[data] replay kwargs (realized): "
          + " ".join(f"{k}={v!r}" for k, v in sorted(replay_kwargs.items())))

    window_steps, n_windows, cadence_problem = train_window_plan(
        steps=int(args.steps), window=int(args.train_window_steps),
    )
  # ⚑⚑ SNAPPED TO A WINDOW BOUNDARY, not `int(frac * steps)` clamped to the
  # interior. See `mid_step_on_window_boundary`: the fraction guard bounds a
  # PROXY (±0.01 is ±2 whole windows at the arm's budget), so a passing knob
  # could still put MID at LR scale 1.0 against LAST at 0.1. Snapping makes MID
  # sit at the bottom of a release cycle, which is where LAST sits and where two
  # production checkpoints sit.
    mid_step = mid_step_on_window_boundary(
        steps=int(args.steps), window=int(window_steps),
        frac=float(args.mid_checkpoint_frac),
    )
    mid_ckpt = out_dir / "checkpoint_mid.pt"
    print(f"[train] {n_windows} window(s) of {window_steps} step(s) "
          f"(production cadence ~{PRODUCTION_TRAIN_WINDOW_STEPS}/iteration; the "
          "LR release cycle is derived from each call's step count)")
    if cadence_problem is not None:
        print(f"⚑ CADENCE: {cadence_problem}")
    if mid_step:
        print(f"[train] mid-budget checkpoint at step {mid_step} of "
              f"{int(args.steps)} ({mid_step / max(int(args.steps), 1):.4f} of "
              f"the budget, window boundary "
              f"{mid_step // max(int(window_steps), 1)}/{n_windows})")

  # ⚑⚑ THE LAUNCH-TIME REFUSAL. EVERY entry below is knowable from the
  # arguments, the config and the plan — and until this call existed they were
  # computed only AFTER the last step, so TWICE a full run ended
  # `valid_control: false` for a reason its own command line already contained
  # (`--steps 20001`, then `--steps 20000` against the divisibility rule). The
  # mid step passed here is the PLANNED one; the post-run call re-evaluates with
  # the REALIZED one, and `test_launch_refusal_lists_the_same_problems_as_the_run`
  # pins that a clean run's two readings agree.
  #
  # ⚑ `--allow-invalid-control` is what a plumbing smoke passes. It does NOT
  # make the run valid — every problem still lands in `validity_problems` and
  # `compare` still refuses the artifact with no waiver — it only says the
  # operator knows and wants the checkpoints anyway.
    live_game_frac, game_frac_provenance = live_production_game_frac(
        live_config=live_production_config_path(),
    )
    print(f"[preflight] production game_frac {live_game_frac:.4f} derived from "
          f"{game_frac_provenance}")
    planned_problems = control_validity_problems(
        allow_leak=bool(args.allow_leak),
        allow_arch_drift=bool(args.allow_arch_drift),
        has_purity_receipt=receipt is not None,
        steps=int(args.steps),
        warmup_steps=int(kwargs["warmup_steps"]),
        mid_saved_at_step=mid_step or None,
        mid_checkpoint_frac=float(args.mid_checkpoint_frac),
        window_steps=int(window_steps),
        cadence_problem=cadence_problem,
        device=str(kwargs["device"]),
        configured_device=configured_device,
        batch_size=batch_size,
        configured_batch_size=configured_batch_size,
        live_config_unread=(_LIVE_FILE_UNREAD in arch_provenance
                            or _LIVE_FILE_UNREAD in trainer_provenance),
        live_game_frac=live_game_frac,
    )
    if planned_problems and not args.allow_invalid_control:
        raise SystemExit(
            "REFUSING TO LAUNCH — this run cannot be a valid control, and every "
            "reason is already knowable from the arguments:\n  "
            + "\n  ".join(planned_problems)
            + "\n⚑ `lc0_control_eval compare` refuses valid_control: false with "
            "NO WAIVER, so finishing this run would spend its whole budget to "
            "produce an artifact that cannot be quoted. Fix the arguments, or "
            "pass --allow-invalid-control if you want the checkpoints anyway "
            "(a plumbing smoke).",
        )
    if planned_problems:
        print("⚑⚑ --allow-invalid-control: THIS RUN IS NOT A VALID CONTROL and "
              "its artifact cannot be quoted:\n  " + "\n  ".join(planned_problems))
    with CaptureRealizedLosses(
        rebuild_categorical=bool(kwargs["rebuild_categorical_target"]),
        categorical_params=kwargs["categorical_target_params"],
    ) as capture, SaveMidBudgetCheckpoint(
        trainer, at_step=mid_step, path=mid_ckpt,
    ) as mid:
  # ⚑⚑ WINDOWED AT PRODUCTION'S CADENCE, NOT ONE MONOLITHIC CALL. See
  # `PRODUCTION_TRAIN_WINDOW_STEPS`: with `lr_release_cycle_steps: 0` the release
  # cycle is derived from each CALL's `steps`, so one call of N gave the arm a
  # single flat-then-anneal trajectory while production runs a ~88-step sawtooth —
  # and the deciding MID->LAST slope then contained an anneal production never
  # places between two of its own checkpoints.
  #
  # ⚑ AND THIS IS NOT THE SPLIT `SaveMidBudgetCheckpoint`'s DOCSTRING WARNS
  # ABOUT. That warning is about TWO calls of N/2, which puts MID halfway up one
  # ramp and LAST at the bottom of another — two different phases. W windows of
  # `window` steps puts BOTH on a window BOUNDARY, i.e. both at the bottom of a
  # release cycle, which is exactly where two production checkpoints sit. The mid
  # checkpoint is still taken from inside the run by the same wrapper (it counts
  # optimizer steps, not calls), so there is still ONE trajectory.
  #
  # ⚑ The LAST window's metrics are the ones reported and banked, as before: they
  # are the end-of-run reading. `train_windows` in the artifact is what tells a
  # reader that `metrics` describes the final window rather than the whole budget.
  # ⚑ Declared before the loop, and a zero-window plan is a REFUSAL rather than a
  # possibly-unbound `metrics` further down: `--steps 0` would otherwise reach the
  # summary writer with no training at all having happened.
        metrics: Any = None
  # ⚑⚑ THE LAST WINDOW IS SHORT WHEN THE BUDGET IS NOT A WHOLE NUMBER OF WINDOWS,
  # so the REALIZED step count equals the REQUESTED one. The previous revision
  # ran `steps // window` full windows and DISCARDED the remainder while
  # `summary["steps"]` reported the request; running a ceiling number of FULL
  # windows would make the opposite error and train up to `window - 1` steps MORE
  # than the operator asked for. Both are "a value accepted and then silently
  # ignored". A short final window is safe for the LR: `_scale_for_window_step`
  # returns `min_scale` at `local_step == cycle_steps - 1` for ANY cycle length
  # (measured), so LAST still sits at the bottom of a release cycle.
        steps_done = 0
        for window_index in range(n_windows):
            this_window = min(int(window_steps), int(args.steps) - steps_done)
            if this_window <= 0:
                break
            metrics = trainer.train_steps(
                _as_replay_buffer(buf), batch_size=batch_size,
                steps=this_window,
            )
            steps_done += this_window
            if n_windows > 1:
                # flush=True is load-bearing: stdout redirected to a file is
                # 8KB block-buffered, and at ~60 bytes/line ~136 windows sat
                # invisible between flushes — the 2026-08-27 2x run read as
                # "stalled for 3 hours" while perfectly healthy.
                print(f"[train] window {window_index + 1}/{n_windows} "
                      f"({this_window} steps) done, "
                      f"{steps_done} of {int(args.steps)}", flush=True)

    if metrics is None:
        raise SystemExit(
            f"no training window ran (--steps {int(args.steps)}, "
            f"--train-window-steps {int(args.train_window_steps)}): the budget "
            "must be positive.",
        )
    if capture.calls == 0:
        raise SystemExit("compute_loss was never called — no step ran")

    print_realized(capture, metrics)

    readout = capture.worst
  # ⚑ NOT gated on --allow-leak. Guard 1 refuses every sf_wdl_frac > 0 config
  # and nothing on this call graph can raise the frac afterward, so skipping
  # the assert here left it with no reachable input that could fire it. The
  # flag opens the LAUNCH gate; this one still closes.
    try:
  # ⚑ THE BAR IS DERIVED, NOT DEFAULTED. `max_outcome_borne` defaults to the
  # hand-written `PRODUCTION_GAME_FRAC`; `live_game_frac` is the same quantity
  # read off the recipe the preflight judged against, and the two disagreeing is
  # itself a `validity_problems` entry. Passing it explicitly means a production
  # blend change moves the bar rather than leaving a stale constant in force.
        assert_no_silent_outcome_fallback(
            readout, max_outcome_borne=live_game_frac,
            context="realized training step",
        )
  # ⚑ The SAME defect on the sibling value head, and it needs its own read
  # because `compute_loss`'s blend fracs cannot see it: the categorical
  # rebuild runs in `_prepare_arrays`, upstream of the loss. Inert on today's
  # corpus for a reason that lives in the CONVERTER (no `categorical_target`
  # column) rather than in the config, so it is measured every step.
        assert_categorical_rebuild_is_inert(
            capture.worst_categorical, context="realized training step",
        )
    except ValueBlendMisconfigured as exc:
  # ⚑ The mid-budget checkpoint is written DURING the run, i.e. before this
  # guard can have run. "No checkpoint written" has to stay literally true, or
  # a failed run leaves a scorable artifact behind for `lc0_control_eval` to
  # pick up — and nothing in that file's metadata would say the run was refused.
        if mid.saved_at_step is not None:
            mid_ckpt.unlink(missing_ok=True)
        raise SystemExit(
            f"REALIZED VALUE-BLEND GUARD FAILED — no checkpoint written "
            f"(including the mid-budget one).\n{exc}",
        ) from exc
    print("\n[guard] PASS: no SF-to-outcome leak, no all-outcome value target, "
          "and no outcome-borne categorical rebuild on any observed step")

    ckpt = out_dir / "checkpoint.pt"
    trainer.save(ckpt)
  # ⚑ AFTER both saves, because the identity IS the file bytes. See
  # `checkpoint_identities`: `valid_control` was a verdict about a RUN with
  # nothing tying it to a FILE, so `lc0_control_eval score` could bank a valid
  # run's summary against an unrelated checkpoint, and `compare` could pair two
  # LAST checkpoints from different trajectories as LAST-vs-MID-BUDGET.
    run_id, checkpoint_records = checkpoint_identities(
        last=ckpt, mid=mid_ckpt if mid.saved_at_step is not None else None,
    )
  # ⚑ THE SAME FUNCTION THE LAUNCH REFUSAL CALLED, with the ONE input that could
  # not be known in advance replaced by its realized value. Two copies of this
  # list would let the artifact and the refusal disagree, which is the shape of
  # every finding in this review; `test_launch_refusal_lists_the_same_problems`
  # asserts the two readings are identical for a clean run.
    validity_problems = control_validity_problems(
        allow_leak=bool(args.allow_leak),
        allow_arch_drift=bool(args.allow_arch_drift),
        has_purity_receipt=receipt is not None,
        steps=int(args.steps),
        warmup_steps=int(kwargs["warmup_steps"]),
        mid_saved_at_step=mid.saved_at_step,
        mid_checkpoint_frac=float(args.mid_checkpoint_frac),
        window_steps=int(window_steps),
        cadence_problem=cadence_problem,
        device=str(kwargs["device"]),
        configured_device=configured_device,
        batch_size=batch_size,
        configured_batch_size=configured_batch_size,
        live_config_unread=(_LIVE_FILE_UNREAD in arch_provenance
                            or _LIVE_FILE_UNREAD in trainer_provenance),
        live_game_frac=live_game_frac,
    )
    summary = {
        "steps": int(args.steps),
  # ⚑ THE STEPS THAT ACTUALLY RAN. `steps` above is the REQUEST; a windowed loop
  # can train fewer (the old floor plan discarded the remainder) or more (a
  # ceiling plan of full windows), and nothing in the artifact said which. These
  # are equal by construction now, and that is exactly why the number is banked
  # rather than assumed: a reader can check it instead of trusting this comment.
        "steps_realized": int(steps_done),
  # ⚑ THE SEED, BANKED. It seeds `torch.manual_seed` before `build_model` AND the
  # replay buffer's RNG, and it appeared in neither the summary nor the
  # checkpoint: `run_id` is content-derived, so it IDENTIFIES a trajectory but
  # cannot REPRODUCE one. The replay deviation `deterministic_refresh: true` is
  # justified as "a pure function of the seed", which is only auditable if the
  # seed is in the artifact.
        "seed": int(args.seed),
  # ⚑ The realized LR cadence — see PRODUCTION_TRAIN_WINDOW_STEPS. `metrics`
  # below describes the LAST window, not the whole budget, and this is what says
  # so.
        "train_window_steps": int(window_steps),
        "train_windows": int(n_windows),
        "batch_size": batch_size,
        "configured_batch_size": configured_batch_size,
        "warmup_steps": int(kwargs["warmup_steps"]),
  # ⚑ THE TRAJECTORY IDENTITY, so `valid_control` is a statement about THESE
  # FILES and not about "a run". `lc0_control_eval` hashes the --checkpoint it
  # was given and refuses a summary that does not name it, and `compare`
  # requires one run id with the roles {mid, last}.
        "run_id": run_id,
        "checkpoints": checkpoint_records,
        "trainable_params": params,
        "architecture_judged_against": arch_provenance,
        "trainer_judged_against": trainer_provenance,
  # ⚑ The outcome bar the realized guard actually enforced, and where it came
  # from. `PRODUCTION_GAME_FRAC` is banked beside it so a reader can see whether
  # the constant in the tree still agrees with the recipe production runs.
        "production_game_frac": live_game_frac,
        "production_game_frac_judged_against": game_frac_provenance,
        "pinned_production_game_frac": PRODUCTION_GAME_FRAC,
        "replay_judged_against": replay_provenance,
  # ⚑ The two DECLARED buffer deviations are applied after guard 0d, so the
  # realized values go in the artifact for the same reason
  # `realized_after_guard` does one entry down: a recipe that is certified and
  # then changed must not be able to look identical to one that was not.
        "realized_replay_after_guard": replay_kwargs,
        "mid_checkpoint": (
            None if mid.saved_at_step is None else {
                "path": str(mid_ckpt.resolve()),
                "step": mid.saved_at_step,
                "of_steps": int(args.steps),
            }
        ),
  # ⚑ REVIEW F6. Guard 0c certifies `kwargs` against the live trainer recipe,
  # and THEN the driver overwrites these two from the CLI. Both are in
  # LIVE_TRAINER_PIN ("cuda" / True), so a run can be certified and then execute
  # a different recipe with nothing in the artifact saying so. `--no-compile` is
  # plausibly objective-neutral, but "plausibly neutral" is an argument, not a
  # record: a reader needs the REALIZED values to check it.
  #
  # ⚑⚑ `device` IS NOW A `validity_problems` ENTRY (see above) — this field is the
  # RECORD, not the gate. The previous revision left the gate off and reasoned
  # that "an entry here would disqualify every smoke"; that was measurably false
  # (a clean 4-step CPU smoke already banks four entries), which left
  # `realized_after_guard` banked and unread by every consumer. `use_compile` is
  # still record-only, deliberately: it changes throughput and kernel fusion, not
  # the objective, and unlike `device` it does not change numerics.
        "realized_after_guard": {"device": kwargs["device"],
                                 "configured_device": configured_device,
                                 "use_compile": kwargs["use_compile"]},
        "valid_control": not validity_problems,
        "validity_problems": validity_problems,
  # ⚑ REVIEW F5 — CORPUS IDENTITY. `--shards` was a CLI argument that left no
  # trace in the artifact, so the run's headline number could not be tied to
  # the rows that produced it or to the purity check that cleared them.
        "corpus": {
            "shard_dirs": [str(Path(d).resolve()) for d in shard_dirs],
            "staged_shards": staged,
            "label_coverage": coverage,
            # ⚑ What the history-identity reader saw (round 2, Grok D3): a
            # run launched with --allow-mixed-history says so here.
            "history_identity": history_identity,
        },
        "purity_receipt": (
            None if receipt is None or receipt_path is None else {
                "path": str(receipt_path.resolve()),
                "frozen": receipt.get("frozen"),
                "frozen_sha256": receipt.get("frozen_sha256"),
                "train_shards": receipt.get("train_shards"),
                "exposed_inputs": receipt.get("exposed_inputs"),
                "train_rows": receipt.get("train_rows"),
            }
        ),
        "compute_loss_calls": capture.calls,
        "realized": dict(readout.as_table()),
        "realized_categorical": dict(capture.worst_categorical.as_table()),
        "metrics": {
            f.name: getattr(metrics, f.name)
            for f in dataclasses.fields(type(metrics))
            if isinstance(getattr(metrics, f.name), (int, float, bool, type(None)))
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if mid.saved_at_step is not None:
        print(f"\n[checkpoint] MID  {mid_ckpt} (step {mid.saved_at_step} of "
              f"{int(args.steps)})")
    print(f"\n[checkpoint] {ckpt}")
    print(f"[summary]    {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
