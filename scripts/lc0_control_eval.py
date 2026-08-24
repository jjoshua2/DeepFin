#!/usr/bin/env python3
"""Top-1 agreement with lc0's visit-argmax, on frozen rows, paired (McNemar).

The lc0 positive control's yardstick, exactly as
`docs/lc0_positive_control_prereg.md` defines it and no other metric. Two
subcommands:

  score    --config <yaml> --frozen <frozen.json> --shards <dirs>
           [--checkpoint <ckpt.pt>] [--seed N] [--shuffle-targets]
           --out <scores.npz>
           Per-row hit/miss against `argmax(policy_target)`. Omit
           --checkpoint to score an untrained net — that is prereg guard 2,
           the RANDOM-INIT FLOOR, whose gate is a SEEDED BAND (see below).
           `--shuffle-targets` is prereg guard 1, the NEGATIVE CONTROL: it
           permutes the targets across rows, and the run EXITS 1 when the
           result does not collapse to the floor that control actually has.

  compare  --a <scores_A.npz> --b <scores_B.npz>
           Paired difference on the rows BOTH files scored, with the McNemar
           discordant counts, a Wald CI on the paired difference, and the
           exact binomial p. Refuses to compare files whose row sets differ,
           files whose TARGET PROVENANCE differs (see below), and refuses to
           report a slope whose halfwidth exceeds the pre-committed bar.

⚑⚑ `compare` READS THE NEGATIVE-CONTROL METADATA, IT DOES NOT ONLY CARRY IT.
`score` banks `shuffled_targets_seed` into every artifact, and until 2026-08-17
`compare` loaded that metadata and then used exactly one field of it
(`checkpoint`) for a print. Measured on this tree at HEAD c3490d9c0: a B
artifact carrying `shuffled_targets_seed: 0` — prereg guard 1, the PERMUTED-
TARGET NEGATIVE CONTROL — compared against a real-target A at n=100,000
printed `delta +0.0400 pp, CI [+0.0123, +0.0677]` and **exited 0**, i.e. the
control read as the primary learning slope and cleared every pre-committed
gate. That is this repo's signature defect (a value accepted and then silently
ignored) sitting inside the script written to enforce the prereg.

So the provenance is now a REFUSAL, not a warning, and it is evaluated BEFORE
the arithmetic so the number never reaches the screen to be quoted:

  * the two artifacts' `shuffled_targets_seed` differ  -> refuse, naming the
    field and BOTH values;
  * either artifact IS a negative control (seed not None) -> refuse;
  * either artifact FAILED its own negative-control gate (`negative_control_z`
    above the `negative_control_max_z` that run was gated at) -> refuse, and
    this one is NOT waivable;
  * either artifact was gated MORE LENIENTLY than `NEGATIVE_CONTROL_Z` -> refuse,
    NOT waivable. The bar is banked by the run itself and `--negative-control-z`
    has no ceiling, so `score --shuffle-targets --negative-control-z 1e9` wrote
    an artifact whose control sat 41.7 sigma above the floor and was recorded as
    PASSING — which turned the refusal above into a decoration. The bar is a
    CLAIM, not a measurement;
  * either artifact came from a run the DRIVER DISQUALIFIED (`valid_control:
    false` in summary.json: `--allow-arch-drift`, `--allow-leak`, no purity
    receipt, no mid checkpoint, `--steps` under `warmup_steps`) -> refuse, NOT
    waivable;
  * either artifact carries NO validity record at all -> refuse, waivable by
    `--allow-unrecorded-validity`. Absent is not clean;
  * the two artifacts' `run_id` DIFFER, or their `checkpoint_role` pair is not
    {mid, last} -> refuse, NOT waivable. `valid_control: true` is a verdict about
    a RUN and said nothing about WHICH checkpoint: until 2026-08-17 a summary had
    no file hash and no run identity, so a valid run's summary could vouch for an
    unrelated (or disqualified) checkpoint, and two independently initialised
    trajectories' LAST checkpoints could be paired and reported as the prereg's
    LAST vs MID-BUDGET slope. `score` now hashes the `--checkpoint` it was given
    and REFUSES a summary that does not name it. ⚑ These two are judged on the
    artifacts that CARRY the field, INDEPENDENTLY of the waiver below — the first
    version chained all three as `if/elif/elif`, so waiving the absent-identity
    branch skipped both of these and `--allow-unverified-trajectory` cleared a
    LAST-vs-LAST pair and a two-trajectory pair at exit 0;
  * either artifact carries no `run_id`/`checkpoint_role` at all -> refuse,
    waivable by `--allow-unverified-trajectory`, which concedes only that the
    MISSING field's binding is unverified;
  * either artifact's frozen set was not built from the preregistered SIX hourly
    tars -> refuse, waivable by `--allow-non-prereg-heldout`. `freeze` gated the
    row COUNT at 100,000 and not the POPULATION, and `score` recomputes the
    source count from the artifact rather than trusting a stamp;
  * either artifact carries NO held-out population record at all -> refuse,
    waivable by its own `--allow-unrecorded-heldout`. Absent is not clean here
    either: reading the field as `... or ()` made a pre-field artifact read as
    PREREGISTERED, the same direction `score`'s recompute exists to avoid.

`--allow-shuffled-contrast` waives the first two, for the one case that wants
them: deliberately reading the shuffled contrast. It does not waive the failed
control, because "I meant to compare the shuffled control" is a statement about
intent and a failed control is a statement about the RIG. And it does not waive
an unrecorded validity, an unverified trajectory or a non-preregistered held-out
population either — each has its own flag, so no waiver here clears more than
its name says. Every refusal carries the name of the ONE flag that clears it
(`ProvenanceProblem.waiver`), rather than a flag matching on the message text.

⚑⚑ THE NEGATIVE CONTROL'S FLOOR IS **NOT** `E[1/n_legal]`, AND IT IS NOT A
CONSTANT. `--shuffle-targets` scores each prediction against ANOTHER ROW'S REAL
TARGET, so under a uniformly random permutation its expectation is the marginal
collision rate

    E[hit rate] = SUM_m p_pred(m) * p_tgt(m)

— a property of the SCORED RUN, not of the position set. Measured on 100,000
real converted lc0 rows: `E[1/n_legal]` = 0.063622 while `SUM_m p_tgt(m)^2` =
0.003283, i.e. **19x apart**, so a correctly-behaving control could never have
hit the band the prereg named (review F2b; prereg AMENDMENT 2). Getting the
JENSEN direction right — `E[1/n]` vs `1/E[n]`, the correction already in
`heldout.py chance` — did not make it the right QUANTITY: both are floors for a
UNIFORM MOVER, and this control does not produce one.

So the floor is computed from the same score, printed next to the observed
rate, and GATED: the run exits 1 when the shuffled rate sits more than
`--negative-control-z` standard errors ABOVE its collision rate. That is the
direction with a failure mode behind it — a shuffled control that still agrees
is a rig leaking row identity into the prediction, which is the exact thing
guard 1 exists to detect.

⚑ THE RANDOM-INIT FLOOR IS SEEDED, AND THE SEED IS IN THE OUTPUT. Four
unseeded draws of the identical command on the identical 100,000 frozen rows
read 0.058770 / 0.062050 / 0.071630 / 0.073500 / 0.080590 — a 2.18 pp spread,
5.6x the 0.392 pp material bar. A guard whose own value moves further than the
effect it gates on is not a guard. `--seed` names the draw; the prereg's band
must be derived from the spread across seeds, NOT from one invocation.

⚑ AND A RANDOM-INIT NET IS NOT EXPECTED TO SIT AT ANY PARTICULAR POINT. It is a
fixed arbitrary function over a highly non-uniform argmax distribution, so it
beats uniform guessing on the frequent moves. "Above chance" is not evidence the
evaluator manufactures agreement; only the seeded band can say that, and that
band is NOT YET MEASURED — it needs a trained checkpoint (prereg AMENDMENT 2).

⚑ WHY PAIRED, AND WHY IT REFUSES TO PAD. Two independent top-1 rates at
n=100,000 resolve ~0.30 pp; the same rows scored by both checkpoints resolve
~0.20 pp because the row-to-row difficulty variance cancels. That is the whole
reason the row set is frozen. If the two score files do not cover the same
rows the pairing is a fiction, so `compare` exits rather than intersecting and
quietly reporting a smaller n against the prereg's 0.392 pp bar — which was
derived AT n=100,000.

⚑ THIS SCRIPT MEASURES POLICY AGREEMENT AND NOTHING ELSE. It says nothing
about Elo, and per the prereg a held-out slope alone is not a verdict: it is
read jointly with the same statistic on a frozen sample of ALREADY-TRAINED
rows. Both use this script; only the `--frozen`/`--shards` inputs differ.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chess_anti_engine.eval.lc0_control_rows import (
    iter_shard_arrays,
    load_frozen,
    row_ids,
    sha256_file,
    source_selection_problems,
)
from chess_anti_engine.replay.dataset import collate_arrays
from chess_anti_engine.train.losses import apply_policy_mask_to_logits
from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config
from chess_anti_engine.model import build_model, model_config_from_flat_config
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


Z_95 = 1.959963985


def paired_halfwidth_pp(*, n: int, discordance: float) -> float:
    """95% halfwidth (in pp) of the McNemar paired difference at this n.

    ⚑ THE PREREG'S CONSTANTS, DERIVED RATHER THAN QUOTED. `--max-halfwidth-pp`
    and `--sample` were hard-coded defaults citing a file that had never been
    committed (review F2a). The file is committed now, and this reproduces its
    resolution table so the numbers are checkable rather than remembered:

        n=100,000 discordance 0.10 -> 0.196 pp   (the paired resolution)
        n=100,000 discordance 0.20 -> 0.277 pp
        2x the first               -> 0.392 pp   (MATERIAL_BAR_PP below)

    Same estimator as `cmd_compare`: with `b + c = discordance * n` and the
    worst case `c - b = 0`, `var = (b + c) / n^2` and `half = 1.96 * sqrt(var)`.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return 100.0 * Z_95 * float(np.sqrt(max(0.0, discordance) / n))


# The prereg's resolution point and material bar. Both are DERIVED, not typed:
# 100,000 paired rows resolve ~0.2 pp, and the bar is 2x the resolution so an
# effect the instrument cannot see is never material.
PREREG_SAMPLE_ROWS = 100_000
PREREG_DISCORDANCE = 0.10
MATERIAL_BAR_PP = round(2.0 * paired_halfwidth_pp(
    n=PREREG_SAMPLE_ROWS, discordance=PREREG_DISCORDANCE,
), 3)

# How far above its own floor a shuffled-target control may land before the run
# is refused. 5 sigma rather than 2: at n=100,000 and a collision rate of
# ~0.0033 one SE is ~0.018 pp, so 2 sigma would refuse on ordinary permutation
# noise while 5 sigma still refuses anything the leak this guards against would
# produce (a rig leaking row identity moves the rate by whole percentage
# points, hundreds of sigma).
NEGATIVE_CONTROL_Z = 5.0


def shuffled_control_expectation(
    predicted: np.ndarray, target: np.ndarray,
) -> float:
    """`SUM_m p_pred(m) * p_tgt(m)` — the shuffled control's TRUE floor.

    ⚑ NOT `E[1/n_legal]`. Scoring row `i`'s prediction against row `pi(i)`'s
    target, with `pi` uniform over all permutations (fixed points included,
    each with probability `1/n`), gives

        E[hits] = SUM_i SUM_j (1/n) * 1[tgt_j == pred_i]
                = (1/n) SUM_m count_pred(m) * count_tgt(m)

    exactly — no large-n approximation and no derangement correction. For a
    trained net `p_pred ~ p_tgt`, so this is close to `SUM_m p_tgt(m)^2`:
    0.003283 on 100,000 real lc0 rows, against an `E[1/n_legal]` of 0.063622.

    The marginals are permutation-invariant, so it does not matter whether the
    shuffled or the original target array is passed.
    """
    width = int(max(predicted.max(initial=0), target.max(initial=0))) + 1
    n = int(predicted.size)
    if n == 0:
        return 0.0
    pred_counts = np.bincount(predicted, minlength=width).astype(np.float64)
    tgt_counts = np.bincount(target, minlength=width).astype(np.float64)
    return float((pred_counts * tgt_counts).sum() / (n * n))


def negative_control_problem(
    *, observed: float, expected: float, rows: int, max_z: float,
) -> tuple[float, str | None]:
    """`(z, problem)` for a shuffled-target run against its own floor.

    ONE-SIDED, deliberately. Landing BELOW the collision rate is what an
    unusually lucky permutation does and has no failure mode behind it; landing
    ABOVE it is the rig agreeing with a target it was not shown, which is the
    only thing this control exists to detect.

    The SE is the independent-Bernoulli one. Sampling the permutation is
    sampling WITHOUT replacement, whose variance is no larger, so this is the
    conservative side of the approximation.
    """
    if rows <= 0:
        return 0.0, "the negative control scored ZERO rows, so it checked nothing"
    se = float(np.sqrt(max(expected * (1.0 - expected), 0.0) / rows))
    if se <= 0.0:
        z = 0.0 if observed <= expected else float("inf")
    else:
        z = (observed - expected) / se
    if z <= max_z:
        return z, None
    return z, (
        f"the shuffled-target control scored {observed:.6f}, which is {z:.1f} "
        f"standard errors ABOVE its own floor {expected:.6f} (SUM_m "
        f"p_pred(m)*p_tgt(m) over {rows} rows, SE {se:.6f}). A control scored "
        "against OTHER ROWS' targets cannot beat the marginal collision rate "
        "unless the prediction carries row identity — i.e. the rig is "
        "manufacturing agreement, and no verdict off it is a verdict in either "
        "direction. ⚑ The floor is NOT E[1/n_legal]: that is the uniform-mover "
        "reference and sits ~19x higher (prereg AMENDMENT 2)."
    )


def target_provenance(meta: Mapping[str, Any]) -> tuple[str, Any]:
    """``(label, shuffled_targets_seed)`` — WHICH TARGETS this artifact scored.

    ⚑ The seed is the whole discriminant, and ``None`` is a value here, not a
    missing key: ``score`` writes ``shuffled_targets_seed: null`` for a real-
    target run and the seed for a permuted one. An artifact written before that
    field existed also reads ``None``, which is the safe direction — it is
    treated as real targets and the SLOPE gate still applies.
    """
    seed = meta.get("shuffled_targets_seed")
    label = (
        "real lc0 targets" if seed is None
        else f"SHUFFLED targets (prereg guard 1, negative control), seed {seed}"
    )
    return label, seed


def _failed_negative_control(meta: Mapping[str, Any]) -> tuple[bool, float, float]:
    """``(failed, z, bar)`` for an artifact's own negative-control gate.

    ⚑ The BAR is read off the artifact, not off ``NEGATIVE_CONTROL_Z``. The run
    that wrote the file may have been given a different ``--negative-control-z``,
    and re-judging its z against this module's constant would answer a question
    nobody asked. ``negative_control_max_z`` is banked by ``cmd_score`` for
    exactly this; a file predating it falls back to the constant.
    """
    z = meta.get("negative_control_z")
    if z is None:
        return False, 0.0, 0.0
    bar = float(meta.get("negative_control_max_z", NEGATIVE_CONTROL_Z))
    return float(z) > bar, float(z), bar


# The waiver flags, as constants: `ProvenanceProblem.waiver` names one of these
# and `cmd_compare` looks each up on the parsed args, so a refusal and the flag
# that clears it cannot drift apart in prose. ⚑ ONE FLAG PER REASON — a waiver
# that clears more than its name says is how a gate becomes a decoration.
SHUFFLED_CONTRAST_WAIVER = "--allow-shuffled-contrast"
UNRECORDED_VALIDITY_WAIVER = "--allow-unrecorded-validity"
UNVERIFIED_IDENTITY_WAIVER = "--allow-unverified-trajectory"
NON_PREREG_HELDOUT_WAIVER = "--allow-non-prereg-heldout"
# ⚑ SEPARATE from the one above, on the same reasoning that keeps
# `--allow-unrecorded-validity` separate from `valid_control: false`: "this set is
# declaredly not the prereg's six" and "nothing in this artifact says which
# population it scored" are different claims, and a flag named for the first must
# not clear the second.
UNRECORDED_HELDOUT_WAIVER = "--allow-unrecorded-heldout"
# ⚑⚑ THE POPULATION ROLE. The prereg reads TWO deltas — `Delta_heldout` and
# `Delta_train` — and `cmd_score` classified EVERY artifact with the HELD-OUT
# six-source rule while banking no statement of which population it scored. So a
# train-side sample that happened to span six directories banked an empty
# `heldout_source_selection_problems` and could be presented as the held-out
# slope, while an honest train sample spanning more directories demanded a
# MISLEADING held-out waiver. Reported by the third independent review of #438
# (thread 3795733605). The role is now banked, `compare` is told which comparison
# it is making, and the six-source rule applies only to the held-out one.
UNRECORDED_POPULATION_WAIVER = "--allow-unrecorded-population"
# ⚑ Partial cluster keys. `cluster_keys_complete` was banked by `score` and read
# by NOTHING — the wave-5 mutant that set it to False changed no output and no
# exit code, which is this repo's signature defect inside the very field added to
# make an assumption visible. It gates here: a `rows/cluster` ratio computed over
# a key set with holes UNDERSTATES the clustering (the missing rows read as one
# empty cluster and are dropped from `distinct_clusters`), so the number is not
# merely unproven, it is wrong in a known direction.
PARTIAL_CLUSTERS_WAIVER = "--allow-partial-clusters"
POPULATIONS = ("heldout", "train")


def checkpoint_identity(
    data: Mapping[str, Any], checkpoint: Path | None, summary_path: Path,
) -> dict[str, Any]:
    """Tie THIS ``--checkpoint`` to the summary that is about to vouch for it.

    ⚑⚑ THE SUMMARY HAD NO IDEA WHICH FILE IT DESCRIBED. ``valid_control`` is a
    verdict about a RUN and the artifact carried nothing naming a FILE, so
    ``--summary <valid run>/summary.json --checkpoint <other run>/checkpoint.pt``
    banked ``valid_control: true`` for a checkpoint that summary had never seen —
    including one from a run the driver disqualified. A hash mismatch is a HARD
    ERROR here rather than a recorded problem, because there is no reading of it
    under which the operator got what they asked for.

    ``role`` and ``run_id`` come from the matched record: ``compare`` needs them
    to require MID and LAST **of one trajectory** instead of two runs' LASTs.
    """
    banked = list(data.get("checkpoints") or ())
    if checkpoint is None:
        return {
            "checkpoint_sha256": None, "checkpoint_role": None, "run_id": None,
            "checkpoint_identity": (
                "no --checkpoint: a random-init floor has no trajectory identity"
            ),
        }
    digest = sha256_file(Path(checkpoint))
    if not banked:
        return {
            "checkpoint_sha256": digest, "checkpoint_role": None, "run_id": None,
            "checkpoint_identity": (
                f"{summary_path} banks no `checkpoints`, so it cannot say "
                "whether it describes this file (written before the field "
                "existed)"
            ),
        }
  # ⚑ ALL matches, not the first. A first-match lookup on a CONTENT key silently
  # labels a byte-identical LAST as `mid` (and the pair is then refused as
  # `{mid}`, for a reason that names the wrong thing). Two identical checkpoints
  # mean the optimizer changed nothing between the mid step and the end, which is
  # a defect in its own right, so it is named here rather than resolved by
  # iteration order.
    matches = [r for r in banked if str(r.get("sha256")) == digest]
    if len(matches) > 1:
        raise SystemExit(
            f"{summary_path} banks {len(matches)} checkpoints with the SAME "
            f"sha256 {digest[:16]} (roles "
            + ", ".join(str(r.get("role")) for r in matches)
            + f"), so {checkpoint} cannot be assigned a role. Byte-identical "
            "checkpoints mean no weight changed between them — the pair carries "
            "no slope at all, and `compare` would refuse it for zero "
            "discordance after reporting the wrong role.",
        )
    for record in matches:
        return {
            "checkpoint_sha256": digest,
            "checkpoint_role": str(record.get("role")),
            "run_id": data.get("run_id"),
            "checkpoint_identity": (
                f"verified: role {record.get('role')!r}, sha256 "
                f"{digest[:16]}, named by {summary_path}"
            ),
        }
    raise SystemExit(
        f"{summary_path} does not describe {checkpoint}: that file's sha256 is "
        f"{digest}, and the summary banks "
        + ", ".join(
            f"{record.get('role')}={str(record.get('sha256'))[:16]}"
            for record in banked
        )
        + ". A summary's `valid_control` is a verdict about the run that wrote "
        "THESE checkpoints; banking it against another file would let a valid "
        "run vouch for an unrelated — or disqualified — checkpoint.",
    )


def read_training_validity(
    summary: Path | None, checkpoint: Path | None,
) -> dict[str, Any]:
    """The training run's own ``valid_control`` verdict, for banking.

    ⚑ ANNOUNCED FROM WHAT WAS ACTUALLY READ. ``source`` records the path this
    returned from, or why it returned nothing — an explicit ``--summary`` that
    does not exist is a hard error rather than a silent fallback, because the
    operator naming a path and getting the default instead is the shape this
    module keeps closing.

    With no ``--summary``, the checkpoint's own directory is tried, since
    ``lc0_control_train.py`` writes ``summary.json`` beside the checkpoints it
    produces. A miss returns ``valid_control: None``, which ``compare`` refuses
    unless ``--allow-unrecorded-validity`` is passed.

    ⚑ AND IT VERIFIES THE BINDING — see ``checkpoint_identity``. Until
    2026-08-17 the ``--summary``/``--checkpoint`` pair was two free-floating CLI
    paths, exactly the shape the purity receipt exists to stop one file over.
    """
    path: Path | None = None
    if summary is not None:
        path = Path(summary)
        if not path.is_file():
            raise SystemExit(
                f"--summary {path} does not exist. Point it at the training "
                "run's summary.json, or omit it and accept an unrecorded "
                "validity (which `compare` will refuse by default).",
            )
    elif checkpoint is not None:
        candidate = Path(checkpoint).resolve().parent / "summary.json"
        path = candidate if candidate.is_file() else None
    if path is None:
        return {
            "valid_control": None, "validity_problems": None,
            "source": "no summary.json found",
            "checkpoint_sha256": (
                None if checkpoint is None else sha256_file(Path(checkpoint))
            ),
            "checkpoint_role": None, "run_id": None,
            "checkpoint_identity": "no summary.json found",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"--summary {path} is not readable JSON: {exc}") from exc
    if "valid_control" not in data:
        raise SystemExit(
            f"{path} has no `valid_control` field, so it is not a "
            "lc0_control_train summary. Banking it would record a validity "
            "this file never claimed.",
        )
    return {
        "valid_control": bool(data["valid_control"]),
        "validity_problems": list(data.get("validity_problems") or []),
        "source": str(path),
        **checkpoint_identity(data, checkpoint, path),
    }


def _negative_control_readout(meta: Mapping[str, Any]) -> str:
    """The z and the bar, for the PASSING path's printout.

    ⚑ This file's own standard, applied to itself: "PRINTED, not merely
    checked — a gate that stopped running looks identical to a gate that ran
    and passed." `compare` printed the target-provenance label and then judged
    the negative control in silence, so a reader of a passing readout could not
    see that the second gate ran at all, let alone what bar it used. Both
    numbers, on the line the verdict is read off.
    """
    z = meta.get("negative_control_z")
    if z is None:
        return "none (real targets)"
    bar = float(meta.get("negative_control_max_z", NEGATIVE_CONTROL_Z))
    return f"z={float(z):+.2f} vs bar {bar:g}"


@dataclasses.dataclass(frozen=True)
class ProvenanceProblem:
    """One reason a pair is refused, and the ONE flag that clears it.

    ⚑ The waiver is a FIELD, not a substring of the message. The first version
    of the ``--allow-unrecorded-validity`` waiver filtered
    ``unwaivable`` by matching ``"carries NO validity record" in p``, which makes
    a waiver's scope a property of prose: reword the message and the flag either
    stops working or starts clearing a neighbouring refusal. ``waiver=None`` is
    unwaivable and no flag in this file can reach it.
    """

    message: str
    waiver: str | None = None


def _identity_readout(meta: Mapping[str, Any]) -> str:
    """``role``/``run_id`` for the PASSING path's printout — same rule as
    ``_negative_control_readout``: a gate that ran and passed must not look
    identical to a gate that stopped running."""
    role = meta.get("checkpoint_role")
    run_id = meta.get("run_id")
    return (
        f"role: {role or 'UNRECORDED'}   "
        f"run: {str(run_id)[:8] if run_id else 'UNRECORDED'}   "
        f"pop: {meta.get('population') or 'UNRECORDED'}"
    )


def score_provenance_problems(
    meta_a: Mapping[str, Any], meta_b: Mapping[str, Any],
    *, population: str = "heldout",
) -> list[ProvenanceProblem]:
    """Every reason these two artifacts cannot be paired, each with its waiver.

    ⚑⚑ THE ONE THING THIS FUNCTION MUST NOT DO IS RETURN AN EMPTY LIST FOR A
    SHUFFLED ARTIFACT. `--shuffle-targets` permutes the targets across rows, so
    a shuffled score is a measurement of the marginal collision rate and NOT of
    top-1 agreement. Differencing it against a real-target score produces a
    number with the units of the primary yardstick and the meaning of noise,
    and every downstream gate in this file (the n floor, the halfwidth bar, the
    zero-discordance refusal) is satisfied by it.

    ⚑ Both values are named in the message, per the review: "the provenance
    differs" tells an operator to go and open two npz files, which is the step
    at which people stop.
    """
    label_a, seed_a = target_provenance(meta_a)
    label_b, seed_b = target_provenance(meta_b)
    problems: list[ProvenanceProblem] = []

    def refuse(message: str, *, waiver: str | None = None) -> None:
        problems.append(ProvenanceProblem(message, waiver))

    if seed_a != seed_b:
        refuse(
            "the two score files did not score the SAME TARGETS: field "
            f"`shuffled_targets_seed` reads {seed_a!r} in --a ({label_a}) and "
            f"{seed_b!r} in --b ({label_b}). A permuted-target run measures the "
            "marginal collision rate, so differencing it against a real-target "
            "run reports a NEGATIVE CONTROL in the units of the primary "
            "yardstick.",
            waiver=SHUFFLED_CONTRAST_WAIVER,
        )
    elif seed_a is not None:
        refuse(
            "BOTH score files are negative controls: field "
            f"`shuffled_targets_seed` reads {seed_a!r} in --a and {seed_b!r} in "
            "--b. Their paired difference is a difference of two collision "
            "rates and says nothing about the arm.",
            waiver=SHUFFLED_CONTRAST_WAIVER,
        )
    for label, meta in (("--a", meta_a), ("--b", meta_b)):
        failed, z, bar = _failed_negative_control(meta)
        if failed:
            refuse(
                f"{label} FAILED its own negative-control gate: field "
                f"`negative_control_z` reads {z:+.2f} against the "
                f"`negative_control_max_z` bar {bar:.1f} that run was gated at. "
                "A rig that agrees with targets it was not shown manufactures "
                "agreement, so no verdict off it is a verdict in either "
                "direction — including this one.",
            )
  # ⚑⚑ AND THE BAR ITSELF IS AN INPUT, WHICH MAKES THE "UNWAIVABLE" REFUSAL
  # WAIVABLE FROM THE SCORE SIDE. Reading the bar off the artifact is right —
  # re-judging a banked z against whatever this module's constant happens to be
  # at read time answers a question nobody asked — but `--negative-control-z`
  # has no ceiling and `cmd_score` banks whatever it is handed. Independent
  # review of #438 demonstrated it end-to-end: `score --shuffle-targets
  # --negative-control-z 1e9` writes an artifact whose control sits 41.7 sigma
  # above this module's floor and is recorded as PASSING, and `compare` then
  # prints the pre-fix slope at exit 0. The refusal above is doing its job
  # perfectly; it is just being asked a question whose answer was chosen by the
  # run under test. A run gated more leniently than NEGATIVE_CONTROL_Z is
  # therefore its own refusal — the bar is the claim, not the measurement.
    for label, meta in (("--a", meta_a), ("--b", meta_b)):
        if meta.get("negative_control_z") is None:
            continue
        bar = float(meta.get("negative_control_max_z", NEGATIVE_CONTROL_Z))
        if bar > NEGATIVE_CONTROL_Z:
            refuse(
                f"{label} was gated MORE LENIENTLY than this module's floor: "
                f"field `negative_control_max_z` reads {bar:g} against "
                f"NEGATIVE_CONTROL_Z {NEGATIVE_CONTROL_Z:g}. The bar is banked "
                "by the run itself, so a loose `--negative-control-z` turns the "
                "negative-control refusal into a decoration: the control can "
                "sit arbitrarily far above the floor and still be recorded as "
                "passing. Re-score with the standard bar.",
            )
  # ⚑⚑ THE FIELD THIS PR ITSELF INTRODUCED, ONE FILE SHORT OF THE GATE THAT
  # NEEDS IT. `lc0_control_train.py` computes `validity_problems` and stamps
  # `valid_control: false` into summary.json for a run launched with
  # `--allow-arch-drift` ("this is NOT production's architecture"), `--allow-leak`,
  # no purity receipt, no mid checkpoint, or `--steps` below `warmup_steps`.
  # `cmd_score` banked twelve meta keys and none of them was that one, so a
  # checkpoint from a run THE DRIVER ITSELF DISQUALIFIED scored clean and
  # reported as the primary slope at exit 0. The pattern is established right
  # above — bank the provenance, refuse before the arithmetic — and it stopped
  # at the field it had just created. Independent review of #438.
    for label, meta in (("--a", meta_a), ("--b", meta_b)):
        valid = meta.get("valid_control")
        if valid is False:
            recorded = meta.get("validity_problems") or ["<not recorded>"]
            refuse(
                f"{label} was produced by a run THE DRIVER DISQUALIFIED: field "
                f"`valid_control` reads false. Recorded reason(s): "
                + "; ".join(str(p) for p in recorded)
                + ". A run that is not a valid control cannot supply either "
                "side of the arm's primary comparison.",
            )
        elif valid is None:
            refuse(
                f"{label} carries NO validity record: field `valid_control` is "
                "absent, so this readout cannot tell a clean run from one "
                "launched with --allow-arch-drift, --allow-leak, no purity "
                "receipt, or too few steps. Re-score with --summary pointing "
                "at the training run's summary.json, or pass "
                f"{UNRECORDED_VALIDITY_WAIVER} to state that you know it is "
                "unrecorded.",
                waiver=UNRECORDED_VALIDITY_WAIVER,
            )
  # ⚑⚑ AND `valid_control: true` SAID NOTHING ABOUT *WHICH* CHECKPOINT. The
  # summary banks a content-derived `run_id` and one sha256 per checkpoint, and
  # `read_training_validity` refuses a summary that does not name the scored
  # file — but a pair of individually-verified artifacts can still be two
  # LASTs from two independently initialised trajectories, which is not the
  # prereg's statistic. The prereg reads the arm as two slopes "both measured
  # LAST vs MID-BUDGET" ON ONE TRAJECTORY, so that is what is required here:
  # one run id, and the roles {mid, last}.
  # ⚑⚑ AND THE THREE CHECKS ARE INDEPENDENT, NOT AN `if/elif/elif` CHAIN. The
  # first version chained them, so when EITHER artifact lacked identity the
  # waivable branch matched and the two UNWAIVABLE branches never executed:
  # `--allow-unverified-trajectory` then cleared a LAST-vs-LAST pair and a
  # two-trajectory pair, both of which this flag's own help and prereg
  # AMENDMENT 6 called unwaivable. Executed by the independent review of #438 at
  # rc 0 with the full slope printed. One gate structurally shadowing two others
  # is the "backstop hollows out the primary guard" shape one level over — and
  # the wave's own tests could not see it, because the mismatch test set BOTH
  # run_ids (so the waivable branch was empty) and the absence test's fixture pair
  # already WAS {mid, last}.
  #
  # ⇒ each field is judged on the artifacts that CARRY it: the mismatch checks
  # fire whenever both sides have the field, whatever the waiver does about the
  # absent one.
    run_a, role_a = meta_a.get("run_id"), meta_a.get("checkpoint_role")
    run_b, role_b = meta_b.get("run_id"), meta_b.get("checkpoint_role")
    unidentified = [
        label for label, run, role in (("--a", run_a, role_a), ("--b", run_b, role_b))
        if run is None or role is None
    ]
    if unidentified:
        missing = ", ".join(sorted({
            field
            for run, role in ((run_a, role_a), (run_b, role_b))
            for field, value in (("run_id", run), ("checkpoint_role", role))
            if value is None
        }))
        refuse(
            f"{' and '.join(unidentified)} carr"
            f"{'y' if len(unidentified) > 1 else 'ies'} NO trajectory identity: "
            f"field(s) {missing} absent, so this readout cannot tell MID-vs-LAST "
            "on one trajectory from two unrelated checkpoints — the prereg's "
            "primary yardstick is the former. ⚑ Waiving this waives ONLY the "
            "checks the missing field makes impossible: a run_id mismatch and a "
            "role pair that is not {mid, last} are still refused on whatever the "
            "two artifacts DO carry. Re-score with --summary pointing at the "
            f"training run's summary.json, or pass {UNVERIFIED_IDENTITY_WAIVER} "
            "and read the pair knowing the binding is UNVERIFIED.",
            waiver=UNVERIFIED_IDENTITY_WAIVER,
        )
    if run_a is not None and run_b is not None and run_a != run_b:
        refuse(
            "the two score files come from DIFFERENT TRAJECTORIES: field "
            f"`run_id` reads {str(run_a)[:16]} in --a and {str(run_b)[:16]} in "
            "--b. The prereg's primary yardstick is LAST vs MID-BUDGET on ONE "
            "trajectory; two runs' checkpoints differ by their whole "
            "initialisation and data order as well as by budget.",
        )
  # ⚑ Judged on the roles that are PRESENT. Two known roles must be {mid, last};
  # one known role cannot form a pair, and the absence refusal above is what
  # covers that state — so a `last` against an unknown is waivable, while a `last`
  # against a `last` is not, whatever else is missing.
    if role_a is not None and role_b is not None \
            and {str(role_a), str(role_b)} != {"mid", "last"}:
        refuse(
            "the two score files are not the prereg's MID/LAST pair: field "
            f"`checkpoint_role` reads {role_a!r} in --a and {role_b!r} in --b. "
            "The deciding statistic is the slope between the mid-budget and the "
            "final checkpoint of one trajectory.",
        )
  # ⚑ AND THE HELD-OUT POPULATION. `freeze` gated the row COUNT at 100,000 and
  # not the SOURCES, so a frozen set built from one hour cleared every gate in
  # this rig. Recomputed by `cmd_score` from the artifact's own sources list
  # rather than trusted from a stamp, so a frozen set predating the stamp is
  # judged too.
  #
  # ⚑⚑ AND AN ABSENT KEY IS NOT A CLEAN ONE. The first version read
  # `meta.get(...) or ()`, so an artifact written before the field existed
  # produced no problem, no banner and rc 0 — absent ⇒ preregistered, the exact
  # direction `cmd_score` recomputes to avoid one level down, and the opposite of
  # how this same function treats an absent `run_id`. Two instances of one shape
  # with two different answers in one commit; found by the independent review.
  # `cmd_score` writes a LIST (empty when clean), so `None` means "no record",
  # which gets its own waiver — the same split as `valid_control` false vs None.
  # ⚑⚑ WHICH POPULATION, ASKED FOR AND RECORDED. See UNRECORDED_POPULATION_WAIVER:
  # the six-source held-out rule was applied to every artifact, so the two
  # preregistered deltas were indistinguishable in the artifact and each one's gate
  # was wrong for the other. `population` is what the CALLER says this comparison
  # is; the field is what the SCORER recorded; a disagreement is unwaivable,
  # because a train-side slope presented as `Delta_heldout` is the prereg's
  # generalisation claim made out of rows the net trained on.
    for label, meta in (("--a", meta_a), ("--b", meta_b)):
        recorded_pop = meta.get("population")
        if recorded_pop is None:
            refuse(
                f"{label} carries NO population record: field `population` is "
                "absent, so nothing in the artifact says whether these rows are "
                "the preregistered HELD-OUT set or the TRAIN-side sample — and "
                "the two are different prereg deltas with different gates. "
                f"Re-score with --population, or pass "
                f"{UNRECORDED_POPULATION_WAIVER}.",
                waiver=UNRECORDED_POPULATION_WAIVER,
            )
        elif str(recorded_pop) != population:
            refuse(
                f"{label} scored the {str(recorded_pop).upper()} population but "
                f"this comparison was asked for {population.upper()} "
                "(--population). The prereg's two deltas are read against "
                "different bars, and a train-side slope quoted as the held-out "
                "one is the generalisation claim made out of rows the net "
                "trained on.",
            )
  # ⚑ THE SIX-SOURCE RULE IS A HELD-OUT RULE, and applying it to a train-side
  # artifact is what forced an honest train sample to pass a held-out waiver.
  # Judged only when the comparison IS the held-out one.
    for label, meta in (("--a", meta_a), ("--b", meta_b)):
        if population != "heldout":
            continue
        heldout = meta.get("heldout_source_selection_problems")
        if heldout is None:
            refuse(
                f"{label} carries NO held-out population record: field "
                "`heldout_source_selection_problems` is absent, so this readout "
                "cannot tell the preregistered six hourly tars from a frozen set "
                "built out of one hour. Re-score it (the scorer recomputes the "
                f"population from the frozen artifact), or pass "
                f"{UNRECORDED_HELDOUT_WAIVER}.",
                waiver=UNRECORDED_HELDOUT_WAIVER,
            )
        elif list(heldout):
            refuse(
                f"{label} scored a NON-PREREGISTERED held-out population: "
                + "; ".join(str(p) for p in heldout),
                waiver=NON_PREREG_HELDOUT_WAIVER,
            )
    return problems


def _policy_logits(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """The base policy head, chosen exactly as ``compute_loss`` chooses it."""
    logits = outputs["policy"] if "policy" in outputs else outputs.get("policy_own")
    if logits is None:
        raise KeyError("model outputs carry neither 'policy' nor 'policy_own'")
    return logits


def _load_trainer(
    config_path: Path, checkpoint: Path | None, device: str, *, seed: int,
) -> Trainer:
    cfg = flatten_run_config_defaults(load_yaml_file(str(config_path)))
    kwargs = trainer_kwargs_from_config(cfg)
    kwargs["device"] = device
  # ⚑ Compilation is off for scoring regardless of the config. It changes
  # nothing about the arithmetic and costs minutes per invocation, and the
  # ruler is run far more often than the trainer.
    kwargs["use_compile"] = False
    model_cfg = model_config_from_flat_config(cfg)
  # ⚑ SEED BEFORE build_model, not after. Without this the random-init floor —
  # a preregistered GATE — is a different net on every invocation and cannot be
  # reproduced from the score metadata. Measured spread 2.18 pp.
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    model = build_model(model_cfg)
  # See lc0_control_train.py: without model_config the trainer cannot read
  # LC0-root history rows at all.
    trainer = Trainer(model, model_config=model_cfg, **kwargs)
    if checkpoint is not None:
  # ⚑ NOT Trainer.load. That calls load_state_dict_tolerant with the default
  # require_complete=False, so a config whose architecture drifted from the
  # checkpoint's scores a HYBRID of trained and freshly-initialised tensors
  # under the checkpoint's name — and the catastrophic-load detector only
  # fires below 50%. Top-1 is this arm's only yardstick; a partial load has to
  # be an error, not a silent discount. Optimizer/scheduler state is
  # deliberately not restored: nothing here steps.
        from chess_anti_engine.model import load_state_dict_tolerant

        payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
        load_state_dict_tolerant(
            trainer.model, payload["model"],
            label=f"lc0-control-score {Path(checkpoint).name}",
            require_complete=True,
        )
    return trainer


def _predict_rows(
    trainer: Trainer,
    shard_dirs: list[Path],
    wanted: list[str],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """``(predicted_move_id, target_move_id, tied_rows)`` aligned to ``wanted``.

    ⚑ The target is ``argmax(policy_target)``, i.e. lc0's own visit-argmax over
    the compact 1858 encoding. `tied` counts rows where the top visit count is
    shared by more than one move: on those, argmax breaks the tie by index and
    "agreement" is partly arbitrary. It is reported rather than dropped —
    dropping rows would change the population the prereg's resolution was
    computed for.

    ⚑ MOVE IDS, not hits. The negative control needs to re-pair predictions
    with somebody else's target, which a hit vector has already thrown away —
    and permuting a hit vector preserves the hit RATE exactly, i.e. produces a
    negative control that cannot fail.
    """
    want = set(wanted)
    predicted_by_id: dict[str, int] = {}
    target_by_id: dict[str, int] = {}
    tied = 0
    rng = np.random.default_rng(0)
    device = trainer.device
    trainer.model.eval()

    for _path, arrs in iter_shard_arrays(shard_dirs):
        ids = row_ids(arrs)
        take = [
            i for i, row_id in enumerate(ids)
            if row_id in want and row_id not in predicted_by_id
        ]
        if not take:
            continue
        n_rows = len(ids)
        row_keys = [
            key for key, value in arrs.items()
            if value.ndim >= 1 and value.shape[0] == n_rows
        ]
  # ⚑ The 0-d metadata scalars (`_input_history_encoding`, `_policy_encoding`,
  # `_policy_size`, `_history_rep_fix`) must be CARRIED, not indexed. Dropping
  # them does not fail quietly: `select_input_history_arrays` then reads the
  # subset as legacy-history rows and refuses the whole batch, because
  # synthesizing LC0-root planes from legacy cannot recover side-to-move
  # (rl_loop_audit M12). Slicing a shard is not the same as sampling it.
        scalars = {key: value for key, value in arrs.items() if value.ndim == 0}
        for start in range(0, len(take), batch_size):
            index = np.array(take[start:start + batch_size], dtype=np.int64)
            rows = {key: np.ascontiguousarray(arrs[key][index]) for key in row_keys}
            rows.update(scalars)
            prepared = trainer._prepare_host_arrays(
                rows, rng=rng, mirror_prob=0.0,
            )
            batch = collate_arrays(prepared, device=device)
            with torch.no_grad(), trainer._amp_context():
                outputs = trainer.model(batch["x"])
                logits = apply_policy_mask_to_logits(
                    _policy_logits(outputs), batch, "legal_mask", "has_legal_mask",
                )
                predicted = logits.float().argmax(dim=-1).detach().cpu().numpy()
            target = batch["policy_t"].float().detach().cpu().numpy()
            best = target.argmax(axis=-1)
            top = target.max(axis=-1, keepdims=True)
            tied += int((((target >= top) & (top > 0.0)).sum(axis=-1) > 1).sum())
            for offset, row_index in enumerate(index):
                predicted_by_id[ids[int(row_index)]] = int(predicted[offset])
                target_by_id[ids[int(row_index)]] = int(best[offset])

    missing = [row_id for row_id in wanted if row_id not in predicted_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(wanted)} frozen rows were not found in the "
            "given shards. Scoring a subset would silently shrink n below the "
            "value the prereg's resolution was computed at.",
        )
    return (
        np.array([predicted_by_id[row_id] for row_id in wanted], dtype=np.int64),
        np.array([target_by_id[row_id] for row_id in wanted], dtype=np.int64),
        tied,
    )


def _score_rows(
    trainer: Trainer,
    shard_dirs: list[Path],
    wanted: list[str],
    *,
    batch_size: int,
    shuffle_targets_seed: int | None = None,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """``(hits, tied, predicted, target)`` aligned to ``wanted``.

    ⚑ ``shuffle_targets_seed`` is PREREG GUARD 1, the NEGATIVE CONTROL: each
    row is scored against another row's lc0 target. Every row keeps a real
    target, just not its own, so a net that learned the position-to-move
    mapping must collapse to the MARGINAL COLLISION RATE between the two
    distributions. A net that does not has been reading something in the RIG
    rather than in the position.

    ⚑ The move ids are returned, not only the hits, because that floor is
    computed FROM THEM (`shuffled_control_expectation`). A hit vector has
    already thrown away everything the floor is a function of, which is how the
    gate ended up being compared against an unrelated constant.
    """
    predicted, target, tied = _predict_rows(
        trainer, shard_dirs, wanted, batch_size=batch_size,
    )
    if shuffle_targets_seed is not None:
        target = target[np.random.default_rng(shuffle_targets_seed).permutation(target.size)]
    return (predicted == target).astype(np.uint8), tied, predicted, target


def cmd_score(args: argparse.Namespace) -> int:
    payload = load_frozen(Path(args.frozen))
    wanted = list(payload["row_ids"])
    trainer = _load_trainer(
        Path(args.config), args.checkpoint, args.device, seed=int(args.seed),
    )
    shuffle_seed = int(args.shuffle_seed) if args.shuffle_targets else None
    validity = read_training_validity(args.summary, args.checkpoint)
    print(f"[score] valid_control {validity.get('valid_control')!r} "
          f"(from {validity.get('source')})")
  # ⚑ PRINTED, like every other gate in this file: "verified" and "the summary
  # banks no checkpoints" are different states and a reader must be able to see
  # which one this score is standing on.
    print(f"[score] checkpoint identity: {validity.get('checkpoint_identity')}")
  # ⚑ RECOMPUTED from the frozen artifact's own `sources`, not read off a stamp
  # the freezer wrote: a frozen set built before the stamp existed has none, and
  # "absent" would then read as "preregistered".
  #
  # ⚑⚑ AND ONLY FOR THE HELD-OUT POPULATION. This call was unconditional, so the
  # prereg's OTHER delta — `Delta_train`, scored on rows the net trained on — was
  # classified by the held-out six-source rule: a train sample spanning six
  # directories banked an empty problem list and could be presented as
  # `Delta_heldout`, and an honest train sample spanning more needed a MISLEADING
  # held-out waiver to compare at all. `--population` is banked so `compare` can
  # require the role the verdict it is building needs.
    population = str(args.population)
    heldout_problems = (
        source_selection_problems(payload) if population == "heldout" else []
    )
    print(f"[score] population {population.upper()}"
          + ("" if population == "heldout" else
             " — the preregistered six-hourly-source rule is NOT applied to a "
             "train-side sample, and `compare --population train` is what reads "
             "this artifact"))
    if heldout_problems:
        print("⚑⚑ [score] NON-PREREGISTERED HELD-OUT POPULATION — "
              + " | ".join(heldout_problems))
    hits, tied, predicted, target = _score_rows(
        trainer, args.shards, wanted, batch_size=int(args.batch_size),
        shuffle_targets_seed=shuffle_seed,
    )
    rate = float(hits.mean())
    collision = shuffled_control_expectation(predicted, target)
    z, problem = (0.0, None)
    if shuffle_seed is not None:
        z, problem = negative_control_problem(
            observed=rate, expected=collision, rows=int(hits.size),
            max_z=float(args.negative_control_z),
        )
  # ⚑⚑ THE CLUSTER KEY, CARRIED INTO THE SCORE. The converter emits many plies per
  # game, so these hit indicators are correlated within a game and the row-level
  # Wald/McNemar CI below (and `MATERIAL_BAR_PP`, derived under independence) is
  # optimistic by the design effect. The clustered estimator is DEFERRED; banking
  # the key is not, because it makes an estimator-only reanalysis possible from the
  # two npz files alone. Empty strings when the frozen set predates the field.
    clusters = list(payload.get("cluster_keys") or ())
    cluster_by_id = dict(zip(payload["row_ids"], clusters, strict=False))
    row_clusters = [cluster_by_id.get(row_id, "") for row_id in wanted]
    np.savez_compressed(
        Path(args.out),
        row_ids=np.array(wanted, dtype="U32"),
        hit=hits,
        cluster_keys=np.array(row_clusters, dtype=object),
        meta=np.array([json.dumps({
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "config": str(args.config),
            "frozen": str(args.frozen),
            "frozen_sha_source": payload.get("row_id_version"),
  # ⚑ The seed is part of the RESULT, not of the invocation. Without it in the
  # file the random-init floor cannot be reproduced from its own metadata.
            "init_seed": int(args.seed),
            "shuffled_targets_seed": shuffle_seed,
  # ⚑ The TRAINING run's own verdict on itself, carried into the score so
  # `compare` can refuse a checkpoint the driver disqualified. `valid_control`
  # is None when no summary was found, which `compare` treats as a refusal
  # waivable by `--allow-unrecorded-validity` -- absent is not clean.
            "valid_control": validity.get("valid_control"),
            "validity_problems": validity.get("validity_problems"),
            "validity_source": validity.get("source"),
  # ⚑ WHICH CHECKPOINT, AND FROM WHICH TRAJECTORY. `valid_control` is a verdict
  # about a RUN, and without these three fields nothing tied it to the file
  # scored here: a valid run's summary could vouch for an unrelated checkpoint,
  # and `compare` could pair two runs' LAST checkpoints as LAST-vs-MID-BUDGET.
            "checkpoint_sha256": validity.get("checkpoint_sha256"),
            "checkpoint_role": validity.get("checkpoint_role"),
            "run_id": validity.get("run_id"),
            "checkpoint_identity": validity.get("checkpoint_identity"),
  # ⚑ WHICH POPULATION this artifact is, so `compare` can require the role the
  # prereg's delta needs instead of inferring one. Absent in artifacts written
  # before this field existed, which `compare` refuses under its own waiver.
            "population": population,
  # ⚑ The HELD-OUT population, recomputed from the frozen artifact rather than
  # trusted from its stamp. `freeze` gated n and not the sources. ⚑ Empty for a
  # TRAIN-side artifact because the rule does not apply to it — `population`
  # above is what distinguishes that from a clean held-out read, and `compare`
  # judges this field only when the comparison is the held-out one.
            "heldout_source_selection_problems": heldout_problems,
  # ⚑ The cluster STRUCTURE, next to the CI it makes optimistic. Recorded, not yet
  # applied — see `cluster_keys` above and `cmd_compare`'s printout.
            "distinct_clusters": len({c for c in row_clusters if c}),
            "cluster_keys_complete": bool(row_clusters) and all(row_clusters),
            "rows": int(hits.size),
            "top1_agreement": rate,
            "tied_argmax_rows": int(tied),
  # ⚑ The negative control's floor, banked WITH the score. It is a property of
  # this run's own prediction marginal, so it cannot be recomputed later from
  # the hit vector alone — and a gate whose reference is not in the artifact is
  # a gate nobody can re-check.
            "shuffled_collision_rate": collision,
            "negative_control_z": z if shuffle_seed is not None else None,
  # ⚑ The BAR, banked next to the z it judged. `compare` re-reads this pair to
  # refuse an artifact that failed its own gate, and re-judging a banked z
  # against whatever `NEGATIVE_CONTROL_Z` happens to be at read time would
  # answer a different question than the run asked.
            "negative_control_max_z": (
                float(args.negative_control_z) if shuffle_seed is not None else None
            ),
        })], dtype=object),
        allow_pickle=True,
    )
    label = args.checkpoint or "NONE (random init — the floor guard)"
    print(f"checkpoint        {label}")
    print(f"init seed         {int(args.seed)}")
    if shuffle_seed is not None:
        print(f"⚑ SHUFFLED TARGETS (negative control), seed {shuffle_seed}")
    print(f"rows scored       {hits.size}")
    print(f"top-1 agreement   {rate:.6f}  ({rate * 100:.4f}%)")
    print(f"tied-argmax rows  {tied}")
    print(f"shuffled floor    {collision:.6f}   "
          "<-- SUM_m p_pred(m)*p_tgt(m); NOT E[1/n_legal]")
    if shuffle_seed is not None:
        print(f"negative control  {z:+.2f} sigma vs that floor "
              f"(bar {float(args.negative_control_z):.1f})")
    print(f"written           {args.out}")
    if problem is not None:
        print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    return 0


def _load_scores(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta"][0]))
    return data["row_ids"], data["hit"].astype(np.int64), meta


def cmd_compare(args: argparse.Namespace) -> int:
    ids_a, hit_a, meta_a = _load_scores(Path(args.a))
    ids_b, hit_b, meta_b = _load_scores(Path(args.b))
  # ⚑⚑ BEFORE THE ARITHMETIC, DELIBERATELY. Printing the slope and refusing
  # afterwards still puts a quotable number on the screen, and the failure this
  # closes is precisely a number being read off a comparison that had no
  # business producing one. Nothing below this block runs on a refused pair.
  # ⚑ Each problem names the ONE flag that clears it (`ProvenanceProblem.waiver`)
  # and that flag is looked up here. Scoping a waiver by matching its message —
  # what the first `--allow-unrecorded-validity` did — makes the waiver's reach a
  # property of prose, and `valid_control: false` (a run the driver disqualified)
  # must stay unreachable by every flag in this file.
    provenance = score_provenance_problems(
        meta_a, meta_b, population=str(args.population),
    )
    passed = {
        SHUFFLED_CONTRAST_WAIVER: bool(args.allow_shuffled_contrast),
        UNRECORDED_VALIDITY_WAIVER: bool(args.allow_unrecorded_validity),
        UNVERIFIED_IDENTITY_WAIVER: bool(args.allow_unverified_trajectory),
        NON_PREREG_HELDOUT_WAIVER: bool(args.allow_non_prereg_heldout),
        UNRECORDED_HELDOUT_WAIVER: bool(args.allow_unrecorded_heldout),
        UNRECORDED_POPULATION_WAIVER: bool(args.allow_unrecorded_population),
    }
    waived = [p for p in provenance if p.waiver is not None and passed[p.waiver]]
    for flag in sorted({p.waiver for p in waived if p.waiver is not None}):
        print(f"⚑⚑ {flag}: "
              + " | ".join(p.message for p in waived if p.waiver == flag))
    refusals = [p for p in provenance if p not in waived]
    if refusals:
        offered = sorted({p.waiver for p in refusals if p.waiver is not None})
  # ⚑ The hint appears when EVERY refusal is waivable, not when the flag count
  # happens to equal the refusal count: two artifacts both non-preregistered are
  # two refusals sharing one waiver, and the earlier `len(offered) ==
  # len(refusals)` test printed no hint for a state that is entirely waivable.
        all_waivable = all(p.waiver is not None for p in refusals)
        print(
            "FAIL: these two score files are not a valid pair. "
            + " | ".join(p.message for p in refusals)
            + (
                f" Waivable by: {', '.join(offered)} — if that IS the comparison "
                "you intend."
                if all_waivable
                else ""
            ),
            file=sys.stderr,
        )
        return 1
    if list(ids_a) != list(ids_b):
        print(
            "FAIL: the two score files do not cover the same rows in the same "
            "order, so they cannot be paired. Re-score both against the SAME "
            "frozen row set instead of intersecting them here.",
            file=sys.stderr,
        )
        return 1
    n = int(hit_a.size)
  # McNemar's discordant cells. b = A right / B wrong, c = A wrong / B right.
    b = int(((hit_a == 1) & (hit_b == 0)).sum())
    c = int(((hit_a == 0) & (hit_b == 1)).sum())
    rate_a, rate_b = hit_a.mean(), hit_b.mean()
    delta = (c - b) / n
  # Wald SE for the difference of CORRELATED proportions. The concordant cells
  # contribute nothing, which is exactly why pairing buys resolution.
    var = (b + c - (c - b) ** 2 / n) / n**2
    se = float(np.sqrt(max(var, 0.0)))
    half = 1.959963985 * se
    p_exact = _exact_mcnemar_p(b, c)

  # ⚑⚑ THE RESOLUTION GATE IS BUILT AND CHECKED **BEFORE** ANY RESULT LINE IS
  # PRINTED. It used to run after the rates, the delta, the CI and the p-value
  # were already on stdout, so a refused comparison still put a quotable
  # primary-yardstick number on the screen — and a pasted readout that drops
  # stderr and the exit code (which is how these numbers travel) shows the slope
  # with nothing beside it. Reported by the third independent review of #438
  # (thread 3795733620). The provenance gate twenty lines up was already built
  # this way; this one had the same policy in a comment and the opposite order in
  # the code.
    half_pp = half * 100.0
  # ⚑ A NON-FINITE OVERRIDE IS NOT AN OVERRIDE. `--max-halfwidth-pp nan` was
  # accepted by `float()`, switched `prereg_bar_in_force` off (which disabled the
  # n floor) and then compared `half_pp > nan` — FALSE for every finite
  # halfwidth. Both gates off, exit 0, slope printed: a clamp is not a validator,
  # and NaN propagates through every comparison as "no". `<= 0` is refused for
  # the mirror reason: it makes the halfwidth gate unpassable rather than
  # unfailable, which is a bar nobody could have re-derived.
  # ⚑ Bound to a LOCAL, because the narrowing has to survive into `bar_pp` below:
  # `args.max_halfwidth_pp` is `Any | None` to the type checker, so re-reading the
  # attribute after the guard re-widens it.
    override: float | None = args.max_halfwidth_pp
    if override is not None and not (
        math.isfinite(float(override)) and float(override) > 0.0
    ):
        print(
            f"FAIL: --max-halfwidth-pp {override!r} is not a finite "
            "positive number of percentage points. A non-finite bar switches the "
            "n floor off and then compares FALSE against every halfwidth, so the "
            "comparison would report a slope with no resolution gate in force at "
            "all.",
            file=sys.stderr,
        )
        return 1
    prereg_bar_in_force = override is None
    bar_pp = MATERIAL_BAR_PP if override is None else float(override)
  # ⚑⚑ THE PRE-COMMITTED BAR IS A PROPERTY OF n, AND NOTHING WAS ENFORCING n.
  # The prereg's 0.392 pp material bar is derived AT n=100,000. The reviewer's
  # end-to-end demonstration paired 5,000 train-side rows and got a halfwidth
  # of 1.0721 pp — 2.7x the bar it would then be judged against — and the rig
  # printed it without comment. A slope whose own CI is wider than the effect
  # size it is compared to cannot decide anything, so this refuses rather than
  # letting prose carry the gate.
  #
  # ⚑⚑ AND THE HALFWIDTH ALONE CANNOT CARRY IT (review F8). `var = (b + c -
  # (c-b)^2/n) / n^2` is exactly 0 when `b == c == 0`, so a comparison of a
  # file against itself — or of two near-identical checkpoints — passes the bar
  # at ANY n, including n=1. The gate that enforces the prereg's n had a state
  # in which it could not fire, which is the same shape as every other finding
  # in this review. n is now checked in its own right, and the zero-discordance
  # case is refused by name rather than by an inequality it satisfies
  # vacuously.
  #
  # ⚑⚑ AND THE n FLOOR IS NO LONGER SWITCHED OFF BY THE MERE PRESENCE OF AN
  # OVERRIDE. `--max-halfwidth-pp` may only be a RE-DERIVATION of the bar, which
  # at a smaller n means a TIGHTER bar; a bar LOOSER than the prereg's material
  # effect size is not a re-derivation, it is "accept a CI wider than the effect
  # I am judging", and it left the small-n gate open with one argument. So the
  # floor is in force whenever the bar in force is not at least as strict as
  # MATERIAL_BAR_PP, whether or not a flag was passed.
    bar_is_relaxed = bar_pp > MATERIAL_BAR_PP + 1e-12
    problems: list[str] = []
    if b + c == 0:
        problems.append(
            f"the two score files are discordant on ZERO of {n} paired rows, so "
            "the McNemar halfwidth is 0.0000 pp BY CONSTRUCTION and clears any "
            "bar at any n. That is not a resolved comparison, it is two files "
            "that agree on every row — check that they are different "
            "checkpoints.",
        )
    if (prereg_bar_in_force or bar_is_relaxed) and n < PREREG_SAMPLE_ROWS:
        problems.append(
            f"n={n} is below the prereg's resolution point "
            f"{PREREG_SAMPLE_ROWS}, at which the {MATERIAL_BAR_PP:.4f} pp "
            f"material bar was derived (bar in force {bar_pp:.4f} pp"
            + ("" if prereg_bar_in_force else
               ", which is LOOSER than the material bar and so is not a "
               "re-derivation at this n")
            + "). Score the full frozen set, or pass --max-halfwidth-pp with a "
            "bar at least as strict as the material one, re-derived at this n.",
        )
    if half_pp > bar_pp:
        problems.append(
            f"halfwidth {half_pp:.4f} pp exceeds the pre-committed bar "
            f"{bar_pp:.4f} pp at n={n}. At this n the comparison cannot resolve "
            "the effect it is being judged against.",
        )
  # ⚑⚑ AND THE CLUSTER-KEY COMPLETENESS GATES SOMETHING NOW. `score` banked
  # `cluster_keys_complete` and NOTHING read it: the wave-5 mutant that forced it
  # False changed no line of output and no exit code — a field added to make an
  # assumption visible, itself invisible. A PARTIAL key set is worse than none:
  # rows without a key collapse into one dropped empty cluster, so
  # `distinct_clusters` UNDERSTATES the clustering and the `rows/cluster` ratio
  # printed below is wrong in a known direction. Refused under its own flag
  # rather than folded into --allow-underpowered, because "the CI is too wide" and
  # "the design effect cannot be estimated" are different claims.
    cluster_line, partial_clusters = _cluster_readout(meta_a, meta_b, n=n)
    if partial_clusters is not None:
        problems.append(partial_clusters)

  # ⚑ EACH AXIS IS WAIVED BY ITS OWN FLAG, and a flag that clears one leaves the
  # others refused — the same rule the provenance waivers follow. Folding both
  # into --allow-underpowered would make one flag clear two different claims,
  # which is how a gate becomes a decoration.
    resolution_problems = [q for q in problems if q != partial_clusters]
    unwaived: list[str] = []
    if resolution_problems and not args.allow_underpowered:
        unwaived += resolution_problems
    if partial_clusters is not None and not args.allow_partial_clusters:
        unwaived.append(partial_clusters)
    if unwaived:
        print(f"FAIL: {' | '.join(unwaived)}", file=sys.stderr)
        return 1
    if problems:
        print(f"⚑⚑ RESOLUTION GATE WAIVED: {' | '.join(problems)}")

  # ⚑ PRINTED, not merely checked. "the provenance gate passed" has to be an
  # observation in the output a reader can point at, otherwise a gate that
  # stopped running looks identical to a gate that ran and passed.
  # ⚑ The IDENTITY is on the line too, for the same reason `neg-control:` is: the
  # prereg requires a pasted readout to be self-describing, and "this pair is
  # mid-vs-last of one trajectory" is now a gate — a gate whose result a reader
  # of the output could not see.
    for label, meta, rate in (("A", meta_a, rate_a), ("B", meta_b, rate_b)):
        print(f"{label}  {meta.get('checkpoint')}   top-1 {rate:.6f}   "
              f"targets: {target_provenance(meta)[0]}   "
              f"neg-control: {_negative_control_readout(meta)}   "
              f"{_identity_readout(meta)}")
    print(f"population           {args.population.upper()}")
    print(f"paired rows          {n}")
    print(f"discordant  b(A only)={b}  c(B only)={c}  "
          f"discordance={(b + c) / n:.4f}")
    print(f"delta (B - A)        {delta * 100:+.4f} pp")
    print(f"95% CI               [{(delta - half) * 100:+.4f}, "
          f"{(delta + half) * 100:+.4f}] pp   (halfwidth {half * 100:.4f} pp)")
    print(f"exact McNemar p      {p_exact:.6g}")
  # ⚑⚑ THE CI ABOVE ASSUMES INDEPENDENT ROWS AND THE ROWS ARE NOT INDEPENDENT.
  # The converter emits many plies per game, so hits are correlated within a game
  # and both the halfwidth and `MATERIAL_BAR_PP` (derived at n=100,000 under
  # independence) are optimistic by the design effect. The clustered estimator is
  # DEFERRED — it needs the real corpus's plies-per-game distribution — so what
  # ships is the KEY (banked by `score`, unrecoverable after the freeze) and this
  # line, which states the assumption instead of leaving it implicit. A readout
  # quoted against the pre-committed bar has to show it.
    print(cluster_line)
    return 0


def _cluster_readout(
    meta_a: Mapping[str, Any], meta_b: Mapping[str, Any], *, n: int,
) -> tuple[str, str | None]:
    """``(the printed line, a refusal message or None)`` for the game clusters.

    ⚑ THE RATIO IS ONLY PRINTED WHEN THE KEY SET IS COMPLETE. `distinct_clusters`
    counts the NON-EMPTY keys, so an artifact whose keys have holes reports fewer
    clusters over the same rows and a HIGHER rows/cluster figure than the truth —
    a number that looks like a stronger statement about the design effect while
    being an artifact of the missing keys. Incomplete is therefore its own state,
    with its own refusal, and never a clean ratio.
    """
    clusters_a = int(meta_a.get("distinct_clusters") or 0)
    incomplete = [
        label for label, meta in (("--a", meta_a), ("--b", meta_b))
        if meta.get("distinct_clusters") and meta.get("cluster_keys_complete") is not True
    ]
    if incomplete:
        return (
            "game clusters        PARTIAL   ⚑ "
            + " and ".join(incomplete)
            + " bank cluster keys for only SOME rows "
            f"(`cluster_keys_complete` is not true), so the {clusters_a} distinct "
            "keys are counted over an unknown subset and no rows/cluster ratio is "
            "shown",
            f"{' and '.join(incomplete)} carr"
            f"{'y' if len(incomplete) > 1 else 'ies'} an INCOMPLETE cluster-key "
            "set: `cluster_keys_complete` is not true, so rows with no key collapse "
            "into one dropped empty cluster, `distinct_clusters` understates the "
            "clustering and any design-effect statement read off this pair is wrong "
            "in a known direction. Re-freeze/re-score so every row carries a key, "
            f"or pass {PARTIAL_CLUSTERS_WAIVER} to read the ROW-level CI knowing "
            "the design effect cannot be estimated from these artifacts.",
        )
    if clusters_a > 0:
        return (
            f"game clusters        {clusters_a} over {n} rows "
            f"({n / clusters_a:.1f} rows/cluster)   ⚑ the CI above is "
            "ROW-level and is OPTIMISTIC by the design effect; the clustered "
            "estimator is not yet implemented", None,
        )
    return (
        "game clusters        UNRECORDED   ⚑ the CI above is ROW-level and "
        "cannot be corrected for within-game correlation from this artifact",
        None,
    )


# Above this many discordant pairs the exact tail is summed in LOG space. Below
# it, `math.comb` is used unchanged, because that path is what the arithmetic
# pins in `tests/test_lc0_control_drivers.py` assert against and an exact
# integer/exact-power-of-two ratio is bit-reproducible.
EXACT_COMB_MAX_DISCORDANT = 1000


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs.

    ⚑ THE ENUMERATING VERSION DID NOT FINISH AT THE PREREG'S OWN n. It summed
    `math.comb(total, k)` for every k up to `min(b, c)` — arbitrary-precision
    integers with tens of thousands of digits, tens of thousands of times — so a
    balanced 20,000-pair comparison took tens of seconds and a 100,000-pair one
    minutes, on the ONE readout the prereg says decides the arm. Reported by the
    third independent review of #438. Above `EXACT_COMB_MAX_DISCORDANT` the same
    tail is accumulated in log space by a stable recurrence on
    `log C(total, k+1) = log C(total, k) + log(total-k) - log(k+1)`, which is
    O(min(b, c)) float operations. It is still the EXACT binomial tail, not a
    normal approximation: only the arithmetic changed.
    """
    total = b + c
    if total == 0:
        return 1.0
    smaller = min(b, c)
    if total <= EXACT_COMB_MAX_DISCORDANT:
        tail = sum(math.comb(total, k) for k in range(smaller + 1)) / 2**total
        return float(min(1.0, 2.0 * tail))
    log_half = -total * math.log(2.0)
    log_comb = 0.0
    tail = math.exp(log_half)
    for k in range(smaller):
        log_comb += math.log(total - k) - math.log(k + 1)
        tail += math.exp(log_comb + log_half)
    return float(min(1.0, 2.0 * tail))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score")
    score.add_argument("--config", type=Path, default=Path("configs/lc0_positive_control.yaml"))
    score.add_argument("--frozen", type=Path, required=True)
    score.add_argument(
        "--summary", type=Path, default=None,
        help="the TRAINING run's summary.json, whose `valid_control` says "
             "whether the driver disqualified that run (--allow-arch-drift, "
             "--allow-leak, no purity receipt, no mid checkpoint, too few "
             "steps). Banked into the score so `compare` can refuse it. "
             "Defaults to <checkpoint dir>/summary.json when that exists.")
    score.add_argument("--shards", type=Path, nargs="+", required=True)
    score.add_argument("--checkpoint", type=Path, default=None)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--batch-size", type=int, default=512)
    score.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    score.add_argument(
        "--seed", type=int, default=0,
        help="⚑ seeds build_model. The random-init floor is a GATE; unseeded it "
             "spans 2.18 pp, 5.6x the 0.392 pp material bar.",
    )
    score.add_argument(
        "--shuffle-targets", action="store_true",
        help="prereg guard 1, the NEGATIVE CONTROL: score each row against "
             "another row's lc0 target. Any net must collapse to the marginal "
             "collision rate SUM_m p_pred(m)*p_tgt(m) — NOT to E[1/n_legal], "
             "which sits ~19x higher and is the uniform-mover reference.",
    )
    score.add_argument(
        "--population", choices=POPULATIONS, default="heldout",
        help="⚑⚑ WHICH of the prereg's two deltas this artifact is for. "
             "`heldout` (default) applies the preregistered six-hourly-source "
             "rule to the frozen set; `train` does not, because that rule is a "
             "held-out rule. The role is banked and `compare --population` must "
             "match it: an unconditional held-out classification let a "
             "six-directory TRAIN sample be presented as Delta_heldout.",
    )
    score.add_argument("--shuffle-seed", type=int, default=0)
    score.add_argument(
        "--negative-control-z", type=float, default=NEGATIVE_CONTROL_Z,
        help="how many SEs above its own floor a shuffled control may land "
             "before the run is refused (exit 1).",
    )
    score.set_defaults(handler=cmd_score)

    compare = sub.add_parser("compare")
    compare.add_argument("--a", type=Path, required=True)
    compare.add_argument("--b", type=Path, required=True)
    compare.add_argument(
  # ⚑ Default None, not the number: "the operator re-derived a bar for this n"
  # and "the operator typed the prereg's own bar" are different claims, and
  # only the first may switch off the n floor. A concrete default cannot tell
  # them apart.
        "--max-halfwidth-pp", type=float, default=None,
        help=f"the prereg's material bar ({MATERIAL_BAR_PP} pp = 2x the paired "
             "resolution at n=100,000; see paired_halfwidth_pp). A slope whose "
             "95%% halfwidth exceeds it is refused rather than reported. "
             "⚑ Passing a bar at least as STRICT as the material one waives the "
             "n floor, because that is a re-derivation at the n you have; a "
             "LOOSER bar does not, because accepting a CI wider than the effect "
             "size is the underpowered state the floor exists for. Must be "
             "finite and positive: `nan` used to switch the floor off AND pass "
             "every halfwidth.",
    )
    compare.add_argument("--allow-underpowered", action="store_true")
    compare.add_argument(
        "--population", choices=POPULATIONS, default="heldout",
        help="⚑⚑ WHICH of the prereg's two deltas this comparison is. Both score "
             "files must have been scored with the SAME --population, and a "
             "mismatch is UNWAIVABLE: the held-out six-source rule is applied "
             "only to the held-out comparison, and a train-side slope presented "
             "as Delta_heldout is the generalisation claim made out of rows the "
             "net trained on.",
    )
    compare.add_argument(
        UNRECORDED_POPULATION_WAIVER, action="store_true",
        help="proceed when a score file carries no `population` field (written "
             "before the field existed). ⚑ It does NOT waive a population "
             "MISMATCH, and for a held-out comparison the six-source refusals "
             "still apply on whatever the artifact records.")
    compare.add_argument(
        PARTIAL_CLUSTERS_WAIVER, action="store_true",
        help="proceed when a score file's `cluster_keys_complete` is not true. "
             "⚑ Its own flag rather than a clause of --allow-underpowered: a CI "
             "that is too wide and a design effect that cannot be estimated are "
             "different claims. With partial keys no rows/cluster ratio is "
             "printed at all, because `distinct_clusters` understates the "
             "clustering in a known direction.")
    compare.add_argument(
        UNVERIFIED_IDENTITY_WAIVER, action="store_true",
        help="proceed when a score file carries no `run_id`/`checkpoint_role`. "
             "⚑ Without them a pair of individually-valid checkpoints can be "
             "two LASTs from two independently initialised trajectories, "
             "reported as the prereg's LAST vs MID-BUDGET slope. It does NOT "
             "waive a run_id MISMATCH or a role pair that is not {mid, last} — "
             "those are measurements, not absences, and they are judged on "
             "whatever the two artifacts DO carry even when this flag is passed. "
             "⚑ What it DOES concede is that the missing field's binding is "
             "UNVERIFIED: with one side's role absent, nothing establishes that "
             "the pair is mid-vs-last at all.")
    compare.add_argument(
        NON_PREREG_HELDOUT_WAIVER, action="store_true",
        help="proceed when a score file's frozen set was not built from the "
             "preregistered six hourly tars. Its own flag, so declaring a "
             "different held-out population does not also clear an unrecorded "
             "validity or a shuffled contrast.")
    compare.add_argument(
        UNRECORDED_HELDOUT_WAIVER, action="store_true",
        help="proceed when a score file carries no "
             "`heldout_source_selection_problems` at all (written before the "
             "field existed). ⚑ Not the same claim as the flag above — absent is "
             "not clean — so it does not waive a set that IS recorded as "
             "non-preregistered.")
    compare.add_argument(
        UNRECORDED_VALIDITY_WAIVER, action="store_true",
        help="proceed when a score file carries no `valid_control` field. ⚑ It "
             "does NOT waive `valid_control: false` -- a run the driver "
             "disqualified stays refused. Its own flag rather than a clause of "
             "--allow-shuffled-contrast, so no waiver waives more than its "
             "name says.")
    compare.add_argument(
        SHUFFLED_CONTRAST_WAIVER, action="store_true",
        help="⚑⚑ compare artifacts whose `shuffled_targets_seed` differ, or "
             "which are BOTH negative controls. Without it such a pair is "
             "REFUSED, because a permuted-target score differenced against a "
             "real-target one reports prereg guard 1 in the units of the "
             "primary yardstick and clears every other gate in this file. It "
             "does NOT waive an artifact that failed its own negative-control "
             "gate.",
    )
    compare.set_defaults(handler=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
