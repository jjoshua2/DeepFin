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
    and REFUSES a summary that does not name it;
  * either artifact carries no `run_id`/`checkpoint_role` at all -> refuse,
    waivable by `--allow-unverified-trajectory`;
  * either artifact's frozen set was not built from the preregistered SIX hourly
    tars -> refuse, waivable by `--allow-non-prereg-heldout`. `freeze` gated the
    row COUNT at 100,000 and not the POPULATION, and `score` recomputes the
    source count from the artifact rather than trusting a stamp.

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
import hashlib
import json
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


def _file_sha256(path: Path) -> str:
    """sha256 of a file's bytes, streamed — checkpoints are ~2 GB in production."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# The waiver flags, as constants: `ProvenanceProblem.waiver` names one of these
# and `cmd_compare` looks each up on the parsed args, so a refusal and the flag
# that clears it cannot drift apart in prose. ⚑ ONE FLAG PER REASON — a waiver
# that clears more than its name says is how a gate becomes a decoration.
SHUFFLED_CONTRAST_WAIVER = "--allow-shuffled-contrast"
UNRECORDED_VALIDITY_WAIVER = "--allow-unrecorded-validity"
UNVERIFIED_IDENTITY_WAIVER = "--allow-unverified-trajectory"
NON_PREREG_HELDOUT_WAIVER = "--allow-non-prereg-heldout"


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
    digest = _file_sha256(Path(checkpoint))
    if not banked:
        return {
            "checkpoint_sha256": digest, "checkpoint_role": None, "run_id": None,
            "checkpoint_identity": (
                f"{summary_path} banks no `checkpoints`, so it cannot say "
                "whether it describes this file (written before the field "
                "existed)"
            ),
        }
    for record in banked:
        if str(record.get("sha256")) == digest:
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
                None if checkpoint is None else _file_sha256(Path(checkpoint))
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


def score_provenance_problems(
    meta_a: Mapping[str, Any], meta_b: Mapping[str, Any],
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
    run_a, role_a = meta_a.get("run_id"), meta_a.get("checkpoint_role")
    run_b, role_b = meta_b.get("run_id"), meta_b.get("checkpoint_role")
    unidentified = [
        label for label, run, role in (("--a", run_a, role_a), ("--b", run_b, role_b))
        if run is None or role is None
    ]
    if unidentified:
        refuse(
            f"{' and '.join(unidentified)} carr"
            f"{'y' if len(unidentified) > 1 else 'ies'} NO trajectory identity: "
            "fields `run_id`/`checkpoint_role` are absent, so this readout "
            "cannot tell MID-vs-LAST on one trajectory from two unrelated "
            "checkpoints — the prereg's primary yardstick is the former. "
            "Re-score with --summary pointing at the training run's "
            f"summary.json, or pass {UNVERIFIED_IDENTITY_WAIVER}.",
            waiver=UNVERIFIED_IDENTITY_WAIVER,
        )
    elif run_a != run_b:
        refuse(
            "the two score files come from DIFFERENT TRAJECTORIES: field "
            f"`run_id` reads {str(run_a)[:16]} in --a and {str(run_b)[:16]} in "
            "--b. The prereg's primary yardstick is LAST vs MID-BUDGET on ONE "
            "trajectory; two runs' checkpoints differ by their whole "
            "initialisation and data order as well as by budget.",
        )
    elif {str(role_a), str(role_b)} != {"mid", "last"}:
        refuse(
            "the two score files are not the prereg's MID/LAST pair: field "
            f"`checkpoint_role` reads {role_a!r} in --a and {role_b!r} in --b. "
            "The deciding statistic is the slope between the mid-budget and the "
            "final checkpoint of one trajectory.",
        )
  # ⚑ AND THE HELD-OUT POPULATION. `freeze` gated the row COUNT at 100,000 and
  # not the SOURCES, so a frozen set built from one hour cleared every gate in
  # this rig. Recomputed by `cmd_score` from the artifact's own `sources` list
  # rather than trusted from a stamp, so a frozen set predating the stamp is
  # judged too.
    for label, meta in (("--a", meta_a), ("--b", meta_b)):
        heldout = list(meta.get("heldout_source_selection_problems") or ())
        if heldout:
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
    heldout_problems = source_selection_problems(payload)
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
    np.savez_compressed(
        Path(args.out),
        row_ids=np.array(wanted, dtype="U32"),
        hit=hits,
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
  # ⚑ The HELD-OUT population, recomputed from the frozen artifact rather than
  # trusted from its stamp. `freeze` gated n and not the sources.
            "heldout_source_selection_problems": heldout_problems,
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
    provenance = score_provenance_problems(meta_a, meta_b)
    passed = {
        SHUFFLED_CONTRAST_WAIVER: bool(args.allow_shuffled_contrast),
        UNRECORDED_VALIDITY_WAIVER: bool(args.allow_unrecorded_validity),
        UNVERIFIED_IDENTITY_WAIVER: bool(args.allow_unverified_trajectory),
        NON_PREREG_HELDOUT_WAIVER: bool(args.allow_non_prereg_heldout),
    }
    waived = [p for p in provenance if p.waiver is not None and passed[p.waiver]]
    for flag in sorted({p.waiver for p in waived if p.waiver is not None}):
        print(f"⚑⚑ {flag}: "
              + " | ".join(p.message for p in waived if p.waiver == flag))
    refusals = [p for p in provenance if p not in waived]
    if refusals:
        offered = sorted({p.waiver for p in refusals if p.waiver is not None})
        print(
            "FAIL: these two score files are not a valid pair. "
            + " | ".join(p.message for p in refusals)
            + (
                f" Waivable by: {', '.join(offered)} — if that IS the comparison "
                "you intend."
                if offered and len(offered) == len(refusals)
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
  # ⚑ PRINTED, not merely checked. "the provenance gate passed" has to be an
  # observation in the output a reader can point at, otherwise a gate that
  # stopped running looks identical to a gate that ran and passed.
    print(f"A  {meta_a.get('checkpoint')}   top-1 {rate_a:.6f}   "
          f"targets: {target_provenance(meta_a)[0]}   "
          f"neg-control: {_negative_control_readout(meta_a)}")
    print(f"B  {meta_b.get('checkpoint')}   top-1 {rate_b:.6f}   "
          f"targets: {target_provenance(meta_b)[0]}   "
          f"neg-control: {_negative_control_readout(meta_b)}")
    print(f"paired rows          {n}")
    print(f"discordant  b(A only)={b}  c(B only)={c}  "
          f"discordance={(b + c) / n:.4f}")
    print(f"delta (B - A)        {delta * 100:+.4f} pp")
    print(f"95% CI               [{(delta - half) * 100:+.4f}, "
          f"{(delta + half) * 100:+.4f}] pp   (halfwidth {half * 100:.4f} pp)")
    print(f"exact McNemar p      {p_exact:.6g}")
  # ⚑ THE PRE-COMMITTED BAR IS A PROPERTY OF n, AND NOTHING WAS ENFORCING n.
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
    half_pp = half * 100.0
    prereg_bar_in_force = args.max_halfwidth_pp is None
    bar_pp = MATERIAL_BAR_PP if prereg_bar_in_force else float(args.max_halfwidth_pp)
    problems: list[str] = []
    if b + c == 0:
        problems.append(
            f"the two score files are discordant on ZERO of {n} paired rows, so "
            "the McNemar halfwidth is 0.0000 pp BY CONSTRUCTION and clears any "
            "bar at any n. That is not a resolved comparison, it is two files "
            "that agree on every row — check that they are different "
            "checkpoints.",
        )
    if prereg_bar_in_force and n < PREREG_SAMPLE_ROWS:
        problems.append(
            f"n={n} is below the prereg's resolution point "
            f"{PREREG_SAMPLE_ROWS}, at which the {bar_pp:.4f} pp bar was "
            "derived. Score the full frozen set, or pass --max-halfwidth-pp "
            "explicitly with the bar you are re-deriving at this n.",
        )
    if half_pp > bar_pp:
        problems.append(
            f"halfwidth {half_pp:.4f} pp exceeds the pre-committed bar "
            f"{bar_pp:.4f} pp at n={n}. At this n the comparison cannot resolve "
            "the effect it is being judged against.",
        )
    if problems:
        message = " | ".join(problems)
        if not args.allow_underpowered:
            print(f"FAIL: {message}", file=sys.stderr)
            return 1
        print(f"⚑⚑ --allow-underpowered: {message}")
    return 0


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs."""
    total = b + c
    if total == 0:
        return 1.0
    from math import comb

    smaller = min(b, c)
    tail = sum(comb(total, k) for k in range(smaller + 1)) / 2**total
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
             "Passing it explicitly also waives the n floor, because it says "
             "you re-derived the bar at the n you have.",
    )
    compare.add_argument("--allow-underpowered", action="store_true")
    compare.add_argument(
        UNVERIFIED_IDENTITY_WAIVER, action="store_true",
        help="proceed when a score file carries no `run_id`/`checkpoint_role`. "
             "⚑ Without them a pair of individually-valid checkpoints can be "
             "two LASTs from two independently initialised trajectories, "
             "reported as the prereg's LAST vs MID-BUDGET slope. It does NOT "
             "waive a run_id MISMATCH or a role pair that is not {mid, last} — "
             "those are measurements, not absences.")
    compare.add_argument(
        NON_PREREG_HELDOUT_WAIVER, action="store_true",
        help="proceed when a score file's frozen set was not built from the "
             "preregistered six hourly tars. Its own flag, so declaring a "
             "different held-out population does not also clear an unrecorded "
             "validity or a shuffled contrast.")
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
