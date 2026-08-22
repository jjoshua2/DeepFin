#!/usr/bin/env python3
"""Offline target-retuning driver: retrain a checkpoint over existing replay
with SF targets REBUILT from sparse MultiPV labels under different params.

For each ``--variant name:k=v,k=v`` this runs a fixed training budget from
the same starting checkpoint and replay window with the given params
overridden, then writes one checkpoint per variant. This is what turns
sf_policy_temp / cp-logistic / label-smoothing and replay-sampling
questions into offline retrains instead of weeks-long live A/Bs — judge the
resulting checkpoints per docs/eval_protocol.md (arena_standard matched_sims
vs the base checkpoint).

SF targets: by default ``train.rebuild_sf_targets`` is forced on (targets
recomputed from the sparse MultiPV labels) so the A/B isolates a target
param. Pass ``--no-rebuild-sf-targets`` to train on the shards' stored
targets exactly as live training does — required when the A/B is a *sampling*
knob (e.g. replay_sf_gap_priority_weight), not a target param.

With the rebuild ON, ``sf_p0_policy_target`` and ``sf_volatility_target`` are
MASKED in every arm (their sources live on other shard rows and no sampled
batch can move them with their source — docs/target_rebuildability.md). The
masking is identical across arms, so variant deltas stay paired; it does mean
the ``w_sf_own`` own-move-teacher leg is absent from ALL arms of a rebuilt
sweep, and the sweep therefore says nothing about that leg.

Overridable param keys: sf_policy_temp, sf_policy_label_smooth,
sf_wdl_use_cp_logistic, sf_wdl_cp_slope, sf_wdl_cp_draw_width, and any
replay_* sampling knob (anything else in the flat config also works, e.g. lr).

Every variant retrains from a COLD optimizer: only the checkpoint's model
weights are restored; optimizer moments / scheduler / step counters are
deliberately discarded so all variants share the identical fresh-AdamW
starting point (warmup per the config). The deltas between variants stay
meaningful, but the absolute trajectories differ from what a live retune
that kept the optimizer state would produce. Each variant is seeded
identically, so dropout masks and the buffer's random draws match across
arms; a replay_* sampling override still (intentionally) changes WHICH rows
those shared draws select. Matched draws need BOTH halves and both are now
enforced, not merely intended:

* ``deterministic_refresh=True`` on the buffer below removes the load-dependent
  refresh race (NOT true before 2026-07-31 — see the comment there); and
* ARM 1's OWN buffer scan is the reference shard list, and every later arm
  asserts its buffer scanned exactly that list. Variants run sequentially and
  each is a full retrain, so a live window or a salvage pool that ingests
  mid-run gives the later arms a different pool — same seed, different rows,
  silently unpaired. Measured 2026-07-31 at ``--steps 800``, flag on, same seed:
  one extra shard moved 617/800 sampled rows (77.1%), one shard fewer moved
  343/800 (42.9%). The run now aborts instead.

The reference is arm 1's ``DiskReplayBuffer._snapshot_shards()`` and NOT a glob
taken earlier in ``main()``. Pairing is defined relative to the pool arm 1
actually drew from, so arm 1 is the only honest reference: a pre-startup glob
sits tens of seconds before arm 1's scan (``torch.load`` + ``build_model`` +
``Trainer()`` + CUDA init all run in between), and a shard landing inside that
window aborts the sweep even though every arm would have been paired. That is
not a stricter gate, it is a false one — it fires on the single case the gate
does not care about while detecting nothing the arm-1 reference misses.

⚑ POINT ``--replay-dir`` AT A FROZEN COPY, not the live window. A shard landing
while arm 1 TRAINS still de-pairs arm 2, and that abort is real: the sweep has
to be re-run from the start. ``cp -r`` the shards, or use a salvage pool nothing
is writing to.

TARGET-SURGERY LADDER (arms R / V / G), all DEFAULT OFF. Three rig-only arms
that use a shallow wide-MultiPV SF label as an EDITOR of the stored search target
rather than as a CE-teacher — the form dose-ladder2 killed (its blend flattened
the student and worsened E[regret] monotonically with dose). Each arm joins the
label file written by ``scripts/rvg_label_pass.py`` by position fingerprint:

* ``rvg_arm=r`` — listed-only regret loss. Serves ``sf_p0_regret`` rebuilt from
  the MPV20 label with UNLISTED MOVES AT 0.0 (June's PR #78 form defaulted them
  to 1.0, making a magnitude arm into a membership prior) and sets
  ``w_sf_own_regret`` from ``--rvg-r-weight``. REQUIRES ``w_sf_policy_floor=0``
  on the arm and its baseline — see ``_apply_rvg_config_side``.
* ``rvg_arm=v`` — veto edit: ``t' ~ t*exp(-lambda*relu(regret-tau))`` on listed
  moves plus a hard zero at ``--rvg-v-veto-cp``; unlisted moves untouched.
* ``rvg_arm=g`` — geometric blend: ``t' ~ t^(1-alpha) * q^alpha`` on the listed
  moves, ``q ~ exp(-regret/T)``; unlisted keep their t-mass.
* ``rvg_arm=vg`` — the COMPOSITION, V FIRST and then G on the veto-edited
  target, reusing the same two edit functions (``RvgArmSpec.STAGES``) rather
  than a third implementation. Order is load-bearing: on a softly-downweighted
  move V-then-G applies V's multiplier to the power ``(1-alpha)`` while
  G-then-V applies it at full strength. Per-stage row counters are reported.

Every arm reports realized coverage ``f`` off its own counters, and an arm that
served zero rows or covered zero rows is a REFUSAL, not a null result.

Usage::

    PYTHONPATH=. python3 scripts/retarget_retrain.py \\
        --config configs/pbt2_small.yaml \\
        --checkpoint <trial>/checkpoint_000123/trainer.pt \\
        --replay-dir <frozen copy>/replay_shards   # NOT the live window \\
        --steps 800 --out-dir runs/retarget \\
        --variant base: \\
        --variant sharp:sf_policy_temp=0.006 \\
        --variant smooth:sf_policy_temp=0.05,sf_policy_label_smooth=0.02

    # ... or one target-surgery ladder, ONE invocation (paired draws):
    PYTHONPATH=. python3 scripts/retarget_retrain.py ... \\
        --rvg-labels <out>/rvg_labels_mpv40_150k.jsonl \\
        --rvg-v-lambda 0.02 --rvg-v-tau 20 --rvg-v-veto-cp 200 \\
        --rvg-g-alpha 0.3 --rvg-g-temp 150 \\
        --variant a000: \\
        --variant v020:rvg_arm=v \\
        --variant g030:rvg_arm=g \\
        --variant vg01:rvg_arm=vg
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Protocol
import json
import time
from pathlib import Path

import numpy as np

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.eval.audit import AUDIT_REGRET_CAP_CP
from chess_anti_engine.eval.rvg_surgery import (
    RvgExternalPolicyIndex,
    RvgLabelIndex,
    apply_geometric_blend,
    apply_veto_edit,
    distribution_entropy_nats,
    listed_only_regret_vector,
    position_fingerprints,
)
from chess_anti_engine.model import build_model, load_state_dict_tolerant, model_config_from_flat_config
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays
from chess_anti_engine.replay.threat_upgrade import V1_INPUT_PLANES
from chess_anti_engine.train.constants import SF_OWN_REGRET_CAP_CP
from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.uci.model_loader import model_config_from_arch
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


def _assert_replay_planes_match(replay_dir: Path, target_planes: int, *, upgrade_v1: bool) -> None:
    """Fail loud if the replay shards' stored plane width can't feed the arch.

    The empty-buffer guard in ``_run_variant`` only catches the stored>target
    case (wider shards are rejected -> empty pool). The stored<target case is
    worse and SILENT: the buffer zero-pads the missing planes, so a v1
    (146-plane) ``--replay-dir`` against a v2_threats (175-plane) checkpoint
    trains on all-zero threat planes and prints a success line — an invalid A/B
    with no error. ``len(buf)`` counts scanned shards regardless of plane width,
    so it can't detect this; check the actual stored width up front.
    """
    paths = iter_shard_paths(replay_dir)
    if not paths:
        return  # no shards: the empty-buffer guard reports this with context
    arrs, _ = load_shard_arrays(paths[0], lazy=True)
    stored = int(arrs["x"].shape[1])
    if stored == target_planes:
        return
    if stored == V1_INPUT_PLANES and stored < target_planes and upgrade_v1:
        return  # v1 shards will have their threat planes recomputed — intended
    detail = (
        "would be SILENTLY ZERO-PADDED (set replay_upgrade_v1_planes to "
        "recompute the threat planes instead)" if stored < target_planes
        else "are wider than the arch and would be rejected"
    )
    raise SystemExit(
        f"replay shards under {replay_dir} store {stored} input planes but the "
        f"checkpoint arch needs {target_planes}; the {stored}-plane shards {detail}. "
        "Point --replay-dir at matching shards — the A/B would otherwise be invalid."
    )


def _assert_shards_unchanged(
    *, observed: list[Path], snapshot: list[Path], name: str,
) -> None:
    """Abort if this arm's replay pool differs from the first arm's.

    ``deterministic_refresh=True`` makes the draw sequence a function of the
    seed AND of the shard list the buffer scanned. Variants run sequentially
    and each is a full retrain (minutes to hours), so a ``--replay-dir`` that
    is the live window or a salvage pool being topped up hands the later arms a
    different pool: identical seed, different rows, and every variant delta the
    script prints is then unpaired with nothing on screen to say so. Refuse to
    train rather than emit a comparison that reads valid.

    ``observed`` is THIS arm's pool and ``snapshot`` is arm 1's. Both are
    keyword-only BECAUSE they are the same type: passed positionally a swap is
    silent, still aborts, and inverts added/removed and the two counts, sending
    the operator hunting a shard that was deleted when one was in fact added.
    Keyword-only makes that swap unexpressible rather than merely tested-for.

    There is deliberately no falsy-``snapshot`` early return: an empty
    ``snapshot`` is a real reference and must still fail against a non-empty
    ``observed``. On today's path ``_run_variant`` cannot actually hand this
    function ``[]`` — an arm-1 pool of ``[]`` means ``len(buf) == 0``, which
    ``SystemExit``s before the summary is returned, so ``main()`` never adopts
    an empty reference. That justification therefore describes a case which
    cannot arise; the choice is still right, and the tests pin it as a decision
    rather than an unverifiable preference.
    """
    if observed == snapshot:
        return
    obs, snap = set(observed), set(snapshot)
    added = sorted(p.name for p in obs - snap)
    removed = sorted(p.name for p in snap - obs)
    detail = (
        f"{len(added)} added ({', '.join(added[:5])}{'...' if len(added) > 5 else ''}), "
        f"{len(removed)} removed ({', '.join(removed[:5])}{'...' if len(removed) > 5 else ''})"
        if (added or removed)
        else "same shards in a different scan order"
    )
    raise SystemExit(
        f"replay shard list CHANGED before variant {name!r}: {detail} "
        f"({len(snapshot)} shards at start, {len(observed)} now). The arms would "
        "be DE-PAIRED — same seed, different pool, so their draws no longer "
        "match and every variant delta this script prints would be unpaired. "
        "Point --replay-dir at a frozen copy of the shards (or a salvage pool "
        "nothing is writing to) and re-run the whole sweep."
    )


def _assert_draws_unchanged(
    *, observed: dict[str, float], snapshot: dict[str, float], name: str,
) -> None:
    """Abort if this arm drew a different NUMBER of batches than the first arm.

    ``_assert_shards_unchanged`` pins the POOL. It does not pin the draw
    sequence, and there is a path that breaks the sequence with the pool
    untouched: ``Trainer.train_steps`` catches a transient CUDA error and calls
    ``batch_iter.add_retry_batches(...)``, which pulls replacement batches and
    so advances the buffer's private RNG. One retry in ONE arm desynchronises
    that arm's rows from every other arm's PERMANENTLY, for every subsequent
    step -- and it used to be visible only as a ``logging.warning`` on a
    thousand-line console. ``retarget_report.json`` recorded nothing, so a
    silently de-paired 21-hour sweep was indistinguishable from a clean one.

    Load-dependent exactly like the shuffle-refresh race this module already
    fixed, and on a shared box over hours it is not a remote possibility.
    Refuse rather than print a comparison that reads valid.

    Both counts come from ``TrainMetrics``: ``batches_drawn`` is the total
    microbatches pulled from the buffer and ``transient_cuda_retry_batches``
    how many of those were replacements. A non-zero retry count in the FIRST
    arm is fatal on its own -- there is no later arm to disagree with it, and
    the reference sequence is already the perturbed one.
    """
    obs_retry = float(observed.get("transient_cuda_retry_batches", 0.0))
    snap_retry = float(snapshot.get("transient_cuda_retry_batches", 0.0))
    obs_drawn = float(observed.get("batches_drawn", 0.0))
    snap_drawn = float(snapshot.get("batches_drawn", 0.0))
    if obs_retry == snap_retry == 0.0 and obs_drawn == snap_drawn:
        return
    raise SystemExit(
        f"replay DRAW SEQUENCE de-paired at variant {name!r}: "
        f"batches_drawn {snap_drawn:.0f} (reference) vs {obs_drawn:.0f} (this arm), "
        f"transient CUDA retry batches {snap_retry:.0f} vs {obs_retry:.0f}. "
        "A transient-CUDA retry draws replacement batches and advances the "
        "buffer's RNG, so from that point on the arms sample DIFFERENT rows "
        "with the same seed and every delta this script prints is unpaired. "
        "Re-run the whole sweep on a quieter box (or with a larger "
        "--gpu-mem-fraction); the counts are in retarget_report.json."
    )


def _override_key_reaches_the_trainer(key: str, base_config: dict) -> bool:
    """Does ``key`` actually change what an arm trains with?

    ⚑ THIS EXISTS BECAUSE THE 2026-07-06 DOSE SCREEN SHIPPED A NO-OP AS A
    RESULT. Its arms were launched with ``sf_gap_priority_signed=...``; the key
    is not a ``TrialConfig`` field, ``TrialConfig.from_dict`` drops unknown keys
    in silence, and ``retarget_report.json`` recorded the override as applied.
    Four arms of GPU time measured the control twice. An independent review
    reproduced the same shape here with a single doubled letter --
    ``policy_target_tempp=1.5`` is accepted, trains at 1.0, and is reported as
    an override -- which would spend a 21-hour paired sweep producing two
    identical nets and a report claiming otherwise.

    The test is BEHAVIOURAL rather than a hand-kept allowlist, because an
    allowlist is one more thing that drifts from the code it describes:

    1. a ``TrialConfig`` field -- reaches the replay buffer; or
    2. a key the flat config already defines -- a real knob of some consumer,
       and refusing it would break every existing sampling/loss sweep; or
    3. a key that demonstrably moves ``trainer_kwargs_from_config``'s output --
       this is what catches ``policy_target_temp``, which is read straight off
       the flat dict by the Trainer and is in neither of the first two sets.

    Anything else cannot reach the Trainer or the buffer, so an arm carrying it
    is the control wearing another name.
    """
    if key in base_config:
        return True
    if hasattr(TrialConfig.from_dict({}), key):
        return True
  # Probe with two values so a key whose default happens to equal the probe
  # cannot read as inert. Comparing the WHOLE kwargs dict, not the same-named
  # entry: `trainer_kwargs_from_config` renames some keys on the way through.
    base_kwargs = trainer_kwargs_from_config(dict(base_config))
    for probe in (0.5, 2.0):
        if trainer_kwargs_from_config({**base_config, key: probe}) != base_kwargs:
            return True
    return False


def _assert_overrides_reach_the_trainer(
    *, name: str, overrides: dict, base_config: dict,
) -> None:
    """Fail the sweep on an override key nothing will read. See above."""
    dead = sorted(
        k for k in overrides if not _override_key_reaches_the_trainer(k, base_config)
    )
    if not dead:
        return
    raise SystemExit(
        f"variant {name!r} sets {len(dead)} key(s) that reach NOTHING: "
        f"{', '.join(repr(k) for k in dead)}. They are not TrialConfig fields, "
        "are absent from the flat config, and do not move "
        "trainer_kwargs_from_config's output, so this arm would train "
        "identically to the control while retarget_report.json recorded the "
        "override as applied -- the 2026-07-06 sf_gap_priority_signed failure. "
        "Check the spelling against the yaml and TrialConfig."
    )


def _sweep_level_keys() -> frozenset[str]:
    """Config keys a per-variant override CANNOT win, derived from the signature.

    ⚑ REVIEW FINDING (codex, P1). ``_run_variant`` takes ``steps``, ``batch_size``,
    ``device``, ``checkpoint``, ``replay_dir`` and ``gpu_mem_fraction`` as explicit
    arguments that ``main()`` resolves ONCE from the CLI and the BASE config, and
    passes identically to every arm --- e.g. ``batch_size`` at
    ``int(args.batch_size or base_config.get("batch_size", 256))``, off
    ``base_config``, never off the per-variant ``config``. So
    ``--variant arm:batch_size=1024`` updates a dict nothing downstream reads,
    trains identically to the control, and lands in ``retarget_report.json`` as
    ``"overrides": {"batch_size": 1024}``.

    That is the 2026-07-06 ``sf_gap_priority_signed`` failure again, and
    ``_override_key_reaches_the_trainer`` does NOT catch it: ``batch_size`` is a
    real ``TrialConfig`` field, so it passes the reachability probe honestly ---
    it reaches the buffer, it just cannot reach the thing an operator writing
    ``batch_size=1024`` means. Reachability is the wrong question for this class;
    SHADOWING is.

    Derived by introspection rather than hand-listed, so a future parameter added
    to ``_run_variant`` is covered the day it is added instead of the day someone
    remembers this list. ``rebuild_sf_targets`` was already refused by hand here
    for the same reason; it now falls out of the same rule.
    """
    import inspect
    plumbing = {"name", "overrides", "base_config", "out_dir", "shard_snapshot",
                "draw_snapshot", "allow_yaml_arch", "allow_partial_load",
  # `rig` carries the sweep-level rvg DEFAULTS and the loaded label file. It is
  # plumbing rather than a shadowed setting because the individual `rvg_*`
  # names are NOT parameters of `_run_variant` — they are rig-only keys the
  # variant string is SUPPOSED to override (that is how one invocation runs a
  # whole ladder over one identically-seeded draw sequence). Listing `rig` here
  # keeps `rvg_g_alpha=0.3` overridable while `steps`/`batch_size` stay refused.
                "rig"}
    return frozenset(inspect.signature(_run_variant).parameters) - plumbing


def _parse_variant(spec: str) -> tuple[str, dict]:
    name, _, body = spec.partition(":")
    if not name:
        raise SystemExit(f"--variant needs a name before ':', got {spec!r}")
    shadowed = _sweep_level_keys()
    overrides: dict = {}
    for pair in filter(None, body.split(",")):
        k, _, v = pair.partition("=")
        if not _:
            raise SystemExit(f"variant override must be k=v, got {pair!r}")
        if k.strip() == "rebuild_sf_targets":
            # A variant override here would silently beat the CLI flag AND
            # could give the two arms different targets, so the A/B would
            # no longer isolate the knob it claims to. One global flag only.
            raise SystemExit(
                "rebuild_sf_targets is not variant-overridable — use the "
                "global --rebuild-sf-targets / --no-rebuild-sf-targets flag "
                "so every arm trains on the same targets"
            )
        if k.strip() in shadowed:
            raise SystemExit(
                f"{k.strip()!r} is a SWEEP-LEVEL setting, not variant-overridable: "
                f"main() resolves it once and passes it to every arm as an explicit "
                f"argument, so this override would reach nothing while "
                f"retarget_report.json recorded it as applied. Use the matching "
                f"--{k.strip().replace('_', '-')} flag, which applies to all arms."
            )
        lowered = v.strip().lower()
        if lowered in ("true", "false"):
            overrides[k.strip()] = lowered == "true"
        else:
            try:
                overrides[k.strip()] = float(v)
            except ValueError:
                overrides[k.strip()] = v.strip()
    return name, overrides


class _SoftPolicyAsMainBuffer:
    """Rig-only buffer view: serve ``policy_soft_target`` as the MAIN policy target.

    Screens the lc0-shape hypothesis (ledger 2026-08-06: our policy_target is
    ~2.4x sharper than real lc0 training targets, while the stored
    ``policy_soft_target`` already matches lc0's entropy) without touching
    production code: the trainer sees batches whose ``policy_target`` IS the
    soft target, so the search-facing head trains on the lc0-shaped
    distribution. Rows lacking a soft target keep their hard target —
    ``has_policy`` stays authoritative for masking; ``has_policy_soft`` only
    gates the swap. Activated per-variant via ``rig_policy_from_soft=1``
    (popped before the config reaches the Trainer).
    """

    def __init__(self, inner: DiskReplayBuffer):
        self._inner = inner
        self.total_rows = 0
        self.eligible_rows = 0

    def rig_active_stamp(self) -> str:
        """This wrapper has ONE shape, so the stamp is a constant.

        ⚑ It used to read ``self._spec.stamp()``, copied from the rvg wrapper
        during the port. ``_spec`` is not an attribute here, so ``__getattr__``
        forwarded the lookup to the wrapped ``DiskReplayBuffer`` and the arm
        died with ``AttributeError`` the first time ``rig_policy_from_soft=1``
        was used — i.e. this branch broke a pre-existing arm, and no test caught
        it because the only existing mention refuses before it ever wraps.
        """
        return "SOFT"

    def rig_description(self) -> str:
        return ("main policy target served from policy_soft_target "
                "(soft-shape screen)")

    def rig_stats(self) -> dict[str, Any]:
        return {
            "total_rows": int(self.total_rows),
            "eligible_rows": int(self.eligible_rows),
            "realized_f": (float(self.eligible_rows) / float(self.total_rows)
                           if self.total_rows else 0.0),
        }

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def __len__(self) -> int:
        return len(self._inner)

    def sample_batch_arrays(self, batch_size: int, **kw) -> dict:
        arrs = self._inner.sample_batch_arrays(batch_size, **kw)
        soft = arrs.get("policy_soft_target")
        hard = arrs.get("policy_target")
        has_soft = arrs.get("has_policy_soft")
        if soft is None or hard is None or has_soft is None:
            raise SystemExit(
                "rig_policy_from_soft=1 but the sampled batch lacks "
                "policy_soft_target/has_policy_soft — this pool predates soft "
                "targets, so the arm would silently train on hard targets"
            )
        swapped = hard.copy()
        mask = has_soft.astype(bool)
        self.total_rows += int(hard.shape[0])
        self.eligible_rows += int(mask.sum())
        swapped[mask] = soft[mask].astype(swapped.dtype, copy=False)
        out = dict(arrs)
        out["policy_target"] = swapped
        return out


# ── target-surgery ladder (arms R / V / G) ───────────────────────────────────


class _SamplesBatches(Protocol):
    """What a rig wrapper actually needs from the thing it wraps.

    Stated as a protocol rather than as ``DiskReplayBuffer`` because that is the
    honest contract — the wrapper calls exactly these two methods and forwards
    everything else — and because a nominal annotation forces every test to cast
    a stub through ``object``, which is a suppression pretending to be a type.
    """

    # Mirrors `DiskReplayBuffer.sample_batch_arrays` exactly, including the
    # keyword-only `wdl_balance`: a `**kw: Any` member would demand that every
    # implementation accept arbitrary keywords, which the real buffer does not,
    # and the protocol would then describe nothing that exists.
    def sample_batch_arrays(
        self, batch_size: int, *, wdl_balance: bool = True,
    ) -> dict[str, np.ndarray]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class RvgArmSpec:
    """One target-surgery arm's fully resolved parameters.

    Resolved ONCE per variant from the sweep-level ``--rvg-*`` defaults and the
    variant's own ``rvg_*`` overrides, then handed to the wrapper. Every number
    the wrapper applies is announced off THIS object by the wrapper itself, not
    off the CLI namespace the operator typed into — a value that is accepted and
    then re-derived somewhere else is this repo's signature defect, and the
    ladder2 wiring proof is exactly the counter-measure.
    """

    arm: str
    r_weight: float
    v_lambda: float
    v_tau_cp: float
    v_veto_cp: float
    g_alpha: float
    g_temp_cp: float

    #: Ordered target-edit STAGES per arm. ``vg`` is a COMPOSITION, expressed as
    #: an ordered list rather than as a third edit function: the chain reuses the
    #: exact ``apply_veto_edit`` / ``apply_geometric_blend`` calls the standalone
    #: arms use, so "VG" cannot drift from "V" and "G" the way a hand-fused
    #: third implementation would. Arm ``r`` has NO target stage — it edits
    #: ``sf_p0_regret`` and a loss weight, not the search target.
    STAGES: ClassVar[dict[str, tuple[str, ...]]] = {
        "r": (), "v": ("V",), "g": ("G",), "vg": ("V", "G"),
    }

    def stages(self) -> tuple[str, ...]:
        return RvgArmSpec.STAGES[self.arm]

    def stamp(self) -> str:
        """Unambiguous arm identity for ``rig_active``: ``R`` / ``V`` / ``G`` /
        ``V+G``. A composition must never read as one of its halves in the
        report — that is how an operator ends up comparing a VG run against a G
        entry in the ladder table."""
        stages = self.stages()
        return "+".join(stages) if stages else self.arm.upper()

    def _describe_v(self) -> str:
        return (f"V: t' ~ t*exp(-{self.v_lambda}*relu(regret-{self.v_tau_cp:g}cp)), "
                f"hard veto at regret >= {self.v_veto_cp:g}cp, unlisted untouched")

    def _describe_g(self, q_source: str = "sf_regret") -> str:
        q = (f"q ~ exp(-regret/{self.g_temp_cp:g}cp) over listed moves"
             if q_source == "sf_regret"
             else f"q from {q_source} (external per-row policy; temp NOT applied)")
        return (f"G: t' ~ t^(1-{self.g_alpha}) * q^{self.g_alpha}, {q}, "
                "unlisted keep their t-mass")

    def describe(self, q_source: str = "sf_regret") -> str:
        if self.arm == "r":
            return (f"R: listed-only regret loss, w_sf_own_regret={self.r_weight} "
                    f"(regret/{SF_OWN_REGRET_CAP_CP:g}cp, unlisted moves EXCLUDED)")
        if self.arm == "v":
            return self._describe_v()
        if self.arm == "g":
            return self._describe_g(q_source)
        # ⚑ ORDER IS PART OF THE ARM, NOT AN IMPLEMENTATION DETAIL. V FIRST, then
        # G on the VETO-EDITED target. The two orders give different answers on
        # every softly-downweighted move: V-then-G raises V's multiplier to the
        # power (1-alpha), G-then-V applies it at full strength. (The HARD veto
        # is order-invariant — zero times anything is zero — so an example built
        # only from hard vetoes cannot tell the two apart, which is exactly the
        # trap the composed unit test is written to avoid.)
        return (f"VG (V FIRST, then G on the veto-edited target) — "
                f"{self._describe_v()} THEN {self._describe_g(q_source)}")

    def _validate_v(self) -> None:
        if self.v_lambda < 0.0 or self.v_tau_cp < 0.0:
            raise SystemExit("rvg V: lambda and tau must be >= 0")
        if self.v_lambda == 0.0 and self.v_veto_cp > AUDIT_REGRET_CAP_CP:
            raise SystemExit(
                f"rvg V with lambda=0 and veto_cp={self.v_veto_cp:g} above the "
                f"{AUDIT_REGRET_CAP_CP:g}cp regret cap: nothing can be downweighted "
                "and nothing can be vetoed, so this arm IS the control"
            )

    def _validate_g(self) -> None:
        if not 0.0 < self.g_alpha < 1.0:
            raise SystemExit(
                f"rvg G with alpha={self.g_alpha}: the band is 0 < alpha < 1. "
                "alpha=0 is the control (and must run UNWRAPPED, bitwise); "
                "alpha=1 makes t**0 == 1 and would GROW the target's support "
                "onto moves the search never visited."
            )
        if not self.g_temp_cp > 0.0:
            raise SystemExit(f"rvg G with temp={self.g_temp_cp}: needs > 0")

    def validate(self) -> None:
        """Refuse an arm that cannot move anything.

        An arm whose parameters make the edit the identity is the CONTROL under
        another name, and it would land in ``retarget_report.json`` as a dose.
        A composition validates EVERY stage it contains — a VG arm with an inert
        G leg is a V arm wearing VG's label in the ladder table.
        """
        if self.arm not in RvgArmSpec.STAGES:
            raise SystemExit(
                f"rvg_arm={self.arm!r}: expected one of "
                f"{', '.join(sorted(RvgArmSpec.STAGES))}"
            )
        if self.arm == "r" and not self.r_weight > 0.0:
            raise SystemExit(
                f"rvg_arm=r with --rvg-r-weight {self.r_weight}: a zero weight "
                "trains the control while reporting an arm"
            )
        for stage in self.stages():
            (self._validate_v if stage == "V" else self._validate_g)()


class _RvgTargetSurgeryBuffer:
    """Serve MPV20-edited targets (arms V / G / VG) or regret vectors (arm R).

    Joins each sampled row to the shallow wide-MultiPV label file by
    ``rvg_surgery.position_fingerprints`` — the SAME function
    ``scripts/rvg_label_pass.py`` keyed the file with, called on the SAME stored
    planes, so the join is exact by construction. The realized coverage ``f`` is
    reported per arm exactly as dose-ladder2 reported its ``sf_p0`` eligibility,
    because ``f`` is a property of the banked corpus and not of the live run.

    ⚑ "UNLABELED ROWS ARE SERVED UNTOUCHED" IS TRUE FOR V/G/VG AND FALSE FOR R,
    and the difference is deliberate. V, G and VG edit ``policy_target`` in
    place and leave an unlabeled row's array bit-identical. Arm R REPLACES the
    whole ``sf_p0_regret`` field with a freshly built vector, so an unlabeled
    row does not keep its stored MPV6 regret — it gets zeros and
    ``has_sf_p0_regret = 0``. That is the point of the arm: the stored vector is
    the MPV6 one whose bad-mass reach is 60.9% and whose warm-TT rows do not
    reproduce, and mixing repaired and unrepaired vectors inside one loss term
    would make the arm a blend of two teachers rather than a test of one.
    ``has = 0`` is the field's own "no label here" state, so the row leaves the
    ``w_sf_own_regret`` term entirely rather than contributing "regret is zero
    on every move", which would be a strong and false signal —
    ``test_arm_r_unlabeled_rows_leave_the_loss_term_entirely`` pins that through
    the real loss, not through the field.

    ⚑ ARM R EDITS ``sf_p0_regret``, NOT ``policy_target``. The loss leg it feeds
    (``w_sf_own_regret``) is a whole-distribution pull ``sum_m p(m) * r(m)``;
    with unlisted moves at 0.0 that sum runs over the listed set only, which is
    the repair the prereg asks for. Two consequences are stated rather than
    discovered:

    * the term's minimum over the UNLISTED set is at "put all mass there" —
      unlisted moves cost nothing. With MPV20 covering 95.3% of the target's bad
      mass the escape hatch is narrow, and the main policy CE anchors the
      distribution, but it is a real property of the literal listed-only form
      and it is what the arm tests;
    * ``sf_policy_floor_deficit`` reads the SAME vector and treats
      ``regret <= delta_cp/cap`` as SF-APPROVED, so unlisted-at-0.0 would floor
      every unlisted legal move. ``_apply_rvg_config_side`` refuses arm R unless
      ``w_sf_policy_floor`` is 0 for that arm.

    ⚑ ARM VG IS A CHAIN, NOT A THIRD EDIT. ``RvgArmSpec.STAGES`` orders it as
    ``V`` then ``G``, and the loop below feeds each stage the PREVIOUS stage's
    output — so V and G run through the identical functions the standalone arms
    call and a fix to either reaches VG automatically. Two chain decisions,
    stated because neither is forced:

    * a stage that FALLS BACK (all mass removed / degenerate) returns its INPUT
      unchanged, so the next stage sees the last good target rather than the raw
      one. The row is still counted against that stage's ``fallback_rows``;
    * a row is written back if ANY stage edited it. A row every stage fell back
      on is served bitwise-unedited and counted in the arm's ``fallback_rows``.
    """

    def __init__(
        self, inner: _SamplesBatches, spec: RvgArmSpec, labels: RvgLabelIndex,
        *, external_q: RvgExternalPolicyIndex | None = None,
    ) -> None:
        spec.validate()
        # ⚑ THE TEST HAS TO BE "HAS A G STAGE", NOT "HAS ANY STAGE". `stages()`
        # is empty only for arm R, so the original form ACCEPTED arm V beside an
        # external q: V has a stage, the guard passed, the q reached nothing, and
        # `rig_stats` still reported `q_source: external_policy_file`. A FALSE
        # PROVENANCE LINE is worse than the silent no-op it describes, because it
        # is the line a reader would use to rule the failure out.
        #
        # ⚑ AND THE ANSWER IS NOT A PER-LEG REFUSAL. A mixed ladder
        # (a000 / v020 / g030) with a q file is legitimate — the q belongs to the
        # G legs and the V leg simply has no use for it — so refusing here would
        # ban a correct sweep. Instead: DETACH it on a leg that cannot use it, so
        # the leg is honest about running on nothing, and let `main` refuse the
        # case that is genuinely an operator error (a q file that reaches NO leg
        # in the whole sweep). Silence is not an option either way; what varies
        # is whether the right response is "stop" or "say so".
        self._external_q_detached = external_q is not None and "G" not in spec.stages()
        if self._external_q_detached:
            external_q = None
            print(
                f"[retarget] rvg arm {spec.arm!r} has no G stage: the external q "
                "source is NOT attached to this leg and its q_source reads "
                "sf_regret. (The q applies to the g/vg legs of this sweep.)"
            )
        self._inner = inner
        self._spec = spec
        self._labels = labels
        self._external_q = external_q
        # Rows whose G stage found NO external-q record. Counted, never
        # silently back-filled from SF: the two sources are different teachers
        # and quietly substituting one for the other is the arm-B version of
        # this repo's signature defect.
        self.external_q_missing_rows = 0
        # Present in the q file, but not one move it names is in this row's
        # listed set -- see the G stage. Counted apart from `missing_rows`
        # because the two point at different repairs.
        self.external_q_zero_overlap_rows = 0
        self.total_rows = 0
        self.eligible_rows = 0
        self.edited_rows = 0
        self.fallback_rows = 0
        # PER-STAGE, so a VG run shows how many rows V touched and how many G
        # then touched. An aggregate alone cannot distinguish "both stages bit"
        # from "V bit and G was inert on every row" — and the second is a V arm
        # reported as a VG arm.
        self.stage_edited: dict[str, int] = dict.fromkeys(spec.stages(), 0)
        self.stage_fallback: dict[str, int] = dict.fromkeys(spec.stages(), 0)
        # ⚑⚑ CALIBRATION INSTRUMENT, NOT A METRIC. Arm G's (alpha, T) are chosen
        # so the post-edit target entropy lands near the ~0.970-nat BT4 anchor;
        # these two sums make that choice CHECKABLE from retarget_report.json
        # instead of asserted. They are never a success criterion and never a
        # gate -- G's temperature IS the gradient of entropy, so an arm scored on
        # entropy wins by construction
        # ([[an_arm_that_is_the_gradient_of_the_metric_always_wins]]). The
        # deciding yardstick stays row-(a) top-1 deep-SF regret with the
        # E[regret] co-criterion.
        #
        # "Before" is the G STAGE's own input, which in arm VG is the
        # VETO-EDITED target, not the raw stored one -- G is the last stage, so
        # "after" is the final target either way. Accumulated only over rows the
        # G stage actually edited; `g_entropy_rows` is the denominator and is
        # reported beside them so nobody has to guess it.
        self.g_entropy_pre_sum = 0.0
        self.g_entropy_post_sum = 0.0
        self.g_entropy_rows = 0

    @property
    def spec(self) -> RvgArmSpec:
        """The realized arm, read off the object that applies it."""
        return self._spec

    def rig_active_stamp(self) -> str:
        """``R`` / ``V`` / ``G`` / ``V+G`` — read off the applying object."""
        return self._spec.stamp()

    def rig_description(self) -> str:
        # Built from `self._spec`, never from the CLI value the factory was
        # handed: the activation line is the operator's proof that the dose
        # arrived, so it is announced by the consumer that will apply it.
        q_source = ("sf_regret" if self._external_q is None
                    else str(self._external_q.header.get("name", "external policy file")))
        return (f"{self._spec.describe(q_source)} | "
                f"{len(self._labels)} labeled positions")

    def rig_stats(self) -> dict[str, Any]:
        return {
            "arm": self._spec.arm,
            "stamp": self._spec.stamp(),
            "stages": [
                {
                    "stage": stage,
                    "edited_rows": int(self.stage_edited[stage]),
                    "fallback_rows": int(self.stage_fallback[stage]),
                }
                for stage in self._spec.stages()
            ],
            "params": {
                "r_weight": self._spec.r_weight,
                "v_lambda": self._spec.v_lambda,
                "v_tau_cp": self._spec.v_tau_cp,
                "v_veto_cp": self._spec.v_veto_cp,
                "g_alpha": self._spec.g_alpha,
                "g_temp_cp": self._spec.g_temp_cp,
            },
            "total_rows": int(self.total_rows),
            "eligible_rows": int(self.eligible_rows),
            "edited_rows": int(self.edited_rows),
            "fallback_rows": int(self.fallback_rows),
            "realized_f": (float(self.eligible_rows) / float(self.total_rows)
                           if self.total_rows else 0.0),
            # See the ctor: calibration check for arm G's (alpha, T) against the
            # ~0.970-nat BT4 anchor. NOT a success metric, NOT a gate. `None`
            # when this arm has no G stage.
            "g_target_entropy_nats": (
                None if not self.g_entropy_rows else {
                    "before": self.g_entropy_pre_sum / self.g_entropy_rows,
                    "after": self.g_entropy_post_sum / self.g_entropy_rows,
                    "rows": int(self.g_entropy_rows),
                    "note": "calibration only -- never a success metric or gate",
                }
            ),
            "q_source": (
                "sf_regret" if self._external_q is None else "external_policy_file"
            ),
            "external_q": (
                {
                    "attached": False,
                    "reason": "this arm has no G stage; the q applies to g/vg legs",
                }
                if self._external_q_detached else
                None if self._external_q is None else {
                    "attached": True,
                    # ⚑ The PATH AND ITS BYTES, not just the path. Recorded at
                    # load time inside the index (`file_provenance`), so the
                    # report names the file the sweep actually read rather than
                    # the argument it was given.
                    "file": self._external_q.header.get("file_provenance"),
                    "n_positions": len(self._external_q),
                    "missing_rows": int(self.external_q_missing_rows),
                    "zero_overlap_rows": int(self.external_q_zero_overlap_rows),
                    "header": self._external_q.header,
                }
            ),
            "labels": {
                "n_positions": len(self._labels),
                "nodes": self._labels.header.get("nodes"),
                "multipv": self._labels.header.get("multipv"),
                "syzygy_matches_production_label_path": self._labels.header.get(
                    "syzygy_matches_production_label_path",
                ),
                "ucinewgame_per_position": self._labels.header.get(
                    "ucinewgame_per_position",
                ),
            },
        }

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def __len__(self) -> int:
        return len(self._inner)

    def _row_keys(self, arrs: dict) -> list[bytes]:
        """Fingerprints for the batch, grouped by history encoding.

        ⚑ A pool mixing encodings would otherwise be keyed with ONE encoding's
        plane offsets, which does not raise — it reads a history plane as a
        castling bit and hands the row a plausible, wrong key.

        In practice ``DiskReplayBuffer`` refuses to build a CHUNK from shards of
        mixed encodings, so a mixed BATCH is not reachable through it today.
        This branch is therefore a guard against a future pool shape, not a
        live bug — and it is kept, and tested, because the failure it prevents
        is silent: the wrong key reads as "this row has no label", i.e. as
        lower coverage rather than as an error.
        """
        x = arrs["x"]
        encs = np.asarray(arrs["_input_history_encoding"]).reshape(-1)
        uniq = np.unique(encs)
        if uniq.size == 1:
            return position_fingerprints(x, input_history_encoding=str(uniq[0]))
        keys: list[bytes] = [b""] * int(x.shape[0])
        for enc in uniq:
            sel = np.flatnonzero(encs == enc)
            part = position_fingerprints(x[sel], input_history_encoding=str(enc))
            for slot, key in zip(sel.tolist(), part, strict=True):
                keys[slot] = key
        return keys

    def sample_batch_arrays(self, batch_size: int, **kw) -> dict:
        arrs = self._inner.sample_batch_arrays(batch_size, **kw)
        target = arrs.get("policy_target")
        if target is None:
            raise SystemExit(
                "rvg arm active but the sampled batch has no policy_target"
            )
        self.total_rows += int(target.shape[0])
        keys = self._row_keys(arrs)
        out = dict(arrs)
        spec = self._spec

        if spec.arm == "r":
            regret = arrs.get("sf_p0_regret")
            has_regret = arrs.get("has_sf_p0_regret")
            if regret is None or has_regret is None:
                raise SystemExit(
                    "rvg_arm=r but the sampled batch lacks sf_p0_regret/"
                    "has_sf_p0_regret — the arm would train the control"
                )
            new_regret = np.zeros_like(regret)
            new_has = np.zeros_like(has_regret)
            for i, key in enumerate(keys):
                label = self._labels.get(key)
                if label is None:
                    continue
                self.eligible_rows += 1
                self.edited_rows += 1
                idx, reg = label
                new_regret[i] = listed_only_regret_vector(
                    int(regret.shape[1]), idx, reg, cap_cp=SF_OWN_REGRET_CAP_CP,
                ).astype(new_regret.dtype, copy=False)
                new_has[i] = 1
            out["sf_p0_regret"] = new_regret
            out["has_sf_p0_regret"] = new_has
            return out

        edited = target.copy()
        stages = spec.stages()
        for i, key in enumerate(keys):
            label = self._labels.get(key)
            if label is None:
                continue
            self.eligible_rows += 1
            idx, reg = label
            # THE CHAIN. `current` starts as the stored target and is replaced by
            # each stage's output, so stage k+1 edits stage k's result — which is
            # what "V FIRST, then G on the veto-edited target" means, and the
            # only place the ORDER lives.
            current: np.ndarray = target[i]
            any_edit = False
            for stage in stages:
                if stage == "V":
                    current, fell_back = apply_veto_edit(
                        current, idx, reg, lam=spec.v_lambda,
                        tau_cp=spec.v_tau_cp, veto_cp=spec.v_veto_cp,
                    )
                else:
                    before = current
                    q_weights = None
                    if self._external_q is not None:
                        q_weights = self._external_q.weights_for(key, idx)
                        if q_weights is None:
                            self.external_q_missing_rows += 1
                            self.stage_fallback[stage] += 1
                            continue
                        # ⚑ PRESENT BUT USELESS IS A THIRD STATE, and it reads
                        # like the other two if it is not counted. The row IS in
                        # the q file, so it is not `missing`; every move the file
                        # names falls outside this row's listed set, so the
                        # aligned q is all zeros and the blend falls back — which
                        # is indistinguishable from a numerical fallback in the
                        # counters. It means something specific and actionable:
                        # the q file and the labels are joined but their MOVE
                        # spaces disagree (a frame or canonicalization mismatch),
                        # which is a different repair from "regenerate the keys".
                        if not float(np.sum(q_weights)) > 0.0:
                            self.external_q_zero_overlap_rows += 1
                    current, fell_back = apply_geometric_blend(
                        current, idx, reg,
                        alpha=spec.g_alpha, temp_cp=spec.g_temp_cp,
                        q_weights=q_weights,
                    )
                    if not fell_back:
                        self.g_entropy_pre_sum += distribution_entropy_nats(before)
                        self.g_entropy_post_sum += distribution_entropy_nats(current)
                        self.g_entropy_rows += 1
                if fell_back:
                    self.stage_fallback[stage] += 1
                else:
                    self.stage_edited[stage] += 1
                    any_edit = True
            if not any_edit:
                self.fallback_rows += 1
                continue
            self.edited_rows += 1
            edited[i] = current.astype(edited.dtype, copy=False)
        out["policy_target"] = edited
        return out


# ── rig-only override keys ───────────────────────────────────────────────────
# Keys consumed by THIS SCRIPT rather than by the Trainer or by TrialConfig.
# They reach training by WRAPPING the replay buffer the Trainer draws from, so
# they are invisible to every test `_override_key_reaches_the_trainer` performs
# — correctly, because none of those three questions is the one being asked.
#
# ⚑ THIS IS THE FIX FOR THE 2026-08-20 DOSE-LADDER REFUSAL, ported here from the
# ladder2 rig branch (52127f817, `feat/sfp0-dose-ladder`, never merged — PR #450
# landed only the LIVE `sf_p0_blend_alpha` train-side leg). The shape is worth
# restating because it is this repo's signature defect wearing a guard's
# uniform: `rig_policy_from_soft` HAD a working consumer, but the pop that
# removed it from the config ran TEN LINES AFTER the dead-knob guard had already
# read it off `overrides`, so any arm using it would have died on a guard
# insisting the knob reached nothing. The knob was wired; the guard just ran
# first.
#
# The mapping is deliberately NOT a hand-kept allowlist bolted onto the guard —
# the guard's own docstring rejects allowlists, and rightly. The VALUE under each
# key is the factory the activation site calls, so this table cannot list a key
# that has no consumer, and cannot omit a key that has one: registering a name
# and wiring it are the same edit.
_RIG_ONLY_WRAPPERS: dict[str, Callable[[Any, Any, _RigContext], Any]] = {
    "rig_policy_from_soft": lambda buf, _v, _ctx: _SoftPolicyAsMainBuffer(buf),
    "rvg_arm": lambda buf, v, ctx: _RvgTargetSurgeryBuffer(
        buf, ctx.resolve_rvg(str(v)), ctx.require_labels(),
        external_q=ctx.external_q,
    ),
}

# Rig-only keys that PARAMETERIZE an active wrapper without activating one.
# Split out of the Trainer-bound overrides for the same reason the activation
# keys are — they reach nothing through the config — but never treated as an
# arm, so `rvg_arm=v,rvg_v_lambda=0.02` is one arm and not two.
_RIG_ONLY_PARAMS: frozenset[str] = frozenset({
    "rvg_r_weight", "rvg_v_lambda", "rvg_v_tau", "rvg_v_veto_cp",
    "rvg_g_alpha", "rvg_g_temp",
})


@dataclass
class _RigContext:
    """Sweep-level state the rig factories need: label file + arm defaults."""

    defaults: RvgArmSpec
    labels: RvgLabelIndex | None
    labels_path: list[Path] | None
    params: dict[str, float]
    #: Arm B's alternative ``q``: an external per-row policy (an lc0/BT4 net's
    #: distribution) instead of the SF-regret-derived softmax. Default None =
    #: the SF source, which is arm G.
    external_q: RvgExternalPolicyIndex | None = None
    external_q_path: Path | None = None

    def resolve_rvg(self, arm: str) -> RvgArmSpec:
        spec = replace(
            self.defaults,
            arm=str(arm).strip().lower(),
            r_weight=self.params.get("rvg_r_weight", self.defaults.r_weight),
            v_lambda=self.params.get("rvg_v_lambda", self.defaults.v_lambda),
            v_tau_cp=self.params.get("rvg_v_tau", self.defaults.v_tau_cp),
            v_veto_cp=self.params.get("rvg_v_veto_cp", self.defaults.v_veto_cp),
            g_alpha=self.params.get("rvg_g_alpha", self.defaults.g_alpha),
            g_temp_cp=self.params.get("rvg_g_temp", self.defaults.g_temp_cp),
        )
        spec.validate()
        return spec

    @staticmethod
    def inert() -> _RigContext:
        """A context with no labels and every arm parameter at its off value.

        The default for callers that never pass ``--rvg-*`` (including every
        pre-existing test). An rvg arm activated against it still refuses, in
        ``require_labels`` — "no labels" must never degrade to "train the
        control quietly".
        """
        return _RigContext(
            defaults=RvgArmSpec(
                arm="", r_weight=0.0, v_lambda=0.0, v_tau_cp=0.0,
                v_veto_cp=float("inf"), g_alpha=0.0, g_temp_cp=100.0,
            ),
            labels=None, labels_path=None, params={},
            external_q=None, external_q_path=None,
        )

    def require_labels(self) -> RvgLabelIndex:
        if self.labels is None:
            raise SystemExit(
                "rvg_arm is set but --rvg-labels was not: every arm joins the "
                "shallow wide-MultiPV label file by position fingerprint, and "
                "without it the arm would train the control"
            )
        return self.labels


def _corpus_policy_encoding(replay_dir: Path) -> str:
    """The policy space THIS corpus stores, read off its own first shard.

    ⚑ ARMS A GATE THAT WAS OTHERWISE DEAD. ``RvgLabelIndex.load`` and
    ``RvgExternalPolicyIndex.load`` both refuse a label file whose policy space
    disagrees with the corpus — and both were called without the argument, so
    the refusal could only ever fire in the tests that passed it explicitly.
    A gate whose only caller declines to arm it is the defect this repo is named
    for; this is the value that arms it.

    Read lazily (headers only, no row data): the corpus is ~800 shards and this
    runs before the first arm.
    """
    paths = iter_shard_paths(replay_dir)
    if not paths:
        raise SystemExit(f"no replay shards under {replay_dir}")
    lazy, _ = load_shard_arrays(paths[0], lazy=True)
    return str(np.asarray(lazy["_policy_encoding"]).reshape(-1)[0])


def _split_rig_overrides(overrides: dict) -> tuple[dict, dict, dict]:
    """Partition a variant's overrides into ``(rig, rig-params, Trainer-bound)``.

    ⚑ CALL THIS BEFORE ``_assert_overrides_reach_the_trainer``. The ordering IS
    the fix: the guard must only ever be asked about keys that are supposed to
    reach the Trainer, because "reaches the Trainer" is the only question it can
    answer.

    ``overrides`` is never mutated. It is what lands in ``retarget_report.json``
    as this arm's applied settings, and a rig knob that quietly vanished from the
    report would trade a loud refusal for a silent one — strictly worse than the
    bug this replaces.
    """
    rig = {k: v for k, v in overrides.items() if k in _RIG_ONLY_WRAPPERS}
    params = {k: v for k, v in overrides.items() if k in _RIG_ONLY_PARAMS}
    trainer_bound = {
        k: v for k, v in overrides.items()
        if k not in _RIG_ONLY_WRAPPERS and k not in _RIG_ONLY_PARAMS
    }
    return rig, params, trainer_bound


def _apply_rig_wrappers(
    buf: Any, rig_overrides: dict, ctx: _RigContext, *, name: str,
    params: dict | None = None,
) -> tuple[Any, str | None]:
    """Wrap ``buf`` for whichever rig knob this arm activated.

    Returns ``(buffer_to_train_on, active_key_or_None)``.

    A FALSY value is the CONTROL leg and runs UNWRAPPED, bitwise — a control
    must not travel through a copy-and-mask path that merely happens to be the
    identity today.

    Two active wrappers is a refusal, not a precedence rule: they redefine
    overlapping batch fields, so the order they were applied in would silently
    decide what the arm trained on.
    """
    active = {k: v for k, v in rig_overrides.items() if v}
    params = dict(params or {})
    # ⚑ RIG PARAMETERS WITHOUT A WRAPPER ARE A SILENT CONTROL. `_split_rig_overrides`
    # keeps the `rvg_*` parameter names away from the dead-knob guard (they are
    # not supposed to reach the Trainer), which means a variant that tunes an arm
    # and forgets to ACTIVATE it —
    #     --variant "g030:rvg_g_alpha=0.30,rvg_g_temp=150"   (no rvg_arm=g)
    # — sails through: the overrides are recorded in the report as applied, the
    # trainer-bound set is empty, no wrapper is built, and the leg trains the
    # control while its row in the ladder table reads "alpha 0.30". Exactly the
    # shape the dead-knob guard exists for, reintroduced one level up by the
    # exemption that guard needed.
    if params and not active:
        raise SystemExit(
            f"variant {name!r} sets rig parameters {sorted(params)} but activates "
            "NO rig wrapper, so they reach nothing and this leg would train the "
            "control while the report records them as applied. Add the arm "
            "(e.g. rvg_arm=g), or drop the parameters."
        )
    if len(active) > 1:
        raise SystemExit(
            f"variant {name!r} activates {len(active)} rig wrappers at once: "
            f"{', '.join(sorted(active))}. Each rewrites the batch the Trainer "
            "sees and their order would silently decide the arm. Run them as "
            "separate arms."
        )
    if not active:
        return buf, None
    key, value = next(iter(active.items()))
    wrapped = _RIG_ONLY_WRAPPERS[key](buf, value, ctx)
    # ⚑ THE STAMP, NOT JUST THE KEY. `rvg_arm` alone cannot tell a VG run from a
    # G run in `retarget_report.json`, and a composition that reads as one of its
    # halves is how an operator ends up comparing the wrong two rows of a ladder
    # table. Taken off the WRAPPER, so it is the applying object that names the
    # arm.
    stamp = getattr(wrapped, "rig_active_stamp", None)
    active_key = f"{key}:{stamp()}" if stamp is not None else key
    print(f"[retarget-{name}] rig wrapper ACTIVE ({active_key}): "
          f"{wrapped.rig_description()}")
    return wrapped, active_key


def _assert_rig_wrapper_took_effect(train_buf: Any, *, name: str) -> None:
    """A dose RECORDED as applied must be PROVABLY applied.

    ⚑ The summary's rig block is written off the wrapper OBJECT, so it would
    report its parameters in perfect confidence even if the Trainer had drawn
    every batch from somewhere else and the wrapper had served nothing. The
    counters below are incremented inside ``sample_batch_arrays`` itself — they
    are the one reading that separates "wrapped" from "wrapped AND USED", and
    checking them is what stops this script from re-committing the 2026-07-06
    ``sf_gap_priority_signed`` failure it was written to prevent.

    Both legs are the same defect reached from opposite sides:

    * ``total_rows == 0`` — the Trainer never drew through the wrapper, so the
      arm trained on unedited batches under a report claiming an edit.
    * ``eligible_rows == 0`` — the wrapper served rows and not one was eligible,
      so the edit was the identity and this arm IS the control under another
      name.
    * ``edited_rows == 0`` — ⚑ THE THIRD DOOR, and eligibility does not close
      it. A row can be eligible (it HAS a label) and still be served bitwise
      unchanged, because every stage fell back. The realistic trigger is arm B
      with a q file whose keys do not match this corpus: every row looks up,
      every lookup misses, every G stage falls back, and the arm trains the
      control while ``realized_f`` reports full coverage. Reported as
      ``fallback_rows`` in the summary — but a number in a report nobody reads
      is not a gate, so it is one here.

    Duck-typed on ``rig_stats`` rather than switched on a class, so a wrapper
    added later is covered the day it is added. A wrapper whose stats carry no
    ``edited_rows`` key (the soft-shape arm) is exempt from that leg rather than
    failed by it — ``.get(..., None)`` distinguishes "did not edit" from
    "does not count edits".
    """
    stats = getattr(train_buf, "rig_stats", None)
    if stats is None:
        return
    s = stats()
    if int(s.get("total_rows", 0)) == 0:
        raise SystemExit(
            f"variant {name!r}: the rig wrapper served ZERO rows — the Trainer "
            "drew its batches from somewhere else, so this arm trained UNEDITED "
            "while retarget_report.json recorded the edit as applied."
        )
    if int(s.get("eligible_rows", 0)) == 0:
        raise SystemExit(
            f"variant {name!r}: the rig wrapper served {s['total_rows']} rows and "
            "NOT ONE was eligible, so the edit was the identity and this arm is "
            "the control under another name. Check that --rvg-labels covers this "
            "--replay-dir (realized f is reported per arm in retarget_report.json)."
        )
    edited = s.get("edited_rows")
    if edited is not None and int(edited) == 0:
        raise SystemExit(
            f"variant {name!r}: {s['eligible_rows']} rows were eligible and NOT "
            "ONE was edited — every stage fell back, so the Trainer saw the "
            "stored targets bit for bit and this arm is the control under "
            f"another name (fallback_rows={s.get('fallback_rows')})."
        )
    ext = s.get("external_q")
    if isinstance(ext, dict):
        missing = int(ext.get("missing_rows", 0) or 0)
        if missing and missing >= int(s.get("eligible_rows", 0)):
            raise SystemExit(
                f"variant {name!r}: the external q file supplied NO row the "
                f"corpus asked for ({missing} misses over "
                f"{s['eligible_rows']} eligible rows). The two files are in "
                "different key spaces — regenerate the q file against this "
                "--replay-dir rather than training a control labelled arm B."
            )


def _apply_rvg_config_side(
    config: dict, rig_overrides: dict, trainer_overrides: dict, ctx: _RigContext,
    *, name: str,
) -> None:
    """Arm R's CONFIG half: the loss weight, and the floor interaction guard.

    Arm R is the only one of the three that is not purely a batch edit — its
    listed-only regret vector is inert unless ``w_sf_own_regret`` is non-zero,
    so the rig sets it from ``--rvg-r-weight`` and records it in the summary.

    ⚑⚑ AND IT REFUSES TO RUN BESIDE THE SF POLICY FLOOR. ``sf_policy_floor_deficit``
    reads the SAME ``sf_p0_regret`` vector and defines its SF-approved set as
    ``regret <= sf_policy_floor_delta_cp / 1000`` (0.02 at the production 20cp).
    Production's vector fills unlisted AND illegal moves with ``>= 0.5``, which
    is what keeps them OUT of that set; arm R's repaired vector puts unlisted
    moves at 0.0, which puts every unlisted LEGAL move INTO it and floors each at
    ``sf_policy_floor_tau`` (0.15 live). The arm would then be measuring a
    corrupted floor, not a repaired regret term, and nothing in the report would
    say so. Production runs ``w_sf_policy_floor: 0.8``, so this fires by default
    and the operator must pass ``w_sf_policy_floor=0`` on the arm — AND on the
    baseline it is compared against, or the comparison moves two knobs.
    """
    arm = str(rig_overrides.get("rvg_arm", "") or "").strip().lower()
    if arm != "r":
        return
    spec = ctx.resolve_rvg(arm)
    floor = float(config.get("w_sf_policy_floor", 0.0) or 0.0)
    if floor != 0.0:
        raise SystemExit(
            f"variant {name!r}: rvg_arm=r needs w_sf_policy_floor=0 but this arm "
            f"resolves to {floor}. Arm R writes 0.0 regret on UNLISTED moves "
            "(the whole repair); sf_policy_floor_deficit reads the same vector "
            "and treats regret <= delta_cp/1000 as SF-APPROVED, so every unlisted "
            "legal move would be floored at sf_policy_floor_tau. Add "
            "'w_sf_policy_floor=0' to this arm AND to its baseline arm."
        )
    if "w_sf_own_regret" in trainer_overrides:
        # Two sources for one number: the rig would win and the variant string
        # would still read as applied in the report. One source only.
        raise SystemExit(
            f"variant {name!r} sets w_sf_own_regret directly AND activates "
            "rvg_arm=r, which sets it from --rvg-r-weight. Drop the explicit "
            "override — the report would otherwise record a value that was "
            "overwritten."
        )
    config["w_sf_own_regret"] = float(spec.r_weight)
    print(f"[retarget-{name}] rvg arm R config: w_sf_own_regret="
          f"{config['w_sf_own_regret']} (from --rvg-r-weight), "
          f"w_sf_policy_floor={floor}")


def build_rig_replay_buffer(
    *,
    config: dict,
    replay_dir: Path,
    target_planes: int,
    rng: np.random.Generator,
) -> DiskReplayBuffer:
    """THE rig's replay buffer — one definition, used by every consumer.

    ⚑ EXTRACTED SO NOTHING RE-DERIVES IT. ``scripts/rvg_label_pass.py
    enumerate-rows`` has to enumerate the EXACT rows a seeded sweep will draw,
    and a second copy of this constructor would drift from this one silently:
    every kwarg below (recency exponent, draw cap, wl ratio, refresh interval,
    ``deterministic_refresh``) changes WHICH rows come out, so an enumeration
    built from a re-typed ctor would name a different row set while looking
    identical. Sampling knobs resolve through ``TrialConfig.from_dict`` — the
    documented single source of truth for defaults — so this cannot drift from
    production when the YAML omits a key.
    """
    tc = TrialConfig.from_dict(config)
    # read_only: this script only samples (`train_steps` never adds, flushes or
    # clears), and it is routinely pointed at a directory that is either the
    # live window or a designated salvage pool. Until now the only thing
    # standing between an offline A/B and an evicted production window was the
    # `capacity=10**9` below being large enough -- safety by magic number. The
    # capacity stays as it is because it also sets the effective shuffle cap,
    # but it is no longer what keeps the shards alive (audit G12).
    buf = DiskReplayBuffer(
        10**9, shard_dir=replay_dir, rng=rng,
        read_only=True,
        input_planes=target_planes,
        upgrade_v1_planes=tc.replay_upgrade_v1_planes,
        shuffle_cap=tc.shuffle_buffer_size,
        shard_size=tc.shard_size,
        refresh_interval=tc.shuffle_refresh_interval,
        refresh_shards=tc.shuffle_refresh_shards,
        # Threaded for the same reason as the refresh knobs: this rig's whole
        # claim is that its buffer draws like production's, and a recency
        # exponent left at the constructor default while the config carried
        # another would make the arms differ from the run they stand in for.
        shard_recency_exponent=tc.replay_shard_recency_exponent,
        draw_cap_frac=tc.shuffle_draw_cap_frac,
        wl_max_ratio=tc.shuffle_wl_max_ratio,
        sf_gap_priority_weight=tc.replay_sf_gap_priority_weight,
        # Unconditional, and the reason the paired claim in this module's
        # docstring is true rather than merely intended. The default refresh
        # picks between an async and a synchronous shuffle-pool refresh by who
        # won a race, and only the synchronous one advances `self.rng`; a
        # single lost race therefore desynchronises this variant's entire draw
        # sequence from every other variant's, permanently, as a function of
        # machine load. "Each variant is seeded identically, so ... the
        # buffer's random draws match across arms" is the load-bearing
        # assumption of every delta this script reports, and without this it
        # was false. Measured 2026-07-31 on the sibling offline probe: 15
        # identical invocations gave 3 distinct draw sequences, 0.0056 nats of
        # held-out CE apart (docs/experiment_ledger.md, that date).
        #
        # No flag: an unpaired offline A/B is not a mode anyone wants, and a
        # switch here would only offer a way to invalidate the comparison
        # quietly. The cost is the refresh's shard reads moving onto the
        # sampling thread, ~+40% of sampling wall time on a loaded machine —
        # bought back many times over by not having to re-run a sweep whose
        # arms turn out not to have shared their rows.
        deterministic_refresh=True,
        # Offline-experiment knob (no TrialConfig field yet): read straight
        # from the flat config until it graduates to production.
        sf_gap_priority_signed=tc.replay_sf_gap_priority_signed,
        fast_low_surprise_priority=tc.replay_fast_low_surprise_priority,
        diff_focus_pol_scale=tc.diff_focus_pol_scale,
        diff_focus_q_weight=tc.diff_focus_q_weight,
    )
    return buf


def _run_variant(
    *,
    name: str,
    overrides: dict,
    base_config: dict,
    checkpoint: Path,
    replay_dir: Path,
    steps: int,
    batch_size: int,
    device: str,
    out_dir: Path,
    shard_snapshot: list[Path] | None,
    draw_snapshot: dict[str, float] | None = None,
    rig: _RigContext | None = None,
    rebuild_sf_targets: bool = True,
    gpu_mem_fraction: float = 0.0,
    allow_yaml_arch: bool = False,
    allow_partial_load: bool = False,
) -> dict:
  # ⚑ THE SPLIT RUNS FIRST, AND THAT ORDERING IS LOAD-BEARING. Rig-only keys
  # reach training by wrapping the buffer, not through the config, so asking the
  # dead-knob guard about them gets a truthful "no" to a question nobody asked —
  # which is precisely how the 2026-08-20 ladder was refused with its wrapper
  # already written and working. The guard now sees exactly the keys that are
  # supposed to reach the Trainer.
    rig_overrides, rig_params, trainer_overrides = _split_rig_overrides(overrides)
    ctx = replace(
        rig if rig is not None else _RigContext.inert(),
        params={k: float(v) for k, v in rig_params.items()},
    )
  # BEFORE torch.load and CUDA init: a dead override key must cost seconds, not
  # the hours a full arm takes to reveal that it trained the control twice.
    _assert_overrides_reach_the_trainer(
        name=name, overrides=trainer_overrides, base_config=base_config,
    )
    config = dict(base_config)
    config["rebuild_sf_targets"] = bool(rebuild_sf_targets)
    config.update(trainer_overrides)
  # Belt and braces: a rig key parked in the YAML would arrive through
  # `base_config` rather than through `overrides`, bypassing the split. It must
  # still never reach the Trainer's kwargs.
    for _rig_key in (*_RIG_ONLY_WRAPPERS, *_RIG_ONLY_PARAMS):
        config.pop(_rig_key, None)
  # Arm R's config half (loss weight + the sf_policy_floor interaction guard),
  # applied BEFORE the Trainer is built because that is the only point at which
  # `w_sf_own_regret` can still reach it.
    _apply_rvg_config_side(
        config, rig_overrides, trainer_overrides, ctx, name=name,
    )

    import torch

    if device.startswith("cuda") and gpu_mem_fraction:
        # Cap the SELECTED device so sidecar retrains can't OOM a live
        # trainer sharing the GPU (same convention as the yardstick scripts).
        tail = device.split(":", 1)[1] if ":" in device else ""
        if tail and not tail.isdigit():
            raise SystemExit(
                f"invalid CUDA device {device!r}: expected 'cuda' or 'cuda:<N>'"
            )
        idx = int(tail) if tail else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(float(gpu_mem_fraction), idx)

    # Identical seed per variant so dropout masks and the buffer's random
    # draws match across arms; sampling-knob overrides still (intentionally)
    # change WHICH rows those draws select.
    torch.manual_seed(int(config.get("seed", 0)))

    # ARCHITECTURE from the checkpoint's own ``arch`` key, NOT the YAML: the
    # config only carries training/target/sampling params, so building the
    # model from it silently drops every checkpoint tensor whose shape the
    # YAML disagrees on (observed: embed.weight + per-layer ffn when the YAML
    # arch drifted from the trained net) — a partially random-init model that
    # invalidates the A/B. The stored arch reconstructs the exact model, and
    # the load is strict: any dropped tensor aborts the run — unless
    # --allow-partial-load is set, for retargeting a historical checkpoint saved
    # under drifted model code (arch topology captures the layout, not the code
    # revision, so a since-added/removed param yields a missing/unexpected key
    # that every live resume path tolerates).
    # map_location stays "cpu": the Trainer moves the model to `device`, and
    # GPU-loading the whole checkpoint (optimizer state included) would spike
    # exactly the VRAM that --gpu-mem-fraction is trying to protect.
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    # Match the sibling loaders (load_model_from_checkpoint, reinit_value_heads,
    # shrink_ffn): a truthy-but-non-dict ``arch`` must NOT take the arch path,
    # or model_config_from_arch(arch).get(...) raises an opaque AttributeError.
    arch_used = isinstance(ckpt, dict) and isinstance(ckpt.get("arch"), dict)
    if arch_used:
        model_cfg = model_config_from_arch(ckpt["arch"])
    elif allow_yaml_arch:
        print(f"[retarget-{name}] WARNING: checkpoint has no embedded arch; "
              "building the model from the YAML config. If the YAML arch has "
              "drifted from the trained net, mismatched tensors keep their "
              "random init and the A/B is invalid.")
        model_cfg = model_config_from_flat_config(config)
    else:
        raise SystemExit(
            f"{checkpoint} has no embedded arch key, so the model would be "
            "built from the YAML config — the exact silent partial-load this "
            "script guards against. Re-save the checkpoint with a current "
            "Trainer, or pass --allow-yaml-arch if you have verified the "
            "YAML architecture matches the trained net."
        )
    model = build_model(model_cfg)

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_state_dict_tolerant(
        model, state, label=f"retarget-{name}",
        require_complete=arch_used and not allow_partial_load,
    )

    kwargs = trainer_kwargs_from_config(config)
    kwargs["device"] = device
    trainer = Trainer(model, model_config=model_cfg, **kwargs)

    rng = np.random.default_rng(int(config.get("seed", 0)))
    # Mirror the production sampling ctor (tune/trainable_init.py) — without
    # these the offline retrain samples with NO surprise weighting / draw cap
    # and any replay_* sampling override in a --variant is a silent no-op,
    # so A/Bs of sampling knobs (e.g. replay_sf_gap_priority_weight) would
    # measure nothing. Values resolve through TrialConfig.from_dict — the
    # documented single source of truth for defaults — so this cannot drift
    # from production when the YAML omits a key. Only `capacity` differs
    # (unbounded: the offline A/B uses the whole provided replay dir, not the
    # live run's growing window) and `input_planes`, which must match the
    # checkpoint arch rather than the YAML (the flat config has no top-level
    # input_extra_features key — reading it there yields None -> 146, which
    # rejects 175-plane v2_threats shards and empties the buffer).
    tc = TrialConfig.from_dict(config)
    target_planes = input_plane_count(model_cfg.input_extra_features)
    # Detect a stored<target plane mismatch that DiskReplayBuffer would silently
    # zero-pad (the len(buf)==0 guard below only catches the stored>target reject
    # case). Must run before the ctor so a v1-dir-vs-v2-checkpoint aborts loudly.
    _assert_replay_planes_match(
        replay_dir, target_planes, upgrade_v1=tc.replay_upgrade_v1_planes,
    )
    buf = build_rig_replay_buffer(
        config=config, replay_dir=replay_dir,
        target_planes=target_planes, rng=rng,
    )
    try:
        # `_snapshot_shards()` is the buffer's OWN post-scan list, so this is
        # what THIS arm will actually draw from — not an independent re-glob,
        # which would agree even if the buffer had filtered. Inside the try so
        # the buffer is closed even if the scan itself raises.
        shard_pool = buf._snapshot_shards()
        # Arm 1 (`shard_snapshot is None`) DEFINES the reference; it cannot be
        # checked against anything, and checking it against a glob taken before
        # `torch.load` + CUDA init only manufactures false aborts from shards
        # that landed during startup — a window in which no arm has drawn yet,
        # so no arm can be de-paired by it. The sentinel is `None`, never `[]`:
        # an empty list is a real reference, not "unset". (It cannot reach here
        # today — an empty arm-1 pool trips the `len(buf) == 0` exit below
        # before any summary is returned — but `is None` is what makes that a
        # property of the code rather than of the empty-pool guard's ordering.)
        if shard_snapshot is None:
            print(f"[retarget] {name}: shard reference = this arm's own scan, "
                  f"{len(shard_pool)} shards under {replay_dir}; "
                  "every later arm must scan exactly these")
        else:
            # Before the empty-pool check: if the reference was non-empty and
            # the directory has since been emptied, "de-paired" is the accurate
            # report and "wrong --replay-dir" is not.
            _assert_shards_unchanged(
                observed=shard_pool, snapshot=shard_snapshot, name=name,
            )
        if len(buf) == 0:
            raise SystemExit(
                f"replay buffer is empty: no shards under {replay_dir} match "
                f"input_planes={target_planes} "
                "(wrong --replay-dir, or plane-count mismatch with the "
                "checkpoint arch)"
            )
        train_buf, rig_active = _apply_rig_wrappers(
            buf, rig_overrides, ctx, name=name, params=rig_params,
        )
        t0 = time.time()
        metrics = trainer.train_steps(train_buf, batch_size=int(batch_size), steps=int(steps))
        duration = time.time() - t0
    finally:
        buf.close()

  # The knob is recorded as applied a few lines below; this is what earns the
  # right to record it. Runs after `buf.close()` so a refusal here cannot leak
  # the buffer's handles.
    _assert_rig_wrapper_took_effect(train_buf, name=name)

  # ⚑ DIRECT ATTRIBUTE READS, NOT `getattr(..., 0.0)`. Both fields are
  # non-optional on `TrainMetrics`, and the 0.0 defaults these used to carry made
  # `_assert_draws_unchanged` a gate that could not fail: delete or rename either
  # field upstream and BOTH arms would read 0.0, so the guard would compare
  # `0.0 == 0.0`, return silently, and certify a sweep whose pairing it had not
  # checked. An AttributeError here is the correct outcome -- it is loud, it is
  # at the top of the run rather than at the end, and a de-pairing guard that
  # cannot observe its own inputs is worth less than no guard at all.
    draws = {
        "batches_drawn": float(metrics.batches_drawn),
        "transient_cuda_retry_batches": float(metrics.transient_cuda_retry_batches),
    }
  # Arm 1 DEFINES the reference, exactly as it does for the shard pool -- but a
  # retry in arm 1 is still fatal, because the reference sequence would already
  # be the perturbed one and no later arm can disagree with it.
    _assert_draws_unchanged(
        observed=draws,
        snapshot=draw_snapshot if draw_snapshot is not None else dict(draws),
        name=name,
    )

    out_path = out_dir / f"{name}.pt"
    trainer.save(out_path)
    summary = {
        "variant": name,
        "overrides": overrides,
        "steps": int(steps),
        "duration_s": round(duration, 1),
        "checkpoint": str(out_path),
        # The pool this arm actually drew from. `main()` reads arm 1's copy back
        # as the reference for arms 2..N, and it lands in the report so a reader
        # can see WHICH shards the printed deltas were paired over — the shard
        # list is the other half of "same seed" and was previously unrecorded.
        "shard_pool": [str(p) for p in shard_pool],
  # The other half of "same seed, same rows": how many batches this arm pulled
  # from the buffer, and how many of those were post-CUDA-error replacements
  # that advanced its RNG. Recorded so a reader can SEE the arms were paired,
  # rather than inferring it from the shard list alone.
        "draws": draws,
  # Which rig-only override actually wrapped the buffer this arm trained on
  # (None = no wrapper, the control leg). `overrides` above records what was
  # ASKED FOR; this records what the rig DID with it, and the two being separate
  # fields is the point — a rig key that silently failed to activate now shows up
  # as a disagreement between them instead of being invisible.
        "rig_active": rig_active,
  # Ladder provenance: the REALIZED eligible-row coverage f, read off the
  # wrapper's OWN counters — never spliced from a live run's coverage figure,
  # which is a different population. `None` when no wrapper is active.
        "rig_stats": (
            train_buf.rig_stats() if hasattr(train_buf, "rig_stats") else None
        ),
  # Arm R changes a LOSS WEIGHT rather than a batch field, so the wrapper's
  # counters cannot prove that half.
  #
  # ⚑ READ OFF THE TRAINER'S OWN ATTRIBUTES, NOT THE CONFIG DICT. The config is
  # the PRODUCER's copy: it proves what was asked for, which is the half we
  # already know. `trainer.w_sf_own_regret` and `trainer.sf_policy_floor_params.w`
  # are the values `compute_loss` is actually called with, so a rename or a
  # dropped key between `from_dict` and the Trainer shows up as a disagreement
  # instead of being invisible. Same rule the trainer's own floor announcement
  # follows ([[announce_from_the_consumers_own_parameter]]).
        "rvg_config_side": {
            "w_sf_own_regret": float(trainer.w_sf_own_regret),
            "w_sf_policy_floor": float(trainer.sf_policy_floor_params.w),
            "asked_for": {
                "w_sf_own_regret": float(config.get("w_sf_own_regret", 0.0) or 0.0),
                "w_sf_policy_floor": float(config.get("w_sf_policy_floor", 0.0) or 0.0),
            },
        },
        "rvg_labels": (
            ctx.labels.header.get("label_file_provenance")
            if ctx.labels is not None else None
        ),
        "rvg_external_q": (
            ctx.external_q.header.get("file_provenance")
            if ctx.external_q is not None else None
        ),
        "final_metrics": {
            k: float(v)
            for k, v in vars(metrics).items()
            if isinstance(v, (int, float))
        },
    }
    print(f"[retarget] {name}: trained {steps} steps in {duration:.0f}s -> {out_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--replay-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="default: train.batch_size from the config")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/retarget"))
    ap.add_argument("--variant", action="append", required=True,
                    help="name:k=v,k=v (':' with empty body = config defaults)")
    ap.add_argument("--rebuild-sf-targets", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="--no-rebuild-sf-targets trains on the shards' stored "
                         "targets exactly as live training does — required when "
                         "the A/B is a sampling knob, not a target param")
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.0,
                    help="cap this process's share of the selected CUDA device "
                         "(0 = uncapped); set when running beside a live trainer")
    ap.add_argument("--allow-yaml-arch", action="store_true",
                    help="permit building the model from the YAML config when "
                         "the checkpoint has no embedded arch key (UNSAFE: a "
                         "drifted YAML arch silently random-inits mismatched "
                         "tensors and invalidates the A/B)")
  # ── target-surgery ladder (arms R / V / G), all default OFF ───────────────
  # These set the SWEEP-LEVEL defaults. An arm is activated per-variant with
  # `rvg_arm=r|v|g`, and any parameter can be overridden per-variant with the
  # matching `rvg_*` key (e.g. `--variant "g030:rvg_arm=g,rvg_g_alpha=0.30"`), so
  # one invocation runs the whole ladder over ONE identically-seeded draw
  # sequence — the pairing property dose-ladder2 depends on.
  # `append`: REPEATABLE, single-file behaviour unchanged. The overlay machinery
  # (`RvgLabelIndex.load` over several files, weakest pass first so the strongest
  # per-move reading wins) was built and tested but had NO WAY TO BE CALLED —
  # a `type=Path` scalar. Tested-but-unreachable is the same defect class as
  # accepted-but-ignored, one layer up, and the composite-target design needs a
  # deeper-narrow overlay later, so this arms it rather than deleting it.
  # ⚑ v1 of the ladder uses ONE pass; a multi-file invocation is not part of the
  # banked command list.
    ap.add_argument("--rvg-labels", type=Path, action="append", default=None,
                    metavar="PATH",
                    help="shallow wide-MultiPV label JSONL from "
                         "scripts/rvg_label_pass.py; required by any rvg arm. "
                         "REPEATABLE: several passes are overlaid per move, "
                         "deeper-per-line wins (v1 uses one file)")
    ap.add_argument("--rvg-r-weight", type=float, default=0.0,
                    help="arm R: w_sf_own_regret applied to the listed-only "
                         "regret vector (arm R also REQUIRES w_sf_policy_floor=0)")
    ap.add_argument("--rvg-v-lambda", type=float, default=0.0,
                    help="arm V: exponential downweight rate per cp above tau")
    ap.add_argument("--rvg-v-tau", type=float, default=0.0,
                    help="arm V: cp regret below which a listed move is untouched")
    ap.add_argument("--rvg-v-veto-cp", type=float, default=float("inf"),
                    help="arm V: listed moves with regret >= this are zeroed")
    ap.add_argument("--rvg-g-alpha", type=float, default=0.0,
                    help="arm G: geometric blend weight on the SF distribution "
                         "(band 0 < alpha < 1)")
    ap.add_argument("--rvg-g-temp", type=float, default=100.0,
                    help="arm G: cp temperature sharpening q ~ exp(-regret/T)")
  # `rvg_arm=vg` needs no flag of its own: it is the COMPOSITION of the V and G
  # parameters above, applied V FIRST. Adding a separate parameter set for the
  # composed arm would let VG's V leg drift from the standalone V arm it is
  # supposed to be comparable with.
    ap.add_argument("--rvg-g-q-source", default="sf",
                    help="arm G/VG: 'sf' (default, q ~ exp(-regret/T)) or "
                         "'file:<path.jsonl>' for an EXTERNAL per-row policy "
                         "({key, policy:{uci: prob}}, same key space as the "
                         "label pass) -- arm B, the blend toward an lc0/BT4 net")
    ap.add_argument("--allow-partial-load", action="store_true",
                    help="tolerate missing/unexpected state-dict keys (logged) "
                         "instead of aborting, for retargeting a historical "
                         "checkpoint saved under drifted model code; the "
                         "mismatched tensors keep fresh init")
    args = ap.parse_args()

    base_config = flatten_run_config_defaults(load_yaml_file(args.config))
    batch_size = int(args.batch_size or base_config.get("batch_size", 256))
    args.out_dir.mkdir(parents=True, exist_ok=True)

  # Loaded ONCE for the whole sweep, not once per arm: the banked corpus is
  # ~1.5M positions and re-parsing the JSONL per arm would add minutes of wall
  # to every leg for an index that is identical across them by construction.
    labels: RvgLabelIndex | None = None
    # The corpus's OWN policy space, passed to BOTH index loaders so their
    # encoding refusals are armed on the production path rather than only in the
    # tests that hand them the argument.
    #
    # ⚑ LAZY, and that is not a micro-optimization. Reading it eagerly made a
    # sweep with NO rvg arguments open the shard directory here and die on an
    # empty one, several guards earlier than the buffer that is supposed to
    # report that — i.e. a change made for the rvg path broke the non-rvg path.
    # It is read once, on first need, and only when there is a file to check.
    corpus_encoding: str | None = None
    if args.rvg_labels is not None:
        corpus_encoding = _corpus_policy_encoding(args.replay_dir)
        t_lab = time.time()
        labels = RvgLabelIndex.load(
            list(args.rvg_labels), policy_encoding=corpus_encoding,
        )
        print(f"[retarget] rvg labels: {len(labels)} positions from "
              f"{args.rvg_labels} in {time.time() - t_lab:.1f}s "
              f"(nodes={labels.header.get('nodes')}, "
              f"multipv={labels.header.get('multipv')}, "
              f"fresh_tt={labels.header.get('ucinewgame_per_position')})")
    external_q: RvgExternalPolicyIndex | None = None
    external_q_path: Path | None = None
    q_source = str(args.rvg_g_q_source or "sf").strip()
    if q_source not in ("", "sf"):
        if not q_source.startswith("file:"):
            raise SystemExit(
                f"--rvg-g-q-source {q_source!r}: expected 'sf' or 'file:<path>'"
            )
        external_q_path = Path(q_source[len("file:"):])
        if corpus_encoding is None:
            corpus_encoding = _corpus_policy_encoding(args.replay_dir)
        external_q = RvgExternalPolicyIndex.load(
            external_q_path, policy_encoding=corpus_encoding,
        )
        print(f"[retarget] rvg external q: {len(external_q)} positions from "
              f"{external_q_path}")
    rig = _RigContext(
        defaults=RvgArmSpec(
            arm="", r_weight=float(args.rvg_r_weight),
            v_lambda=float(args.rvg_v_lambda), v_tau_cp=float(args.rvg_v_tau),
            v_veto_cp=float(args.rvg_v_veto_cp), g_alpha=float(args.rvg_g_alpha),
            g_temp_cp=float(args.rvg_g_temp),
        ),
        labels=labels,
        labels_path=(list(args.rvg_labels) if args.rvg_labels else None),
        params={},
        external_q=external_q,
        external_q_path=external_q_path,
    )
    # ⚑ THE ONE q REFUSAL THAT IS CORRECTLY SWEEP-LEVEL. A leg without a G stage
    # detaches the q and says so (see `_RvgTargetSurgeryBuffer.__init__`), which
    # keeps a mixed ladder legal. But a q file that reaches NO leg in the whole
    # sweep is not a mixed ladder — it is an operator who believes arm B is
    # running. That is a stop, and it can only be decided once every variant is
    # known, which is here.
    if external_q is not None:
        reaches = [
            name for name, ov in (_parse_variant(v) for v in args.variant)
            if "rvg_arm" in ov and "G" in rig.resolve_rvg(str(ov["rvg_arm"])).stages()
        ]
        if not reaches:
            raise SystemExit(
                f"--rvg-g-q-source names {external_q_path} but NO variant in this "
                "sweep has a G stage, so the external policy reaches nothing and "
                "every leg would train on the SF-derived q (or on nothing at all) "
                "while the run was labelled arm B. Add an rvg_arm=g or rvg_arm=vg "
                "variant, or drop --rvg-g-q-source."
            )

    # ONE reference for the whole sweep, and it is arm 1's OWN buffer scan —
    # deliberately not a glob taken here. Each arm re-scans the directory inside
    # its own DiskReplayBuffer ctor, so a per-arm reference would just re-measure
    # the drift it is meant to catch; but a reference read HERE sits tens of
    # seconds (torch.load + CUDA init) before arm 1's scan, and a shard landing
    # in that window aborts a sweep whose arms would all have been paired.
    shard_snapshot: list[Path] | None = None
    draw_snapshot: dict[str, float] | None = None

    report = args.out_dir / "retarget_report.json"
  # ⚑ REVIEW FINDING (codex, P3). The de-pairing guards raise INSIDE
  # `_run_variant`, and the report used to be written only after every arm
  # returned — so the one path that most needs its counts banked (`the sweep
  # aborted, here is what the arms drew`) wrote nothing, and in a REUSED
  # --out-dir the previous run's report survived the abort and read as a clean
  # sweep of the run that had just failed. Two halves: delete the stale file
  # before the first arm, so a missing report can never be mistaken for an old
  # one; and write what we have on the way out whatever happens. `bank the dump,
  # not just the number`.
    report.unlink(missing_ok=True)

    summaries: list[dict] = []
    aborted: str | None = None
    try:
        for spec in args.variant:
            name, overrides = _parse_variant(spec)
            summary = _run_variant(
                name=name, overrides=overrides, base_config=base_config,
                checkpoint=args.checkpoint, replay_dir=args.replay_dir,
                steps=args.steps, batch_size=batch_size, device=args.device,
                out_dir=args.out_dir, shard_snapshot=shard_snapshot,
                draw_snapshot=draw_snapshot, rig=rig,
                rebuild_sf_targets=args.rebuild_sf_targets,
                gpu_mem_fraction=args.gpu_mem_fraction,
                allow_yaml_arch=args.allow_yaml_arch,
                allow_partial_load=args.allow_partial_load,
            )
            if shard_snapshot is None:
                # Subscript, not `.get(..., [])`: a missing field must be a loud
                # KeyError here, not an empty reference that turns every later arm
                # into a false `(0 shards at start, N now)` abort.
                shard_snapshot = [Path(p) for p in summary["shard_pool"]]
            if draw_snapshot is None:
                draw_snapshot = dict(summary["draws"])
            summaries.append(summary)
    except BaseException as exc:
  # A guard's SystemExit carries the message; anything else is a crash. Either
  # way the completed arms' draw/shard provenance is the evidence for WHY the
  # sweep is void, so it goes to disk before the process leaves.
        aborted = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload: object = summaries
        if aborted is not None:
            payload = {"aborted": aborted, "completed_variants": summaries}
        report.write_text(json.dumps(payload, indent=2))
        print(
            f"[retarget] {'PARTIAL ' if aborted else ''}report written to {report} "
            f"({len(summaries)} of {len(args.variant)} variant(s))"
        )


if __name__ == "__main__":
    main()
