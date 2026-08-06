"""Per-iteration era-forgetting probes: a frozen old-era row set, scored live.

The run that ended 2026-07-31 lost 48.6 Elo [-68.2, -29.4] over three weeks
while every live signal stayed flat, and the flatness is why nobody read the
slide. The shape that WAS there, found offline afterwards, is the one this
module makes visible per iteration:

  * a fixed set of OLD-era rows gets steadily worse once its content leaves
    the effective replay window (the forgetting hinge — decay onset tracks
    each set's window-exit iteration, proven across five sets);
  * rows still INSIDE the window keep improving over the same span.

So the treadmill's fingerprint is the DIVERGENCE of the pair, not either
number alone. One number moving means the net moved; the pair separating means
in-window gains are being paid for out of old-era competence. Both halves run
the same code on the same rulers here — ``probe_era_*`` against a frozen
old-era set and ``probe_inwindow_*`` against a set re-cut from the newest
shards — precisely so the difference is attributable to the DATA and not to
two instruments.

**The two rulers.**

``policy_eregret`` is EXPECTED regret under the net's own prior:
``sum_m p_own(m) * regret(m)``, where ``regret(m)`` is the stored per-move
normalized SF cp-loss at THIS position (``sf_p0_regret``, 0 for SF's best
move). It is the same quantity ``train/losses.py`` minimises as
``sf_own_regret`` under ``w_sf_own_regret``, computed on the same legal-masked
``policy_own`` head, so the probe reads the axis training pushes on rather
than a cousin of it.

  ⚑ EXPECTED, never top-1. A standing method rule since 2026-08-02: the argmax
  moves on only ~19% of positions, so a top-1 contrast carries a +/-5.4cp
  median CI against effects of 3-5cp — it read a real, significant history
  benefit as "absent" for a whole session. A ruler whose CI is wider than the
  effect it is aimed at is not a ruler.

  ⚑ ``sf_p0_regret`` and NOT the row's own ``sf_multipv_raw``. SF label queries
  run at P1, AFTER the net's move is pushed, so a row's own MultiPV block
  describes the NEXT position in the OPPONENT's perspective. ``sf_p0_regret``
  is the one-ply-shifted field — SF's read of THIS position, in this row's
  perspective — and is the only stored per-move cp signal that pairs with
  ``policy_own`` at all.

``value_err`` is EXPECTED error under the net's own WDL distribution:
``sum_c p(c) * |score(c) - score(target)|`` with ``score = (1.0, 0.5, 0.0)``
for ``(win, draw, loss)`` and ``target`` the stored ``wdl_target``. Linear in
``p`` exactly as expected regret is linear in the policy, which is what makes
it the value-side twin rather than merely a value metric: no argmax, no
threshold, and no cancellation between over- and under-confident rows (an
``|E[score] - target|`` form would let a hedged prediction score as well as a
right one). Bounded in [0, 1].

**Cost.** One forward pass per set per scored iteration, no backward, under
``inference_mode``, over a row set capped by config. At the production shape
(2048 rows, batch 512) that is 4 forwards per set — against ~88 training steps
per iteration, each of which is a forward AND a backward AND an optimizer step
— so both probes together are well under 1% of an iteration. ``probe_ms`` is
published so the claim is an observation and not this docstring.

**These rows never train — and that is NOT the same as held out.** Two separate
statements, and conflating them will misread the whole pre-hinge segment:

  * *The probe injects nothing.* A ``ProbeSet`` holds its arrays privately and
    the only thing this module does with them is a forward pass; no path here
    appends to a replay buffer, and ``tests/test_era_forgetting_probe.py``
    proves it by marking probe rows and drawing from the sampler until the
    marker would have had to appear (with the positive control that the same
    detector fires when the rows ARE added).
  * *The era set is NOT held out.* It is cut from shards that are, at build
    time, still IN the replay window — ``--oldest N`` over a live shard dir
    returns the oldest still-RETAINED shards, not shards that have aged out.
    So until those shards leave the window, ``probe_era_*`` is reading rows the
    trainer is drawing from: it is an IN-WINDOW MEMORISATION reading, and it is
    expected to IMPROVE. That is not a fault — it is the mechanism. The hinge
    is the turn, at the iteration the era shards exit, and the pre-exit segment
    is the baseline the turn is measured against. Reading the early era curve
    as "old-era competence is fine" is the misread this paragraph exists to
    prevent.

⚑ ``probe_gap_*`` carries an unmeasured POSITION-DIFFICULTY offset: the two legs
are different row sets, so their level difference is not zero even for a net
that has forgotten nothing. Only the gap's TREND is interpretable. This is
stronger than "read the pair, not the level" — it says the pair's level is not
a quantity at all.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from chess_anti_engine.replay.dataset import collate_arrays
from chess_anti_engine.replay.shard import load_shard_arrays
from chess_anti_engine.train.losses import (
    align_action_values,
    apply_policy_mask_to_logits,
    masked_sum_and_count,
)

# The two probe labels. Metric names are built from them, so they are the
# single place the column prefix is decided.
PROBE_ERA = "era"
PROBE_INWINDOW = "inwindow"
PROBE_LABELS: tuple[str, ...] = (PROBE_ERA, PROBE_INWINDOW)

# Expected score of each WDL class, in the row's side-to-move POV. Index order
# is the stored `wdl_target` encoding: 0=win, 1=draw, 2=loss.
WDL_CLASS_SCORES: tuple[float, float, float] = (1.0, 0.5, 0.0)

# Exactly the shard fields a probe set carries. `x`, `policy_target` and
# `wdl_target` are what `collate_arrays` requires of any batch; the legal mask
# puts the policy head in the same masked space training uses; `sf_p0_regret`
# is the policy ruler. Anything else would be weight in the file that no
# reader reads -- and a frozen set should say what it is for.
PROBE_SET_FIELDS: tuple[str, ...] = (
    "x",
    "policy_target",
    "wdl_target",
    "legal_mask",
    "has_legal_mask",
    "sf_p0_regret",
    "has_sf_p0_regret",
)

# Fields whose absence makes the probe set unusable rather than degraded.
#
# The legal mask is in here rather than treated as optional because an unmasked
# softmax spreads probability onto illegal indices, and those indices are NOT
# zero-regret. ``_build_sf_p0_regret_vector``
# (chess_anti_engine/selfplay/finalize.py) pre-fills EVERY uncovered index with
# ``default_regret = (worst_regret + 1) / 2``, which is >= 0.5 by construction,
# and only then overwrites the moves SF actually listed. Measured over 2578 rows
# of ``data/c17_ab/pre`` carrying both flags:
#
#   illegal indices: mean 0.8302, min 0.5000, max 1.0000, frac == 0 -> 0.0000
#   legal indices:   mean 0.3272, min 0.0000, frac == 0 -> 0.0594
#
# So a maskless set reads HIGHER than the net earns — a PESSIMISTIC bias, and a
# large one. (An earlier revision of this comment asserted the opposite: that
# illegal regret is 0 and the bias is optimistic. That was wrong in both the
# premise and the sign; caught in review on PR #315 by measuring the shards
# instead of reading the comment.) The conclusion is unchanged and does not
# depend on the direction — a ruler whose level is set by which illegal moves a
# net happens to like is not a ruler — but the wrong premise is exactly what
# gets confirmed rather than caught next time someone reasons from it.
PROBE_SET_REQUIRED_FIELDS: tuple[str, ...] = (
    "x", "policy_target", "wdl_target", "legal_mask", "has_legal_mask",
    "sf_p0_regret", "has_sf_p0_regret",
)

# Suffix of the sidecar the builder writes next to a probe set.
PROVENANCE_SUFFIX = ".provenance.json"


def provenance_path(set_path: str | Path) -> Path:
    """Sidecar path for a probe set at *set_path*.

    A sidecar rather than a member inside the npz: ``ShardMeta`` is a closed
    dataclass that would refuse the block, and the shard list, reject log and
    gate thresholds are what an operator reads with ``cat`` before trusting a
    curve. The two are bound by the DIGEST, which the sidecar records and
    :func:`load_probe_set` rechecks against the rows it loaded — so a sidecar
    that has drifted from its set says so rather than describing another file.
    """
    p = Path(set_path)
    return p.with_name(p.name + PROVENANCE_SUFFIX)


def probe_set_digest(arrs: dict[str, np.ndarray]) -> str:
    """Content digest of a probe set: the 16-hex identity of the frozen rows.

    Over the FIELDS THE PROBE READS, in a fixed order, including each array's
    shape and dtype. Two sets with the same digest score identically on the
    same weights, which is the property the operator needs: a probe number is
    only comparable across iterations if the ruler behind it is the same
    object, and the digest is how a re-cut set announces that it is not.

    Deliberately not a hash of the file: an npz carries compression state and
    a member order that can move without a single row changing, so a file hash
    would report a new ruler where there is none.
    """
    h = hashlib.sha256()
    for key in PROBE_SET_FIELDS:
        arr = arrs.get(key)
        if arr is None:
            h.update(f"{key}:absent".encode())
            continue
        a = np.ascontiguousarray(arr)
        h.update(f"{key}:{a.shape}:{a.dtype.str}".encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class ProbeSet:
    """A frozen row set plus the identity needed to compare its readings.

    ``arrays`` is private in the sense that matters: nothing in this module
    hands it to a replay buffer, and the only consumer is
    :func:`score_probe_set`.
    """

    label: str
    path: str
    arrays: dict[str, np.ndarray]
    n_rows: int
    n_policy_rows: int
    digest: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeReading:
    """One probe's scoring of one set of weights."""

    n_rows: int = 0
    n_policy_rows: int = 0
    policy_eregret: float = float("nan")
    value_err: float = float("nan")
    seconds: float = 0.0


def _row_count(arrs: dict[str, np.ndarray]) -> int:
    x = arrs.get("x")
    return 0 if x is None else int(np.asarray(x).shape[0])


def _load_provenance(
    set_path: Path, *, label: str, digest: str, truncated: bool,
) -> dict[str, Any]:
    """Read and CHECK the sidecar. Returns {} (loudly) when it cannot vouch.

    Three states, all printed, because "no provenance" and "provenance about
    another file" must not look like "screened and sound":

      * absent — the set was not cut by ``scripts/build_era_probe_set.py``, so
        nothing screened it for SF desync. The frozen holdout is the standing
        example of what that costs: it reads 0.101305 no-MultiPV and, being
        frozen, never ages out;
      * digest mismatch — the sidecar describes a DIFFERENT set of rows;
      * ok.

    A digest mismatch is EXPECTED and benign when the row cap truncated the
    set, since the sidecar's digest is over the whole file. That case is
    reported as a truncation rather than as drift, and the provenance is still
    honoured: the desync screen applied to every row of the file, including
    the prefix that survived.
    """
    sidecar = provenance_path(set_path)
    if not sidecar.exists():
        print(
            f"[probe] {label}: WARNING no {sidecar.name} beside {set_path.name}; "
            f"this set was NOT cut by scripts/build_era_probe_set.py and "
            f"nothing has screened it for SF desync. A poisoned probe set is a "
            f"forgetting curve about detached labels.",
            flush=True,
        )
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[probe] {label}: {sidecar} is unreadable ({exc}); provenance unknown", flush=True)
        return {}
    if not isinstance(data, dict):
        print(f"[probe] {label}: {sidecar} is not a provenance object; ignoring", flush=True)
        return {}
    recorded = str(data.get("digest", ""))
    if recorded and recorded != digest and not truncated:
        print(
            f"[probe] {label}: WARNING {sidecar.name} records digest "
            f"{recorded} but the loaded rows digest {digest}. The sidecar "
            f"describes a DIFFERENT set; treat its desync screening and shard "
            f"list as unverified.",
            flush=True,
        )
        return {}
    if recorded and truncated:
        print(
            f"[probe] {label}: the row cap truncated the set, so the loaded "
            f"digest {digest} differs from the file's {recorded} by "
            f"construction. Provenance still applies (the screen ran over every "
            f"row of the file), but the RULER is the truncated prefix — pin the "
            f"cap for the life of the column.",
            flush=True,
        )
    return data


def load_probe_set(
    path: str | Path | None,
    *,
    label: str,
    max_rows: int,
    expected_planes: int,
    expected_policy_size: int,
    expects_relations: bool = False,
) -> ProbeSet | None:
    """Load a frozen probe set, or return None with a printed reason.

    Every failure is non-fatal and lands on "no probe": a ruler that cannot be
    read must not take the run down. But every failure PRINTS, because the
    thing this whole module exists to prevent is a signal that is silently not
    there — the 2026-07 slide went unread for three weeks behind flat columns.

    The shape checks are the ones ``holdout_state.load_holdout_rows`` makes
    and for the same reason: a plane-count change (146 -> 175) or a policy-width
    change (4672 -> 1858) between the set's cut and now would otherwise be
    discovered mid-iteration inside a forward pass.

    ``print``, not ``logging``: this runs inside a Ray trial actor, which
    installs no logging handler, so an INFO line's presence and its absence are
    indistinguishable on the production path. That defect blocked PR #310.
    """
    raw = "" if path is None else str(path).strip()
    if not raw:
        return None
    if bool(expects_relations):
        # A probe set carries no `relations` matrices (5x64x64 uint8 per row
        # would double the file for a default-off feature). With
        # use_dynamic_relations on, a relation-less forward is a zero bias —
        # a DIFFERENT input than production serves, not a degraded one — so
        # refuse rather than publish a number about another model.
        print(
            f"[probe] {label}: use_dynamic_relations is on, but a probe set "
            f"carries no relation matrices; probe disabled (re-cut the set "
            f"with the relation fields before enabling it under this model)",
            flush=True,
        )
        return None
    p = Path(raw).expanduser()
    if not p.exists():
        print(
            f"[probe] {label}: no probe set at {p} — the {label} probe columns "
            f"will read nan for this run",
            flush=True,
        )
        return None
    try:
        arrs, _meta = load_shard_arrays(p)
    except Exception as exc:
        print(f"[probe] {label}: {p} could not be read ({exc}); probe disabled", flush=True)
        return None

    missing = [k for k in PROBE_SET_REQUIRED_FIELDS if k not in arrs]
    if missing:
        print(
            f"[probe] {label}: {p} is missing required fields {missing}; probe "
            f"disabled (rebuild it with scripts/build_era_probe_set.py)",
            flush=True,
        )
        return None

    x = np.asarray(arrs["x"])
    policy = np.asarray(arrs["policy_target"])
    if x.ndim != 4 or policy.ndim != 2:
        print(
            f"[probe] {label}: {p} has unexpected shapes (x={x.shape} "
            f"policy_target={policy.shape}); probe disabled",
            flush=True,
        )
        return None
    planes, width = int(x.shape[1]), int(policy.shape[1])
    if planes != int(expected_planes) or width != int(expected_policy_size):
        print(
            f"[probe] {label}: {p} was cut at {planes} input planes / policy "
            f"width {width}, but this run uses {int(expected_planes)} / "
            f"{int(expected_policy_size)}; probe disabled (a set cut against a "
            f"different encoding is a different ruler, not a degraded one)",
            flush=True,
        )
        return None

    n_all = int(x.shape[0])
    # A HEAD slice, never a random draw. The set is frozen and the cap only
    # decides how much of it is affordable; taking a random subset would make
    # the ruler depend on the process seed, so two restarts of one run would
    # score different sets under one column name.
    keep = n_all if int(max_rows) <= 0 else min(n_all, int(max_rows))
    rows = {
        k: np.ascontiguousarray(np.asarray(arrs[k])[:keep])
        for k in PROBE_SET_FIELDS
        if k in arrs
    }
    n_rows = _row_count(rows)
    # The digest identifies the FILE's rows, so it is taken BEFORE the
    # eligibility intersection below — otherwise a set whose rows are all fine
    # would still have to match a digest computed from a modified copy.
    digest = probe_set_digest(rows)
    provenance = _load_provenance(p, label=label, digest=digest, truncated=keep < n_all)

    # ⚑ PER-ROW, not just per-field. Requiring the `legal_mask` FIELD is not
    # enough: `apply_policy_mask_to_logits` multiplies the mask by the row's
    # `has_legal_mask`, so a row with that flag clear is scored with a fully
    # UNMASKED softmax and nothing downstream can tell. That is not a small
    # error — uncovered indices are pre-filled with `(worst_regret + 1)/2 >= 0.5`
    # (selfplay/finalize.py::_build_sf_p0_regret_vector), so the illegal mass
    # such a row spreads onto measures 0.83 mean against 0.33 on its legal moves
    # (2578 rows, data/c17_ab/pre). One maskless row lands ~0.5 of expected
    # regret into the mean.
    #
    # Intersected rather than refused: the VALUE ruler needs no legal mask, so
    # such a row is perfectly good for `value_err` and merely ineligible for the
    # policy one. Dropping it from the policy denominator alone loses nothing
    # and makes `probe_*_policy_n` mean exactly what its name says. A
    # builder-cut set cannot reach this path — `_eligible_rows` already requires
    # both flags — so the intersection is a no-op there and the digest above
    # still matches the builder's.
    has_reg = np.asarray(rows["has_sf_p0_regret"]).astype(bool)
    has_mask = np.asarray(rows["has_legal_mask"]).astype(bool)
    n_maskless = int(np.count_nonzero(has_reg & ~has_mask))
    if n_maskless:
        rows["has_sf_p0_regret"] = (has_reg & has_mask).astype(
            np.asarray(rows["has_sf_p0_regret"]).dtype,
        )
        print(
            f"[probe] {label}: WARNING {n_maskless} of {n_rows} rows carry "
            f"sf_p0_regret with has_legal_mask CLEAR; they are excluded from "
            f"probe_{label}_policy_eregret (an unmasked softmax spreads onto "
            f"illegal indices whose stored regret is ~0.83, not 0, so scoring "
            f"them would bias the mean UPWARD). Rebuild with "
            f"scripts/build_era_probe_set.py, which filters on both flags.",
            flush=True,
        )
    n_policy = int(np.count_nonzero(np.asarray(rows["has_sf_p0_regret"]).astype(bool)))

    # PROOF OF EFFECT for the construction-only config keys behind this: the
    # digest and the row count are read off the LOADED ARRAYS, not off the
    # config that named them, so this line cannot print while the probe is
    # running something else.
    print(
        f"[probe] {label} set loaded: path={p} rows={n_rows}/{n_all} "
        f"policy_rows={n_policy} digest={digest} planes={planes} "
        f"policy_width={width} "
        f"lineage={provenance.get('lineage', 'unrecorded')} "
        f"shards={provenance.get('n_shards', 'unrecorded')} "
        f"desync_screened={provenance.get('desync_screened', 'unrecorded')}",
        flush=True,
    )
    if n_policy == 0:
        print(
            f"[probe] {label}: WARNING 0 of {n_rows} rows carry sf_p0_regret, so "
            f"probe_{label}_policy_eregret can only ever read nan. The set was "
            f"cut from shards recorded with selfplay.record_sf_p0_regret off.",
            flush=True,
        )
    return ProbeSet(
        label=str(label), path=str(p), arrays=rows,
        n_rows=n_rows, n_policy_rows=n_policy, digest=digest,
        provenance=provenance,
    )


def eager_module(model: torch.nn.Module) -> torch.nn.Module:
    """Peel ``torch.compile`` / SWA wrappers off ``model``, innermost wins.

    The probe is an auxiliary eval running INSIDE the trainer process, and the
    trainer's ``self.model`` is the ``torch.compile`` ``OptimizedModule``
    (``train/compile_probe.py: apply_compile`` rebinds it). Calling that
    wrapper at a shape the training step never uses sends the probe into
    inductor for a fresh cudagraph capture, and a capture that fails does not
    fail alone: it leaves ``cudagraph_trees`` mid-recording and every later
    capture in the process dies with ``beginAllocateToPool: already recording
    to mempool_id`` (live 2026-08-05, trial d2003 — CUBLAS threw
    ``CUBLAS_STATUS_INTERNAL_ERROR`` mid-capture and both probes were dead for
    the rest of the session). Scoring the eager module is not a workaround for
    that CUBLAS error; it is the rule that an instrument must not be able to
    mutate the trainer's compile state at all.

    Unwrapping is by ATTRIBUTE, never by name-munging: ``AveragedModel`` nests
    the compiled module (``module._orig_mod.*``), so the wrappers come off one
    layer at a time until a module with neither attribute is left. An object
    that is not wrapped is returned unchanged, which is the CPU/test path and
    the ``use_compile: off`` path.
    """
    seen: set[int] = set()
    cur = model
    while id(cur) not in seen:
        seen.add(id(cur))
        inner = getattr(cur, "_orig_mod", None)
        if not isinstance(inner, torch.nn.Module) and isinstance(
            cur, torch.optim.swa_utils.AveragedModel,
        ):
            # AveragedModel.forward delegates to .module, so descending is
            # weight-identical and is the only way to reach a nested _orig_mod.
            inner = getattr(cur, "module", None)
        if not isinstance(inner, torch.nn.Module):
            break
        cur = inner
    return cur


def _probe_forward_eager(module: torch.nn.Module, x: torch.Tensor) -> Any:
    """One probe forward, fenced off from dynamo.

    Belt to :func:`eager_module`'s braces, and the ORDER matters. The unwrap is
    what makes capture structurally impossible — there is no compiled callable
    left to enter. The disable wrapper alone would NOT have fixed the bug:
    ``OptimizedModule.__call__`` installs its own dynamo context, so a
    ``torch.compiler.disable``d caller still compiles it (measured on CPU with
    a counting backend, and pinned by
    ``tests/test_era_probe_no_compile.py::test_the_decorator_alone_does_not_save_us``).
    What the wrapper buys is that dynamo cannot trace INTO the eager module
    if the
    probe is ever called from a compiled frame, so a future caller cannot
    reintroduce the wedge by wrapping its own loop.
    """
    return module(x)


# Applied as a call rather than a decorator, and re-annotated: torch declares
# ``disable`` as returning a one-argument callable, so the decorator form makes
# every call site a type error that could only be silenced with a suppression
# whose validity tracks the installed torch stubs.
_probe_forward: Callable[[torch.nn.Module, torch.Tensor], Any] = cast(
    "Callable[[torch.nn.Module, torch.Tensor], Any]",
    torch.compiler.disable(_probe_forward_eager),
)


@torch.inference_mode()
def score_probe_set(
    model: torch.nn.Module,
    probe: ProbeSet,
    *,
    device: str,
    batch_size: int,
) -> ProbeReading:
    """Score one frozen set with the CURRENT weights. Forward pass only.

    A deterministic full pass in stored row order — every row exactly once, no
    resampling, no priority weighting, no WDL rebalance. Two scorings of the
    same weights over the same set return the same number, which is the
    property a hinge read across iterations depends on; the sampled
    alternative gave the live holdout a floor of sd 0.052 nats with nothing to
    do with the model (rl_loop_audit G14).

    Both means are accumulated as (sum, count) and divided ONCE, not averaged
    over per-batch means: the policy denominator is the number of rows with
    ``sf_p0_regret``, which varies between batches, so a mean of means would
    be a different estimator that silently up-weights sparse batches.

    Runs in fp32 with NO autocast, unlike the training step. Deliberate, and it
    costs about 2x the forward: an amp ruler's reading moves with the autocast
    dtype, so a future amp change would put a step in every probe column that
    had nothing to do with the net. It also means the probe's ABSOLUTE level is
    not the number the trainer's own loss sees — the probe is for the trend and
    for the era/in-window contrast, and neither depends on that offset.

    The forward runs on the EAGER module (see :func:`eager_module`) — never on
    the trainer's compiled wrapper, which the probe's row-cap shape would send
    into a fresh cudagraph capture.
    """
    t0 = time.perf_counter()
    n = int(probe.n_rows)
    bs = max(1, int(batch_size))
    if n <= 0:
        return ProbeReading()

    module = eager_module(model)
    # eval()/train() stay on the object the CALLER handed us: OptimizedModule
    # registers _orig_mod as a submodule, so the mode change reaches the same
    # parameters either way, and restoring the wrapper's own flag keeps the
    # trainer's view of its model exactly as it was.
    was_training = bool(model.training)
    model.eval()
    pol_sum = 0.0
    pol_cnt = 0.0
    val_sum = 0.0
    val_cnt = 0.0
    try:
        for start in range(0, n, bs):
            stop = min(n, start + bs)
            chunk = {k: v[start:stop] for k, v in probe.arrays.items()}
            batch = collate_arrays(chunk, device=device)
            out = _probe_forward(module, batch["x"])
            logits = out["policy"] if "policy" in out else out.get("policy_own")
            if logits is None:
                raise KeyError("model outputs carry neither 'policy' nor 'policy_own'")

            width = int(logits.shape[-1])
            masked = apply_policy_mask_to_logits(
                logits, batch, "legal_mask", "has_legal_mask",
            )
            probs = torch.softmax(masked.float(), dim=-1)
            regret = align_action_values(batch["sf_p0_regret_t"], width).float()
            per_row = (probs * regret).sum(-1)
            s, c = masked_sum_and_count(per_row, batch["has_sf_p0_regret"])
            pol_sum += float(s.item())
            pol_cnt += float(c.item())

            scores = torch.tensor(
                WDL_CLASS_SCORES, dtype=torch.float32, device=batch["wdl_t"].device,
            )
            wdl_p = torch.softmax(out["wdl"].float(), dim=-1)
            target = scores[batch["wdl_t"].clamp(0, len(WDL_CLASS_SCORES) - 1)]
            err = (wdl_p * (scores.unsqueeze(0) - target.unsqueeze(1)).abs()).sum(-1)
            val_sum += float(err.sum().item())
            val_cnt += float(stop - start)
    finally:
        if was_training:
            model.train()

    def _mean(total: float, count: float) -> float:
        return total / count if count > 0.0 else float("nan")

    return ProbeReading(
        n_rows=n,
        n_policy_rows=round(pol_cnt),
        policy_eregret=_mean(pol_sum, pol_cnt),
        value_err=_mean(val_sum, val_cnt),
        seconds=time.perf_counter() - t0,
    )


def probe_metric_defaults() -> dict[str, float]:
    """Every column this module can emit, at its not-measured value.

    Spliced into the report unconditionally so Ray's CSV logger locks the
    header on row 1: it fixes the header from the first row and a resume
    appends without re-heading, so a column that appears only once a probe is
    configured would misalign every later segment of progress.csv.
    """
    out: dict[str, float] = {}
    for label in PROBE_LABELS:
        out[f"probe_{label}_policy_eregret"] = float("nan")
        out[f"probe_{label}_value_err"] = float("nan")
        out[f"probe_{label}_n"] = 0.0
        out[f"probe_{label}_policy_n"] = 0.0
    out["probe_gap_policy_eregret"] = float("nan")
    out["probe_gap_value_err"] = float("nan")
    out["probe_ms"] = 0.0
    return out


def probe_metrics(readings: dict[str, ProbeReading]) -> dict[str, float]:
    """Report columns for a mapping of label -> reading.

    The ``probe_gap_*`` pair is ``era - inwindow`` and is emitted rather than
    left to the reader because it, not either level, is the treadmill
    signature: both levels move with every ordinary thing that moves a net,
    and only their separation says the in-window gain is being paid for out of
    old-era competence. A reader who has to subtract two columns by hand is a
    reader who will compare them across different row counts.
    """
    out = probe_metric_defaults()
    total_s = 0.0
    for label, r in readings.items():
        if label not in PROBE_LABELS:
            raise KeyError(f"unknown probe label {label!r}; expected {PROBE_LABELS}")
        out[f"probe_{label}_policy_eregret"] = float(r.policy_eregret)
        out[f"probe_{label}_value_err"] = float(r.value_err)
        out[f"probe_{label}_n"] = float(r.n_rows)
        out[f"probe_{label}_policy_n"] = float(r.n_policy_rows)
        total_s += float(r.seconds)
    era = readings.get(PROBE_ERA)
    inw = readings.get(PROBE_INWINDOW)
    if era is not None and inw is not None:
        out["probe_gap_policy_eregret"] = float(era.policy_eregret - inw.policy_eregret)
        out["probe_gap_value_err"] = float(era.value_err - inw.value_err)
    out["probe_ms"] = float(total_s * 1000.0)
    return out
