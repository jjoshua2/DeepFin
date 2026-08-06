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

Usage::

    PYTHONPATH=. python3 scripts/retarget_retrain.py \\
        --config configs/pbt2_small.yaml \\
        --checkpoint <trial>/checkpoint_000123/trainer.pt \\
        --replay-dir <frozen copy>/replay_shards   # NOT the live window \\
        --steps 800 --out-dir runs/retarget \\
        --variant base: \\
        --variant sharp:sf_policy_temp=0.006 \\
        --variant smooth:sf_policy_temp=0.05,sf_policy_label_smooth=0.02
"""
from __future__ import annotations

import argparse
from typing import Any
import json
import time
from pathlib import Path

import numpy as np

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.model import build_model, load_state_dict_tolerant, model_config_from_flat_config
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays
from chess_anti_engine.replay.threat_upgrade import V1_INPUT_PLANES
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


def _parse_variant(spec: str) -> tuple[str, dict]:
    name, _, body = spec.partition(":")
    if not name:
        raise SystemExit(f"--variant needs a name before ':', got {spec!r}")
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
        swapped[mask] = soft[mask].astype(swapped.dtype, copy=False)
        out = dict(arrs)
        out["policy_target"] = swapped
        return out


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
    rebuild_sf_targets: bool = True,
    gpu_mem_fraction: float = 0.0,
    allow_yaml_arch: bool = False,
    allow_partial_load: bool = False,
) -> dict:
    config = dict(base_config)
    config["rebuild_sf_targets"] = bool(rebuild_sf_targets)
    config.update(overrides)
    # Rig-only knob: never a Trainer/TrialConfig key, so pop it here. The
    # override stays recorded in the summary's `overrides` for deploy proof.
    rig_policy_from_soft = bool(config.pop("rig_policy_from_soft", 0))

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
        train_buf: Any = buf
        if rig_policy_from_soft:
            print(f"[retarget-{name}] rig_policy_from_soft ACTIVE: main policy "
                  "target served from policy_soft_target (soft-shape screen)")
            train_buf = _SoftPolicyAsMainBuffer(buf)
        t0 = time.time()
        metrics = trainer.train_steps(train_buf, batch_size=int(batch_size), steps=int(steps))
        duration = time.time() - t0
    finally:
        buf.close()

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
    ap.add_argument("--allow-partial-load", action="store_true",
                    help="tolerate missing/unexpected state-dict keys (logged) "
                         "instead of aborting, for retargeting a historical "
                         "checkpoint saved under drifted model code; the "
                         "mismatched tensors keep fresh init")
    args = ap.parse_args()

    base_config = flatten_run_config_defaults(load_yaml_file(args.config))
    batch_size = int(args.batch_size or base_config.get("batch_size", 256))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ONE reference for the whole sweep, and it is arm 1's OWN buffer scan —
    # deliberately not a glob taken here. Each arm re-scans the directory inside
    # its own DiskReplayBuffer ctor, so a per-arm reference would just re-measure
    # the drift it is meant to catch; but a reference read HERE sits tens of
    # seconds (torch.load + CUDA init) before arm 1's scan, and a shard landing
    # in that window aborts a sweep whose arms would all have been paired.
    shard_snapshot: list[Path] | None = None

    summaries = []
    for spec in args.variant:
        name, overrides = _parse_variant(spec)
        summary = _run_variant(
            name=name, overrides=overrides, base_config=base_config,
            checkpoint=args.checkpoint, replay_dir=args.replay_dir,
            steps=args.steps, batch_size=batch_size, device=args.device,
            out_dir=args.out_dir, shard_snapshot=shard_snapshot,
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
        summaries.append(summary)

    report = args.out_dir / "retarget_report.json"
    report.write_text(json.dumps(summaries, indent=2))
    print(f"[retarget] report written to {report}")


if __name__ == "__main__":
    main()
