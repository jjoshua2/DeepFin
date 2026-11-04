"""Make the holdout ruler survive a restart.

The holdout buffer is what `test_loss` — and therefore best-model selection —
is measured against. It used to be rebuilt EMPTY on every process start, so a
restart silently replaced the ruler: measured across two restarts of one live
trial, iter 1 held 210 rows and only reached the 2000-row cap by iter 14, and
iter 42 (after the next restart) was back to 194. The first ~3 iterations of
each segment ran with no holdout eval at all (`len(holdout_buf) >= batch_size`
gates it), the next ~10 were judged against a set that was still growing, and
`best_loss` carried across the boundary defending a number earned on a ruler
that no longer existed. `freeze_holdout_at` cannot fix that on its own: it
fires at `len(holdout_buf) >= freeze_holdout_at`, so after a restart it just
freezes a fresh post-restart sample.

**Where the state lives.** The rows go in the Ray checkpoint directory, beside
`trainer.pt`, and the two scalars (`holdout_frozen`, `holdout_generation`) go
into the checkpoint's existing `trial_meta.json`. The alternative was
`_durable_trial_dir`, where `best.json` lives. The checkpoint wins on three
counts:

  * it is the only location that survives ALL the restart paths that actually
    happen — a normal resume, `train.sh start` auto-resume, AND a salvage
    restore, which restores into a *different* trial dir and would therefore
    never see a durable-dir copy (salvage copies a fixed file list out of the
    checkpoint, so the sidecar rides along by name);
  * it keeps the ruler in lockstep with the weights it judges, so rolling back
    to an older checkpoint restores the holdout that checkpoint's `test_loss`
    was measured on, instead of pairing old weights with a newer ruler;
  * Ray already syncs and prunes checkpoints, so there is no new file
    lifecycle to own. Pruning can never orphan the sidecar because a resume
    always restores from a checkpoint that still exists.

**Cost.** Measured on 2000 real production rows (175 planes, lc0_1858):
102.2 MB dense in memory, **2.09 MB** on disk as a compressed npz, 1.2 s to
write and 0.9 s to read. Against a ~40-minute iteration and a checkpoint whose
`trainer.pt` is three orders of magnitude larger, that is free; six retained
checkpoints (`tune_num_to_keep: 6`) cost ~13 MB in total.

**Degradation.** Row save/read failures remain non-fatal and permit an empty
holdout that refills. The writer first records zero saved rows in existing
checkpoint metadata, before mutating weights, then commits the actual row count
with the new checkpoint identity. An interrupted write cannot authorize an old
sidecar. Failure to invalidate metadata aborts BEFORE weights change; failure to
write final metadata leaves restore invalidated. Salvage uses the same rule and
commits metadata after its copies complete.

Missing metadata or an absent row-count field retains genuinely legacy behavior.
Existing unreadable/malformed metadata, an invalid count, or a zero count refuses
the rows. Old readers do not understand this marker: rolling back code can restore
a stale file when cleanup failed. This is not a multi-file checkpoint transaction,
nor protection against deleting authoritative metadata or concurrent source writers.
An absent/corrupt/shape-mismatched sidecar also returns zero rows. The caller then
treats the holdout as new — see `restored_holdout_scalars`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chess_anti_engine.replay import ArrayReplayBuffer
from chess_anti_engine.replay.shard import load_shard_arrays, save_npz_arrays
from chess_anti_engine.tune._utils import SIDECAR_HOLDOUT_ROWS, SIDECAR_TRIAL_META
from chess_anti_engine.utils.atomic import atomic_write_text

  # `print`, not `logging`: this runs inside a Ray actor, whose stdout is what
  # train.sh captures. A `logging` call reaches only the Ray session logs --
  # the same reason `load_optional_json` prints. Every message here explains a
  # ruler that silently did not come back, which is exactly the class of thing
  # this audit found by reading the training log.


def _holdout_checkpoint_meta(ckpt_dir: Path) -> dict:
    try:
        meta = json.loads((ckpt_dir / SIDECAR_TRIAL_META).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(meta, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    return meta


def invalidate_holdout_restore(*, ckpt_dir: Path) -> None:
    """Refuse old rows before mutating a reused checkpoint or salvage slot.

    Preserve existing identity fields: an interrupted save must not label old
    weights with a new iteration. Failure propagates BEFORE weights change.
    The checkpoint writer commits the new row count with its final metadata.
    """
    meta = _holdout_checkpoint_meta(Path(ckpt_dir))
    meta["holdout_rows_saved"] = 0
    atomic_write_text(
        Path(ckpt_dir) / SIDECAR_TRIAL_META,
        json.dumps(meta, sort_keys=True, indent=2),
    )


def save_holdout_rows(*, ckpt_dir: Path, holdout_buf: ArrayReplayBuffer) -> int:
    """Write the holdout's rows into *ckpt_dir*. Returns the row count.

    An empty holdout DELETES any existing sidecar rather than leaving one
    behind: `ckpt_dir` is a single reused directory (`work_dir/ckpt`), so a
    stale file from before a drift reset would otherwise be restored as if it
    were the current ruler.

    Save failures remain non-fatal. The caller MUST record the returned count
    in trial_meta.json: cleanup can fail, leaving a stale sidecar on disk.
    Checkpoint writers invalidate that metadata before changing weights.
    """
    path = Path(ckpt_dir) / SIDECAR_HOLDOUT_ROWS
    try:
        rows = holdout_buf.export_arrays()
        if rows:
            save_npz_arrays(path, arrs=rows, meta=None, compress=True)
            return int(np.asarray(rows["x"]).shape[0])
    except Exception as exc:
        print(
            f"[trial] failed to save the holdout sidecar to {path} ({exc}); "
            f"checkpoint metadata must record zero saved holdout rows",
            flush=True,
        )
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        print(
            f"[trial] could not remove stale holdout sidecar {path} ({exc}); "
            f"checkpoint metadata must exclude it from restore",
            flush=True,
        )
    return 0


def load_holdout_rows(
    *,
    state_dir: Path | None,
    holdout_buf: ArrayReplayBuffer,
    expected_planes: int,
    expected_policy_size: int,
) -> int:
    """Refill *holdout_buf* from the sidecar in *state_dir*. Returns rows added.

    Returns 0 — leaving the buffer untouched — when there is no sidecar, when
    it cannot be read, or when the stored rows do not match the model this
    process is about to train. New checkpoint metadata must authorize a positive
    row count matching the file; absent legacy metadata remains supported.

    The shape checks are not defensive padding. A plane-count change (v1's 146
    to v2_threats' 175) or a policy-width change (az_4672 to lc0_1858) between
    the checkpoint and now would otherwise be discovered at eval time, and a
    mixed-width holdout raises out of `_gather_rows` mid-iteration rather than
    degrading. Refusing the restore keeps that a logged line at startup.
    """
    if state_dir is None:
        return 0
    path = Path(state_dir) / SIDECAR_HOLDOUT_ROWS
    if not path.exists():
        return 0
    try:
        meta = _holdout_checkpoint_meta(Path(state_dir))
    except (OSError, UnicodeError, ValueError) as exc:
        print(
            f"[trial] cannot establish holdout sidecar identity from checkpoint "
            f"metadata in {state_dir} ({exc}); starting from an empty holdout",
            flush=True,
        )
        return 0
    saved_rows = meta.get("holdout_rows_saved")
    if "holdout_rows_saved" in meta and (type(saved_rows) is not int or saved_rows <= 0):
        print(
            f"[trial] checkpoint metadata in {state_dir} does not authorize "
            f"holdout rows; starting from an empty holdout",
            flush=True,
        )
        return 0
    try:
        arrs, _meta = load_shard_arrays(path)
    except Exception as exc:
        print(
            f"[trial] holdout sidecar {path} exists but could not be read "
            f"({exc}); starting from an empty holdout",
            flush=True,
        )
        return 0

    x = np.asarray(arrs.get("x"))
    policy = np.asarray(arrs.get("policy_target"))
    if x.ndim != 4 or policy.ndim != 2:
        print(
            f"[trial] holdout sidecar {path} has unexpected array shapes "
            f"(x={x.shape} policy_target={policy.shape}); starting from an "
            f"empty holdout",
            flush=True,
        )
        return 0
    if saved_rows is not None and int(x.shape[0]) != saved_rows:
        print(
            f"[trial] holdout sidecar {path} row count does not match checkpoint "
            f"metadata; starting from an empty holdout",
            flush=True,
        )
        return 0
    planes = int(x.shape[1])
    policy_size = int(policy.shape[1])
    if planes != int(expected_planes) or policy_size != int(expected_policy_size):
        print(
            f"[trial] holdout sidecar {path} was written with {planes} input "
            f"planes / policy width {policy_size}, but this run uses "
            f"{int(expected_planes)} / {int(expected_policy_size)}; discarding "
            f"it and starting from an empty holdout",
            flush=True,
        )
        return 0

    before = len(holdout_buf)
    try:
        holdout_buf.add_many_arrays(arrs)
    except (ValueError, KeyError) as exc:
        print(
            f"[trial] holdout sidecar {path} could not be loaded into the "
            f"buffer ({exc}); starting from an empty holdout",
            flush=True,
        )
        holdout_buf.clear()
        return 0
    return len(holdout_buf) - before


def restored_holdout_scalars(
    *, rows_restored: int, stored_frozen: bool, stored_generation: int,
    had_stored_state: bool,
) -> tuple[bool, int]:
    """Reconcile the restored `(frozen, generation)` with what actually loaded.

    Two rules, both about not lying to the best-model comparison:

    * `frozen` is only honoured when rows came back. Trusting a stored
      ``frozen=True`` over an empty buffer would wedge the holdout shut
      forever: ingest skips the holdout while frozen, so it would never fill,
      `len(holdout_buf) >= batch_size` would never hold, and the run would
      never evaluate a holdout again.
    * losing the rows of a checkpoint that HAD them bumps the generation. The
      counter's job is to say which holdout a `test_loss` was measured on, and
      a restart that could not restore the set really is a different ruler.
      The bump is what lets `_update_best_model` hand the record over instead
      of defending a number earned on a set that no longer exists.
      ``had_stored_state`` keeps a fresh start (no checkpoint at all, nothing
      lost) at generation 0.
    """
    if rows_restored > 0:
        return bool(stored_frozen), int(stored_generation)
    return False, int(stored_generation) + (1 if had_stored_state else 0)
