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

**Degradation.** Every failure here is non-fatal and lands on the old
behaviour: an empty holdout that refills. A checkpoint written before this
existed has no sidecar; a corrupt or truncated npz raises on read; and an
input-plane or policy-width change makes the stored rows unusable against the
current model. All three log and return 0 rows. The caller must then treat the
holdout as new — see `restored_holdout_scalars`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from chess_anti_engine.replay import ArrayReplayBuffer
from chess_anti_engine.replay.shard import load_shard_arrays, save_npz_arrays
from chess_anti_engine.tune._utils import SIDECAR_HOLDOUT_ROWS

log = logging.getLogger(__name__)


def save_holdout_rows(*, ckpt_dir: Path, holdout_buf: ArrayReplayBuffer) -> int:
    """Write the holdout's rows into *ckpt_dir*. Returns the row count.

    An empty holdout DELETES any existing sidecar rather than leaving one
    behind: `ckpt_dir` is a single reused directory (`work_dir/ckpt`), so a
    stale file from before a drift reset would otherwise be restored as if it
    were the current ruler.

    Never raises. A checkpoint that cannot carry the holdout is still a
    perfectly good checkpoint, and failing the save would cost the run its
    model state to protect an eval set.
    """
    path = Path(ckpt_dir) / SIDECAR_HOLDOUT_ROWS
    try:
        rows = holdout_buf.export_arrays()
        if not rows:
            path.unlink(missing_ok=True)
            return 0
        save_npz_arrays(path, arrs=rows, meta=None, compress=True)
        return int(np.asarray(rows["x"]).shape[0])
    except (OSError, ValueError, KeyError) as exc:
        log.warning("[trial] failed to save the holdout sidecar to %s: %s", path, exc)
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
    process is about to train.

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
        arrs, _meta = load_shard_arrays(path)
    except Exception as exc:
        log.warning(
            "[trial] holdout sidecar %s exists but could not be read (%s); "
            "starting from an empty holdout",
            path, exc,
        )
        return 0

    x = np.asarray(arrs.get("x"))
    policy = np.asarray(arrs.get("policy_target"))
    if x.ndim != 4 or policy.ndim != 2:
        log.warning(
            "[trial] holdout sidecar %s has unexpected array shapes "
            "(x=%s policy_target=%s); starting from an empty holdout",
            path, x.shape, policy.shape,
        )
        return 0
    planes = int(x.shape[1])
    policy_size = int(policy.shape[1])
    if planes != int(expected_planes) or policy_size != int(expected_policy_size):
        log.warning(
            "[trial] holdout sidecar %s was written with %d input planes / "
            "policy width %d, but this run uses %d / %d; discarding it and "
            "starting from an empty holdout",
            path, planes, policy_size, int(expected_planes), int(expected_policy_size),
        )
        return 0

    before = len(holdout_buf)
    try:
        holdout_buf.add_many_arrays(arrs)
    except (ValueError, KeyError) as exc:
        log.warning(
            "[trial] holdout sidecar %s could not be loaded into the buffer (%s); "
            "starting from an empty holdout",
            path, exc,
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
