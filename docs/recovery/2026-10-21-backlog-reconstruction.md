# Retained backlog composition for PR1110

This source checkpoint composes six repair scopes over recovery commit
`377e5d59662ea6051df5487bb1824fb0efcfd13e`, whose public-main base is
`1fdd4c854ace0d63ac08e18e87513741ea0eefec`. It is a new candidate. It is not
the missing final v8 artifact, and inherits no historical execution credit.

## Retained repair provenance

Only patch hunks and their regression cases were composed; stale whole-file
postimages were not transplanted. The three first patches apply without source
adaptation at this base. Their source-review reports remain historical source
evidence and do not qualify this combined candidate.

| Scope | Retained manifest | Patch SHA256 |
| --- | --- | --- |
| Replay writer lock inode checks | `2026-10-16/task-9/writer-lock-manifest.json` | `09178827642df23a311bdce0874a5bb8a97a03815b8df187a82a02f72fd4963b` |
| Quarantine restore containment | `2026-10-16/task-9/quarantine-containment-manifest.json` | `b7948115d9f4d69e5a0416612826f6c8c3e765388ee1efaabc662cb862bf6699` |
| Arena checkpoint architecture | `2026-10-13/task/arena-checkpoint-identity-20261019-v1/handoff-v1.json` | `4b21df34bc74810adf5857ab8f9229f047ab20f5aaf0ad93e604d452c5733cdc` |
| Dropped-value warning privacy | `2026-10-16/task-10/home-codex-adapter/dropped-value-logging-20261019/source-packet.json` | `b5f526f8e4016b770cdc1eaae6f7880a517c47a2d7c17559cd6bffba366ab0de` |

The warning packet was based on a retained pause-helper candidate, not this
main revision. Only `_warn_on_keys_dropped_from_yaml` and its two warning tests
are composed here. Its saved old warning used different dash punctuation;
the current composition removes the deferred value argument and preserves
the warning's instructions. No pause-helper changes are included.

The writer lock protects the opened regular inode before flock/PID stamping;
it does not promise atomicity against all namespace mutations or exclude hard
links. Quarantine restore validates the whole journal before mutation and
rechecks immediate entries. It requires trusted, quiescent roots; it does not
claim recursive symlink or concurrent adversarial snapshot protection.

Arena-best loading opts into the checkpoint's embedded architecture and
requires a complete weight load for that architecture. Manifest/current-model
loading retains the previous caller-config behavior. This does not establish
that a broker supports cross-width arena models.

## Newly reconstructed scopes

### Worker watcher handoff

The recovered daemon checked `_selfplay_session_active` before taking the
model-update lock. Session teardown cleared that flag without draining an
in-flight update. Consequently an old callback could overlap the next outer
manifest/model setup. The new session transition takes the same lock, and both
ply and watcher callbacks check active-session eligibility under that lock.
Watcher freshness checks share this boundary. Cleanup signals an interruptible
wait, drains the callback and joins the retained watcher before closing clients.
Existing update tests now stop/join their test watchers rather than leaving
daemons behind.

`tests/test_worker_model_watch_handoff.py` exercises actual WorkerSession
methods with fake callback/client seams. It includes the original inactive-ply
entry-point regression, an in-flight callback/owner handoff ordering witness,
freshness lock ownership, next-session activation, and cleanup/client ordering.
The thread waits are bounded in the tests. Production handoff deliberately
waits for a callback to finish rather than releasing state under a live swap;
it introduces no forced cancellation or new network timeout policy.

### PID node budget invariants

Construction previously clamped unchecked bounds, while live edits could
mutate the minimum before a maximum parse failed. Factory `int()` conversion
also hid fractional or Boolean input before validation. A shared node parser
now requires positive finite integral counts, preserves integer strings and
integral numeric inputs, and refuses integers not exactly representable by the
existing floating node lever. Constructor and live bounds consistently keep
the existing policy that an inverted floor yields to the ceiling.

The complete proposed live bounds pair is checked before changing any state.
Restore validates node and frozen metric bounds before changing node state.
Valid frozen metric bounds retain their saved values; live ratchets do not
move them. Unchanged bounds leave the fractional accumulator untouched. No
lever gains, fitting, regret controls, or valid metric semantics are changed.

`tests/test_pid_node_invariants.py` covers constructor/factory refusals,
representability, integer-string compatibility, live mutation atomicity,
unchanged-bound fractional accumulation and restored-vs-live metric identity.

## Qualification still required

No subject imports, AST/compile execution, tests, lint, builds, model reads,
WSL commands, GPU work or live generation actions were run by this source
author. Only native source inspection, patch application and Git whitespace
checking were performed. The new cases are uncollected and unrun.

Use the exact combined candidate under the CPU operator's qualified route.
At minimum include the new inode, containment, arena, warning, watcher and
PID cases plus their affected existing regressions:

- `tests/test_replay_writer_lock_inode.py` and existing writer lock lifecycle
  cases in `test_replay_disk_buffer.py` / `test_replay_window_safety.py`.
- `tests/test_quarantine_restore_containment.py` and existing quarantine
  roundtrip, partial-run, tamper and reservation cases.
- `tests/test_worker_arena_checkpoint_arch.py`, current model-load/update and
  broker input-width tests.
- The dropped-warning cases in `tests/test_trainable_config_ops.py`, including
  existing YAML reload behavior.
- `tests/test_worker_model_watch_handoff.py`, `test_worker_model_update.py`
  and `test_worker_broker_warmup_gate.py`.
- `tests/test_pid_node_invariants.py`, `test_pid_inverse_regret.py` and
  `test_pid_live_bounds_zero_outcome.py`.

The direct Worker/model/tune imports require the real project/native startup
closure; the retained arena tests additionally use extracted actual methods
with fake model/device/match boundaries. Source evidence alone is insufficient
to claim runtime/provider/custody compatibility. Independent combined source
review and qualified execution remain outstanding.
