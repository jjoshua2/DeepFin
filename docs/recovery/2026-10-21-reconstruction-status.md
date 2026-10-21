# PR1110 reconstruction checkpoint

This is a newly reconstructed candidate on the published recovery branch. It is
not the lost final v8 artifact. No old 197-case or v7 qualification transfers.
The original recovered manifest and its source identities remain unchanged.

Source checkpoints composed so far:

- `66555285f33d3c083db443fd9c98eb0502f86c37`: metadata-first replay minimum
  112-plane admission, with eager/lazy refusal and supported-width cases.
- `2f29111`: four retained fixes composed from their original patches, plus
  reconstructed watcher handoff and PID budget invariants. Retained provenance
  and explicit scope limits are recorded in the backlog reconstruction document.
- `fb2f5ae`: selfplay adopts `GenerationSessionRunner`; STOP finalization does
  not recycle finished games. Torch adopts shared FIFO routing and the two-slot
  pump, preserving request grouping, barriers and owned output snapshots.

The actual shared-owner callback affinity and finish-before-next-execute repair
is still being reconstructed in a separate source worktree. It must land before
any production-owner callback or model qualification. An executor-thread guard
callback is not accepted as an owner-thread checkpoint/control substitute.

Independent read-only source review found no actionable issue in the replay
increment. Backlog and selfplay/Torch independent reviews are pending. Every new
test remains uncollected and unrun. Git whitespace checks are the only executed
source checks so far; no Python imports, native builds, model loads or GPU tests
have qualified these changes.

The local CPU environment proposal uses a fresh Linux checkout and locked
CPython 3.13.15/uv 0.12.10 `dev`+`cpu` environment. Review corrections sanitize
the build environment and explicitly materialize/verify the tracked skill
symlink. Environment installation has not started at this checkpoint. Native
extension origins/backend freshness, actual-import regressions, whole static
checks and ordinary CPU compatibility remain outstanding.

Actual BT4 qualification remains separate: existing canonical exclusive custody,
exact successor controls/source bindings, unique nontraining outputs, independent
launch review and a qualified restoration path are required. No generation
restart, PAUSE/STOP removal, live adoption, merge or evaluation is performed by
this source checkpoint. Targets 320/340 remain hypotheses; Ceres physical32
evidence remains separate.
