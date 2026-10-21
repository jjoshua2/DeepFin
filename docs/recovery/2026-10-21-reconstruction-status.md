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

The shared-owner callback repair is now composed through `55f28a1`. Dispatcher
execution and finish acceptance share one owner, preparation may overlap, and
background guard checks rendezvous with the original controller during both
progress and shutdown. Independent review found a serviced-expiry ordering issue;
the correction records sticky failure before publishing the callback error.
Checkpoint/ACK ownership and retained timeout custody remain mandatory.

Independent read-only review found no remaining actionable source issue in the
replay, backlog, selfplay/Torch and corrected affinity increments. Source review
alone does not qualify the reconstructed runtime.

The local CPU environment proposal uses a fresh Linux checkout and locked
CPython 3.13.15/uv 0.12.10 `dev`+`cpu` environment. Review corrections sanitize
the build environment and explicitly materialize/verify the tracked skill
symlink. Installation completed in a fresh disposable Linux root. The first
focused run on exact `55f28a1295ada48618524ac5fa49e82504cd1b2a` authenticated
CPython3.13.15, Torch2.14.0+cpu (no CUDA build), NumPy2.2.6, pytest9.1.1,
Zarr2.18.7, CPU affinity29/30 and local fresh magic native extensions.

That first run completed **624 tests: 608 passed, 7 failed, 9 skipped**, with
zero collection errors and JUnit time69.235s. It is a failed qualification run.
Stockfish discovery returned none, so engine-dependent skips are missing e2e
evidence. The new owner-thread/dispatcher/STOP regression controls passed.
The failures exposed stale warning-value assertions, bypassed worker constructor
fixtures missing watcher state, omitted retained Ceres source pins, an obsolete
constructor-failure seam, and the old scalar-encoder monkeypatch after bulk
encoding adoption. Test-only corrections preserve or strengthen the original
oracles; their independent source reviews found no actionable issues.

Complete first-run stdout SHA256:
`6ddc6758b3edad4e58288efabed489c3808bdd11ee11fbcd472e64cc7d70d6e7`.
Full JUnit/metrics and native streams are retained outside the checkout under
the isolated qualification report roots. The corrected focused run on exact
`416a4189a35e731c1f864478787a3a1ec7f3b45d` completed **625 tests: 616 passed,
9 skipped, zero errors/failures** in JUnit83.737s. All nine skips require the
missing Stockfish engine; they are not passing e2e evidence.

Whole Ruff/basedpyright/vulture subsequently passed on
`1e459966b21dcccf3a88fc9d5e4da1af632f843a`, with zero type errors/warnings.
That commit changes only two test files relative to the passing focused run;
all three affected cases passed a separate rerun. These are overlapping cases,
not three additional unique passes. The two actual-import affinity/ordering
controls fail for their intended reasons on recovered `377e5d5` and pass on the
reconstruction. All earlier failed runs and the first pre-test Git packing
failure remain retained.

The whole ordinary CPU suite remains unrun: present host-dependent fixtures
would import a historical frozen runtime and automatically use an installed,
unbound Bend toolchain. Neither was executed, hidden or removed. No retained
model or GPU execution has occurred. Compact identities and scope are in
`2026-10-21-cpu-qualification.json`; remaining gates are in
`2026-10-21-bt4-next-gates.md`.

Actual BT4 qualification remains separate: existing canonical exclusive custody,
exact successor controls/source bindings, unique nontraining outputs, independent
launch review and a qualified restoration path are required. No generation
restart, PAUSE/STOP removal, live adoption, merge or evaluation is performed by
this source checkpoint. Targets 320/340 remain hypotheses; Ceres physical32
evidence remains separate.
