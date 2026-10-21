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

At that earlier checkpoint the whole ordinary CPU suite remained unrun: host-dependent fixtures
would import a historical frozen runtime and automatically use an installed,
unbound Bend toolchain. Neither was executed, hidden or removed. At that checkpoint no retained
model or GPU execution has occurred. Compact identities and scope are in
`2026-10-21-cpu-qualification.json`; remaining gates are in
`2026-10-21-bt4-next-gates.md`.

Actual BT4 qualification remains separate: existing canonical exclusive custody,
exact successor controls/source bindings, unique nontraining outputs, independent
launch review and a qualified restoration path are required. No generation
restart, PAUSE/STOP removal, live adoption, merge or evaluation is performed by
this source checkpoint. Targets 320/340 remain hypotheses; Ceres physical32
evidence remains separate.
## Hosted ordinary suite: failed qualification, reviewed test repairs

The later hosted run on `20836c925fe195971ac40a9a755eb7028f26c989` completed
with **105 failed, 16,100 passed, 145 skipped and four setup errors** (plus four
deselected and four expected failures) in 1,747.97 seconds. Hosted lint and
PEXT passed. The ordinary suite failed; its passing cases do not qualify the
whole reconstructed candidate.

Complete failure attribution separates five families:

| Family | Hosted result | Source correction scope |
| --- | ---: | --- |
| Replay fixture widths | 95 failures | Valid positive fixtures use at least 112 planes; row, label, seal, dtype and negative-input oracles remain. |
| Watchdog fake interface | 3 failures | Supply the stop Event read by the actual watcher method. |
| Ceres fixture source closure | 6 failures | Add required consequential source pins; the three child exits may still expose further failures. |
| Public-document paths | 1 failure | Correct documentation paths without changing runtime source. |
| Native component suite compatibility | 4 setup errors | Permit model-runtime modules already loaded by normal conftest; reject newly introduced modules per component case and initialize encoder mode before each case. |

Eleven test files have reviewed corrections. The replay minimum-width admission
rule, production watcher and Ceres checks, normal conftest, native equality/RNG
oracles and ownership cleanup are unchanged. The tiny holdout model still reads
its original four planes despite valid 112-plane storage. The narrow-shard
negative checks both writer rejection and the dump reader's refusal of a
malformed raw store. The separate isolated native controller retains its strict
no-model-runtime import guarantee.

At that stage these were **source-reviewed repairs, not a passing rerun**. The
later hosted successor result is recorded below; the original failed run remains
retained and is not relabeled successful.

## Dependency inspection and remaining runtime gate

Six synthetic sparse-ELF parser controls passed and were independently reconciled.
They exercise bounded table/string parsing and refusal mechanisms only. They did
not read installed dependency payloads or qualify a loader, normal runtime import,
model session or accelerator. The separate sparse metadata observation identified
130 installed file records; it is not a complete dependency identity/import proof.
The earlier full census stopped before completion and remains failed evidence.

The selected aggregate runtime still needs its complete normal-import and native
loader closure under the admitted resource limits. The full Torch startup that
failed under a 4 GiB address-space limit remains a real boundary; parser controls
and prior low-dependency component results do not waive it. Material audit
approval and exact runtime/custody/action review remain outstanding before another
aggregate model trial. No larger cap, audit bypass, live adoption, merge or
training credit is authorized by this note.
## Hosted successor passes; utility candidate remains separately scoped

Hosted [run 39646964682](https://github.com/jjoshua2/DeepFin/actions/runs/39646964682)
on published `f32feb7611460263e7d2fa7c6f3ef668467fafec` passed lint, PEXT and
the CPU job. The ordinary suite reported **16,209 passed, 145 skipped, four
deselected and four expected failures**. The separate capped regression reported
**24 passed**. These are results for that published revision, following the
failed `20836` run; skips and expected failures are not passing coverage.

A separate local utility-boundary candidate passed eight selected normal-import
and metadata-caller controls, with independent saved-result acceptance. The
controls cover the selected utility interfaces and constructor boundaries. The
owner and pooled-worker entry points were imported and checked as callable,
not executed. This is not full-worker, model-session, GPU, gameplay, throughput,
or adoption qualification, and those eight cases are not added to the hosted
suite's pass count.

The four utility production changes are outside the hosted `f32feb76` result.
Their separate focused static successor passed Ruff and Basedpyright with zero
errors, warnings or notes. The eight-case execution used the V5 test; the final
V7 test differs only by separately reviewed style/type edits and passed static
validation. Those eight cases were not rerun on V7, and the hosted pass does not
transfer to these postimages.

Activation of the proposed ten-second control
audit remains disabled pending the material-tradeoff decision; candidate GPU
admission remains disabled. Any subsequent model trial still requires its exact
runtime, canonical custody and action review. These records change no ordinary
generation control, audit policy, resource limit or training admission.
