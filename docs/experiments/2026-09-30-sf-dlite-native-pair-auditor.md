# Independent Selected-E / D-lite native-pair pack audit

Status: prepared CPU-only code and tiny native fixtures. No production pack,
label corpus or raw Stockfish source has been scanned by this auditor. Its
qualification status must not be used until a fresh successful label terminal, the
independent all-label audit and the paired builder terminal are sealed, and
this auditor completes every shard and final readback.
The current successor source pins bind the fresh full07 worker, authorization, builder v12 and core v3.
Full04, full05 and full06 labels and receipts are ineligible. Production admission
still requires a completed full07 label terminal and independently monitored
all-label audit, followed by a sealed native builder terminal.

The builder's terminal is deliberately unadmitted. This auditor reconstructs
the control from the qualified saved base and Selected-E overlay, then
reconstructs the candidate by replacing only `search_wdl` with the fixed
float32 `(D8 + 2 × Selected-E) / 3` operation followed by one float16 round.
It compares all 17 physical fields, raw original game/ply IDs, source-qualified
180-byte roster rows, roster-index sidecars, native file bytes, and every
declared digest for both arms. It independently rehashes the source base and
overlay content seals and every label block used for the local index. The
label-index block set and exact receipt/data hashes must match the frozen plan
behind the independent all-label PASS terminal; a self-consistent later
replacement of label files is refused. The
separate independent label audit is the authority for the raw Stockfish
search/source proof; this pack auditor checks transport from those qualified
labels to the paired training data and does not rerun raw searches.

The runner requires an exact SHA-pinned JSON plan with schema
`sf_dlite_independent_paired_pack_audit_plan_v1` and status
`REVIEWED_ZERO_CREDIT_PLAN`. It contains `{path,sha256}` references named
`census`, `builder_terminal`, `label_terminal`, `label_audit`, `roster_audit`,
`label_audit_session`, `selected_e_qualification`, `injectivity`, and
`target_scheme`, plus
`auditor_sources` and `auditor_runtime` equal to the runner's read-only
`source_provenance()` and `runtime_provenance()` results. Its
`qualification_path` is exactly
`<paired-pack-root>/PAIRED_PACK_QUALIFICATION.json`; the auditor's `--output`
is a separate owned directory under the artifact root. Freeze the complete
plan file and pass its whole-file SHA through `--plan-sha256`. The final
trainer launch must separately pin the resulting qualification file SHA.
The `label_audit_session` reference pins a separate full07 monitor-session
`COMPLETE.json`. Admission reopens its `FINISH.json`, frozen PLAN and
freeze-session completion, verifies the logged finish step and source hashes,
and refuses any session or audit `FAILED.json`. The builder terminal must bind
the same completion path and SHA in its `source` object; an independent child
audit `TERMINAL.json` alone has no pack credit.

Each label-index, shard-audit, shard-readback and final unit runs in an owned
child with a 1,800-second parent watchdog, parent-death termination and its
own 1,800-second alarm. The parent enforces a sampled 6-GiB child RSS ceiling, 32-GiB
host available-memory floor, 400-GiB shared-cgroup physical-I/O bound per
attempt, 1-GiB auditor-output ceiling and 50-GiB free-space floor. The shared
cgroup counter is physical host I/O, not process logical bytes. A newly visible
device is charged from zero; a vanished or reversed device holds the run. One
baseline and prior-sample map span every unit in an attempt. The trace and
resource receipts retain the baseline and per-device counters. The runner
holds the existing shared heavy-I/O lease; do not start it while full07 or
another owner holds that lease. A label index and 384 audit/verify receipt
pairs are published by fsynced no-replace links. Every accepted unit also has
a durable monitor-completion receipt linked to its output bytes. If the parent
dies between a shard proof and monitor acceptance, the next owner independently
recomputes and matches that orphan proof before recording acceptance. Every
restart checks the contiguous receipt chain; a crash loses at most the active
bounded unit. The
final readback checks current source/output metadata stamps, sidecars,
manifests, the complete roster permutation, pinned roster bytes, and the
sealed label blocks. The final child seals an unadmitted candidate in the
auditor checkpoint root. Only after the child exits and its monitor receipt is
durable does the parent publish PASS inside the paired pack root, binding all
unit monitor receipts. A metadata stamp is a change guard after the
earlier complete byte checks, not a substitute for those checks.

The current fixture uses three synthetic rows and two native `.zarr` arms. It
tests all-field and target byte parity, seven source/pack/label corruptions,
label-index source rechecks, receipt-chain gap/tamper rejection, exact
SIGKILL before and after the receipt link followed by a fresh-process resume,
parent-SIGKILL child termination before and after Python watchdog setup,
a pre-exec child alarm, an actual killed monitor after proof emission
followed by fresh-process resource acceptance, orphan shard proof readback, and parent-only
qualification publication outside the auditor checkpoint root. Four physical-I/O
counter cases cover stable, new, vanished and reversed devices, plus a device
appearing in one unit and disappearing or reversing before the next.
These tests do not measure production runtime or prove actual WSL power-loss
behavior. Production execution remains held until plan/source review and the
shared heavy-I/O lease are explicitly scheduled.

## October 1 correctness review

The label index now hashes the exact JSONL bytes consumed by its line parser,
including length, before publishing a durable index. Receipt JSON is parsed from
the same SHA-pinned bytes. A transient valid WDL replacement, a byte append or a
truncated trailing newline must fail even if the original source is restored
before final readback. Partial staging files carry no index or pack credit.

A label index published before its parent monitor accepted the unit is now
independently reconstructed from the qualified sources on restart. Its table,
bitmap and receipt identity must match; successful reconstruction preserves the
original receipt and removes only the new comparison staging directory. A
self-consistent unmonitored table hash alone cannot establish source fidelity.
Monitor-backed indexes retain their source and staged-byte rechecks.

The added synthetic regressions exercise consumed-byte substitution and
unmonitored index reconstruction. No production archive, label scan, GPU work,
pack admission or strength measurement is part of this code review. The full07
completion and independent monitor chain remain prerequisites, not results
established by merging this implementation. Existing plans pin the auditor source
bytes, so these fixes require a newly reviewed plan rather than silently reusing
an older implementation's checkpoints.

Independent review also identified a live-memmap gap between label-index
verification and shard consumption. Roster and staged-label arrays now consume
immutable in-memory snapshots hashed from those exact bytes on every shard pass.
The production sizes are 450 MB for the roster and 30 MB for labels; existing
6-GiB child RSS, 30-minute wall and shared-I/O gates still apply. This deliberately
adds per-pass hashing and memory rather than assuming a source path stays frozen.
Tiny tests mutate the files after snapshot acquisition and substitute bytes at
read time to distinguish immutability from a later source rehash.


## October 1 local restart and training-eligibility fixes

The all-label audit completed for all 2,500,000 full07 labels before the paired
build began. Two bounded build attempts exited before sealing a native shard:
the first native writer rejected its surrounding process alarm; the next writer
rejected three custom provenance keys outside the native ShardMeta schema.
The successor preserves the external deadline across both writes, uses supported
native metadata, and retains source/census identity and the source metadata hash
in the sealed receipt. Resume checks that hash. Failed attempt identities and
partial outputs were preserved; they carry no native-pack or training credit.

Independent tests include seven real timer/writer cases and two full guarded
build-path cases: both native arms, stage seal, publication, repeated reopen,
independent decoded-array audit, and refusal of a forged source metadata hash.
A fresh production attempt has published its first shard. That observation
does not qualify the complete pack or establish training/playing strength.

The auditor successor uses a fresh standard-library exec guard before NumPy
imports, avoiding preexec callbacks in the multithreaded parent. It also requires
all selected source rows to have both `is_network_turn=1` and
`has_is_network_turn=1`, matching the actual donor loss filter. Per-shard
source-derived counts are bound through both receipt passes, checked again at
final readback, summed to exactly 2,500,000, and included in qualification.
Inactive rows fail admission; their flags are never rewritten or filtered away.

The final integration starts from hosted-tested revision
`25f1e08242c84f8f1e128c876b9038a0d781b0e6`, preserving its consumed-stream
hashing, unmonitored-index reconstruction, immutable roster/label snapshots and
portable artifact root. The combined focused CPU suite passed 41 tests both
locally and independently. Subsequent explicit integer narrowing and a negative
test annotation address two introduced typecheck findings; affected eligibility
fixtures were rerun (seven passed). Full and scoped Ruff/Vulture pass; explicit
path-scoped basedpyright passes with zero errors and warnings. The whole-repository
default typecheck reports 2,502 errors and 1,562 warnings in this host environment,
including unresolved installed dependencies. That full gate is not a clean pass;
the source base already has an exact hosted CI pass, and this successor requires
its own hosted validation.

Each native/auditor unit retains its 30-minute bound and durable completion
receipts. Long-running jobs reuse only sealed work with the same reviewed
identity after rechecking it. Training and arena remain later, separately
qualified serial GPU jobs.
