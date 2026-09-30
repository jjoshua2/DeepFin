# Independent Selected-E / D-lite native-pair pack audit

Status: prepared CPU-only code and tiny native fixtures. No production pack,
label corpus or raw Stockfish source has been scanned by this auditor. Its
qualification status must not be used until a fresh successful label terminal, the
independent all-label audit and the paired builder terminal are sealed, and
this auditor completes every shard and final readback.
The source pins bind the fresh full07 worker, authorization and builder v8.
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
cgroup counter is physical host I/O, not process logical bytes. The runner
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
qualification publication outside the auditor checkpoint root.
These tests do not measure production runtime or prove actual WSL power-loss
behavior. Production execution remains held until plan/source review and the
shared heavy-I/O lease are explicitly scheduled.
