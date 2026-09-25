# Persistent live-root lifecycle — PR5d

This opt-in headless entrypoint keeps one process alive while adding, replacing
and removing roots. It is based on #874 at `42d4c78ebfca102e146720c818f5dba9db61fd57`,
including the owning FIFO, completion notifications and configurable arenas.
Existing fixed-cohort/UCI behavior and synchronous defaults remain unchanged.

[The September 25 record](../../../docs/experiments/2026-09-25-live-cohort-recovery.md)
contains completed native callback qualification and limitations. The September 24
candidate record is retained historically, not as this implementation's status.

## Ownership and protocol

The separate `live.bend` entrypoint starts with no roots and keeps the model,
policy tables and buffers alive. It requires `DEEPFIN_COHORT_ASYNC=1`; existing
entrypoints and synchronous defaults do not change. The native callback remains
the existing CPU-F32 fixed-batch backend. Only Bend owns roots, full histories,
legal-policy mappings, deadlines and tree updates.

There are at most sixteen occupied slots and one pending lifecycle operation.
Generations are process-global, monotonically increasing, never reused and bounded
at 65,535 reservations. Abandoned reservations still consume an ID. Internal tree
epochs remain bounded slot IDs; the external identity is `(slot, generation)`.
The extra generation is a control-protocol identity, not a change to Search's ABI.

Commands are one line each. `POSITION` has the same grammar as the fixed-cohort
runner: `startpos` or `fen <six fields>`, optionally followed by a complete legal
`moves ...` history.

```text
add POSITION
replace SLOT GENERATION POSITION
remove SLOT GENERATION
cancel SLOT GENERATION
deadline SLOT GENERATION MILLISECONDS
isready
stop
quit
```

`add` chooses the lowest empty slot. Replacement validates the complete new
position before cancelling the old root. Invalid positions leave the old root
unchanged. A successful request first announces `pending-install SLOT GENERATION`.
It announces `admitted SLOT GENERATION` only when installed. A second lifecycle
request while one is queued is rejected. Controls referencing an old or not-yet-
installed generation are rejected; a delayed cancel cannot stop a new generation.

**Replacement and removal commit only when no physical batch is pending.** The
old result is retired/reported before its pool is dropped or its slot reused.
If a row was admitted before cancellation, its execution remains charged and its
output cannot update the replaced tree. Unaffected roots keep their accepted work.
The new tree gets fresh counters, deadlines and stop/expiry/report flags. This is
a conservative full-batch barrier, not arbitrary overlapping generation storage.

`cancel` and `deadline` require an active matching identity. Deadlines use the
existing processing-time start and shortening-only rule. `stop` cancels current
roots and abandons a queued lifecycle change, but keeps intake open. `quit` does
the same and closes intake, joining outstanding work. EOF closes intake but lets
current roots and an already queued replacement finish normally. A finished root
continues to occupy its slot until explicitly replaced or removed.

## Arena storage and execution scope

`DEEPFIN_COHORT_ARENA_NODES` preserves the validated 1..65,536 logical-node contract,
with 4,096 the default and a power-of-two backing array of at least 4,096 slots.
Every replacement retains the configured capacity. Root reports identify logical
and physical capacity. There is one physical batch slot and one pending leaf per
root. Control handling overlaps inference, not multiple model forwards. FIFO sweeps
and the completion-notification wait are inherited; removal filters the queue only
at the physical drain barrier. Every installed generation gets a fresh tree.

## Reporting

Each settled generation emits `live_result_begin SLOT GENERATION`, optional
existing `cohort_node` rows, and one `live_root` object with schema
`deepfin.live-root.v1`. Node rows are scoped by that enclosing generation marker,
not merely their internal slot/epoch field. Automatic deadline notices retain
the existing slot-based form; clients must interpret them in stream order between
admission and result records.

A root record distinguishes accepted and cancelled physical evaluations, manual
cancellation and deadline expiry, completed simulations, budgets and the result.
No expanded continuation uses `searched_move=false,bestmove="0000"`, as in the
offline cohort contract; it is not UCI's bestmove or a silently invented fallback.

The final `deepfin.live-cohort-work.v1` record sums **all reported generations**,
including removed/replaced ones. Slot reuse does not refund prior computation.
It requires executed = accepted + wasted, physical = calls × fixed batch size,
and physical = real + padding. It is emitted only after drain. Idle time, setup,
first inference and drain are in the lifetime clock; useful EPS is deliberately
null. Do not treat this as an equal-wall-time or warmed-throughput benchmark.

## Component checks

From an isolated checkout at the declared base with this patch applied:

```sh
bash native/bend_engine/multi_root/qualify_live_registry.sh \
  /path/to/verified/bend /tmp/new-live-registry-check
python -m pytest tests/test_bend_live_cohort.py -q
```

The registry gate compiles/runs the actual Bend functions in normal and UBSan
modes. It covers sixteen-slot capacity, stale identities, 128 replacement cycles,
generation exhaustion and deadline/expiry reset. Two isolated source mutations
(ignore generation identity; retain an expired timer) must compile and then fail
with the intended invariant error in each mode. Mutants never alter actual source.
The Python cases are synthetic transcript checks, not model or search execution.

## Full-runner qualification

The following source-only gate generates both the unchanged reference and the
new live entrypoint, compiles normal/UBSan test executables at both input widths,
and exercises held-callback replacement/removal/stop/quit/EOF, capacity, stale
controls, timer reset, startup errors, failed callbacks and serial tree parity.
Its status file starts unqualified and cannot report success after failed
generation or a failed behavior check. Source-only CI invokes these gates with
freshly generated code; the dedicated qualification record distinguishes the
verified generated-C reuse in that run.

```sh
bash native/bend_engine/multi_root/qualify_live.sh \
  /path/to/verified/bend /tmp/new-live-owner-check
```

A separate CPU-F32 model build is provided; a build alone is not numerical
or lifecycle qualification of that model:

```sh
bash native/bend_engine/multi_root/build_live.sh /tmp/new-live-model \
  /path/to/trusted.pt2 /path/to/libtorch/share/cmake /path/to/verified/bend
DEEPFIN_BEND_MODEL_PACKAGE=/path/to/trusted.pt2 DEEPFIN_COHORT_ASYNC=1 \
  /tmp/new-live-model/build/deepfin-bend-live --threads 1 -- 4 2 0 0
```

No neural model was executed in this PR5d recovery. The native gates use
deterministic or explicitly held test callbacks. Trained-network fidelity, CUDA,
multiple batch slots, persistent self-play/data generation, adaptive dispatch and
playing strength remain separate work. Encoding, validation, backup and output
are cooperative operations; a wedged callback is not preempted. Self-review only.
EOF, stop and graceful quit must not be described as guarantees for all fatal
process exits. See the September 25 qualification record for exact evidence and failures.
