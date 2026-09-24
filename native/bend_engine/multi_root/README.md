# Bounded multi-root search (PR5a-PR5c)

This is an explicit **CPU-F32, fixed-batch, offline cohort runner**, not a replacement
for the UCI application. It owns 1–16 independent Bend search trees, visits roots
in rotating order, gathers at most one leaf per root per sweep, executes the existing
native batch backend and returns each result to its original tree/ticket.

```sh
bash native/bend_engine/multi_root/build.sh /tmp/new-cohort \
  /path/to/checkpoint.pt2 /path/to/libtorch/share/cmake /path/to/verified/bend
DEEPFIN_BEND_MODEL_PACKAGE=/path/to/checkpoint.pt2 \
  /tmp/new-cohort/build/deepfin-bend-multi-root --threads 1 -- \
  4 2 0 0 'startpos' 'startpos moves e2e4 e7e5'
```

Arguments after `--`: maximum completed simulations per root (1–256), maximum
search depth (1–32), per-root real-neural-row budget (0 disables, otherwise 1–256),
and diagnostics (0/1), then one quoted position per root. Use `startpos` or
`fen <six FEN fields>`, optionally followed by `moves <complete legal history>`.
All positions/history are validated before loading the model. Roots use fresh
4,096-node arenas and immutable input order IDs; the runner exits after the cohort.
Model load/binding must match the existing trusted CPU package contracts. One process
binds one package; incompatible models, encodings or dtypes cannot share a batch.

Each gather pass stops at the physical batch size or the end of the available
root sweep. Partial batches dispatch immediately; there is no wait-to-fill heuristic.
Visited roots rotate behind unvisited roots. Leaf histories, legal entries and
inputs are reconstructed independently; transpositions do not merge paths. Completed
and rule-draw simulations cost no neural row. All real output logits are checked
for finiteness before any neural row is backed up. A failed or malformed model
call terminates with error rather than emitting a successful cohort summary.

`cohort_root` records per-root completed simulations, accepted/executed neural rows,
rule-draw replies, used nodes, stop code and budgets. An unmet neural budget remains
explicitly false: a terminal root, depth/simulation limit or exhausted arena is not
an equal-neural-budget comparison. `searched_move=false,bestmove="0000"` means no
expanded continuation exists, including adjudicated roots that still have legal
moves. This is an offline result, **not a UCI bestmove protocol** or an implicit
legal fallback. Diagnostic mode prints all tree nodes and selected-leaf paths/replies.

`cohort_work` uses `deepfin.multi-root-work.v1`, distinct from PR1's UCI schema but
with the same real/physical/accepted-work definitions. One backend call is not one
simulation; physical padding is never accepted. Forward calls and batch histograms
are global, while accepted rows belong to roots. Do not sum aggregate and per-root
rows. Failure exits have no successful summary. There are no cancelled or in-flight
rows in the default synchronous mode. The opt-in async mode reports cancellation
separately and drains all admitted physical work before its final summary.

The millisecond-resolution cohort clock begins after position validation/model
loading, before shared policy-map/buffer initialization; it includes the first
inference (no excluded warmup) and optional per-leaf diagnostics, but excludes final
node/report formatting. Gathering and normalization/backup are composite CPU phases.
Queue wait and H2D/GPU/D2H timings remain null. These numbers are not an external
command-to-decision benchmark, fixed-wall comparison or steady-state EPS study.

## Explicit qualification

`qualify.sh GENERATED_RUNNER.c ORACLE_BINARY NEW_OUTPUT` compiles test-only callbacks
at five batches and both input widths, plus two UBSan configurations. It compares
all final tree bits and decisions with batch one. `test_backend.c` is never linked
into the LibTorch product. `verify.py` validates actual selected paths/tickets,
CBoard input bits, every real logit, legal priors/WDL, all final tree fields and
real/physical accounting against independent existing references. For real models,
supply both `--package` and `--checkpoint`; otherwise it expects the test callback.

```sh
python -m native.bend_engine.multi_root.verify \
  --binary /tmp/new-cohort/build/deepfin-bend-multi-root --oracle /path/to/oracle \
  --batch 4 --channels 175 --package /path/to/checkpoint.pt2 \
  --checkpoint /path/to/checkpoint.pt --report /tmp/cohort-model.json
```

No production defaults change. PR5b adds async polling/cancellation and PR5c adds
per-root deadline controls below. Live root arrival/removal, persistent self-play
integration, service-time-driven admission/bucket selection, trained-model CUDA
qualification and same-tree concurrency remain later work. PR4's UCI decision
semantics do not apply to this separate headless executable. Fewer forward calls
are not a speedup or Elo claim.

## Qualification evidence

[The experiment record](../../../docs/experiments/2026-09-22-native-multi-root.md)
contains the completed deterministic/UBSan matrix, actual CPU selected-leaf model
checks, exact source/package identities and limits. Deterministic final trees are
bit-identical across batch sizes; real model outputs are qualified numerically,
not claimed bit-identical. The primary real batch-four cohort executes 34 accepted
rows in 9 forwards with 2 padding rows, compared with 34 singleton forwards.
This is a work-count observation, not a measured speedup.

## Optional asynchronous cohorts (PR5b)

The same build wrapper includes a CPU batch worker, but synchronous execution is
still the default. Enable asynchronous control processing at launch:

```sh
DEEPFIN_BEND_MODEL_PACKAGE=/path/to/checkpoint.pt2 DEEPFIN_COHORT_ASYNC=1 \
  /tmp/new-cohort/build/deepfin-bend-multi-root --threads 1 -- \
  64 8 32 0 'startpos' 'startpos moves e2e4'
```

Send newline-terminated commands on stdin while the cohort runs:

| Command | Behavior |
| --- | --- |
| `isready` | Returns `info string cohort_ready` when processed, including during a pending forward. |
| `cancel 1` | Cancels root 1 (IDs are immutable input order, 1 through root count); other batch rows remain live. |
| `stop` | Cancels all remaining roots; final reports follow physical drain. |
| `quit` | Cancels all remaining roots, stops accepting controls and exits after physical drain/reporting. |

Cancellation is irreversible for this cohort. Duplicate cancellation is harmless.
Accepted tree work stays banked; an undispatched cancelled root consumes no row.
A pending cancelled row still executes and counts as wasted rather than useful.
Malformed commands are acknowledged as errors without changing state. EOF stops
further control reads but does not implicitly cancel; a no-stdin batch invocation
still completes normally. There is no live root add/replace command in this slice.

The native worker snapshots inputs and delivers a batch only after physical
completion. Bend owns row-to-root identities and filters cancelled rows before
normalization/search resume. Every real raw logit is checked before any neural
backup, including logits of cancelled rows; cancellation cannot hide backend or
nonfinite-output failure. There is one slot, not overlapping model execution.

PR5b introduced `deepfin.multi-root-async-work.v1`; PR5c extends it to
`deepfin.multi-root-async-work.v2` with the deadline fields below. The synchronous
schema is unchanged. `cohort_root` gains `dispatched_real_rows`, `cancelled_rows` and
`cancel_requested`. `cancelled_rows` counts discarded admitted evaluations, not
cancel commands or undispatched roots. On a successful final report:
`dispatched = executed = accepted + cancelled`, with padding separate and no
unresolved work. Budgets are not refunded. Failed runs have no successful summary.
Do not sum per-root and aggregate row counts.

Control acknowledgments may precede physical completion. Unlike UCI `stop`, this
runner does not emit an early final tree/bestmove; root summaries are produced
after drain. Its final clock includes that wait, first inference and setup, not an
external decision-time measurement. Async phase/queue/GPU/transfer timings are
null. Encoding, gathering, backup and output backpressure can still delay controls;
quit cannot preempt a wedged callback. No GPU or hard-latency guarantee is implied.

Use `--asynchronous` with the existing model verifier to require the new mode and
schema. Complete serial-versus-async/control qualification is explicit:

```sh
bash native/bend_engine/multi_root/qualify_async.sh \
  /path/to/generated/probe.c /path/to/oracle /tmp/new-async-cohort-check
```

This runs deterministic sync/async matrices and blocked-callback controls with the
actual generated coordinator and worker, plus normal/sanitized worker checks.
Real models are separately checked by `verify --asynchronous` using the exact
checkpoint/package. The [PR5b record](../../../docs/experiments/2026-09-23-async-cohort.md)
contains completed results and limits. Real no-cancel model tests and deterministic
cancellation tests are distinct evidence; neither establishes trained/GPU speed.

## Per-root deadlines (PR5c)

In asynchronous mode, `deadline ROOT MS` sets a root's timeout relative to the
moment the command is processed. ROOT is the existing 1-based cohort ID; MS is an
integer from 0 to 3,600,000. Zero expires immediately. A later command may shorten
an existing deadline but cannot extend it. A completed, manually stopped or expired
root cannot be revived. This is a control command, not a startup wall-time budget.
Synchronous mode and its report schema are unchanged.

The coordinator checks deadlines before gathering, after gathering/compaction and
before processing each returned neural row. Expired preadmission rows are removed
and the surviving inputs compacted in stable physical-row order. They consume no
forward reservation. Already admitted rows must still finish physically and are
charged as cancelled/wasted without reaching normalization or Search.resume.
Unaffected roots in the same batch continue, and previously accepted work remains.
All raw logits, even expired rows, still pass the whole-batch validation gate.

Timers continue after stdin EOF and between buffered commands. Notifications use
`cohort_control deadline ROOT MS` and one `cohort_control expired ROOT`. A manual
cancellation does not later become a deadline-expiry reason. Final summaries wait
for physical drain, exactly as in PR5b; expiry is not an early UCI bestmove.

Async final reports now use `deepfin.multi-root-async-work.v2`. Each root adds
`deadline_offset_ms` (null or due time relative to the cohort clock's origin) and
`deadline_expired`. The aggregate adds `deadline_expired_mask`, with bit ROOT set
for each expired root. `cancel_requested` retains its logical-stop meaning and is
true for both manual cancellation and expiry; use `deadline_expired` to distinguish
them. Existing counters keep their units and all admitted rows remain charged.
The verifier accepts historic async v1 reports, but rejects unacknowledged or
contradictory deadline data. Do not compare final drain time with time-to-decision.

Individual encoding, normalization/backup and output operations remain synchronous,
so no exact hard deadline or maximum stop latency is promised. A model callback is
not preempted. This does not add dynamic roots, multiple forwards, live self-play,
measured dispatch choices, CUDA inference or a larger search arena.

After the explicit `qualify_async.sh` matrix, run the deadline-specific tests:

```sh
bash native/bend_engine/multi_root/qualify_deadlines.sh \
  /path/to/verified/bend /path/to/async-matrix /tmp/new-deadline-check
```

This checks actual timer/compaction functions and the compiled coordinator with
held callbacks in normal and UBSan modes. See the
[deadline experiment record](../../../docs/experiments/2026-09-23-cohort-deadlines.md)
for scope and completed evidence. Test-only held callbacks are never linked into
the model product. No GPU, throughput or Elo claim follows from functional passes.


## FIFO ready-root scheduling

The ready set and unvisited tail now use the owning two-list FIFO qualified in
PR #864. Gathered roots remain in their existing Local/Pending tasks until
retirement; completed tasks join the back after all unvisited roots. No root
is revisited within an available sweep. Cancellation/live-mask scans preserve
the queue split; flattening occurs only for final reporting. The initial
1..16-root admission cap, physical worker, tickets, deadlines, budgets and
output schemas are unchanged. This does not enable live admission or same-tree
concurrency, and it is not an end-to-end speedup claim.

Run `qualify_fifo.sh VERIFIED_COMPILER PINNED_PARENT.c NEW_OUTPUT` explicitly
for the deterministic native matrix and parent scheduling comparison. See
[the experiment](../../../docs/experiments/2026-09-23-fifo-cohort-integration.md)
for source/build identities, controls and qualification status.


## Matched FIFO timing

The opt-in `build_fifo_benchmark.sh` and `benchmark_fifo.py` compare exact qualified list/FIFO snapshots using the unchanged deterministic callback. See [the completed experiment](../../../docs/experiments/2026-09-23-fifo-runner-timing.md) for source hashes, commands, raw observations and measurement limits. Ordinary pytest exercises admission logic only; this is not a new production mode or real-model performance claim.


## Completion wait

The asynchronous cohort waits for native completion notification, with a one-millisecond requested wait budget before returning to Bend command/deadline service. Poll/take still owns retirement; notification never copies output or releases a batch slot. The [completion-wait experiment](../../../docs/experiments/2026-09-23-bounded-completion-wait.md) records validation and scope. `benchmark_wait.py` compares qualified callback runners with wall and child CPU observations; it is not an ordinary pytest workload or model/GPU benchmark.
