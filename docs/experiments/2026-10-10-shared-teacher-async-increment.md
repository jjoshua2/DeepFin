# Shared teacher async plumbing: isolated first increment

This opt-in increment connects asynchronous row dispatch and immediate finite
game replacement to the existing BT4 worker. It also supplies a pipelined Ceres
JSON-line service function. Production defaults and the current owner are not
changed. No GPU, benchmark, model, dataset or training run was launched.

## Actual paths and reuse

`bt4_root_policy_worker.run_worker(..., shared_dispatch=(target_rows, max_rows,
batch_wait_ms, max_writes))` calls `run_pooled_games`. The worker CLI requires all
four `--shared-*` controls explicitly. The canonical per-game stepper retains
board history, encoding, legal policy sampling, native teacher targets and outcome
rules; the canonical raw NPZ writer retains its existing payload. The adapter
validates an existing raw orphan against that exact outcome before retrying its
checkpoint, and fsyncs directory renames before acknowledgment.

The new `TeacherDispatcher` is a narrow feed/head adapter. Its FIFO, condition
and gather/scatter structure follows `ThreadedDispatcher`, but it does not import
or claim literal reuse of the Torch dispatcher or selfplay manager. The existing
dispatcher owns Torch buffers, model updates and a two-head transport, so sharing
that hot path would expand this change. The teacher adapter instead accepts an
injected backend, bounds admitted rows, splits complete game requests across row
targets and scatters only complete ordered results. Its single backend thread
does not establish CUDA overlap. Fill waits are explicit and finite; flush drains
all queued rows without subsequent fill waits. Histogram keys are bounded by the
row target and recent queue waits retain at most 128 samples.

`BT4GamePool` owns one canonical stepper per game. IDs and RNGs bind to the finite
budget and seed, and a FIFO ready queue prevents fresh roots overtaking previously
prepared roots under backpressure. Slots refill only after durable commit. This
follows the manager's per-slot finalize/refill pattern; it does not import its
selfplay labels or its finalizer. Done outcomes remain held through commit errors.

`DurableWriter` is separately bounded and single-threaded. Unlike fatal backend
failure, an individual write failure is recoverable while its owning game remains
held. Finite close waits retain ownership on timeout rather than interrupting
fsync or silently releasing it. `CompletedGameLabels` owns completed-game results
until durable publication acknowledgment; it is used by the actual
`serve_ceres_stream` wire loop, rather than being an unused scheduler layer.

The Ceres service accepts the owner's immutable history loader, inference
adapter and atomic companion publisher as callbacks. It introduces no model loader,
root encoder or companion schema. It checks root/feed routing and all three raw
FP16 heads. Its response poll and fill deadline are separate controls. The current
production `Companion` client remains serial; adoption must pipeline that client
and supply the existing pinned callbacks. CPU tests pipeline both requests on a
socket before consuming either response.

Redundant backend and resume wrappers were removed, and raw publication/restart
science checks live in the existing worker rather than importing it back from the
pool module. Startup-cache work is outside this increment.

## Restart and STOP

STOP freezes actor admission and move/RNG application while completed publications
can finish. The pool API retains prepared work for in-process resume. The worker's
opt-in loop watches its output's `STOP` marker and returns a paused result.

Persisted restart is the `run_pooled_games(..., resume=True)` API, not a new CLI
resume mode. It binds committed checkpoints to launch SHA, seed, model, history,
temperature, ply cap, source/native code and table provenance before skipping any
ID. Unfinished games replay from their original initial board and per-game seed;
matching raw orphans reconcile without a second raw file.
Complete fsynced `.npz.writing` stages left by process death are checked against
exact replay metadata and arrays, then published exclusively without overwriting
the final file. Incomplete or mismatched stages remain preserved and fail closed;
automatic recovery from a partially written archive remains outside this draft.
Active native handles are not serialized. CPU mocks verify deterministic replay; actual ORT numerical
behavior across batching remains an independent qualification requirement.

## CPU evidence and remaining work

Focused integration and existing canonical regression tests exercise native
`history_rep_fix=True`, four-ply retained history, temperature sampling, exact raw
NPZ equivalence, finite replacement, STOP, checkpoint provenance rejection,
failure before publication and after raw publication, orphan replay, 300+300
pipelined label rows split as logical 512+88, exact raw three-head routing,
real process death after stage fsync, preservation of unknown incomplete stages,
low-volume draining, duplicate/backpressure/fatal coverage and bounded telemetry.

The actual actor entry point retains the existing normal 32-game and research
128-game budgets, with research capacity at most 64 games. Requesting a row target
of 512 does not produce 512 useful actor rows at that capacity.

The opt-in `run_pooled_units` API now shares one dispatcher and one bounded writer
across a frozen, finite roster of existing canonical units. Each unit retains its
local game IDs, SeedSequence, launch manifest, raw files and per-game checkpoints.
Unit names namespace async routing only. A rotating controller gives every unit
admission turns under row and write backpressure; aggregate live actors, rows,
units, writes and operation time have explicit bounds. STOP on the coordinator or
any unit pauses the entire roster before move/RNG application. Resume checks the
exact ordered roster, geometry and each unit's existing science contract.

CPU tests prepare distinct full 512/768/1024-root batches using 8/12/16 canonical
research units of capacity64, then STOP before application. Smaller completed
units establish exact raw/history/RNG parity, durable restart, post-publication
failure recovery, and asymmetric one/four-ply progress with row capacity1. The
roster is copied before control callbacks can mutate the caller's mapping. These
are CPU protocol profiles, with explicit fixture fill waits; they do not qualify
production timing, headroom or GPU throughput. The API has no production CLI or
resource-owner binding. Existing CUDA evaluator proof publication targets one
output; shared-session proof binding across units remains caller integration work.

Logical cross-game Ceres batching does not itself change the injected backend's
physical batch shape; the frozen fixed32 backend remains fixed32 unless a
separately reviewed shape adapter is supplied. CPU tests do not establish teacher
fit, VRAM, CPU headroom, fairness under production load, throughput or playing
strength. Real dual-owner wiring, production Ceres callbacks/client adoption and
exclusive GPU qualification remain before deployment. The full distributed
selfplay/replay/train smoke is not run under this task's no-training-launch scope.

## Isolated physical Ceres callback increment

`ceres_raw_backend.py` retains the reviewed parameterized own-root adapter from
`ceres_parameterized_fixture_backend_20261010.py` (source SHA256
`bda20a7e844fb3da68764e5d62294599e8f984ad20b4897c0f7cd5d4392df44b`).
The inference implementation retains its packing/gather/scatter path, with modern
imports, a boundary type annotation and legal-index validation moved to inference.
Unused actor-selection/encoding code
and static model/batch constants were removed from the label-only module; a model
constant would not enforce session identity. No model/session loader or second
encoder was introduced. The original frozen fixed32 runtime remains unchanged.

`bind_ceres_raw_backend` adapts its outputs/receipt return value to the existing
pipelined service callback, retaining four fixed successful validated inference
accounting fields. Failed attempted session calls are excluded; these fields
cannot serve as a compute-budget meter. Physical32 is the default; physical256/512
require an explicit argument.
The adapter preserves uint8 TPG packing, repeat-last tail padding, the independent
legal move oracle, immutable prepared board/history checks and full FP16
policy/value/value2 heads. The caller supplies the existing admitted session and
gather functions; session/provider/source proof validation remains its obligation.

CPU fake-session tests run the actual service with two300-row game requests,
logical512+88 dispatch and physical32/256/512 calls. Exact ordered game/root/feed
identity and all three raw heads reach the publication callback; padded rows do
not become labels. Invalid geometry, legal oracle mismatch, board mutation and
wrong head dtype fail closed. These synthetic tests do not open ORT, models or
data and do not qualify actual GPU packing, numerical parity, fit or throughput.
The prepared GPU correctness probe remains blocked by the canonical exclusive
training lease (errno11). Production immutable-history loader/publisher, client
pipelining, shared CUDA proof and resource admission still require owner binding.
