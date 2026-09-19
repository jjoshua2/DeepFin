# Bend evaluator: bounded cross-search batching

## Scope and decision

Continue #778's composition work, based on
`4d5285eb016dec3f36f9eb0a1842c27d770eea97`, rather than tune perft.
The Bend compiler stays at `57bc84edc0df32780e2c4dde44e3a2ee1a500cc9`.
Bend chess/search code, production native code/configuration and perft depths are
unchanged. No merge, deployment, GPU run or trained checkpoint is involved.

Question: can several real Bend searches share a fixed-shape native evaluator,
retain their distinct histories, and resume correctly after cancellation without
adding within-tree parallel-search semantics? Batching belongs to the external
Python coordinator in this gate; each search remains a separate native Bend
process. The model executes in a persistent native C++ process, not Python.

## Contract recorded before hosted confirmation

Use the same fixed-seed **untrained** project TinyNet as #778, exported as a
batch-four CPU F32 package, one native forward per batch. Output rows compare
against independent eager **singleton** forwards at atol 2e-6 / rtol 2e-5.
This numerically tests padding and row independence for this model, not all models.

Five concurrent roots: start, Kiwipete, EP, black promotion and a root with
repeated pre-root history; budgets 8/12/4/6/10. Compare one-real-row padded control
to batches of up to four real rows, using the same package. Compare full tree
structure, boards, visits, best move and per-reply statistics. Each accepted
reply is independently checked against the diagnostic PUCT reference. This is
not production Gumbel search parity.

Inject cancellation at request three: one queued, one after native submission
but before scatter. The cancelled epoch must preserve exactly two completed
simulations. Restart the same Bend processes with epoch two; complete normal
budgets and match the control. Old in-flight rows must never commit into newer
epochs even when node/request IDs are reused. No native-kernel interruption claim.

Deterministic no-model tests cover FIFO, fixed-shape zero padding, partial flush,
capacity/backpressure, identity/replay, input ownership, atomic output validation,
queued/in-flight expiry, late replies and bounded session registration. Reserved
capacity includes cancelled native rows until their containing batch returns.
The C++ worker's malformed input checks run for both default batch one and four.

Run generic/portable/native/UBSan Bend modes. UBSan does not instrument LibTorch
or the compiled model. Retain #778's full single-row neural and session fault
qualification after modifying the native evaluator, not just the new happy path.
No latency/speed acceptance gate: inference call counts are not throughput.

Bounded coordinator: fixed batch four, eight reserved real rows, eight registered
session IDs, one native batch in flight, one pending request per search. Queue
flush age is 2 ms in the test driver, not a hard real-time guarantee: verification,
encoding and pipe parsing also run in the coordinator. Per-request deadline is
30 seconds; a group has a 120-second bound, native pipe IO has its own deadline.
Worker forward continues after per-row cancellation; stale outputs are discarded.

Budget: focused CPU local/hosted confirmation, 15-minute hosted job bound, one
compiler process at a time, two inference threads, single-threaded Bend peers.
No timing/perft run added to ordinary pytest. Existing native-neural workflow is
extended, not duplicated. Recovery: discard this isolated branch, preserving #778.

## Representation and interface

Version-one packages remain batch-one only. New batched sidecars use
`deepfin-tuple-policy-wdl-cpu-f32-batched-v2`, explicit static batch and an explicit
`row_independent` declaration. Supported static bucket sizes are 1/2/4/8/16;
actual model execution here qualifies 1 and 4, not every bucket. No dynamic shape,
automatic checkpoint conversion or cross-package mixing is added.

The binary header counts flattened policy/WDL values. Native tensor shapes must
match the declared static batch. The coordinator keeps row identity as immutable
(session, epoch, request, node) keys; the worker uses a separate monotonic batch
sequence. Inputs and action maps are copied on queue admission. Entire batch
output validation precedes any completion. Partial batches pad unused rows with
zeros, and padding can never produce a search reply.

## Local readout

Initial Clang 17 / Torch 2.10.0 CPU native smoke passed. Forty real requests used
40 one-real-row calls versus 20 batched calls in that run. Both had identical
complete search outcomes. Cancellation/recovery used 45 real native rows and
preserved two completed simulations in each cancelled epoch. Largest observed
logit difference from eager singleton: 2.9802322387695312e-8. These counts are
not a speedup: the control intentionally still executes a batch-four package.

Full local four-mode qualification subsequently passed: 68 search epochs (60
normal completions plus eight deliberately cancelled epochs), 500 real input
rows, max singleton-logit difference 2.9802322387695312e-8. All fill sizes 1/2/3/4
occurred in each mode. Six malformed batch-four native inputs and 80 cheap
contracts passed separately. The host confirmation is still pending.

## Limits and next decision

This is an external coordinator qualification, not a new Bend-internal scheduler
or a production service. Still unqualified: representative trained transformer,
real BF16/CUDA model, multiple outstanding requests within one tree, virtual loss,
root advance, draw history/adjudication, UCI stop and end-to-end speed. CPU row
independence of untrained TinyNet does not establish every model's batching rules.

Self-review only; no independent reviewer or universal proof. Use the trained
model/CUDA contract as the next gate rather than assuming fewer forwards means
higher throughput. See the native-neural README for reproducible commands.
