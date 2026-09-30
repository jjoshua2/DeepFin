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
contracts passed separately. Hosted confirmation is recorded below.

## Hosted readout: validation PASS

[Run 35407998499](https://github.com/jjoshua2/DeepFin/actions/runs/35407998499),
job **105801646990**, verified the exact implementation patch SHA-256
`43c25e5ee45e35d39210baaca494605a8527ca50e08041aabdbe10b673fef045`
before applying and testing it. The clean tested implementation is commit
`3e79d1e0a6e33932455bfe0e34609e6bd306df8f`, directly on #778's head.
This readout is a documentation-only follow-up.

Locked Torch **2.14.0+cpu**, Python 3.13.15, Bun 1.4.2. All four Bend modes
passed: generic C, portable U64, native target, and UBSan. In total the new
batching qualification covered **68 search epochs**: 60 normal completions and
eight deliberately cancelled partial epochs, with **500 real input rows** and
**337 native batch calls**. Maximum absolute logit difference versus independent
eager singleton inference: **5.21540641784668e-8**. All final control/batched
search outcomes and cancellation/recovery accounting matched.

Every mode exercised fill sizes 1/2/3/4. Normal up-to-four batching took 22
native calls for 40 real rows; the one-real-row control took 40 calls for those
same 40 rows. Both execute the same static batch-four package, padding unused
rows. This is a correctness contrast, NOT a throughput or batch-one speedup.

| Mode | Normal calls / rows | Cancellation + recovery calls / rows | One-real-row calls / rows | Peak reserved rows in fault case |
| --- | ---: | ---: | ---: | ---: |
| Generic | 22 / 40 | 20 / 45 | 40 / 40 | 6 |
| Portable | 22 / 40 | 20 / 45 | 40 / 40 | 6 |
| Native | 22 / 40 | 24 / 45 | 40 / 40 | 6 |
| UBSan | 22 / 40 | 25 / 45 | 40 / 40 | 5 |

Different fault-run batch fills reflect arrival timing, not different search
results. Both cancelled sessions retained exactly two completed simulations;
epoch two in the same native process then completed its normal budget. The
remaining searches completed normally, and no old result committed into a new
epoch. Queued and submitted-but-not-scattered cancellation are distinguished;
no claim is made that cancellation interrupted a native model invocation.

| Root | Completed | Nodes | Best private key |
| --- | ---: | ---: | ---: |
| Start | 8 | 165 | 1153 |
| Kiwipete | 12 | 543 | 204 |
| En passant | 4 | 24 | 772 |
| Black promotion | 6 | 33 | 3324 |
| Repeated pre-root history | 10 | 206 | 1153 |

These are untrained-model outcomes, not good-chess recommendations. Each accepted
reply and complete final tree is checked, not just the best key. Six malformed
batch-four native inputs also failed as expected. Deterministic queue-contract
checks include backpressure, expiry, input ownership and atomic failure handling.

The full original regressions remained green: **38 sessions per mode**, 207
python-chess oracle positions; and **40 batch-one neural epochs / 640 native
calls**, 64 encoding comparisons plus the distinct-history case, and six malformed
batch-one native inputs. Maximum batch-one logit difference was
`2.2351741790771484e-8`. Ruff, Basedpyright (zero errors/warnings), and all **80
cheap tests** passed: 27 new batching cases plus 53 inherited boundary/session
cases. No test suppression, numerical tolerance change or perft-depth increase.

The overall development workflow is red only because its final publishing step
was denied permission to modify the existing workflow file. All validation stages
had already passed. The connected GitHub tool subsequently published the exact
existing tested commit, without a code rewrite or an unnecessary validation rerun.
The review branch contains no temporary development workflow or patch payloads.
Broader PR-triggered checks are separate; no repository-wide green claim is made.

Evidence artifact: **bend-batching-confirmation**, ID **10572907809**, 30-day retention.
ZIP SHA-256: `46abf3b5fb4bf368ede4481d26573fdcebd00c8b60db7a4bc5e92e33631a0549`.
Batch report SHA-256: `f7438eb445e4d4af8c9c8afb0a8d975117e42ca3e5a70f8e175be6a1f36a9a25`.
Single-row report SHA-256: `6d9d01254b6bf3211a81d2b7ad85650ef5abc964b45534a4949aeb450a817e44`.
Session report SHA-256: `ec9721bb5791aa3a23e3c9ea763372ac6013628c4037b74542f6f284f0bf4bf2`.
Executed batch-four package SHA-256:
`e96de91df9e3a03ebe747783f27eff8391f1bddd1ea8f2a2936910624ca3df70`.
The package uses seed 20260918, root-legacy-meta/v2_threats/repetition-fix-on,
175 input planes and compact policy 1858; its manifest declares row independence.
Packages are disposable, untrained native-code test artifacts, not committed or
a reproducible-archive-hash promise. Rebuild only trusted code/packages.

## Limits and next decision

This is an external coordinator qualification, not a new Bend-internal scheduler
or a production service. Still unqualified: representative trained transformer,
real BF16/CUDA model, multiple outstanding requests within one tree, virtual loss,
root advance, draw history/adjudication, UCI stop and end-to-end speed. CPU row
independence of untrained TinyNet does not establish every model's batching rules.

Self-review only; no independent reviewer or universal proof. Use the trained
model/CUDA contract as the next gate rather than assuming fewer forwards means
higher throughput. See the native-neural README for reproducible commands.
