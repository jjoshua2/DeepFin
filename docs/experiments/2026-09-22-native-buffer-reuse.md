# Native policy-map and inference-buffer reuse

## Scope and preregistration — September 22, 2026

PR2 of the neural-search scaling plan, on PR1 #828 at
`ba18279fffdb13beb894e1d4f49cd60601f38583`. No training, live configuration changes,
merge, deployment, search-policy change, compiler upgrade, CUDA or strength claim.

Hypothesis: immutable policy-map construction, float-list marshaling, and repeated
bridge input-tensor allocation can be removed without changing any selected-leaf
input, legal policy mapping, probability, completion count, or chosen move.

Control: the parent's list-based numerical converter and existing selected-leaf
oracle; reuse the exact saved untrained CPU-F32 batch-one package from PR1's
qualification artifact rather than exporting another model. One resource-bounded
hosted CPU qualification, two Torch threads, one Bend runtime thread, and small
local compiled probes. No arena, training or production GPU compute.

Success requires unchanged numerical tolerances; bit-identical list/array numeric
cases; rejection of malformed capacities/counts/nonfinite outputs; stable addresses
across changing inputs and repeated roots; zero per-forward bridge input-tensor
allocations after startup; unchanged material UCI/perft; quiet/diagnostic raw traces
and search results identical. Any functional mismatch blocks qualification. No
speed threshold is claimed from this fixture or shared runner.

## Ownership and implementation

The Bend application holds a linear `NativeEvaluation.Cache` across go, stop,
position changes and ucinewgame. Its maps are built once for the neural process;
material mode has a disabled cache and does not construct them. Only the geometric
mapping tables persist. Legal entries and history-dependent input are fresh on every
leaf. Full input capacity is cleared before encoding, including unused cells.

`ModelBuffers.bend` declares a narrow synchronous effect. Input and output buffers
are consumed by the effect and returned after the model finishes. The C transport
checks logical input size separately from exact physical capacity, requires the
pinned compiler's packed scalar representation, and rejects aliasing. It never
constructs per-float input/output lists. `LogitReply.convert_buffer` checks capacity,
exact output count and all 1,861 finite values before legal gathering; unused output
capacity is not part of that shape. It returns the output array for the next call.

The C++ bridge creates one model input tensor, then copies new input into that tensor.
It still copies AOTI outputs into the distinct Bend workspace and validates their
shape/device/dtype. The saved package contract and fail-closed behavior are unchanged.
No promise is made about allocations inside AOTI, legal-policy arrays, temporary
history encoding, or Bend's generic runtime. This is not zero-copy or async-safe
multi-slot scheduling. No buffer may be reused before synchronous completion.

Detailed per-leaf protocol diagnostics are now explicitly opt-in. The existing
external neural verifier enables them; ordinary native play need not serialize a
path and a pair of numbers for every legal move. Rule-draw notices and final PR1
work accounting remain available. Raw traces stay separately opt-in, exclusive-create,
owner-only, and outside the normal execution path.

## Local checks

The unchanged pinned Bend 2.0.21 + U64 compiler accepts the complete application.
Normal and UBSan compiled probes each pass 16 transport round trips (both 146- and
175-plane widths, diagnostics off/on), seven transport/configuration failures,
11 bit-identical list/array logit cases and four additional capacity/needs failures.
The test-only transport backend checks every input value and actual input/output
pointer reuse. It is not linked by any engine build and is not a model fallback.

The array-logit cases poison spare capacity with NaN; valid logical outputs still
pass. In-range NaN/infinity and missing/extra logical outputs still fail. Original
numeric probe files and their tolerances are unchanged.

Local full C generation exceeded the 4 GiB container budget; no compiler pin or
code-generation settings were weakened. Hosted qualification supplies full native
execution evidence when complete. Local self-review is not independent review.

## Reproduction

Use the repository's locked CPU development environment and verified compiler.
After emitting C from `buffer_probe.bend`, link it with `buffer_probe.c` and
`-DDEEPFIN_BEND_NATIVE_MODEL`; do not use that test backend in the application.
Emit/compile `logit_probe.bend` and `buffer_logit_probe.bend` without that define.
Then run:

```sh
python native/bend_engine/standalone/verify_buffers.py \
  --transport /path/to/buffer-probe --array-logits /path/to/buffer-logits \
  --list-logits /path/to/list-logits --report /tmp/buffers.json
python -m native.bend_engine.standalone.verify_neural \
  --checkpoint /path/to/fixture.pt --package /path/to/fixture.pt2 \
  --oracle /path/to/cboard-reference --report /tmp/neural.json \
  --command /path/to/deepfin-bend-neural --threads 1
python -m native.bend_engine.standalone.verify_reuse \
  --package /path/to/fixture.pt2 --reference /tmp/neural.json \
  --report /tmp/reuse.json --command /path/to/deepfin-bend-neural --threads 1
```

No permanent workflow, recurring model export, automatic native traversal, binary,
model package, or raw trace is part of this change.
