# BT4 in-memory root-policy stepper — September 23, 2026

This slice joins the reviewed BT4 root-output adapter and outcome policy in a
batch-friendly, in-memory game state. It is an interface and correctness check
for future opt-in generation, not an active generator. It adds no CLI, shard
writer, replay schema, real model inference, or played game.

`BT4RootPolicyStepper.prepare_roots()` checks natural and claimable endings,
six-man theoretical adjudication, missing tablebase coverage, and the unresolved
ply cap before exposing any root to inference. The caller must preflight the
two Syzygy directories once per worker. It must also configure the explicit
`history_rep_fix` process mode before constructing native boards. The stepper
checks that mode at construction, before preparation/encoding, and before
consumption, and checks the same mode on every evaluator output. It never
changes the process-global flag. Prepared boards and float32 inputs are
copies, ordered by private slot IDs and generation. An external caller may pass
`PreparedBatch.inference_inputs()` to
`BT4OnnxEvaluator.evaluate_roots(boards, x_batch)` and return its outputs with
one explicit temperature per slot. The stepper does not select a temperature.

`apply_root_outputs()` checks the complete batch's identity, teacher tensors,
legal compact-policy mapping, output contract, and temperatures before drawing
from any RNG or changing a board. Reused batches, altered prepared copies, and
wrong-order outputs fail. The named heads and native WDL dtype remain fixed
across later plies. Equal legal probabilities are valid. At temperature
zero, the existing shared sampler chooses the first maximum in legal-move order.
The native float32 T=1 teacher policy and native WDL stay separate from the
sampled actor move and final game WDL.

Preparation stages all outcomes and encoded roots before changing game status,
counters, or batch generation. If a later root fails to encode, a retry still
returns an earlier terminal game's result exactly once.

Each buffered ply retains an immutable, exact float32 root input with its full
encoded history, raw `input_key`, source fingerprint, teacher output, move,
side to move, and temperature. The input bytes are retained so a future writer
can reproduce the observation even after board history advances; a FEN alone
cannot do so. The raw float32 input key must not be conflated with a later
quantized corpus key. The stepper keeps full board history while playing, but
does not retain a full board copy per ply. A resolved outcome backfills WDL
targets from each ply's mover's perspective. An unresolved cap or missing
six-man probe discards the whole buffered game before any labeled row is
returned. Separate counters expose completed/discarded games and rows and
discard reasons.

Twelve CPU fake-evaluator/tablebase tests cover seven-to-six capture before the
next inference, natural draw claims, both mover colors, terminal roots, an
unresolved cap after buffering, immutable source inputs, policy ties, and
all-batch rejection, head drift, and repetition-mode drift without RNG
consumption. These
tests do not validate ONNX
weights, production throughput, corpus writer compatibility, or tablebase
coverage on a real game stream. The model hash in the adapter remains a caller
assertion; final model/provider and corpus provenance belong to future writer
integration.
