# BT4 in-memory root-policy stepper — September 23, 2026

This slice joins the reviewed BT4 root-output adapter and outcome policy in a
batch-friendly, in-memory game state. It is an interface and correctness check
for future opt-in generation, not an active generator. It adds no CLI, shard
writer, replay schema, real model inference, or played game.

`BT4RootPolicyStepper` requires an explicit `outcome_mode`. The historical
`theoretical_wdl` mode keeps the original six-man training-label convention,
including decisive cursed wins and discarding a game if a material probe is
missing. The future `rule50_match_v1` mode requires a caller-owned handle from
`open_strict_match_tablebase`; it checks six-man WDL/DTZ capacity at
construction. The caller must separately retain the verified table-file
identities; a path and table counts alone do not attest the files. A missing
eligible WDL or DTZ probe raises and fails the run rather than quietly
selecting a different accepted-game subset.

`prepare_roots()` checks natural and claimable endings before Syzygy, and
covered six-man outcomes before inference or the ply cap. In rule50 mode,
cursed wins and blessed losses are draws. A decisive WDL at a zero halfmove
clock resolves; a decisive WDL at a positive clock discards the whole game as
`rule50_unresolved` because earlier history can permit a future repetition
claim. Every completed or discarded in-memory game carries an outcome-mode,
path, capacity-count and handle-contract stamp; the stepper also exposes it
before any game is prepared. The stepper remains unwired to a manifest/writer.
The caller must configure the explicit
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
targets from each ply's mover's perspective. An unresolved cap or ambiguous
match result discards the whole buffered game before any labeled row is
returned. Separate counters expose completed/discarded games and rows and
discard reasons. The native teacher arrays remain unchanged by the outcome
choice; old experimental labels are not rewritten.

Focused CPU fake-evaluator/tablebase tests cover seven-to-six capture before the
next inference, natural draw claims, both mover colors, terminal roots, an
unresolved cap after buffering, immutable source inputs, policy ties, and
all-batch rejection, head drift, and repetition-mode drift without RNG
consumption. They also check rule50 WDL/DTZ classes, positive-clock abstention,
missing-probe failure before batch finalization, explicit mode and capacity,
and provenance stamps. These tests do not validate ONNX
weights, production throughput, corpus writer compatibility, or tablebase
coverage on a real game stream. The model hash in the adapter remains a caller
assertion; final model/provider and corpus provenance belong to future writer
integration.

This root-policy actor samples the retained teacher prior without BT4 search.
The outcome policy determines whether game rows can receive a final result;
it does not make neural BT4 predictions on seven-or-more-piece roots
Syzygy-informed. An optional searched BT4 actor would need the separately
reviewed rule-aware `SyzygyProbe` wired to its C search leaves. That route and
the SF/Ceres bootstrap sources remain separate integration work.
