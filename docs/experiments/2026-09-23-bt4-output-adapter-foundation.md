# BT4 output-adapter foundation

The raw BT4 sidecar collector already converted each teacher policy row into a
legal, dense `lc0_1858` float32 label, but that conversion lived inside its shard
loop. The existing `legal_move_probabilities` helper provided only the Leela-index
gather and legal-move softmax. A future per-position teacher producer would have
had to repeat the compact-index scatter and its validation.

`compact_legal_policy` now shares that missing conversion. It returns the ordered
legal `chess.Move` objects, their float32 probabilities, and the dense float32
label. It reuses the existing Leela gather, checks compact-index uniqueness and
range, and checks finite, nonnegative, normalized probabilities. By default it
allocates a dense result; the current collector passes its already allocated row
as a validated writable, contiguous output buffer. The helper clears that row before writing
legal slots, including when it previously held unrelated values.

The collector still chooses the policy head by width or explicit name, requests
native WDL only under an explicit output contract, verifies history before
inference, and writes the same input/source identities and provenance. This
change does not choose a policy/value mixture, generate a new corpus, or wire a
teacher into a generator. Direct reuse by a future generator will also need its
own verified history/input, WDL head and POV, model/precision, and provenance
contract.

CPU fake-session tests compare the dense bytes and metadata against the prior
collector projection across both colors, castling, promotion, en passant and
nonfinite logits. They also cover invalid mappings, dirty and malformed output
buffers, history rejection before inference, reversed policy/value output order,
and native float64 WDL retention. The focused and neighboring BT4 tests pass
(119 cases); scoped Ruff and basedpyright report no findings. This is a
compatibility foundation, with no measured throughput or playing-strength result.
