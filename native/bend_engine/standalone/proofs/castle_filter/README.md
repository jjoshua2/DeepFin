# Full legal-move castling provenance

This opt-in suite connects the actual ordinary scan, both `Chess.castle_side`
calls, and every branch of the actual final filter to the complete returned
`Chess.legal_moves` list. It changes no production function.

## Six public contracts

`scan_excludes_castling` proves that scanning preserves the absence of flag-two
moves in its accumulator. Ordinary destination generation constructs only flags
zero or one, and promotion prefixes have flag zero.

`final_filter_preserves_predicate` preserves any supplied per-Ply property through
actual full-check and fast-filter paths. It is a provenance theorem, not a proof
that a skipped king-safety test was semantically unnecessary.

`both_castling_sides_provenance` and `filtered_castling_provenance` classify each
returned occurrence as an inherited input-tail value or a complete guarded
king-side/queen-side Ply. Arbitrary tails and duplicates remain supported.

`legal_castle_guard_and_route` applies to an actual flag-two member of the entire
`Chess.legal_moves` result. It derives one of the two exact producer guards and the
complete corresponding move. It does not assume absence from an intermediate
list: the actual scan owns an initially empty accumulator and cannot make flag two.

`legal_castle_preserves_representation` adds only input Board consistency and
concludes consistency of actual `Chess.make_move(b,m)`, reusing the previously
proved guard-to-rook-freshness and castling-update preservation results.

All source statements retain arbitrary actual affine tables and raw Board metadata.
No caller-supplied attack oracle, table correctness, output consistency, chosen
route, or rook landing freshness is used to manufacture the final result.
Continuation-based dependent proofs pass each real table once; they do not clone
arrays or assume two independent executions have the same results.

## Explicit boundaries

Source provenance does not establish independently specified king safety, correctness
of the fast filter's sensitive mask, metadata legality, historical rights validity,
legal reachability, or general move-generation soundness/completeness. It proves
representation preservation only for flag-two result members, not all move kinds.
Positive results concern the list, not full returned-array identity or native lifetime.

The consumer contains full symbolic uses of every law, a two-castle nonempty example
using a small arbitrary zero table, and an old external-tail counterexample. That
zero table is a satisfiability witness, not a correct attack-table certificate.

## Reproduction

With the compiler fixed by `standalone/toolchain.json`:

```sh
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts \
  native/bend_engine/standalone/proofs/castle_filter/consumer.bend
bun native/bend_engine/standalone/proofs/castle_filter/focused.js \
  /path/to/pinned/bend --report /tmp/castle-filter-focused.json
CC=clang bun native/bend_engine/standalone/proofs/castle_filter/verify_native.js \
  /path/to/pinned/bend --report /tmp/castle-filter-native.json
```

The focused gate checks six public laws and importing consumer plus seventeen
controls: eight semantic/refinement, eight manifest/import-policy, and one synthetic
warning-output unit (not another compiler execution). Safe source success requires
exit zero and exactly `All terms check.`; crashes, malformed imports, affine errors
and timeouts do not count as semantic rejection.

The native probe constructs actual current `Tables.build()` and executes three
stages per request: both producers with an arbitrary selected tail, those producers
followed by the final filter, and actual complete `legal_moves`. It reports ordered
move lists and all nineteen U32 fields of each actual child Board. The independent
reference uses a 64-square set model, coordinate movement/attacks, and a separate
slow king-safety filter comparison for the finite full-list fixtures. No reference
answer, proof predicate or proof representation is provided to the candidate.

Four modes repeat 402 distinct requests: generic, forced-portable, native-target
and UBSan. There are 11,699 child Board records (222,281 U32 Board fields), 292
full-list castling occurrences, and seven malformed requests per mode. Four
specific destination-attacked castles survive the intermediate producer but are
absent after both the final-filter stage and actual complete legal generation.
Fixtures are representation-consistent with one king per color, but not claimed
legally reachable. Raw metadata and arbitrary inherited tails remain diagnostic
inputs. Full-list equality on these fixtures is not universal legality/completeness.

Three actual-code mutants manufacture flag two in scan, invent a move on the fast
filter path, or bypass the final `filter_after` check. All must compile/run and then
fail independent values. The last also fails the unchanged continuation refinement
at `Filter.after_then`; that rejects changed branching, not an independently proved
meaning of `in_check`. No native counterexample is a discovered production bug.

The full inherited aggregate is not executed by this focused gate. Self-review only;
pinned checker/Base, native lowering/storage, ABI, toolchain and hardware remain
trust boundaries. No model, GPU, search, training, benchmark or perft increase.
