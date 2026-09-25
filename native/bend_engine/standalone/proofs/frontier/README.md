# Accepted parser frontiers and fresh insertion

Opt-in continuation of the exact qualified parser-prefix suite (#881). No
production function, compiler input or earlier law is changed.

```sh
bun native/bend_engine/standalone/proofs/frontier/focused.js /path/to/pinned/bend --report /tmp/frontier-proofs.json
bun native/bend_engine/standalone/proofs/frontier/verify_native.js /path/to/pinned/bend --report /tmp/frontier-native.json
# Expensive complete inherited chain, not implied by modular evidence:
bun native/bend_engine/standalone/proofs/frontier/verify.js /path/to/pinned/bend --report /tmp/frontier-all.json
```

## Actual state invariant

`Spec.safe` imposes no Board invariant on a rejected raw Layout. When its actual
validity flag is true, `Spec.state` requires file0..8, rank0..7, the existing
independent Board partition, and no color occupancy in any unvisited square.
`Spec.pending` defines those squares independently using Nat file/rank tests:
lower ranks, plus the current rank at or beyond the file cursor. It contains no
expected output Board, assumed trace, returned piece value or freshness witness.

The initial state uses the real `Position.empty()` at file0/rank7/valid=true.
A caller-supplied true flag alone is not sufficient: the consumer preserves
counterexamples for an overflowing initial file and preoccupied upcoming square.

## Four public laws

- `layout_preserves_frontier`: arbitrary actual String traversal preserves a safe
  input frontier, including paths that become invalid.
- `accepted_prefix_has_frontier`: if a full concatenated initialized placement is
  accepted, its selected actual prefix has bounded cursor, consistent Board and
  empty unvisited region. The full-parse acceptance premise is explicit.
- `live_typed_insertion_is_fresh`: after any initialized prefix, a typed actual
  piece transition with a true output flag has an input target below64 and a mask
  fresh with respect to the actual input Board. No input freshness or bounded
  cursor is assumed by the caller.
- `accepted_placement_has_consistent_board`: the actual optional placement result
  contains a partition-consistent Board whenever it is Some. `None` is represented
  explicitly, not silently replaced by an arbitrary default Board.

All four quantify over actual Strings and imported Position functions. The last
law is a result predicate; it does not claim every string is accepted or supply
an expected parsed Board. The consumer checks an accepted nonempty complete
placement and final-square freshness as well as rejected overfull ranks and raw
invalid temporary Boards. Rejection need not roll back the temporary Layout.

## Proof construction

`Fresh` proves bit-mask containment and untouched-region preservation by Word
induction and transfers through the existing exact actual insertion bridge.
`Facts` proves the finite bounded cursor arithmetic, mask inclusion/disjointness
and target bounds inside Bend. Its explicit cases are scalar certificates, not a
host-generated table of expected Boards. `Finite` derives the U32/Nat connection
from earlier machine-word proofs. No mathematical unbounded arithmetic is assumed
for arbitrary U32 input cursors.

`Step` uses the unchanged actual parser dispatch. Digits shrink the unvisited
mask; valid slashes move to a lower rank after a completed rank; valid pieces
occupy a fresh square and remove it from the mask. Invalid states remain invalid
by the existing parser law. `Traversal` inducts over the real String recursion,
then combines initialized safety with accepted-prefix validity and the actual
optional-result finalizer. Complete-state behavior is inherited, not replaced by
an oracle-generated trace.

## Test scope and trust

The focused command checks the four obligations and importing consumer, then
requires eight ordinary refinement failures, eight manifest/import-policy
failures and one synthetic warning-output rejection. That unit check is not
another compiler execution. Two implementation mutations reject in existing
actual-transition bridges; they are not mislabeled isolated new frontier failures.
No missing imports, affine/termination errors, crash or timeout counts as an
accepted semantic rejection. Safe success means exit0 and exactly All terms check.

The native probe executes actual initialized `layout_char` and `layout_result`
at every character boundary; the proved String induction links this traversal to
actual `layout`. It imports no proof specification. The independent driver uses
64 square sets and a cursor interpreter, compares every actual Layout field,
and checks the frontier and pre-insertion target on actual observed states.
Freshness applies only to live typed transitions. Invalid intermediate Boards are
retained and compared, not assumed consistent.

Native strings are CLI Unicode without embedded NUL, length<=256, at most32 per
invocation. Fixtures repeat across modes, not disjoint or exhaustive strings.
Two malformed probe-budget cases are rejected; this is not full FEN validation.
The prior parent suite and its malformed fixtures are unchanged.

These laws establish initialized placement partition consistency and fresh
accepted piece targets. They do not yet establish the intended complete square
contents for all strings, six-field FEN metadata semantics, legal positions,
reachability, transactional rollback of raw Layouts, or move application. They
also do not fix the older structural snapshot lowering or closed-builder limits.
Self-review only. Checker/Base, native lowering/storage, ABI, toolchain, OS and
hardware remain trust boundaries. No model, training, search, benchmark or perft
workload is added.
