# Complete-generator table preservation

This opt-in suite proves a storage property of actual `Chess` operations. It does
not implement a replacement generator or make a new legality/completeness claim.

## Public laws

`ordinary_scan_preserves_table` covers the actual ordinary scan with arbitrary
finite source-key lists, Boards and existing move-list tails.

`optimized_filter_preserves_table` covers actual `filter_prepare`, including its
initial check, full checked branch, blocker/ray preparation and optimized branch,
with arbitrary candidate lists and Boards.

`complete_generator_preserves_table` covers the actual full `legal_moves` call:

```text
returned_table(Chess.legal_moves(table, board)) = table
```

Every law quantifies over the actual `Array<U64>` datatype. No initialized-content,
array-depth/balance, valid-Board, unique-king, turn, rights or square-bound premise
is required for this **source storage equality**. This does not turn unsupported
native calls into validated inputs. The native fixtures use balanced arrays,
bounded source squares and Boards with moving kings.

The consumer explicitly specializes the full-generator result to the actual
symbolic initializer, with arbitrary seed and construction parameters. It does
not assume the resulting masks are valid; the accepted initialization/geometry
proofs remain separate and unchanged.

## Argument

`Bind` transports a returned pair through a verified equality of its array
component. `Query` reuses the accepted arbitrary-storage read proof for all six
piece classes used by the generator, including the queen's two queries and the
complete attack/check callback chain. `Scan` handles the actual piece decoder,
pawn targets, destination enumeration and list recursion. `Castle` handles both
true and false producer guards. `Filter` handles both branches of every optimized
filter decision. `Generator` composes the real scan, kingside, queenside and final
filter. `Lift` uses the accepted reification theorem to cover any actual array,
not only a caller-supplied specification array.

Internal continuations are proof functions, not runtime copies of the table.
The public statements assert equality of the complete table component; they do
not constrain the returned move list or establish physical read-only behavior,
pointer identity, allocation safety, concurrency safety or native lifetimes.

## Reproduce

Use the unchanged pinned compiler from standalone/toolchain.json and Bun 1.4.2:

```sh
BUN=bun python3 native/bend_engine/standalone/proofs/table_preservation/focused.py \
  /path/to/pinned/compiler --report /outside/checkout/focused.json
BUN=bun CC=clang python3 native/bend_engine/standalone/proofs/table_preservation/verify_native.py \
  /path/to/pinned/compiler --report /outside/checkout/native.json
```

The source gate checks the complete importing consumer and seventeen controls:
eight actual-source/refinement failures, eight manifest/import checks and one
synthetic warning-output unit. Missing dependencies, parser/ownership errors,
crashes and timeouts are not accepted as semantic rejection. A supplementary
queen-OR-to-AND mutation still passes the storage proofs, demonstrating that
storage preservation is not proof of correct attack answers. It is not counted
as a rejection control or new law.

The native probe enumerates every logical slot before and after the actual scan,
both castling producers, optimized filter and complete generator. It uses five
balanced-array contexts: one leaf, 8 patterned slots, 512 patterned slots, the
real initialized 131072-slot table with an unused marker, and 131072 patterned
slots that are deliberately not a valid attack table. The driver independently
checks the raw patterns and compares every post-stage slot to its pre-operation
value. For empty input tails it also compares ordered move lists from an explicit
scan/castles/filter composition and actual `legal_moves`; this is same-code
pipeline parity, not an independent chess oracle.

Each build mode repeats 29 context/Board/tail requests from eight base cases,
with 2,638,112 U64-slot comparisons. The contexts do not multiply the number of
distinct Boards. Three actual-code corruptions alter unused slot 131071 at the
scan, unchecked-filter or full-generator stage; all move lists remain correct
relative to the unmutated baseline, while complete-slot comparisons reject them.
Mutation builds are generic only. Native observations establish logical-slot
values on these balanced fixtures, not the complete heap or arbitrary ragged
array layout. Bulk native output is streamed into temporary files; reports retain
snapshot hashes, observed move lists, counts and first mismatch diagnostics.
