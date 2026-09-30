# Computed slider data through later blocks and extras

This opt-in suite composes actual computed fills, later metadata/fill framing,
and final extras. It leaves production code and every earlier accepted proof
unchanged. The candidate is based on #869, not a rewritten implementation.

```sh
bun native/bend_engine/standalone/proofs/data/focused.js /path/to/pinned/bend --report /tmp/data-focused.json
bun native/bend_engine/standalone/proofs/data/verify_native.js /path/to/pinned/bend --report /tmp/data-native.json
# More expensive complete chain, not implied by a focused or modular result:
bun native/bend_engine/standalone/proofs/data/verify.js /path/to/pinned/bend --report /tmp/data-aggregate.json
```

## Public contracts

`later_tables_preserve_data` preserves an arbitrary incoming value below the
current certified prefix but at or above 512, across the actual remaining
metadata writes and full fills. All addresses are bounded by 131072 and the
array must have complete depth-17 shape. No per-write path or value certificate
is supplied by the caller.

`stored_data_after_tables` establishes a selected block entry after all blocks
in a bounded call. The selected key is `before + start_key`; the call contains
`before + 1 + after` blocks, with that count plus start_key at most 128. The Nat
relative index is strictly below the actual block size. It returns the complete
actual final array and the computed value for the existing subset recurrence.

`stored_data_after_extras` establishes the same result after actual allocation
and the final extras loop. Its symbolic depth is certified equal to 17; the seed
is arbitrary and the extras square budget is at most 64. The actual allocation
and table operations supply complete shape, rather than assuming their result.

`Index` proves exact Nat-to-U32 conversion for these interior offsets and obtains
actual address bounds from the existing relative/prefix certificates. `Preserve`
proves metadata and later blocks cannot change the selected earlier data. The
`OneBlock`, `Stored` and `Pipeline` compositions reuse the actual fill, certified
array representation, previous table-loop equality and final extras preservation.

The expected value remains `Tables.slider` for the proved enumerator state.
Independent ray-geometry equality and the actual Chess lookup composition are
separate remaining obligations. The laws neither assume those results nor count
native comparisons as source proofs. Full returned-pair equality is not native
pointer identity or allocation/lifetime safety.

## Checks and trust

The focused gate checks all three obligations and an importing consumer, rejects
semantic mutations at named locations, and enforces manifests/imports/no unsafe
or holes. Status zero alone is insufficient: exact `All terms check.` output is
required. Crashes, missing files and malformed terms are not semantic rejection.
Controls-only results explicitly do not claim the consumer or inherited chain.

The complete wrapper invokes the unchanged 103-law header aggregate followed by
this suite. Modular qualification may instead retain exact-source parent results;
such a record must say so and must not claim a newly executed full wrapper.

The native wrapper newly executes the unchanged independent full-buffer verifier
from `headers/verify_native.js`, then requires a disposable later metadata write
to cell 512 to fail as a wrong-value mismatch. No new duplicate ray oracle or
proof representation runs in the candidate. Every original fixture and malformed
input is retained. Modes repeat fixtures, not disjoint or exhaustive input sets.
