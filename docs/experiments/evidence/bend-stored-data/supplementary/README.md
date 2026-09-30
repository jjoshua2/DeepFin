# Supplementary actual Chess.slide composition

This review extends source commit `1ee5d0534a3ac8c2ed3514141f0e89be58ef500b`
(tree `2b0dc51432a9f609a8bf7ede86846ed20978f2cd`) without changing its three
public laws, earlier proofs, production functions or qualification counts.
These are checked derived source statements, not additional registered LAWS
obligations or a newly executed combined aggregate. They are archived as text
so a subsequent continuation can promote them deliberately with their own gate.

Compiler remains `jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`,
84 inputs, fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

## Checked statements

- `Route.model/actual` transfers actual `Chess.slide` to the actual data read,
  given explicit mask/prefix header certificates. Both returned components are
  preserved. This helper alone is conditional on those header certificates.
- `Pipeline.actual` derives those headers and the computed data from the actual
  allocation/table/extras operations and existing proof producers. Its caller
  does not assume correct incoming headers, data, relative-index bounds or a
  desired return value. The result contains the complete final updated array.
- `Canonical.masked_extract/state` uses the accepted U64 and compact-index laws
  to identify the selected recurrence state as the occupancy masked by the
  actual relevant mask. No ray-geometry assumption is introduced.
- `MaskedLookup.actual` composes these into the source-to-source equality:

```text
Chess.slide(actual_pipeline, selected_key, occupancy)
  = (actual_pipeline,
     Tables.slider(selected_key % 64, selected_key >= 64,
                   occupancy & actual_relevant_mask, False))
```

The domains are symbolic allocation depth equal to 17, arbitrary seed and U64
occupancy, selected key `before + start_key`, table-call budget
`before + 1 + after + start_key <= 128`, and extras count plus start at most 64.
The table call begins at the certified prefix for its starting key. Complete
shape, actual stored-header values and data bounds are derived, not additional
caller hypotheses. Universal parameters include the full table/extras budgets;
there is no separately normalized closed `Tables.build()` source proof here.

All four modules completed with status zero and exactly `All terms check.`.
Recorded times were 0.517, 281.710, 178.093 and 294.555 seconds respectively.
`source-receipts.json` retains exact original command, output and timings.
The complete 113-file transitive proof-source closure was checked for regular
files, exact hashes, and no unsafe/foreign/hole dependencies. Its SHA-256
manifest is in the downloadable review archive. The four new source files and
all their hashes are preserved here; inherited sources remain at the exact
qualified commit above.

## Newly executed native and mutation checks

`review_native.py.txt` is the complete external reproduction driver. It calls
actual `Tables.build()` once per execution and repeatedly invokes actual
`Chess.slide` while threading the returned array. The candidate receives only
key/occupancy inputs, never expected masks, prefixes, arrays or attacks.
The reference independently walks signed file/rank rays with blocker inclusion.

Generic and UBSan builds each passed 1,024 query rows over all 128 keys and six
invalid requests. These are selected queries, not full-buffer comparisons, four
modes, 1,024 guaranteed-distinct inputs, or exhaustive arbitrary occupancies.
The modes repeat fixtures. Common output SHA-256:
`051545922146762d9c772fb4c7de6334ed42b4c583c0b9611885dd7b8ed48d86`.

A disposable mutation adds one to the actual stored prefix used by
`Chess.slide_offset`. It compiles and executes, then the independent native
reference rejects row 0, key 0, occupancy 0: observed `0 16843009 16843010`,
expected `0 16843009 16843262`. The same mutation fails source checking at
`Route.model` with ordinary expected/observed diagnostics, status 1. Its
1,455-byte diagnostic SHA-256 is
`6be31abeac7d65a7ba3f412ed32791dd7b70078cb6d4f4defdc647175fc4c51b`.
Neither rejection inflates the main public source-control count.

The final portable driver completed successfully in 6.749 seconds, using Bun
1.4.2 and Clang 17. Source and compiler identities were checked before and after.
`native-review.json` retains exact outputs, mode counts and identities.

## Reproduction

In a disposable checkout of the qualified source above, copy the five archived
`*.bend.txt` files to `native/bend_engine/standalone/proofs/lookup_review/`,
removing only their `.txt` suffix. Verify the file hashes below. Then:

```sh
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts \
  native/bend_engine/standalone/proofs/lookup_review/MaskedLookup.bend
python /path/to/review_native.py.txt /path/to/checkout /path/to/pinned/bend \
  --bun /path/to/bun --cc clang --report /tmp/lookup-review.json
```

The `.txt` driver is ordinary Python source. It performs bounded temporary
builds and disposable mutations and deletes those temporary directories. It
never modifies the original checkout or compiler. Source checking accepts only
safe exact success; the mutation must produce an ordinary type-refinement
failure at the intended route statement, not crash or fail to find an import.

## Exact archived source hashes

| File | SHA-256 |
| --- | --- |
| Route.bend.txt | b2b6293c93eca3be2d91d37dae87437a8a1d213f2fd68ad9582dd31f6b907341 |
| Pipeline.bend.txt | 7d123ca77391aaaa630f7e04f4f8df793efb9c6f22b806e81f1ba7c7536b4aee |
| Canonical.bend.txt | bbc3e499a1afb0a8e795487bc3fe941a03c56b9f30e3a4aca5b40b1f81d339f9 |
| MaskedLookup.bend.txt | 66b17f7fc2dca58a20dd51118baa4c91758246fe0d1f09c2ab8e34e13152afad |
| probe.bend.txt | e40bd95607b7dbc62db84107d59642b0d589e5d3f18d99e190e5cf24a87691e9 |
| review_native.py.txt | e4a9c4983cf024ce168e0047818d74f9a610887313e2a4c9db49b6e269caaaff |

Transitive proof-source manifest SHA-256:
`c1c7aaa24318d2aa77032185f95bf12f67e5798bfc20a436cbffd73fc535ff1a`.
Native report SHA-256:
`8f9030728b5455739b46e801e77707960411dc4d0d406ebc4c0f228ee1784830`.

## Failures and remaining obligation

Draft attempts with a computed nested pattern, a compiler-identity text/JSON
mismatch and an output-path collision failed before valid native qualification.
They were corrected before the final successful run; none is counted as a valid
wrong-behavior rejection. The main three-law development's earlier affine-use,
proof-product and stopped-run receipts are also retained in the downloadable
review archive. There is no claim that invalid drafts passed.

**Still unproved independently:** the production relevant mask and masked
`Tables.slider` value equal a separately defined blocker-ray specification for
every square and occupancy. The native geometry reference tests that behavior
but is not a universal source theorem. This record establishes the actual lookup
composition, not the final independent P2 geometry requirement.

Self-review only, not independent review. The pinned checker/Base, native
lowering, allocation/ownership/lifetime, ABI, toolchain, libraries, OS and
hardware remain trust boundaries. No production source, old accepted proof,
compiler, model/GPU, routine perft, training or benchmark workload changed. No
additional Python application responsibility moved into Bend; export/references,
data/control/training and transitional C++/LibTorch/AOTI remain dependencies.
