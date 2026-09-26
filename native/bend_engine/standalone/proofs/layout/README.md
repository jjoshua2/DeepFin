# Slider-mask populations and scalar sizes

This opt-in suite adds eight contracts without changing production runtime code.
It targets the standalone stack through PR #807. `LAWS.bend` imports scalar
observations in `Spec.bend`, which calls the actual `Tables.slider`. The size
observer is the exact `U32.shln(1, U32.to_nat(U64.popcount(mask)))` expression
currently in `Tables.tables`; the gate checks that source link explicitly.
`lookup_index_bound` calls the actual imported `Sliders.pext_index`.

## Accepted domain and guarantees

The domain is every U32 key whose mathematical Nat value is below 128. Keys 0..63
are rook squares, 64..127 bishop squares. Occupancy is any U64, not a sampled
subset. `Domain.recover` connects the complete Nat case split to arbitrary U32
keys through the existing numeric-observation injectivity proof.

| Contract | Guarantee |
| --- | --- |
| `mask_population` | Actual popcount equals the independent file/rank ray-length count. |
| `tight_population_bound` | At most 12 rook bits or 9 bishop bits. |
| `shift_in_range` | The actual population used as exponent is below 32. |
| `size_positive` | The actual U32 shift produces a nonzero block size. |
| `tight_size_bound` | Block sizes are at most 4096 for rooks and 512 for bishops. |
| `size_matches_capacity` | The shifted U32 size equals the previously proved mathematical capacity. |
| `full_index_bound` | Full U64 PEXT value is below that size, for every occupancy. |
| `lookup_index_bound` | The actual U32 lookup index is below that size, for every occupancy. |

`Cases.bend` proves the count identity by all 128 source-level cases. The checker
normalizes the real mask builder and independent count formula; there is no
foreign proof, precomputed answer table, or host certificate. This finite proof
is exhaustive over the accepted key domain. The occupancy laws use the prior
universal extraction theorem, not occupancy enumeration.

`Facts.bend` handles small geometric counts and bounded powers. `Low.bend`
structurally proves that a low-half projection cannot increase a mathematical
word value. `Consequences.bend` transfers these facts to the exact public
operations; its count parameter is discharged by `Cases`, not assumed.
The importing consumer retains all previous ordinal laws and witnesses
satisfiable corner, center, last-key and arbitrary-occupancy uses.

## Run

Use the verified compiler checkout, not the fork's upstream-only main:

```sh
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/layout/verify.js /path/to/bend --report /tmp/layout-proofs.json
BEND_NO_TELEMETRY=1 bun native/bend_engine/standalone/proofs/layout/verify_native.js /path/to/bend --report /tmp/layout-native.json
```

The source gate invokes the unchanged 40-law successor gate and its 42 negative
controls, then checks the new importing consumer. It requires both status zero
and exactly `All terms check.`. New semantic mutations target affected lemmas;
manifest/import rejection controls are separately labelled. The complete valid
case split is relatively expensive and is not added to ordinary CI or perft.
Each checker invocation is bounded; only one compiler process runs at a time.

The native probe imports neither the count specification nor any proof file.
An external signed-coordinate ray walk and BigInt bit gather supply the reference.
Generic, forced-portable, native-target and UBSan runs repeat the same bounded
fixtures. Their results are execution tests, not a whole-native-program theorem.

## Explicit limits

These are scalar obligations. No assertion is made here that every Array region
is initialized, that prefix offsets fit the allocated 2^17 entries, that regions
are disjoint, or that the actual builder/lookup equals a full blocker-ray model.
Population equality alone is not mask-geometry equality. The actual U32 lookup
index is bounded, but universal equality between its low projection and the full
PEXT value is not a new contract here. Native samples compare that equality.

The unchanged ordinal theorem still needs to be connected to actual affine
writes, offsets and lookup. Proofs trust the pinned checker and Base semantics;
native lowering/runtime/ABI/toolchain/hardware remain separate trust boundaries.
No Python responsibility moves into Bend, and no model, GPU, engine-performance,
training, deployment or complete-migration qualification is claimed.
