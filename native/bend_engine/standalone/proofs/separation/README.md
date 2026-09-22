# Normalized storage-path separation

Opt-in continuation of #821. Four source contracts add value-sensitive framing
for actual `Array.set/get` and the actual `Tables.fill` loop. No production code,
compiler input, previous proof or ordinary test budget changes.

```bash
bun native/bend_engine/standalone/proofs/separation/verify.js /path/to/pinned/bend --report /tmp/separation-proofs.json
bun native/bend_engine/standalone/proofs/separation/verify_native.js /path/to/pinned/bend --report /tmp/separation-native.json
```

`verify.js` runs the unchanged 64-law storage gate, retaining all 90 prior
controls, then the four-law/17-control `focused.js` gate. The complete aggregate
accepts 68 laws and 107 controls. `focused.js` alone labels its report
`focused_gate`, explicitly says `inherited_gate_run:false`, and never claims the
aggregate. It is for bounded development feedback, not full qualification.

## Precise domain and observation

`Spec.separate(a,i,j)` first observes the actual array constructor shape. It
computes the source API's size from the left spine, applies the API's U32 mask to
both indices, and compares the two routes through that shape. It is false for a
shared leaf and true at a branch divergence. The predicate contains **no read
values, expected outputs or equality witnesses**. Numeric inequality is not a
substitute: 0 and 2 alias in a two-cell array. Source arrays may be nonuniform;
a differing normalized integer alone is not assumed to identify a different leaf.

`write_preserves_other_location` proves the **entire** returned read pair equals
the actual updated array paired with the original query value, given separation.
`write_preserves_separation` proves any write leaves all separation decisions
unchanged. `Spec.clear(shape,n,at,j)` checks each of the `n` actual U32-incrementing
write addresses against query `j`. `fill_preserves_unwritten_location` proves
that a true clear condition preserves that query's value through actual fill;
`fill_preserves_separation` proves fill leaves the path decisions unchanged.

The loop theorem quantifies over arbitrary Nat counts, U32 addresses and slider
parameters, including address wrap. It does not assume mathematical addition is
interchangeable with U32 addition. The consumer supplies concrete positive-count
certificates, a U32_MAX-to-zero wrap, a nonuniform tree, and overlap counterexamples.

Structural induction proves the frame statement for the previously certified
storage model, and existing `Roundtrip`/`Representation` lemmas connect it to the
real affine APIs. Only the proof witness duplicates the Data image. It does not
change production ownership or turn source equality into pointer/lifetime safety.

## Evidence and limits

Nine new controls require ordinary semantic/refinement failures; eight preserve
manifests/import policy and exact successful checker output. Missing files,
crashes and timeouts do not count as semantic rejection. The compiler/checker is
unchanged; successful checking requires exactly `All terms check.` and status zero.

Native checks reuse the unchanged storage probe, comparing every output cell
with a flat-array / BigInt / signed-coordinate reference. They test real APIs,
not the proof representation or proof predicates. Complete bounded arrays are
native-tested; arbitrary and nonuniform shapes are source-proved only.

The `clear` precondition is a computable, satisfiable **frame condition**, not an
assumed correctness theorem. Deriving it from all actual chess prefix intervals
is unfinished. U32 prefix no-overflow, allocation capacity, all initialized final
slider entries and independent blocker-ray lookup equality remain separate P2
obligations. No model/GPU, training, engine strength or speedup claim follows.
