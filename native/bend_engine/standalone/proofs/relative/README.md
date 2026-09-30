# Relative table-address bounds

Six additive source contracts connect actual U32 prefix-plus-relative-index
addition to allocation bounds, complete-tree routing and public array framing.
No production runtime or prior proof changes. Bounded opt-in commands:

```sh
bun native/bend_engine/standalone/proofs/relative/verify.js /path/to/pinned/bend --report /tmp/relative-proofs.json
CC=clang-18 bun native/bend_engine/standalone/proofs/relative/verify_native.js /path/to/pinned/bend --report /tmp/relative-native.json
```

`verify.js` runs the unchanged 83-law/156-control parent followed by the full new
six-law/20-control gate: 89 laws and 176 controls. These numbers describe the
required successful aggregate, not a claim that every development command ran it.
`focused.js` alone explicitly excludes inherited-gate execution. Its optional
`--controls-only` mode also excludes the source consumer, emits `control_gate`
instead of `focused_gate`, and cannot pass the combined wrapper. Use that mode
only for mutation-driver development.

## Actual expressions and domains

`Spec.address(k,r)` is U32.add of the proved prefix and r. For k<128 and
r<the actual Tables-derived block size, the laws establish
`prefix(k) <= address < prefix(k+1) < 131072` and exact U32-to-U64 widening of
the addition. `lookup_address_bounds` derives r's bound from the actual
Sliders.pext_index for arbitrary U64 occupancy; callers do not assume it.

For two keys i<j<128 and valid interior offsets, addresses are strictly ordered.
On an actual complete depth-17 affine array, the existing complete-tree theorem
then derives separated normalized routes. `lookup_other_block_write` proves the
complete returned read pair: the updated array and the query's original value.
Its caller supplies only complete shape, ordered valid keys, occupancies and the
write value, not assumed separation, index bounds, or desired read results.

`Add` uses structural Word induction with an invariant relating addition and
subtraction carries. The finite cases are local Boolean full-adder combinations,
not enumerated addresses. `Order` proves unsigned transitivity and its Nat
observation bridge by Word induction and a finite local comparison table. Existing
prefix and actual mask-population proofs provide the endpoint certificates;
there is no new host-generated prefix table or assumed arithmetic theorem.

## What is not proved here

The source address uses the certified prefix expression. Equality of every
stored metadata header to that prefix, the whole fill loop's clear-write
certificate, final computed table contents and independent blocker-ray lookup
refinement remain separate obligations. Complete shape is necessary; capacity
alone does not exclude ragged source aliases. Source equalities do not establish
physical pointer identity, allocation success or native lifetime safety.

The native probe deliberately goes further operationally: it builds the actual
table, reads its real prefix/mask headers, computes PEXT and addition, writes one
address, and observes the other query and write-back. An independent signed-ray /
BigInt reference supplies expected values only to the verifier. 288 cases cover
all 128 keys, zero/max compact indices, both cross-block orientations, and
same-block/same-address controls. Every mode repeats these cases. The probe
observes metadata and selected values, not every returned cell; tests do not
establish the remaining quantified header/content theorems.

The gate requires exact `All terms check.` with exit zero. Twenty controls include
ten affected-refinement failures and ten manifest/import/output guards. Crashes,
timeouts and missing-file errors are not ordinary semantic rejection. The pinned
checker is not modified; disposable Base copies test actual arithmetic mutations.
Semantic controls additionally reject inference/malformed-program errors;
actual Base mutations must reach the named new refinement rather than an earlier
Base contract. Compiler stderr remains strict. Select Clang only on the native command, not on
the Python development install (which needs its usual OpenMP build environment).
