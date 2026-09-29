# Bounded address normalization

This opt-in increment follows PR #826. It proves that every U32 address below the
actual depth-17 capacity (131072) survives the API mask, not only block endpoints.
The numeric mask is injective within that domain. Actual affine read/write wrappers
and the existing path-separation predicate can therefore use the original indices.

```bash
bun native/bend_engine/standalone/proofs/normalization/verify.js /path/to/pinned/bend --report /tmp/normalization-proofs.json
bun native/bend_engine/standalone/proofs/normalization/verify_native.js /path/to/pinned/bend --report /tmp/normalization-native.json
```

The full gate retains the unchanged 74-law/126-control prefix gate and adds five
laws plus 14 controls. `focused.js` checks only this increment and says so explicitly.
Successful proof checking requires exit zero and exactly `All terms check.`. Six
semantic controls require normal expected/observed failures, including mutated real
Base read/write masks in disposable compiler copies. Eight controls guard the law,
proof and import manifests, unsafe output, holes, foreign inputs and symlinks.

`Bound.bend` reasons directly over unsigned Word comparison, avoiding normalization
of huge unary numeric bounds. The fixed machine mask is the actual U32 expression
`U32.sub(131072,1)`. `Actual.size` certifies the complete result of the real Array.size
API against the existing shape observation, using certified affine reification.
No caller supplies a desired read equality. Capacity and address bounds are explicit.
The consumer checks real Array.new capacity for arbitrary seeds, boundary/interior
addresses, injectivity use, and the excluded endpoint that aliases zero.

**Limits:** distinct normalized integers are not yet proved to reach distinct tree
leaves. This is not a balanced-tree route-injectivity theorem, nor a proof that every
prefix-plus-lookup addition stays in range. Those remain necessary to derive all
clear-path certificates and complete the table-content/ray-lookup refinement.
The API correspondence is about complete source array values, not physical pointers,
allocation success or lifetime. Arbitrary shapes with the observed capacity are
permitted; that capacity alone does not imply a balanced tree.

The native probe tests supported public Array wrappers on actual 131072-cell
allocations. It observes initial/post-write queries, capacity and mask values, not
every returned cell. The 74 rows include 64 in-range cases and ten out-of-range
wrapper cases; six malformed requests fail in each of four build modes. The oracle
uses independent unsigned remainder and seed/write expectations. No proof model is
executed by the candidate. Direct native calls to internal Array.get.go/swap.go
failed with `an open Array element type`; the original diagnostic/source is retained
in the experiment evidence. Those internal routines are source-checked, not claimed
native-qualified. No compiler change or error suppression was made.
