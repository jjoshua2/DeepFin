# Actual table-prefix arithmetic and schedule refinement

This opt-in increment follows #822. It leaves the original compiler, earlier
accepted laws and production `Tables` code unchanged.

```bash
bun native/bend_engine/standalone/proofs/prefix/verify.js /path/to/pinned/bend --report /tmp/prefix-proofs.json
bun native/bend_engine/standalone/proofs/prefix/verify_native.js /path/to/pinned/bend --report /tmp/prefix-native.json
```

The aggregate retains the unchanged 68-law/107-control parent before checking
six new public contracts and 19 new controls. `focused.js` alone is explicitly
labeled development evidence (`inherited_gate_run:false`), not the full aggregate.
Success requires status zero and exactly `All terms check.`. Semantic mutation
controls must fail with ordinary expected/observed diagnostics, not missing files,
crashes, timeouts or an unsafe warning with raw status zero.

## Actual source connection

`Spec.prefix` accumulates sizes from the existing independent geometric population
formula, starting at the actual 512-entry reserved region. `Facts` covers all
129 endpoints and 128 transitions inside the source checker; no host-generated
prefix answer table is supplied. Widened U64 addition independently observes the
same accumulation, keeping source comparisons away from enormous unary Nat
machine-limit literals. Endpoint bounds, strict transition order and exact widened
sums are checked, not inferred from native counts.

`Certified.size` obtains the already discharged mask-population law by importing
its original `PROOF.bend`. The public caller supplies only the valid key/count
domain, not assumed geometric size equalities. `Certified.make` constructs the
finite chain of certificates that `Builder.refine` consumes inductively.
`tables_follow_prefix` equates the complete actual `Tables.tables` result with an
explicit-prefix schedule that calls the same real array writes and `Tables.fill`.
It quantifies over arbitrary affine initial arrays and every Nat count/start pair
with `n + k <= 128`. It is not an ideal-array replacement or native test oracle.

The six laws cover cumulative widening, allocation endpoint bounds, actual-size
step arithmetic, ordered block endpoints, the full-array loop refinement, and
the exact logical endpoint 108,160. `blocks_ordered` quantifies over all ordered
key pairs; induction and numeric-order refinement compose the single-step facts.

## Limits

The source theorem remains general in the loop count; it includes n=128, k=0
without enumerating all buffer values. A proposed extra closed `Tables.build()`
equality and concrete full-count consumer exhausted a bounded local check during
construction. That extra closed theorem was never accepted and is not counted.
Its actual runtime behavior is tested independently by the native gate. No
previously accepted law is removed or weakened.

This schedule deliberately retains the actual fill implementation. Therefore its
full-array equality does not itself prove every computed attack, independent
blocker-ray geometry or safe separation of all interior write paths. The next
step is connecting the now bounded numeric intervals to normalized path/clear
certificates and using the existing frame and ordinal laws for final contents.
Endpoint masking facts alone are not that interior-path theorem.

Native checks compile only the original `Tables` operations and an IO probe.
They compare all 131,072 returned cells for zero, one, 64 and 128 slider-block
builds, plus actual `Tables.build()`, with a flat-array, independent coordinate-ray
and direct bit-deposition reference. Proof schedules and certificates never
compute the candidate's contents. Five executions and 655,360 cell comparisons
are repeated in each of four modes; they are not disjoint input sets or exhaustive
arbitrary-U64 coverage. Seven invalid requests are rejected per mode.

Source equalities trust the pinned checker/Base; they are not native pointer,
allocation, ownership/lifetime or full compiler correctness guarantees. No Python
application responsibility moves into Bend, and no perft, model/GPU, training,
strength or performance claim is added.
