# Actual fill interval frames

Six contracts extend the relative-address and complete-tree proofs to every
write in the actual `Tables.fill` invocation. No production implementation or
previous accepted law changes. Commands are bounded, opt-in qualification:

```sh
bun native/bend_engine/standalone/proofs/fill_frame/verify.js /path/to/pinned/bend --report /tmp/fill-frame-proofs.json
CC=clang-18 bun native/bend_engine/standalone/proofs/fill_frame/verify_native.js /path/to/pinned/bend --report /tmp/fill-frame-native.json
```

The aggregate requires the unchanged 89-law/176-control relative-address gate,
then this six-law/20-control gate: 95 laws and 196 controls upon success.
`focused.js` alone never claims the inherited run. `--controls-only` additionally
omits the consumer and emits `control_gate`, not `focused_gate`.

## Contracts and non-vacuous premises

`interval_clear` derives the existing `Separation.clear` predicate from complete
depth-17 shape, bounded query/end, a mathematical count-plus-start not exceeding
the end, and a query outside the half-open interval. The caller does not assume
that each path is separated or that a read is preserved. `interval_fill_preserves_query`
uses that certificate with the actual affine fill frame theorem.

`chess_fill_clear` derives the count relation and endpoint bound from the existing
actual-size/prefix proof producers. It covers every valid key and the full actual
block size. `chess_fill_preserves_query` applies this result to the same slider-fill
invocation made by the table builder, for any bounded query outside its interval.
`fill_preserves_later_lookup` derives the query's range/exclusion from ordered keys
and the actual PEXT index for arbitrary occupancy. `fill_preserves_reserved_query`
protects every query below 512 from a complete slider fill. No relative-index,
clear-path or expected-value equality is imposed on those last two callers.

Generic interval counts are mathematical Nat. The actual `U32.inc` is proved to
increase the observed Nat address while below 131072; the count invariant keeps
all writes within their interval. `AddNat` refines actual ripple addition when
its final carry is absent. The previously proved wide endpoint relation supplies
that carry certificate for every actual chess block, including its full count.
No U32/64 overflowing power or host-generated prefix table is used.

The consumer covers positive-count queries before/after an interval, arbitrary
full blocks, later-block lookups, and a reserved query on an actual allocation.
It keeps depth symbolic while the allocator supplies complete shape. First-write
and overlong-count counterexamples remain. Zero-count behavior is separately
checked. Generic interval end is strictly below 131072; the actual chess ends
satisfy this stronger bound. No arbitrary wrapping schedule is certified here.

## What the source conclusions do and do not say

The fill laws observe the value from a read on the actual returned array; they
are not a new full-returned-pair or physical-identity theorem. Earlier exact
returned-array and shape laws remain in the inherited gate. The frame theorem
works for arbitrary complete arrays, not only all-zero fixtures.

`Spec.block` is the actual slider-fill component, **excluding the two metadata
header writes** performed before it. Prefix/header equality of every final stored
entry, preservation through all other blocks/extras, each computed in-block value,
and independent blocker-ray lookup refinement are still distinct obligations.
A preserved reserved value is not a proof that its header was initialized correctly.
No model/GPU, training, perft increase, strength or speedup claim follows.

## Controls and native observations

Eight new arithmetic/schedule controls require an ordinary failure at the new
refinement. Two mutations of actual fill are caught by the already-accepted
`storage/Build.fill` body refinement; these are intentionally reported as
inherited-layer implementation sensitivity, not new-layer failures. Ten further
controls enforce manifests, proof-producer imports, no holes/foreign/symlinked
inputs and rejection of unsafe output. Timeouts, crashes, inference errors and
linearity errors are not semantic rejections. Success is exactly `All terms check.`
with status zero; no protected checker modification is allowed.

The native candidate imports only production Base, Tables and Text. It reads
actual builder metadata, then runs a full/one/zero fill on a seeded complete
allocation and observes the query, first/last block values and capacity. The
reference independently derives signed-coordinate masks/rays, ordinary integer
prefixes and direct BigInt compact deposition; no expected data reaches the
candidate. 1024 rows per mode cover all 128 keys, with 768 protected and 256
intentionally overwritten queries. All returned cells are not compared here.
Four modes repeat fixtures, not disjoint datasets or exhaustive native U64 coverage.
