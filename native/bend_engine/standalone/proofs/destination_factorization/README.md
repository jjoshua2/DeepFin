# Actual destination factorization

This is a checked structural foundation for the ordinary generator. It factors the
actual `Chess.destinations` through the actual `Chess.bit_squares` and
`Chess.put_move`, then connects that emission to an independent move-block encoding
and full-Ply multiplicity specification. It does not assert independent bitboard
inventory, uniqueness, legal destination geometry or whole-generator correctness.

The importing `consumer.bend` checks four public contracts:

```text
destinations(n,empty,bb,src,pawn,ep,emit(keys,src,pawn,ep,tail))
  == emit(bit_squares(n,empty,bb,keys),src,pawn,ep,tail)

destinations(n,empty,bb,src,pawn,ep,tail)
  == Spec.expand(bit_squares(n,empty,bb,Nil),src,pawn,ep,tail)

Spec.count(destinations(n,empty,bb,src,pawn,ep,tail),query)
  == Spec.tally(bit_squares(n,empty,bb,Nil),src,pawn,ep,query,Spec.count(tail,query))

scan_after(src,pawn,ep,tail,(table,targets))
  == (table,Spec.expand(bit_squares(64,is_zero(targets),targets,Nil),src,pawn,ep,tail))
```

All statements use the real optimized `legal_probe/Chess.bend`. The first three
quantify over arbitrary Nat budget, Bool empty flag, U64 bitboard, U32 source and
EP square, Bool pawn flag and arbitrary Ply tail. The first additionally allows an
arbitrary U32 key accumulator. Counts compare all four fields of an arbitrary Ply
query and retain tail multiplicity. These statements need no bound, consistency,
validity, legality or table-initialization premise. The last statement quantifies
over an arbitrary affine `Array<U64>` and uses the actual scan's fixed 64-step budget
and consistent initial empty flag. Existing checked `Representation.reify` derives
the internal complete Cells witness; callers supply no certificate.

The order is a right fold: the actual square scan prepends popped low bits to its
accumulator, while destinations prepends each move block to its tail. No sorting or
permutation claim substitutes for the exact list equality. The independent block
specification uses accepted `promotion.Spec.choices` for rank 0/7 pawn destinations;
`Emission.put` reuses the checked `promotion_choices_exact` body. Other destinations
emit exactly `(src,dst,0,pawn && dst==ep)`. Promotion contributes exactly qp 1,2,3,4
with flag 0, and **no qp=0 ordinary occurrence**. `Count.promotion_excludes_zero`
proves this for arbitrary other query fields. No new law declarations, axioms,
unsafe annotations, proof holes or compiler changes are introduced.

An independent full-bitboard count theorem cannot quantify over arbitrary budget
and arbitrary empty flag: budget zero returns the unchanged tail even for a
nonempty bitboard; empty=True likewise stops immediately. `zero_budget_stops`
checks a concrete actual counterexample boundary. A future inventory theorem must
use budget 64 with `empty=is_zero(bb)`, or sufficient budget plus consistent empty,
and prove bounded coverage and distinctness of the actual `bit_squares` result.
Ordinary destination geometry, filtering, skip completeness, history, public-builder
and native lowering obligations remain separate. Trust remains the existing pinned
checker/normalizer and Base, including its U64 operations.

## Reproduce locally

From repository root with the unchanged pinned compiler source checkout and Bun:

```bash
export BUN="${BUN:-$HOME/.bun/bin/bun}"
nice -n 10 python3 -m native.bend_engine.standalone.proofs.destination_factorization.qualify \
  "$BEND_COMPILER_SOURCE" --report /tmp/destination-factorization.json
python3 -m unittest native.bend_engine.standalone.proofs.destination_factorization.test_qualify
```

Set `BEND_COMPILER_SOURCE` to a checkout matching `standalone/toolchain.json`
(`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`). The gate verifies the existing 84-file
compiler fingerprint, checks the transitive closure for unsafe/foreign imports,
preserves all 249 inherited inputs, records exact output and source hashes, and
requires unchanged sources before PASS. Each checker call is at most 180 seconds
(controls 120), below the user limit 900. Child affinity and numerical thread
settings are capped at two; cumulative checker CPU is bounded at 5400 seconds per
gate invocation. Repeated development invocations must also fit the task's total
budget; the experiment receipt accounts for them. Outputs use explicit paths
outside the checkout.

Five controls mutate actual ordinary flags, promotion rank, EP tags and bit-square
keys, or disconnect the symbolic consumer. Only semantic expected/observed errors
at the intended obligation count; timeout, backend, ownership, kind, warning and
missing-file failures receive no control credit. Failed starts invalidate stale
PASS receipts. Wrapper checks are also run under -O and -OO.


## Actual arbitrary-mask scan_after counts

`AfterCount.bend` composes the actual `Chess.scan_after` pair with the checked
full-Ply destination tally and PR1000's actual `bit_squares` arbitrary-U32
frequency and below-64 range theorems. It preserves the exact input table and
retains caller-tail multiplicity. See the importing consumer and the dated
[scan_after count record](../../../../../docs/experiments/2026-10-03-bend-scan-after-count.md).


## Closed full-Ply occurrence count

QueryFactor.bend closes the arbitrary-mask query count. For arbitrary targets,
source, pawn flag, EP square, full-Ply query, exact input table and arbitrary tail,
the count of scan_after is:

    count(scan_after(src,pawn,ep,tail,(table,targets)), query)
      = count(tail,query)
        + pick(in_range(query.dst) && test_bit(targets,query.dst),
            query_emission_count(src,pawn,ep,query), 0)

query_emission_count compares every non-destination Ply field against the
actual query. It enumerates promotions 1 through 4 with flag 0 for pawn
back-rank destinations; otherwise it compares the ordinary promotion-0 move
with flag Bool.to_u32(pawn && dst == ep). The gate is an actual destination
whose bit is set. The proof uses PR1000's all-U32 bit_squares frequency theorem
and range result, and it preserves the tail as Spec.count(tail,query), so
duplicates remain multiplicative. The result is composed through
Count.destinations, Emission.destinations, and #1006's exact-table
AfterCount.scan_after_count theorem. It adds no board-validity, target-geometry,
or full legal-move correctness premise.
