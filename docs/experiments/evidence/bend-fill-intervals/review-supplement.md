# Whole-fill interval supplementary self-review

The six public source contracts and their inherited aggregate are separate from
these supplementary checks. This is self-review, not independent review.

## Pipeline-to-metadata query composition

For reproduction, place the following snippet as `review.bend` beside the
new `fill_interval/LAWS.bend` in a disposable checkout. Place the later
cross-block snippet as `cross_block_review.bend` in the same directory.
The public proof/test sources remain unchanged.

The following consumer imports the final candidate unchanged. It proves that
for every valid key and query below 512, a complete slider-block fill preserves
the query. Its shape certificate is derived from actual Array.new, Tables.tables
and Tables.extras; the caller does not supply shape, a clear-path condition or a
desired read equality. Depth remains symbolic until an explicit equality sets
it to the actual depth 17, avoiding concrete-tree expansion during composition.
This does not prove the initial metadata's value or every final lookup result.

Command: `BEND_NO_TELEMETRY=1 bun PIN/bend2/main.ts review.bend` using the exact
same 84-input compiler fingerprint and candidate dependencies.
Result: exit 0, exactly `All terms check.`; elapsed 290.043489 seconds.
Consumer SHA-256: `e2c6b106c32631d60cf7c0c444555eb42a3ee51dc5a3bc8f8ce7ae5b7d34ba42`.

```text
import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as Fill
import ../prefix/Spec.bend as Prefix
import ../prefix/LAWS.bend as PrefixLaws
import ../prefix/PROOF.bend as PrefixProof
import ../prefix/Nats.bend as Nats
import ../relative/Order.bend as Order
import ../complete/Spec.bend as C
import ../complete/Actual.bend as Complete
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S
import ../../Tables.bend as Tables

def lower(+k: Nat,p: {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool} & {U32.is_lt(Prefix.prefix(k),131072) == True{} : Bool}) ->
  {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool}:
  (lo,hi) = p
  lo

# Every metadata/extras query (<512) is protected by every complete slider fill.
def header(a: Array<U64>,+k: Nat,+q: U32,+valid: {Nat.is_lt(k,128n) == True{} : Bool},
  shape: {O.shape(a) == C.full(17n) : O.Shape},qb: {U32.is_lt(q,512) == True{} : Bool}) ->
  {S.value(Array.get(U64,Fill.block(k,a),q)) == S.value(Array.get(U64,a,q)) : U64}:
  Laws.block_fill_preserves_before(a,k,q,valid,shape,
    Order.lt_le(q,512,Prefix.prefix(k),qb,lower(k,PrefixLaws.prefix_in_allocation(k,Nats.weaken(k,128n,valid)))))

# Depth is kept symbolic while composing actual allocation and both real loops.
# Its final equality fixes the production depth without constructing 131072 leaves
# inside the intermediate proof. Shape is produced here, never supplied by caller.
def pipeline_header(+d: Nat,+n: Nat,+extra: Nat,+key: U32,+at: U32,+sq: U32,+seed: U64,
  +k: Nat,+q: U32,depth: {d == 17n : Nat},valid: {Nat.is_lt(k,128n) == True{} : Bool},qb: {U32.is_lt(q,512) == True{} : Bool}) ->
  {S.value(Array.get(U64,Fill.block(k,Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,d,seed)))),q)) ==
    S.value(Array.get(U64,Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,d,seed))),q)) : U64}:
  header(Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,d,seed))),k,q,valid,
    Equal.trans(O.Shape,O.shape(Tables.extras(extra,sq,Tables.tables(n,key,at,Array.new(U64,d,seed)))),C.full(d),C.full(17n),
      Complete.pipeline(d,n,extra,key,at,sq,seed),Equal.cong(Nat,O.Shape,t => C.full(t),d,17n,depth)),qb)
```

## Behavioral sensitivity

A disposable mutation to actual Tables.fill adds a zero write in its zero-count
base case. The unchanged native verifier rejects generic-mode row 1: a query at
the first rook block's exclusive endpoint 4608 is overwritten instead of retaining
its injected value. This is an actual wrong-value mismatch, not a crash or parser
failure. The raw result is retained in native-overrun-mutation.json and the local
review package. It is not an additional public law or gate control.

## Reapplication

The source-only patch was applied in a fresh detached #833 baseline worktree.
The complete resulting tree is `1b7a58263f7f5c17c8acb88b9426bfe05b73279f`; all
269 native source entries match the inspected candidate. Both patch application
and the ordinary Git whitespace check passed. This verifies bytes, not an
independent proof review or native lifetime theorem.


## Audit of actual semantic-rejection diagnostics

A supplementary scratch driver added only raw diagnostic capture and reran the
19 new controls. All control records match the final candidate's control report.
All ten semantic results contain ordinary expected/observed equality failures;
none is a parsing, inference, missing-file, variable-consumption, stack-overflow
or crash result. The public driver and proof sources were not edited by this
audit. The complete raw diagnostics and result JSON are in the review package.

| Mutation | Actual first affected location |
| --- | --- |
| `increment-observation-stuck` | `inc_value` |
| `positive-budget-omits-first-write` | `current` |
| `nat-sum-one-ahead` | `u32` |
| `block-budget-one-extra` | `count` |
| `inclusive-before-query` | `before` |
| `inclusive-after-allocation-bound` | `after` |
| `clear-omits-first-address` | `before` |
| `clear-repeats-write-address` | `before` |
| `actual-fill-writes-next-address` | `../storage/Build.fill` |
| `actual-zero-count-fill-writes` | `../storage/Build.fill` |

Eight of these fail new arithmetic/interval refinements. The last two fail the
imported `storage/Build.fill` refinement, as the gate and readout already state;
they are not counted as sensitivity exclusively located in a new theorem. The
separate native overrun mutation above supplies an actual outside-query
wrong-value test. This supplementary audit adds no accepted law or gate count.


## Cross-block actual-lookup composition in both directions

This additional consumer composes the six new block-fill laws with the existing
relative-address and prefix proofs. For any two valid ordered keys, it proves
that a complete later-block fill preserves an earlier block's PEXT-indexed query,
and that a complete earlier-block fill preserves the later block's query. The
caller supplies complete array shape, both key-domain bounds and key ordering,
but no relative-index bound, clear certificate, or assumed queried-value equality.
The index is the actual Sliders.pext_index; the prefix is the source-certified
expression, not a newly established read of every stored offset header.

It passed source checking with exit 0 and exactly `All terms check.` in 289.555201 seconds.
All 74 existing source imports match the inspected candidate byte for byte.
These are supplementary composition checks, not additional public laws or
aggregate control counts.

Consumer SHA-256: `20a21a74c1f0ac090c0980d6a60573a2cbeca29afff57e6eb4a9e881031ce606`.

```text
import Base
import ./LAWS.bend as FillLaws
import ./PROOF.bend as FillProof
import ./Spec.bend as Fill
import ../relative/LAWS.bend as RelativeLaws
import ../relative/PROOF.bend as RelativeProof
import ../relative/Bounds.bend as Bounds
import ../relative/Spec.bend as Relative
import ../relative/Order.bend as Order
import ../prefix/LAWS.bend as PrefixLaws
import ../prefix/PROOF.bend as PrefixProof
import ../prefix/Spec.bend as Prefix
import ../prefix/Order.bend as PrefixOrder
import ../layout/Spec.bend as Layout
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S
import ../complete/Spec.bend as Complete

def endpoints(+i: Nat,+j: Nat,+iv: {Nat.is_lt(i,128n) == True{} : Bool},
  +jv: {Nat.is_lt(j,128n) == True{} : Bool},before: {Nat.is_lt(i,j) == True{} : Bool}) ->
  {U32.is_le(Prefix.prefix(1n+i),Prefix.prefix(j)) == True{} : Bool}:
  Equal.trans(Bool,U32.is_le(Prefix.prefix(1n+i),Prefix.prefix(j)),
    U32.is_le(U32.add(Prefix.prefix(i),Layout.size(U32.from_nat(i))),Prefix.prefix(j)),True{},
    Equal.cong(U32,Bool,x => U32.is_le(x,Prefix.prefix(j)),Prefix.prefix(1n+i),
      U32.add(Prefix.prefix(i),Layout.size(U32.from_nat(i))),
      Equal.sym(U32,U32.add(Prefix.prefix(i),Layout.size(U32.from_nat(i))),Prefix.prefix(1n+i),
        Bounds.endpoint(Prefix.prefix(i),Layout.size(U32.from_nat(i)),Prefix.prefix(1n+i),PrefixLaws.step_no_overflow(i,iv)))),
    PrefixLaws.blocks_ordered(i,j,before,jv))

# Complete later fill, actual earlier PEXT query; no clear or index-bound premise.
def later_fill(a: Array<U64>,+i: Nat,+j: Nat,+occ: U64,
  +iv: {Nat.is_lt(i,128n) == True{} : Bool},+jv: {Nat.is_lt(j,128n) == True{} : Bool},
  before: {Nat.is_lt(i,j) == True{} : Bool},shape: {O.shape(a) == Complete.full(17n) : O.Shape}) ->
  {S.value(Array.get(U64,Fill.block(j,a),Relative.address(i,Relative.index(i,occ)))) ==
    S.value(Array.get(U64,a,Relative.address(i,Relative.index(i,occ)))) : U64}:
  FillLaws.block_fill_preserves_before(a,j,Relative.address(i,Relative.index(i,occ)),jv,shape,
    Order.lt_le(Relative.address(i,Relative.index(i,occ)),Prefix.prefix(1n+i),Prefix.prefix(j),
      Bounds.inside(i,Relative.index(i,occ),RelativeLaws.lookup_address_bounds(i,occ,iv)),endpoints(i,j,iv,jv,before)))

# Complete earlier fill, actual later PEXT query, also deriving allocation bounds.
def earlier_fill(a: Array<U64>,+i: Nat,+j: Nat,+occ: U64,
  +iv: {Nat.is_lt(i,128n) == True{} : Bool},+jv: {Nat.is_lt(j,128n) == True{} : Bool},
  before: {Nat.is_lt(i,j) == True{} : Bool},shape: {O.shape(a) == Complete.full(17n) : O.Shape}) ->
  {S.value(Array.get(U64,Fill.block(i,a),Relative.address(j,Relative.index(j,occ)))) ==
    S.value(Array.get(U64,a,Relative.address(j,Relative.index(j,occ)))) : U64}:
  FillLaws.block_fill_preserves_after(a,i,Relative.address(j,Relative.index(j,occ)),iv,shape,
    PrefixOrder.trans(Prefix.prefix(1n+i),Prefix.prefix(j),Relative.address(j,Relative.index(j,occ)),endpoints(i,j,iv,jv,before),
      Bounds.lower(j,Relative.index(j,occ),RelativeLaws.lookup_address_bounds(j,occ,jv))),
    Bounds.allocation(j,Relative.index(j,occ),RelativeLaws.lookup_address_bounds(j,occ,jv)))
```


## Verified hosted provenance

All stages of hosted run **35790621404**, job **106957935441**, passed on the
exact candidate. Qualified publication commit:
`c54ad2f1080f11a1885a83d49429fd863bc5463a`. The full source aggregate checked
95 laws and 195 rejection controls, retaining the unchanged 89-law parent.
The six-law consumer was rechecked on its final bytes, including the documented
EOF whitespace adjustment after the earlier local command.

Artifact **10722639257**, `bend-fill-interval-qualification`, was downloaded
and verified against ZIP SHA-256
`095dae2e6dfe9395694d03d9c7dc2694c6142e32af7998a27853b440db31caf4`.
All 269 native-source manifest entries match the inspected local candidate.
The 19 new control records and their source identities match the final local
report exactly. The complete 469-row/four-mode native report matches locally
except for the separately recorded C compiler identity (hosted Clang 18.1.3).
Original compiler source/pin checks and unchanged whole-repository lint passed;
Basedpyright reported zero errors, warnings and notes.

This supplementary document is added after qualification without changing any
executable, proof, test or compiler source. No extra law or gate control is
counted for the composition checks and diagnostic audit above. The remaining
P2 obligations are stored metadata, final computed inside-block contents and
independent blocker-ray lookup refinement. None is inferred from report hashes,
selected native query values or these outside-location preservation results.
