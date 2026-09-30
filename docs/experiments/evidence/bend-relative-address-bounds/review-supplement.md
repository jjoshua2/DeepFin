# Relative-address supplementary self-review

## Verified hosted publication

Run **35777583835**, job **106914593255**, completed successfully at every stage
on source commit `ce9d5e5442cf44cde50a4c8da66e3ea9ffdd5bdc`. The full source gate
passed **89 laws / 176 controls**; all four native modes passed 288 rows and six
malformed requests each; original compiler source/pin checks and unchanged
whole-repository Ruff/Basedpyright/Vulture passed. Hosted evidence commit:
`7248372c9baba4c2b705eaf006d7a46b6a166012`.

Artifact **10717588714**, ZIP SHA-256
`a9023abf40212fb324a399fb249eae61b7591ece29536823fa4f5d11780c449c`, was downloaded
and checked. All 253 native-source manifest entries matched the local candidate.
The reviewed control locations, normalized diagnostic hashes and proof-source
identities match exactly. Every native report field matches except the explicitly
recorded C compiler identity (local Clang 17; hosted Ubuntu Clang 18.1.3).
The source commit object was recovered exactly from its retained raw metadata.
This comparison is not an independent code review.

## Source and scope

Reviewed candidate tree: `4ee76c5d996b16d6ca84b0053b2a65f80d90611e`, on PR #829
`fc1fdd7cbe4d7e84543a14df9b7336c39ab098f2`. The reviewed change to the original
candidate affects the new mutation harness, documentation and evidence only:
**none of the six law/proof Bend sources changed**. Compiler pin remains
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, 84 sources, fingerprint
`d9550e30dbf17f12013aa5db89b27cf6724957fe68213c9c7e409e99f1405ef4`.

**Self-review only, not independent review.** This record adds no public law or
routine test. The source checks below are separate importing consumers, not
extra entries in the six-law aggregate. Native mutation checks are separate from
the 20 source/policy controls and six malformed requests per mode.

## Mutation review finding and corrected gate

The first control run reported success even though two tests failed too early:
an untyped literal in U64.add caused an inference error, and an altered U32.add
failed the preexisting U32.add_comm proof. Those outputs do not demonstrate the
new arithmetic refinement. Original logs are preserved losslessly in
`mutation-review-finding.json`; `historical-initial-controls.json` is explicitly
not the final semantic qualification.

The corrected controls use typed `U32.from_nat(0n)` to remove U64 carry and a
typed U64 low-half increment. They require ordinary diagnostics at
`safe_from_wide` and `widening`, respectively. The gate now rejects inference,
parse, missing-name/file, linearity, crash and timeout errors as semantic passes.
All 20 reviewed controls pass locally. Their exact locations and normalized
diagnostic SHA-256 values are retained in `local-controls.json`.

Obsolete run **35776424965** was cancelled before publication. Its saved source
log is empty and it has no completed aggregate report. Artifact **10716517638**,
ZIP SHA-256 `d5e319ac2e9bc221d097e8f71485e64187b08c582f551e7a5ad30570c1947ad3`,
preserves its unpublished source identity and cancellation-era records. No pass
is inferred from that aborted aggregate.

## Additional source composition checks

Both consumers below exited zero with exactly `All terms check.` using local
Bun 1.4.2 and the pinned checker. The reverse consumer took
317.435 seconds; the metadata consumer took
318.928 seconds. These are observed check times,
not engine performance measurements. All 63 imported existing source files match
the reviewed candidate byte-for-byte; their manifest and raw status/output files
are in `bend-relative-address-review.zip`.

The first consumer derives the **reverse direction**: writing a later block
preserves an earlier block's query. The second proves one certified slider write
preserves a query anywhere below 512, including the metadata/extra-table region.
Both retain the complete-tree shape, valid-key and strict-bound assumptions.
Neither assumes relative PEXT bounds or path separation from its caller.

These are **single-write** statements. They do not establish that every
incrementing Tables.fill address meets the bounds, that each stored header equals
the certified prefix, or that the final table has the required computed contents.
Preserving a value is not proof that the value was initialized correctly.

To reproduce, save these three files beside the relative suite in a **disposable**
checkout of the exact candidate. Run the pinned checker on each consumer, keeping
status and exact output checks. The files are embedded here rather than changing
the already-qualified executable source set.

```sh
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts native/bend_engine/standalone/proofs/relative/reverse_consumer.bend
BEND_NO_TELEMETRY=1 bun /path/to/pinned/bend/bend2/main.ts native/bend_engine/standalone/proofs/relative/metadata_consumer.bend
```

### `reverse_helpers.bend`

SHA-256 `d20b56b65bc54a3036307280d451f35774d0be5a7d91f7790fa1a49ff552690e`.

```text
# Supplemental self-review lemma; not one of the six public laws.
import Base
import ./Order.bend as Order
import ../normalization/Compare.bend as Compare
import ../u64/Words.bend as Old

def finish(+ab: Cmp,+ba: Cmp,
  lt: {Cmp.is_lt(ab) == True{} : Bool},
  chain: {Order.coherent(ab,ba,EQ{}) == True{} : Bool}) ->
  {Cmp.is_eq(ba) == False{} : Bool}:
  match ab ba:
    case LT{} LT{}: {==}
    case LT{} EQ{}: Old.false_true({True{} == False{} : Bool},chain)
    case LT{} GT{}: {==}
    case EQ{} LT{}: {==}
    case EQ{} EQ{}: Old.false_true({True{} == False{} : Bool},lt)
    case EQ{} GT{}: {==}
    case GT{} LT{}: {==}
    case GT{} EQ{}: Old.false_true({True{} == False{} : Bool},lt)
    case GT{} GT{}: {==}

def reverse_ne(+a: U32,+b: U32,lt: {U32.is_lt(a,b) == True{} : Bool}) ->
  {U32.is_eq(b,a) == False{} : Bool}:
  match a b:
    case U32{x} U32{y}:
      finish(Word.cmp(32n,x,y),Word.cmp(32n,y,x),lt,
        Equal.trans(Bool,
          Order.coherent(Word.cmp(32n,x,y),Word.cmp(32n,y,x),EQ{}),
          Order.coherent(Word.cmp(32n,x,y),Word.cmp(32n,y,x),Word.cmp(32n,x,x)),True{},
          Equal.cong(Cmp,Bool,c => Order.coherent(Word.cmp(32n,x,y),Word.cmp(32n,y,x),c),
            EQ{},Word.cmp(32n,x,x),Equal.sym(Cmp,Word.cmp(32n,x,x),EQ{},Compare.reflexive(32n,x))),
          Order.chain(32n,x,y,x)))
```

### `reverse_consumer.bend`

SHA-256 `5397b3f4625c91b558d07c79f246cdb6e5fa9c64663ed59e7127067a47260415`.

```text
# Supplemental composition: a later-block write preserves an earlier-block query.
# This is a checked importing consumer, not an extra law in the six-law aggregate.
import Base
import ./reverse_helpers.bend as Reverse
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ./Bounds.bend as Bounds
import ./Certified.bend as Certified
import ../complete/Spec.bend as Complete
import ../complete/Actual.bend as Actual
import ../storage/Observe.bend as O
import ../separation/Spec.bend as Separation

def reverse_lookup_write(a: Array<U64>,+i: Nat,+j: Nat,+occi: U64,+occj: U64,+v: U64,
  shape: {O.shape(a) == Complete.full(17n) : O.Shape},
  +before: {Nat.is_lt(i,j) == True{} : Bool},+valid: {Nat.is_lt(j,128n) == True{} : Bool}) ->
  {Array.get(U64,Array.set(U64,a,S.address(j,S.index(j,occj)),v),S.address(i,S.index(i,occi))) ==
    (Array.set(U64,a,S.address(j,S.index(j,occj)),v),
      Separation.value(Array.get(U64,a,S.address(i,S.index(i,occi))))) : Array<U64> & U64}:
  +vi = Proof.first_valid(i,j,before,valid)
  +ri = S.index(i,occi)
  +rj = S.index(j,occj)
  Actual.frame(a,S.address(j,rj),S.address(i,ri),v,shape,
    Bounds.allocation(j,rj,Laws.lookup_address_bounds(j,occj,valid)),
    Bounds.allocation(i,ri,Laws.lookup_address_bounds(i,occi,vi)),
    Reverse.reverse_ne(S.address(i,ri),S.address(j,rj),
      Laws.ordered_block_addresses(i,j,ri,rj,before,valid,Certified.lookup_index(i,occi,vi),Certified.lookup_index(j,occj,valid))))
```

### `metadata_consumer.bend`

SHA-256 `e4d1c4ae69a0b294072790da6f7551a4b4bb045053b719fa54055975e37d00ca`.

```text
# Supplemental composition: a certified slider write cannot clobber metadata.
# This proves one write, not every address in the full Tables.fill schedule.
import Base
import ./reverse_helpers.bend as Reverse
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as S
import ./Bounds.bend as Bounds
import ./Order.bend as Order
import ../prefix/Spec.bend as Prefix
import ../prefix/LAWS.bend as PrefixLaws
import ../prefix/PROOF.bend as PrefixProof
import ../prefix/Nats.bend as Nats
import ../complete/Spec.bend as Complete
import ../complete/Actual.bend as Actual
import ../storage/Observe.bend as O
import ../separation/Spec.bend as Separation

def floor(+k: Nat,pair: {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool}
  & {U32.is_lt(Prefix.prefix(k),131072) == True{} : Bool}) ->
  {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool}:
  (lo,hi) = pair
  lo

def preserves_metadata(a: Array<U64>,+k: Nat,+occ: U64,+m: U32,+v: U64,
  shape: {O.shape(a) == Complete.full(17n) : O.Shape},
  +valid: {Nat.is_lt(k,128n) == True{} : Bool},
  +mb: {U32.is_lt(m,512) == True{} : Bool}) ->
  {Array.get(U64,Array.set(U64,a,S.address(k,S.index(k,occ)),v),m) ==
    (Array.set(U64,a,S.address(k,S.index(k,occ)),v),Separation.value(Array.get(U64,a,m))) : Array<U64> & U64}:
  +r = S.index(k,occ)
  +lower = floor(k,PrefixLaws.prefix_in_allocation(k,Nats.weaken(k,128n,valid)))
  +order = Order.lt_le(m,Prefix.prefix(k),S.address(k,r),Order.lt_le(m,512,Prefix.prefix(k),mb,lower),
    Bounds.lower(k,r,Laws.lookup_address_bounds(k,occ,valid)))
  Actual.frame(a,S.address(k,r),m,v,shape,
    Bounds.allocation(k,r,Laws.lookup_address_bounds(k,occ,valid)),
    Order.lt_lt(m,512,131072,mb,{==}),Reverse.reverse_ne(m,S.address(k,r),order))
```

## Native behavior mutation checks

Each mutation ran in a separate disposable source snapshot with the **unchanged**
reference expectations. The real probe compiled and executed; the generic-mode
comparison rejected row zero's values with exit one. The driver stopped there,
so no passing/failing verdict is attributed to later modes in these mutation runs.

| Mutation | Observed failure | Raw log SHA-256 |
| --- | --- | --- |
| `metadata-header-one-ahead` | Actual Tables offset header stores `U64.from_u32(U32.inc(at))`; observed first prefix 513 rather than 512, changing the queried address/value. | `39873b8b6edaa74eaa055dea65fff53ee05099544cec316bb6dfc91d9f0ef5e8` |
| `native-write-corrupts-query` | Actual probe writes at `aj` rather than `ai`; the supposedly protected query changes and the intended write slot remains unchanged. | `938715278fd11fe59a0944836982b80618a977dc1d5199e400040bcced38feb8` |

These diagnostics confirm value-sensitive reference checks. The probe observes
low32 stored prefixes, PEXT indices, addresses, selected array values and capacity;
it does not inspect every cell or certify full U64 header representation. No
expected table values are supplied to the native candidate.

## Reapplication and retained limits

The combined reviewed patch is 134,558 bytes, SHA-256
`93b8307811f82ace341c5612003a1cdd2de1e409f260e96a11440c7e7dc8a961`. Applying it to a fresh exact-parent worktree
reproduces tree `4ee76c5d996b16d6ca84b0053b2a65f80d90611e`, all 253 native-source
manifest entries and a clean Git whitespace check.

Supplement construction initially had an incorrect scratch path, then a match
binder error and an undefined convenience conversion. Those attempts were not
counted as successful source checks or semantic mutation controls. The final
helper uses explicit U32 constructor matching and the existing Word comparison
proofs; its exact source check passes before the two consumer checks.

No production code, old accepted statement, compiler input, model/GPU/perft,
training or benchmark change. Source laws still trust the pinned checker/Base;
native lowering, allocation/lifetime, ABI, C toolchain, OS/hardware and transitional
model backends remain separate trust boundaries. Remaining P2 work is actual
fill-clear schedules, stored metadata and final computed contents, then independent
blocker-ray lookup refinement. The earlier compiler diagnostics and direct raw
Array-routine C-lowering limitation are not fixed by these checks.
