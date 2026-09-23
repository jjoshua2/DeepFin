# Supplementary fill-contents self-review — September 23, 2026

This is an additional bounded author review of the unchanged saved three-law candidate,
not independent review, new public aggregate laws, or replacement hosted qualification.
Base: `0d0b325be4e8b0790b482aebbe096000fe989819`.
Original patch SHA-256: `ba473c4cc0ec106f7c8cd5bfddeeaebfaf78c293c9dab6d586dc729975b9b866`.
The pinned compiler's 84-file fingerprint is unchanged. All 82 files in the candidate's
reported proof/driver closure still match their saved hashes after these tests.

## Arbitrary-start complete returned-pair composition

A separate importing consumer lifts the already-proved arbitrary-start value theorem
to the complete affine read pair using the saved `Actual.read_pair` helper and existing
certified reification. It quantifies over the actual initial Array, arbitrary starting
subset and the same complete-shape/index/end/count budget premises as `bounded_fill_entry`.
There is no assumed desired read value. The output's first component is the complete
final fill result, not the original array.

The original pinned CLI returned status 0 and exactly `All terms check.`. This is a
supplementary derived statement, not a fourth public LAWS obligation or a new execution
of the complete inherited aggregate. It uses the verified dependency snapshot; the
separate hosted job is responsible for full-checkout qualification.

## Nonzero and outside-mask native checks

Fresh source C generation and generic / UBSan compilations execute 54 fixture rows
per mode on actual 131072-cell allocations. Six keys (0,27,63,64,91,127) cover rook and
bishop corners/centers. Each uses three starting states: compact index one, last compact
index, and all 64 bits set (including bits outside the relevant mask). A three-step actual
fill is inspected at entries 0,1,2, each on fresh poisoned storage. The last-index case
crosses the subset cycle. Both modes match independent signed-coordinate ray geometry
and direct compact bit deposition. For the all-ones start, its next state is zero
because unsigned MAX minus the mask is its complement; subsequent compact indices
start at zero. The oracle does not copy the candidate's carry-rippler expression.

Both output digests: `39ae8be214d725a8f122620ccd88c6d307234d095b5a4729d57b6184133040b7`.
The probe observes initial state and selected entry values, not every returned cell.
Modes repeat the fixtures; these are not 108 independent cases or exhaustive arbitrary
starting subsets. No proof predicate or host-generated expected value enters the probe.
Local toolchain: Bun 1.4.2 and Clang 17.0.0; these are not the hosted Clang 18 checks.

A disposable mutation changes the actual fill's written attack to the zero-occupancy
attack instead of the supplied subset's attack. It still compiles and exits zero,
but the unchanged reference rejects 42 of 54 values, beginning at row zero. Initial
state metadata agrees in that row: the rejection is for the computed value, not a
malformed fixture or changed identity. The original source is untouched.

A second disposable mutation makes actual `Tables.fill` skip every write while
retaining its count/address/subset traversal. The unchanged original
`contents/verify_native.js` rejects its first computed-entry row in generic mode;
the later modes are not reached. This specifically tests that the fresh poisoned
storage prevents a no-op from hiding behind previously correct table entries.
It is an additional native failure diagnostic, not one of the 17 source controls.

## Construction failures and limits

The first supplementary probe failed parsing because a match followed a local binding;
a helper with the matched parameter was used instead. The second used `Cons` rather than
the language's list-pattern spelling; it was corrected to `k <> rest`. Both nonzero
code-generation failures and their diagnostics are retained in the conversation review
archive. Neither was a product failure or counted as a passing negative control. The
final probe and both native modes were executed only after those corrections.

No production or published proof source changes are introduced by this review. No
compiler/generated-C patch, warning suppression, numerical relaxation, GPU/model,
perft, training, strength or benchmark result. Stored headers, survival through all
later builder phases and independently specified full lookup geometry remain separate.

## Full-tree recovery and exact patch reapplication

Read-only recovery run 35898058655 produced artifact 10768225491, ZIP SHA-256
`56c7c697ec84c619895e14cbfb8347f67cfce3a1f2d2d27f99a4bdeec6adbe3e`.
It archived source without executing it. Reconstructing a Git index from all archive
bytes and executable/symlink modes reproduces the complete base tree
`540dc223e40a84f4bb3a2626e0591d8b3ff0b621`. The original commit object hashes to
`0d0b325be4e8b0790b482aebbe096000fe989819`.

The saved patch applies cleanly on that full baseline and yields candidate tree
`731bf2a90704324e1dd34727d3f1aaad07162a67`. All 50 changed files exactly match the
saved source overlay, all local report source hashes match, and the Git whitespace
check passes. This closes the previous complete-tree reconstruction gap; it is not
another source/native/lint execution. Supplementary test closure bytes also match
that full candidate. The historical selected-snapshot checks remain correctly labeled.

## Exact supplemental sources and results

### General-pair source

```text
# Supplementary composition only, not an additional public aggregate law.
import Base
import ./Actual.bend as Actual
import ./Spec.bend as Spec
import ../../Tables.bend as Tables
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R

def model(+c: O.Cells,+n: Nat,+i: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  ib: {Nat.is_lt(i,n) == True{} : Bool},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,O.pack(c)),U32.add(at,U32.from_nat(i))) ==
    (Tables.fill(n,at,sq,bishop,mask,subset,O.pack(c)),Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{})) : Array<U64> & U64}:
  Actual.read_pair(Tables.fill(n,at,sq,bishop,mask,subset,O.pack(c)),U32.add(at,U32.from_nat(i)),Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{}),
    Actual.entry(O.pack(c),n,i,at,end,sq,bishop,mask,subset,shape,ib,eb,budget))

def lift(-a: Array<U64>,+n: Nat,+i: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  ib: {Nat.is_lt(i,n) == True{} : Bool},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),U32.add(at,U32.from_nat(i))) ==
    (Tables.fill(n,at,sq,bishop,mask,subset,a),Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{})) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,_),U32.add(at,U32.from_nat(i))) ==
    (Tables.fill(n,at,sq,bishop,mask,subset,a),Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{})) : Array<U64> & U64}
  %same : {Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,O.pack(c)),U32.add(at,U32.from_nat(i))) ==
    (Tables.fill(n,at,sq,bishop,mask,subset,_),Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{})) : Array<U64> & U64}
  model(c,n,i,at,end,sq,bishop,mask,subset,
    Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),Equal.cong(Array<U64>,O.Shape,t => O.shape(t),O.pack(c),a,same),shape),ib,eb,budget)

def arbitrary_start_pair(a: Array<U64>,+n: Nat,+i: Nat,+at: U32,+end: U32,+sq: U32,+bishop: Bool,+mask: U64,+subset: U64,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  ib: {Nat.is_lt(i,n) == True{} : Bool},
  eb: {U32.is_lt(end,131072) == True{} : Bool},
  budget: {Nat.is_le(Nat.add(n,U32.to_nat(at)),U32.to_nat(end)) == True{} : Bool}) ->
  {Array.get(U64,Tables.fill(n,at,sq,bishop,mask,subset,a),U32.add(at,U32.from_nat(i))) ==
    (Tables.fill(n,at,sq,bishop,mask,subset,a),Tables.slider(sq,bishop,Spec.state(i,mask,subset),False{})) : Array<U64> & U64}:
  lift(a,n,i,at,end,sq,bishop,mask,subset,shape,ib,eb,budget,R.reify(a))
```

### General-pair result

```json
{
  "command": [
    "/mnt/data/contents-supplement/bun",
    "/mnt/data/contents-supplement/compiler/bend/bend2/main.ts",
    "/mnt/data/contents-supplement/work/native/bend_engine/standalone/proofs/contents/general-pair-review.bend"
  ],
  "exit_code": 0,
  "elapsed_seconds": 2.9479718289999255,
  "source_sha256": "912bfcf32960c9dc8f359687b117467820c06f2ba9bdb0a367def277147b1e80",
  "exact_success": true
}
```

### Nonzero native probe

```text
# Supplementary self-review only; no proof model or external answer inputs.
import Base
import ../../Tables.bend as Tables

def start(mode: U32,+m: U64) -> U64:
  match mode:
    case 0: U64.pdep(U64.from_u32(1),m)
    case 1: U64.pdep(U64.from_u32(U32.sub(U32.shln(1,U32.to_nat(U64.popcount(m))),1)),m)
    case _: U64.from_parts(4294967295,4294967295)

def emit(key: U32,mode: U32,i: U32,+initial: U64,r: Array<U64> & U64) -> IO(Unit):
  (a,+v) = r
  IO.print(U32.show(key) ++ " " ++ U32.show(mode) ++ " " ++ U32.show(i)
    ++ " " ++ U32.show(U64.high(initial)) ++ " " ++ U32.show(U64.low(initial))
    ++ " " ++ U32.show(U64.high(v)) ++ " " ++ U32.show(U64.low(v)))

def check(+key: U32,+mode: U32,+i: U32) -> IO(Unit):
  +m = Tables.slider(U32.mod(key,64),U32.is_ge(key,64),U64.zero(),True{})
  +initial = start(mode,m)
  emit(key,mode,i,initial,Array.get(U64,
    Tables.fill(3n,512,U32.mod(key,64),U32.is_ge(key,64),m,initial,
      Array.new(U64,17n,U64.from_parts(3735928559,2309737967))),U32.add(512,i)))

def indices(n: Nat,+key: U32,+mode: U32,+i: U32) -> IO(Unit):
  match n:
    case 0n: IO.pure(Unit,Unit{})
    case 1n+p:
      do IO<Unit>:
        check(key,mode,i)
        indices(p,key,mode,U32.inc(i))

def modes(n: Nat,+key: U32,+mode: U32) -> IO(Unit):
  match n:
    case 0n: IO.pure(Unit,Unit{})
    case 1n+p:
      do IO<Unit>:
        indices(3n,key,mode,0)
        modes(p,key,U32.inc(mode))

def keys(xs: List<U32>) -> IO(Unit):
  match xs:
    case Nil{}: IO.pure(Unit,Unit{})
    case k <> rest:
      do IO<Unit>:
        modes(3n,k,0)
        keys(rest)

def main() -> IO(Unit):
  keys([0,27,63,64,91,127])
```

### Native result

```json
{
  "status": "PASS",
  "kind": "Supplementary native self-review, not new public laws or aggregate fixtures",
  "cases": 54,
  "keys": [
    0,
    27,
    63,
    64,
    91,
    127
  ],
  "starting_states": [
    "compact index 1",
    "last compact index",
    "all U64 bits set including outside mask"
  ],
  "queried_indices": [
    0,
    1,
    2
  ],
  "fill_count": 3,
  "seed": "0xdeadbeef89abcdef",
  "candidate": "actual Tables.fill and Array APIs, generated C unchanged",
  "oracle": "independent signed-coordinate rays and direct bit deposition; all-ones start advances to zero by complement identity",
  "modes": [
    {
      "mode": "generic",
      "rows": 54,
      "output_sha256": "39ae8be214d725a8f122620ccd88c6d307234d095b5a4729d57b6184133040b7"
    },
    {
      "mode": "ubsan",
      "rows": 54,
      "output_sha256": "39ae8be214d725a8f122620ccd88c6d307234d095b5a4729d57b6184133040b7"
    }
  ],
  "source_sha256": "aadd227b1132a487b6493f49d6a7c0ec0213e4ee019b69608ffb680c7235c9ff",
  "generated_c_sha256": "4139e9f01a610505025ab68a563461a0246f38602d854215fc4a5fae6d3e12f2"
}
```

### Mutation result

```json
{
  "status": "REJECTED_WRONG_VALUE",
  "mutation": "actual Tables.fill writes the zero-occupancy slider result instead of its actual subset result",
  "candidate_exit_code": 0,
  "oracle_pass": false,
  "rows": 54,
  "mismatched_rows": 42,
  "first_mismatch": {
    "row": 0,
    "actual": [
      0,
      0,
      0,
      0,
      2,
      16843009,
      16843262
    ],
    "expected": [
      0,
      0,
      0,
      0,
      2,
      16843009,
      16843010
    ]
  },
  "source_sha256": "1fea71c63106e7dec87b6f1def7c73627cdd26bdff80dbf32ced76e770a3b9a2",
  "scope": "Disposable supplementary native mutation, not an aggregate source control; original candidate/compiler unchanged"
}
```

### No-op fill diagnostic

```json
{
  "status": "REJECTED_WRONG_VALUE",
  "command": [
    "/mnt/data/contents-supplement/bun",
    "/mnt/data/contents-supplement/noop-work/native/bend_engine/standalone/proofs/contents/verify_native.js",
    "/mnt/data/contents-supplement/compiler/bend"
  ],
  "exit_code": 1,
  "elapsed_seconds": 2.9193214549998174,
  "mutation": "actual Tables.fill skips every Array.set; retains count/subset/address recurrence",
  "verifier": "unchanged saved contents/verify_native.js",
  "first_failure": "computed entry row 0",
  "mode": "generic, later modes not reached",
  "scope": "Supplementary no-op negative control, not a passing native run or an aggregate source control",
  "mutated_tables_sha256": "4215e98d4ffa200a2c04c856eff81f4301b226022544612909fc704df5de1693",
  "first_actual_row": [
    0,
    0,
    0,
    512,
    65793,
    16843134,
    4096,
    4096,
    0,
    2590861344,
    1103838041,
    131072
  ],
  "first_expected_row": [
    0,
    0,
    0,
    512,
    65793,
    16843134,
    4096,
    4096,
    0,
    16843009,
    16843262,
    131072
  ],
  "only_selected_value_differs": true
}
```
