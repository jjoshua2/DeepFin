# Supplementary extras and stored-header self-review — September 23, 2026

Author self-review only, not independent review or extra public aggregate laws.
Original two-law source: `c60a5e83958acef579f50593e970d40b1a0019e9`, tree
`07d1c051564a7c8df128ee524ee1e6f644d7cac5`, on PR #861 at
`6f2c71fa9bd1d1537c1f5ae0ff6e70e31e495c53`. Compiler remains
`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`, with the same 84-input fingerprint.
The public aggregate remains 100 laws / 226 controls.

## Complete reconstruction and two qualification stages

Read-only archive run 35921079224, artifact 10777396209, ZIP SHA-256
`ebb20e48daab7e5343936b8ea9f259dbdd2cf8b0d0a4baca3fc17e34f7388718`,
reproduces the complete current #861 tree
`99da6fbe20da28e99b4dff4dc5fe3908fdb67616` and original commit from all
3,363 tracked archive paths/modes, including tracked paths covered by ignore rules.
Applying the unchanged 128,899-byte saved patch reproduces the original candidate
above and all 39 changed files. All 283 inherited native-source entries match.

Downloaded full-qualification artifact 10778342626 has ZIP SHA-256
`43c0e59c7e379a674d4c05703c5e94e7183bd12ac26644c407d4fae3cfbdb6a8`.
Its patch, original commit object, all 297 candidate native-source entries, all
14 control records/diagnostics, and native report except compiler identity match
the inspected local source. Run 35920944024 actually executed and passed the
complete 100-law / 226-control aggregate. These reconstruction checks are not a
second aggregate execution.

Test reinforcement source: `3c52e6469b3ba9cf133077f73d8f96ca8dcc76cd`.
Hosted reinforcement run 35924465983, job 107396095994, passed every stage:
focused two-law/14-control recheck, strengthened four-mode native suite, copy
regression, 12 compiler-pin tests, unchanged whole-repository lint, and fast-forward
publication. Its downloaded artifact 10777789302 has ZIP SHA-256
`89ce408464943a4947aabeabee5645a169c4c9d8a34200c5657f65eb8147ffc2`.
All 298 native-source manifest entries match the locally tested reinforcement.
The entire focused report equals the local JSON; native JSON differs only in
recorded C compiler identity. The logical Bend proofs and public statements did
not change. The full aggregate was not redundantly rerun after native-only changes.

## Additional extras compositions

A supplementary importing consumer proves that an arbitrary U64 value explicitly
written to a bounded protected slot using actual Array.set survives extras, with
the entire updated array returned. The desired read value is not assumed: existing
same-location read/write and extras preservation laws derive it. Source SHA-256:
`5bf75532ad11e3c2fbc2d20bf2d4f93a2c94bdf41f6baaf167758a9179a30e4f`.
The pinned checker returned status 0 and exactly `All terms check.` in 5.47 seconds.

A second consumer proves `extras(n+m,k,a)` equals
`extras(m,after(n,k),extras(n,k,a))`, where `after` uses repeated actual U32.inc,
retaining wraparound. It quantifies over arbitrary source Nat counts and affine
arrays without promising feasible native allocation/execution at huge counts.
Source SHA-256:
`9b29de4a5b4d6af33ee52df307024353103ba0881440828d9f07c47ce6ee8f34`.
It passed exact checker output in 0.55 seconds. Neither is an added public-law count.

## Native coverage blind spot and repair

A disposable mutation copying protected cell 6 into cell 7 after each actual extras
iteration passes all original uniform-seed native fixtures because both cells have
the same seed value. This is a native-fixture coverage defect, not a production bug
or a passing qualification of corrupted source. Existing source checking rejects
it at `storage/Build.extras`; the expected source manifest also differs.

Every-cell-distinct fixtures reject the same mutation at `unique case 1 cell 7`.
All five original buffers and seven malformed requests remain. Three distinct-cell
buffers are additionally checked, for eight buffers / 1,048,576 cells per mode.
The added output digest in every mode is
`51b2702b6dd1e4cc3823981001b3a8d0d95a15003d5d89ba613ba8bf91d2adbf`;
the original digest remains
`0cf635bddb79e02ebb6a9d64e42284129bc699f1214f1a01e7c01b58eb85eef3`.
The same fixtures repeat across modes, not disjoint datasets or exhaustive inputs.

The initial supplemental probe failed a duplicable-value binder check. A `+v`
formatting helper fixed that diagnostic before successful native executions.
This was not counted as a valid negative control. The strengthened gate changes
only its native driver, a new unique-value probe, and the focused source manifest;
all previously accepted laws, mutations, checker output requirements and runtime
implementation remain unchanged. The final source-copy rejection diagnostic has
SHA-256 `a88cd38cf484b17b8ff606fe872e28414b7db93578c23f70283c3e8adc9bc115`.

## Stored header survives its own actual block fill

The three modules below prove the supplementary derived function
`header_after_own_fill(a,k,prefix,valid,shape)`. For every `k < 128`, complete
depth-17 actual array `a`, and selector `prefix`, reading the corresponding header
after actual `Tables.tables(1n,key(k),Prefix.prefix(k),a)` returns the complete
updated array and the actual mask or widened prefix. The caller supplies complete
shape and a valid key, not assumed header values, block sizes or noninterference.

`Facts` derives finite key/offset bounds by in-checker reduction. `Header`
establishes the two actual public-write values. `OneBlock` uses accepted prefix
producers and the real full-block-before preservation law through the entire fill.
Certified reification transfers the pair equality to arbitrary actual arrays. The
module explicitly imports `Prefix/PROOF`; it does not replace the actual builder
with a host oracle. Its target values remain the actual mask and certified prefix,
not independently proved geometric masks.

Reproduction after restoring the exact text below in an isolated checkout:

```sh
BEND_NO_TELEMETRY=1 bun /pinned/bend/bend2/main.ts native/bend_engine/standalone/proofs/header_review/OneBlock.bend
```

Final result: status 0, exactly `All terms check.`, 310.471 seconds under a separately
registered 600-second supplementary review budget. Earlier short/240-second
attempts timed out; the same final source and compiler were retained. `Header`
alone passed in 2.78 seconds. This is checked supplementary source proof code,
not a third public LAWS obligation or a new complete aggregate execution.

Mutating the actual stored prefix to `U64.from_u32(U32.inc(at))` is rejected in the
inherited `storage/Build.tables` implementation bridge with ordinary
expected/observed diagnostics and status 1. This is not claimed to fail only the
new value theorem. The first disposable mutation omitted
`bitboard_probe/Sliders.bend` and was correctly classified as an invalid missing-file
attempt. The retry restored the complete dependency tree. Final diagnostic SHA-256:
`f829e37fab64a331226b0c08a79b23b8f2b4f04ed5a93ca019162cc0726a4754`.

This does not prove survival across all later blocks, complete stored-header
correctness after Tables.build, or independent blocker-ray geometry. Those remain
the next P2 targets. No compiler/production changes, model/GPU/perft/training runs,
or strength/performance claims. Checker/Base and native allocation/lifetime,
ABI/toolchain/OS/hardware remain trust boundaries. Sources follow as text so future
work can promote them deliberately instead of reconstructing them from chat.

## Reconstruct Facts.bend exactly

Save all three modules under `standalone/proofs/header_review` in an isolated
checkout. The generator writes source case proofs; the checker evaluates the
arithmetic. No expected read values or host-generated attack answers enter the code.

```python
from pathlib import Path
import hashlib
pre='# Finite header-address facts, no external answer table or assumed array values.\nimport Base\nimport ../layout/Domain.bend as Domain\n\ndef key(k: Nat) -> U32: U32.from_nat(k)\ndef offset(k: Nat) -> U32: U32.add(128,key(k))\n\ndef bounds(+k: Nat) -> Type:\n  {U32.is_lt(key(k),256) == True{} : Bool}\n  & {U32.is_lt(offset(k),256) == True{} : Bool}\n  & {U32.is_eq(offset(k),key(k)) == False{} : Bool}\n\ndef facts(+k: Nat,e: {Nat.is_lt(k,128n) == True{} : Bool}) -> bounds(k):\n  match k:\n'
post='    case 128n+p: Domain.impossible(bounds(128n+p),p,e)\n\ndef key_bound(+k: Nat,p: bounds(k)) -> {U32.is_lt(key(k),256) == True{} : Bool}:\n  (kb,rest) = p\n  kb\n\ndef offset_bound(+k: Nat,p: bounds(k)) -> {U32.is_lt(offset(k),256) == True{} : Bool}:\n  (kb,(ob,ne)) = p\n  ob\n'
text=pre+"".join(f"    case {i}n: ({{==}},({{==}},{{==}}))\n" for i in range(128))+post
assert hashlib.sha256(text.encode()).hexdigest()=="20f3c18270e859fc068c0643f37b7ce80efaa21b026abef1f00adbbd464555ef"
Path("Facts.bend").write_text(text)
```

## Header.bend

SHA-256 `36e7c9251069bc9db4b8d2a82951ab785c4f3599641f924315e6e381d7145264`.

```text
# Same two public-array writes used in actual Tables.tables. No lookup oracle.
import Base
import ./Facts.bend as F
import ../layout/Spec.bend as Layout
import ../complete/Spec.bend as C
import ../complete/Actual.bend as Complete
import ../storage/Observe.bend as O
import ../storage/Roundtrip.bend as Round
import ../storage/Write.bend as Write
import ../separation/Spec.bend as S
import ../relative/Order.bend as Order

def first(a: Array<U64>,+k: Nat) -> Array<U64>:
  Array.set(U64,a,F.key(k),Layout.mask(F.key(k)))

def headers(a: Array<U64>,+k: Nat,+at: U32) -> Array<U64>:
  Array.set(U64,first(a,k),F.offset(k),U64.from_u32(at))

def first_shape(+c: O.Cells,+k: Nat,shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {O.shape(first(O.pack(c),k)) == C.full(17n) : O.Shape}:
  Equal.trans(O.Shape,O.shape(first(O.pack(c),k)),O.shape(O.pack(c)),C.full(17n),
    Write.model(c,F.key(k),Layout.mask(F.key(k))),shape)

def shape(+c: O.Cells,+k: Nat,+at: U32,sh: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {O.shape(headers(O.pack(c),k,at)) == C.full(17n) : O.Shape}:
  Equal.trans(O.Shape,O.shape(headers(O.pack(c),k,at)),O.shape(first(O.pack(c),k)),C.full(17n),
    Write.write(first(O.pack(c),k),F.offset(k),U64.from_u32(at)),first_shape(c,k,sh))

def mask_value(+c: O.Cells,+k: Nat,+at: U32,
  sh: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},facts: F.bounds(k)) ->
  {S.value(Array.get(U64,headers(O.pack(c),k,at),F.key(k))) == Layout.mask(F.key(k)) : U64}:
  (kb,(ob,ne)) = facts
  Equal.trans(U64,S.value(Array.get(U64,headers(O.pack(c),k,at),F.key(k))),
    S.value(Array.get(U64,first(O.pack(c),k),F.key(k))),Layout.mask(F.key(k)),
    Equal.cong(Array<U64> & U64,U64,r => S.value(r),Array.get(U64,headers(O.pack(c),k,at),F.key(k)),
      (headers(O.pack(c),k,at),S.value(Array.get(U64,first(O.pack(c),k),F.key(k)))),
      Complete.frame(first(O.pack(c),k),F.offset(k),F.key(k),U64.from_u32(at),first_shape(c,k,sh),
        Order.lt_lt(F.offset(k),256,131072,ob,{==}),Order.lt_lt(F.key(k),256,131072,kb,{==}),ne)),
    Equal.cong(Array<U64> & U64,U64,r => S.value(r),Array.get(U64,first(O.pack(c),k),F.key(k)),
      (first(O.pack(c),k),Layout.mask(F.key(k))),Round.model(c,F.key(k),Layout.mask(F.key(k)))))

def prefix_value(+c: O.Cells,+k: Nat,+at: U32) ->
  {S.value(Array.get(U64,headers(O.pack(c),k,at),F.offset(k))) == U64.from_u32(at) : U64}:
  Equal.cong(Array<U64> & U64,U64,r => S.value(r),Array.get(U64,headers(O.pack(c),k,at),F.offset(k)),
    (headers(O.pack(c),k,at),U64.from_u32(at)),Round.write_read(first(O.pack(c),k),F.offset(k),U64.from_u32(at)))
```

## OneBlock.bend

SHA-256 `e40b249d5862d7574bf66f7f49d48d5ac1dd208618eb0650ebda127dab0c737b`.

```text
# Supplementary one-real-block theorem, not all later blocks or ray geometry.
import Base
import ./Facts.bend as F
import ./Header.bend as H
import ../../Tables.bend as Tables
import ../layout/Spec.bend as Layout
import ../prefix/Spec.bend as Prefix
import ../prefix/LAWS.bend as PrefixLaws
import ../prefix/PROOF.bend as PrefixProof
import ../prefix/Nats.bend as Nats
import ../fill_interval/Certified.bend as Fill
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../separation/Spec.bend as S
import ../relative/Order.bend as Order
import ../contents/Actual.bend as Pair

def run(a: Array<U64>,+k: Nat) -> Array<U64>:
  Tables.tables(1n,F.key(k),Prefix.prefix(k),a)

def query(+k: Nat,prefix: Bool) -> U32:
  Bool.pick(U32,prefix,F.offset(k),F.key(k))

def expected(+k: Nat,prefix: Bool) -> U64:
  Bool.pick(U64,prefix,U64.from_u32(Prefix.prefix(k)),Layout.mask(F.key(k)))

def lower_pair(+k: Nat,pair: {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool}
  & {U32.is_lt(Prefix.prefix(k),131072) == True{} : Bool}) ->
  {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool}:
  (lo,hi) = pair
  lo

def lower(+k: Nat,valid: {Nat.is_lt(k,128n) == True{} : Bool}) ->
  {U32.is_le(512,Prefix.prefix(k)) == True{} : Bool}:
  lower_pair(k,PrefixLaws.prefix_in_allocation(k,Nats.weaken(k,128n,valid)))

def frame(+c: O.Cells,+k: Nat,+q: U32,+v: U64,
  +valid: {Nat.is_lt(k,128n) == True{} : Bool},
  sh: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  qb: {U32.is_lt(q,256) == True{} : Bool},
  placed: {S.value(Array.get(U64,H.headers(O.pack(c),k,Prefix.prefix(k)),q)) == v : U64}) ->
  {Array.get(U64,run(O.pack(c),k),q) == (run(O.pack(c),k),v) : Array<U64> & U64}:
  Pair.read_pair(run(O.pack(c),k),q,v,
    Equal.trans(U64,S.value(Array.get(U64,run(O.pack(c),k),q)),
      S.value(Array.get(U64,H.headers(O.pack(c),k,Prefix.prefix(k)),q)),v,
      Fill.before(H.headers(O.pack(c),k,Prefix.prefix(k)),k,q,valid,H.shape(c,k,Prefix.prefix(k),sh),
        Order.lt_le(q,512,Prefix.prefix(k),Order.lt_lt(q,256,512,qb,{==}),lower(k,valid))),placed))

def model(+c: O.Cells,+k: Nat,prefix: Bool,
  +valid: {Nat.is_lt(k,128n) == True{} : Bool},
  +shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {Array.get(U64,run(O.pack(c),k),query(k,prefix)) ==
    (run(O.pack(c),k),expected(k,prefix)) : Array<U64> & U64}:
  match prefix:
    case False{}:
      frame(c,k,F.key(k),Layout.mask(F.key(k)),valid,shape,F.key_bound(k,F.facts(k,valid)),H.mask_value(c,k,Prefix.prefix(k),shape,F.facts(k,valid)))
    case True{}:
      frame(c,k,F.offset(k),U64.from_u32(Prefix.prefix(k)),valid,shape,F.offset_bound(k,F.facts(k,valid)),H.prefix_value(c,k,Prefix.prefix(k)))

def lift(-a: Array<U64>,+k: Nat,+prefix: Bool,
  valid: {Nat.is_lt(k,128n) == True{} : Bool},
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,run(a,k),query(k,prefix)) ==
    (run(a,k),expected(k,prefix)) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Array.get(U64,run(_,k),query(k,prefix)) == (run(a,k),expected(k,prefix)) : Array<U64> & U64}
  %same : {Array.get(U64,run(O.pack(c),k),query(k,prefix)) == (run(_,k),expected(k,prefix)) : Array<U64> & U64}
  model(c,k,prefix,valid,Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,x => O.shape(x),O.pack(c),a,same),shape))

def header_after_own_fill(a: Array<U64>,+k: Nat,+prefix: Bool,
  valid: {Nat.is_lt(k,128n) == True{} : Bool},
  shape: {O.shape(a) == C.full(17n) : O.Shape}) ->
  {Array.get(U64,run(a,k),query(k,prefix)) ==
    (run(a,k),expected(k,prefix)) : Array<U64> & U64}:
  lift(a,k,prefix,valid,shape,R.reify(a))
```
