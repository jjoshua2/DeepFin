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
