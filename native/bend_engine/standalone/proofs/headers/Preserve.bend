# Later metadata and full slider blocks preserve both headers of every earlier key.
import Base
import ./Facts.bend as F
import ./Header.bend as H
import ./Addresses.bend as A
import ./OneBlock.bend as One
import ../prefix/Spec.bend as P
import ../prefix/LAWS.bend as PL
import ../prefix/PROOF.bend as PP
import ../prefix/Nats.bend as N
import ../layout/Low.bend as Low
import ../complete/Spec.bend as C
import ../complete/Actual.bend as Complete
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../storage/Build.bend as Build
import ../separation/Spec.bend as S
import ../relative/Order.bend as Order
import ../fill_interval/Certified.bend as Fill
import ../contents/Actual.bend as Pair
import ../layout/Spec.bend as Layout
import ../../Tables.bend as Tables

def headers(+c: O.Cells,+k: Nat,+q: U32,+valid: {Nat.is_lt(k,128n) == True{} : Bool},
  +shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},+qb: {U32.is_lt(q,256) == True{} : Bool},
  ne: {U32.is_eq(F.key(k),q) == False{} : Bool} & {U32.is_eq(F.offset(k),q) == False{} : Bool}) ->
  {S.value(Array.get(U64,H.headers(O.pack(c),k,P.prefix(k)),q)) == S.value(Array.get(U64,O.pack(c),q)) : U64}:
  (first,second) = ne
  Equal.trans(U64,S.value(Array.get(U64,H.headers(O.pack(c),k,P.prefix(k)),q)),
    S.value(Array.get(U64,H.first(O.pack(c),k),q)),S.value(Array.get(U64,O.pack(c),q)),
    Equal.cong(Array<U64> & U64,U64,r => S.value(r),Array.get(U64,H.headers(O.pack(c),k,P.prefix(k)),q),
      (H.headers(O.pack(c),k,P.prefix(k)),S.value(Array.get(U64,H.first(O.pack(c),k),q))),
      Complete.frame(H.first(O.pack(c),k),F.offset(k),q,U64.from_u32(P.prefix(k)),H.first_shape(c,k,shape),
        Order.lt_lt(F.offset(k),256,131072,F.offset_bound(k,F.facts(k,valid)),{==}),Order.lt_lt(q,256,131072,qb,{==}),second)),
    Equal.cong(Array<U64> & U64,U64,r => S.value(r),Array.get(U64,H.first(O.pack(c),k),q),
      (H.first(O.pack(c),k),S.value(Array.get(U64,O.pack(c),q))),
      Complete.frame(O.pack(c),F.key(k),q,Layout.mask(F.key(k)),shape,
        Order.lt_lt(F.key(k),256,131072,F.key_bound(k,F.facts(k,valid)),{==}),Order.lt_lt(q,256,131072,qb,{==}),first)))

def block(+c: O.Cells,+k: Nat,+j: Nat,+which: Bool,
  +valid: {Nat.is_lt(k,128n) == True{} : Bool},+earlier: {Nat.is_lt(j,k) == True{} : Bool},
  +shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape}) ->
  {S.value(Array.get(U64,P.block(O.pack(c),k),A.query(j,which))) == S.value(Array.get(U64,O.pack(c),A.query(j,which))) : U64}:
  +q = A.query(j,which)
  +qb = A.query_bound(j,which,Low.le_lt(j,k,128n,N.lt_to_le(j,k,earlier),valid))
  Equal.trans(U64,S.value(Array.get(U64,P.block(O.pack(c),k),q)),
    S.value(Array.get(U64,H.headers(O.pack(c),k,P.prefix(k)),q)),S.value(Array.get(U64,O.pack(c),q)),
    Fill.before(H.headers(O.pack(c),k,P.prefix(k)),k,q,valid,H.shape(c,k,P.prefix(k),shape),
      Order.lt_le(q,512,P.prefix(k),Order.lt_lt(q,256,512,qb,{==}),One.lower(k,valid))),
    headers(c,k,q,valid,shape,qb,A.distinct(j,k,which,earlier,valid)))

def transfer(-a: Array<U64>,+n: Nat,+k: Nat,+q: U32,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  rule: @+c: O.Cells -> {O.shape(O.pack(c)) == C.full(17n) : O.Shape} ->
    {S.value(Array.get(U64,P.scheduled(n,k,O.pack(c)),q)) == S.value(Array.get(U64,O.pack(c),q)) : U64},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {S.value(Array.get(U64,P.scheduled(n,k,a),q)) == S.value(Array.get(U64,a,q)) : U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {S.value(Array.get(U64,P.scheduled(n,k,_),q)) == S.value(Array.get(U64,a,q)) : U64}
  %same : {S.value(Array.get(U64,P.scheduled(n,k,O.pack(c)),q)) == S.value(Array.get(U64,_,q)) : U64}
  rule(c)(Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,x => O.shape(x),O.pack(c),a,same),shape))

def value(+n: Nat,a: Array<U64>,+k: Nat,+j: Nat,+which: Bool,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  +budget: {Nat.is_lt(Nat.add(n,k),129n) == True{} : Bool},
  +earlier: {Nat.is_lt(j,k) == True{} : Bool}) ->
  {S.value(Array.get(U64,P.scheduled(n,k,a),A.query(j,which))) == S.value(Array.get(U64,a,A.query(j,which))) : U64}:
  match n:
    case 0n: {==}
    case 1n+p:
      transfer(a,1n+p,k,A.query(j,which),shape,+c => +sh =>
        Equal.trans(U64,S.value(Array.get(U64,P.scheduled(p,1n+k,P.block(O.pack(c),k)),A.query(j,which))),
          S.value(Array.get(U64,P.block(O.pack(c),k),A.query(j,which))),S.value(Array.get(U64,O.pack(c),A.query(j,which))),
          value(p,P.block(O.pack(c),k),1n+k,j,which,
            Equal.trans(O.Shape,O.shape(P.block(O.pack(c),k)),O.shape(O.pack(c)),C.full(17n),Build.block_shape(c,F.key(k),P.prefix(k)),sh),
            N.next_bound(p,k,budget),N.weaken(j,k,earlier)),block(c,k,j,which,N.strip(p,k,128n,budget),earlier,sh)),R.reify(a))
