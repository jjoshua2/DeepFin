# Transfer scheduled framing to the full pair returned by the actual builder.
import Base
import ./Preserve.bend as Preserve
import ../../Tables.bend as Tables
import ../prefix/Spec.bend as P
import ../prefix/LAWS.bend as PL
import ../prefix/PROOF.bend as PP
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../storage/Representation.bend as R
import ../separation/Spec.bend as S
import ../contents/Actual.bend as Pair

def model(+c: O.Cells,+n: Nat,+k: Nat,+q: U32,
  shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  +budget: {Nat.is_lt(Nat.add(n,k),129n) == True{} : Bool},
  lower: {U32.is_le(512,q) == True{} : Bool},qb: {U32.is_lt(q,131072) == True{} : Bool},
  before: {U32.is_lt(q,P.prefix(k)) == True{} : Bool}) ->
  {Array.get(U64,Tables.tables(n,U32.from_nat(k),P.prefix(k),O.pack(c)),q) ==
    (Tables.tables(n,U32.from_nat(k),P.prefix(k),O.pack(c)),S.value(Array.get(U64,O.pack(c),q))) : Array<U64> & U64}:
  Pair.read_pair(Tables.tables(n,U32.from_nat(k),P.prefix(k),O.pack(c)),q,S.value(Array.get(U64,O.pack(c),q)),
    Equal.trans(U64,S.value(Array.get(U64,Tables.tables(n,U32.from_nat(k),P.prefix(k),O.pack(c)),q)),
      S.value(Array.get(U64,P.scheduled(n,k,O.pack(c)),q)),S.value(Array.get(U64,O.pack(c),q)),
      Equal.cong(Array<U64>,U64,a => S.value(Array.get(U64,a,q)),Tables.tables(n,U32.from_nat(k),P.prefix(k),O.pack(c)),
        P.scheduled(n,k,O.pack(c)),PL.tables_follow_prefix(n,k,O.pack(c),budget)),
      Preserve.value(n,O.pack(c),k,q,shape,budget,lower,qb,before)))

def lift(-a: Array<U64>,+n: Nat,+k: Nat,+q: U32,
  shape: {O.shape(a) == C.full(17n) : O.Shape},budget: {Nat.is_lt(Nat.add(n,k),129n) == True{} : Bool},
  lower: {U32.is_le(512,q) == True{} : Bool},qb: {U32.is_lt(q,131072) == True{} : Bool},
  before: {U32.is_lt(q,P.prefix(k)) == True{} : Bool},cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,Tables.tables(n,U32.from_nat(k),P.prefix(k),a),q) ==
    (Tables.tables(n,U32.from_nat(k),P.prefix(k),a),S.value(Array.get(U64,a,q))) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Array.get(U64,Tables.tables(n,U32.from_nat(k),P.prefix(k),_),q) ==
    (Tables.tables(n,U32.from_nat(k),P.prefix(k),a),S.value(Array.get(U64,a,q))) : Array<U64> & U64}
  %same : {Array.get(U64,Tables.tables(n,U32.from_nat(k),P.prefix(k),O.pack(c)),q) ==
    (Tables.tables(n,U32.from_nat(k),P.prefix(k),_),S.value(Array.get(U64,a,q))) : Array<U64> & U64}
  %same : {Array.get(U64,Tables.tables(n,U32.from_nat(k),P.prefix(k),O.pack(c)),q) ==
    (Tables.tables(n,U32.from_nat(k),P.prefix(k),O.pack(c)),S.value(Array.get(U64,_,q))) : Array<U64> & U64}
  model(c,n,k,q,Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,x => O.shape(x),O.pack(c),a,same),shape),budget,lower,qb,before)

def actual(a: Array<U64>,+n: Nat,+k: Nat,+q: U32,
  shape: {O.shape(a) == C.full(17n) : O.Shape},budget: {Nat.is_lt(Nat.add(n,k),129n) == True{} : Bool},
  lower: {U32.is_le(512,q) == True{} : Bool},qb: {U32.is_lt(q,131072) == True{} : Bool},
  before: {U32.is_lt(q,P.prefix(k)) == True{} : Bool}) ->
  {Array.get(U64,Tables.tables(n,U32.from_nat(k),P.prefix(k),a),q) ==
    (Tables.tables(n,U32.from_nat(k),P.prefix(k),a),S.value(Array.get(U64,a,q))) : Array<U64> & U64}:
  lift(a,n,k,q,shape,budget,lower,qb,before,R.reify(a))
