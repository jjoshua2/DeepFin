# Every selected header survives all later real header and slider-block writes.
import Base
import ./Facts.bend as F
import ./Addresses.bend as A
import ./OneBlock.bend as One
import ./Preserve.bend as Preserve
import ../prefix/Spec.bend as P
import ../prefix/LAWS.bend as PL
import ../prefix/PROOF.bend as PP
import ../prefix/Nats.bend as N
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../storage/Build.bend as Build
import ../storage/Representation.bend as R
import ../separation/Spec.bend as S
import ../contents/Actual.bend as Pair
import ../../Tables.bend as Tables

def below_successor(+n: Nat) -> {Nat.is_lt(n,1n+n) == True{} : Bool}:
  match n:
    case 0n: {==}
    case 1n+p: below_successor(p)

def count(before: Nat,after: Nat) -> Nat:
  Nat.add(before,1n+after)

def first(+c: O.Cells,+n: Nat,+k: Nat,+which: Bool,
  +shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  +budget: {Nat.is_lt(Nat.add(1n+n,k),129n) == True{} : Bool}) ->
  {S.value(Array.get(U64,P.scheduled(1n+n,k,O.pack(c)),A.query(k,which))) == One.expected(k,which) : U64}:
  Equal.trans(U64,S.value(Array.get(U64,P.scheduled(n,1n+k,P.block(O.pack(c),k)),A.query(k,which))),
    S.value(Array.get(U64,P.block(O.pack(c),k),A.query(k,which))),One.expected(k,which),
    Preserve.value(n,P.block(O.pack(c),k),1n+k,k,which,
      Equal.trans(O.Shape,O.shape(P.block(O.pack(c),k)),O.shape(O.pack(c)),C.full(17n),Build.block_shape(c,F.key(k),P.prefix(k)),shape),
      N.next_bound(n,k,budget),below_successor(k)),
    Equal.cong(Array<U64> & U64,U64,r => S.value(r),Array.get(U64,One.run(O.pack(c),k),One.query(k,which)),
      (One.run(O.pack(c),k),One.expected(k,which)),One.model(c,k,which,N.strip(n,k,128n,budget),shape)))

def transfer(-a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,+which: Bool,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  rule: @+c: O.Cells -> {O.shape(O.pack(c)) == C.full(17n) : O.Shape} ->
    {S.value(Array.get(U64,P.scheduled(count(before,after),k,O.pack(c)),A.query(Nat.add(before,k),which))) == One.expected(Nat.add(before,k),which) : U64},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {S.value(Array.get(U64,P.scheduled(count(before,after),k,a),A.query(Nat.add(before,k),which))) == One.expected(Nat.add(before,k),which) : U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {S.value(Array.get(U64,P.scheduled(count(before,after),k,_),A.query(Nat.add(before,k),which))) == One.expected(Nat.add(before,k),which) : U64}
  rule(c)(Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,x => O.shape(x),O.pack(c),a,same),shape))

def selected(+before: Nat,+after: Nat,+k: Nat,a: Array<U64>,+which: Bool,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  +budget: {Nat.is_lt(Nat.add(count(before,after),k),129n) == True{} : Bool}) ->
  {S.value(Array.get(U64,P.scheduled(count(before,after),k,a),A.query(Nat.add(before,k),which))) == One.expected(Nat.add(before,k),which) : U64}:
  match before:
    case 0n:
      transfer(a,0n,after,k,which,shape,+c => +sh => first(c,after,k,which,sh,budget),R.reify(a))
    case 1n+p:
      transfer(a,1n+p,after,k,which,shape,+c => +sh =>
        %N.add_suc(p,k) :
          {S.value(Array.get(U64,P.scheduled(count(p,after),1n+k,P.block(O.pack(c),k)),A.query(_,which))) == One.expected(1n+Nat.add(p,k),which) : U64}
        %N.add_suc(p,k) :
          {S.value(Array.get(U64,P.scheduled(count(p,after),1n+k,P.block(O.pack(c),k)),A.query(Nat.add(p,1n+k),which))) == One.expected(_,which) : U64}
        selected(p,after,1n+k,P.block(O.pack(c),k),which,
          Equal.trans(O.Shape,O.shape(P.block(O.pack(c),k)),O.shape(O.pack(c)),C.full(17n),Build.block_shape(c,F.key(k),P.prefix(k)),sh),
          N.next_bound(count(p,after),k,budget)),R.reify(a))

def actual_model(+c: O.Cells,+before: Nat,+after: Nat,+k: Nat,+which: Bool,
  shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  +budget: {Nat.is_lt(Nat.add(count(before,after),k),129n) == True{} : Bool}) ->
  {Array.get(U64,Tables.tables(count(before,after),F.key(k),P.prefix(k),O.pack(c)),A.query(Nat.add(before,k),which)) ==
    (Tables.tables(count(before,after),F.key(k),P.prefix(k),O.pack(c)),One.expected(Nat.add(before,k),which)) : Array<U64> & U64}:
  Pair.read_pair(Tables.tables(count(before,after),F.key(k),P.prefix(k),O.pack(c)),A.query(Nat.add(before,k),which),One.expected(Nat.add(before,k),which),
    Equal.trans(U64,S.value(Array.get(U64,Tables.tables(count(before,after),F.key(k),P.prefix(k),O.pack(c)),A.query(Nat.add(before,k),which))),
      S.value(Array.get(U64,P.scheduled(count(before,after),k,O.pack(c)),A.query(Nat.add(before,k),which))),One.expected(Nat.add(before,k),which),
      Equal.cong(Array<U64>,U64,a => S.value(Array.get(U64,a,A.query(Nat.add(before,k),which))),
        Tables.tables(count(before,after),F.key(k),P.prefix(k),O.pack(c)),P.scheduled(count(before,after),k,O.pack(c)),
        PL.tables_follow_prefix(count(before,after),k,O.pack(c),budget)),selected(before,after,k,O.pack(c),which,shape,budget)))

def actual_lift(-a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,+which: Bool,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_lt(Nat.add(count(before,after),k),129n) == True{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,Tables.tables(count(before,after),F.key(k),P.prefix(k),a),A.query(Nat.add(before,k),which)) ==
    (Tables.tables(count(before,after),F.key(k),P.prefix(k),a),One.expected(Nat.add(before,k),which)) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Array.get(U64,Tables.tables(count(before,after),F.key(k),P.prefix(k),_),A.query(Nat.add(before,k),which)) ==
    (Tables.tables(count(before,after),F.key(k),P.prefix(k),a),One.expected(Nat.add(before,k),which)) : Array<U64> & U64}
  %same : {Array.get(U64,Tables.tables(count(before,after),F.key(k),P.prefix(k),O.pack(c)),A.query(Nat.add(before,k),which)) ==
    (Tables.tables(count(before,after),F.key(k),P.prefix(k),_),One.expected(Nat.add(before,k),which)) : Array<U64> & U64}
  actual_model(c,before,after,k,which,Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,x => O.shape(x),O.pack(c),a,same),shape),budget)

def actual(a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,+which: Bool,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_lt(Nat.add(count(before,after),k),129n) == True{} : Bool}) ->
  {Array.get(U64,Tables.tables(count(before,after),F.key(k),P.prefix(k),a),A.query(Nat.add(before,k),which)) ==
    (Tables.tables(count(before,after),F.key(k),P.prefix(k),a),One.expected(Nat.add(before,k),which)) : Array<U64> & U64}:
  actual_lift(a,before,after,k,which,shape,budget,R.reify(a))
