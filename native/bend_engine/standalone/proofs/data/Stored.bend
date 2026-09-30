# Every selected computed entry survives all later metadata and slider blocks.
import Base
import ../headers/Facts.bend as F
import ./Spec.bend as Spec
import ./Index.bend as Index
import ../relative/Bounds.bend as Bounds
import ../layout/Spec.bend as Layout
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

def first(+c: O.Cells,+n: Nat,+k: Nat,+i: Nat,
  +shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  +budget: {Nat.is_lt(Nat.add(1n+n,k),129n) == True{} : Bool},
  +ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(k)))) == True{} : Bool}) ->
  {S.value(Array.get(U64,P.scheduled(1n+n,k,O.pack(c)),Spec.query(k,i))) == Spec.expected(k,i) : U64}:
  +valid = N.strip(n,k,128n,budget)
  Equal.trans(U64,S.value(Array.get(U64,P.scheduled(n,1n+k,P.block(O.pack(c),k)),Spec.query(k,i))),
    S.value(Array.get(U64,P.block(O.pack(c),k),Spec.query(k,i))),Spec.expected(k,i),
    Preserve.value(n,P.block(O.pack(c),k),1n+k,Spec.query(k,i),
      Equal.trans(O.Shape,O.shape(P.block(O.pack(c),k)),O.shape(O.pack(c)),C.full(17n),Build.block_shape(c,F.key(k),P.prefix(k)),shape),
      N.next_bound(n,k,budget),Index.lower(k,i,valid,ib),Bounds.allocation(k,U32.from_nat(i),Index.interval(k,i,valid,ib)),Bounds.inside(k,U32.from_nat(i),Index.interval(k,i,valid,ib))),
    Equal.cong(Array<U64> & U64,U64,r => S.value(r),Array.get(U64,P.block(O.pack(c),k),Spec.query(k,i)),
      (P.block(O.pack(c),k),Spec.expected(k,i)),One.model(c,k,i,valid,shape,ib)))

def transfer(-a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,+i: Nat,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  rule: @+c: O.Cells -> {O.shape(O.pack(c)) == C.full(17n) : O.Shape} ->
    {S.value(Array.get(U64,P.scheduled(count(before,after),k,O.pack(c)),Spec.query(Nat.add(before,k),i))) == Spec.expected(Nat.add(before,k),i) : U64},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {S.value(Array.get(U64,P.scheduled(count(before,after),k,a),Spec.query(Nat.add(before,k),i))) == Spec.expected(Nat.add(before,k),i) : U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {S.value(Array.get(U64,P.scheduled(count(before,after),k,_),Spec.query(Nat.add(before,k),i))) == Spec.expected(Nat.add(before,k),i) : U64}
  rule(c)(Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,x => O.shape(x),O.pack(c),a,same),shape))

def selected(+before: Nat,+after: Nat,+k: Nat,a: Array<U64>,+i: Nat,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  +budget: {Nat.is_lt(Nat.add(count(before,after),k),129n) == True{} : Bool},
  +ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k))))) == True{} : Bool}) ->
  {S.value(Array.get(U64,P.scheduled(count(before,after),k,a),Spec.query(Nat.add(before,k),i))) == Spec.expected(Nat.add(before,k),i) : U64}:
  match before:
    case 0n:
      transfer(a,0n,after,k,i,shape,+c => +sh => first(c,after,k,i,sh,budget,ib),R.reify(a))
    case 1n+p:
      transfer(a,1n+p,after,k,i,shape,+c => +sh =>
        %N.add_suc(p,k) :
          {S.value(Array.get(U64,P.scheduled(count(p,after),1n+k,P.block(O.pack(c),k)),Spec.query(_,i))) == Spec.expected(1n+Nat.add(p,k),i) : U64}
        %N.add_suc(p,k) :
          {S.value(Array.get(U64,P.scheduled(count(p,after),1n+k,P.block(O.pack(c),k)),Spec.query(Nat.add(p,1n+k),i))) == Spec.expected(_,i) : U64}
        selected(p,after,1n+k,P.block(O.pack(c),k),i,
          Equal.trans(O.Shape,O.shape(P.block(O.pack(c),k)),O.shape(O.pack(c)),C.full(17n),Build.block_shape(c,F.key(k),P.prefix(k)),sh),
          N.next_bound(count(p,after),k,budget),
          Equal.trans(Bool,Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(p,1n+k))))),
            Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(1n+Nat.add(p,k))))),True{},
            Equal.cong(Nat,Bool,j => Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(j)))),
              Nat.add(p,1n+k),1n+Nat.add(p,k),N.add_suc(p,k)),ib)),R.reify(a))

def actual_model(+c: O.Cells,+before: Nat,+after: Nat,+k: Nat,+i: Nat,
  shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  +budget: {Nat.is_lt(Nat.add(count(before,after),k),129n) == True{} : Bool},
  +ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k))))) == True{} : Bool}) ->
  {Array.get(U64,Tables.tables(count(before,after),F.key(k),P.prefix(k),O.pack(c)),Spec.query(Nat.add(before,k),i)) ==
    (Tables.tables(count(before,after),F.key(k),P.prefix(k),O.pack(c)),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}:
  Pair.read_pair(Tables.tables(count(before,after),F.key(k),P.prefix(k),O.pack(c)),Spec.query(Nat.add(before,k),i),Spec.expected(Nat.add(before,k),i),
    Equal.trans(U64,S.value(Array.get(U64,Tables.tables(count(before,after),F.key(k),P.prefix(k),O.pack(c)),Spec.query(Nat.add(before,k),i))),
      S.value(Array.get(U64,P.scheduled(count(before,after),k,O.pack(c)),Spec.query(Nat.add(before,k),i))),Spec.expected(Nat.add(before,k),i),
      Equal.cong(Array<U64>,U64,a => S.value(Array.get(U64,a,Spec.query(Nat.add(before,k),i))),
        Tables.tables(count(before,after),F.key(k),P.prefix(k),O.pack(c)),P.scheduled(count(before,after),k,O.pack(c)),
        PL.tables_follow_prefix(count(before,after),k,O.pack(c),budget)),selected(before,after,k,O.pack(c),i,shape,budget,ib)))

def actual_lift(-a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,+i: Nat,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_lt(Nat.add(count(before,after),k),129n) == True{} : Bool},
  ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k))))) == True{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,Tables.tables(count(before,after),F.key(k),P.prefix(k),a),Spec.query(Nat.add(before,k),i)) ==
    (Tables.tables(count(before,after),F.key(k),P.prefix(k),a),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Array.get(U64,Tables.tables(count(before,after),F.key(k),P.prefix(k),_),Spec.query(Nat.add(before,k),i)) ==
    (Tables.tables(count(before,after),F.key(k),P.prefix(k),a),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}
  %same : {Array.get(U64,Tables.tables(count(before,after),F.key(k),P.prefix(k),O.pack(c)),Spec.query(Nat.add(before,k),i)) ==
    (Tables.tables(count(before,after),F.key(k),P.prefix(k),_),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}
  actual_model(c,before,after,k,i,Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,x => O.shape(x),O.pack(c),a,same),shape),budget,ib)

def actual(a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,+i: Nat,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_lt(Nat.add(count(before,after),k),129n) == True{} : Bool},
  +ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k))))) == True{} : Bool}) ->
  {Array.get(U64,Tables.tables(count(before,after),F.key(k),P.prefix(k),a),Spec.query(Nat.add(before,k),i)) ==
    (Tables.tables(count(before,after),F.key(k),P.prefix(k),a),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}:
  actual_lift(a,before,after,k,i,shape,budget,ib,R.reify(a))
