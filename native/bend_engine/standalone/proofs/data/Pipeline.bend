# Compose computed data with the actual final extras frame.
import Base
import ../headers/Facts.bend as F
import ./Spec.bend as Spec
import ./Index.bend as Index
import ../relative/Bounds.bend as Bounds
import ../layout/Spec.bend as Layout
import ./Stored.bend as Stored
import ../prefix/Spec.bend as P
import ../prefix/Nats.bend as N
import ../complete/Spec.bend as C
import ../complete/Actual.bend as Complete
import ../storage/Observe.bend as O
import ../storage/Build.bend as Build
import ../storage/Representation.bend as R
import ../separation/Spec.bend as S
import ../contents/Actual.bend as Pair
import ../extras/Spec.bend as ExtrasSpec
import ../extras/Actual.bend as Extras
import ../relative/Order.bend as Order
import ../../Tables.bend as Tables

def selected_valid(+before: Nat,+after: Nat,+k: Nat,
  e: {Nat.is_lt(Nat.add(Stored.count(before,after),k),129n) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(before,k),128n) == True{} : Bool}:
  match before:
    case 0n: N.strip(after,k,128n,e)
    case 1n+p:
      %N.add_suc(p,k) : {Nat.is_lt(_,128n) == True{} : Bool}
      selected_valid(p,after,1n+k,N.next_bound(Stored.count(p,after),k,e))

def or_right(+b: Bool) -> {Bool.or(b,True{}) == True{} : Bool}:
  match b:
    case False{}: {==}
    case True{}: {==}

def outside(+q: U32,e: {U32.is_le(512,q) == True{} : Bool}) ->
  {ExtrasSpec.outside(q) == True{} : Bool}:
  Equal.trans(Bool,ExtrasSpec.outside(q),Bool.or(U32.is_lt(q,256),True{}),True{},
    Equal.cong(Bool,Bool,b => Bool.or(U32.is_lt(q,256),b),U32.is_le(512,q),True{},e),or_right(U32.is_lt(q,256)))

def run(a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,extra: Nat,start: Nat) -> Array<U64>:
  Tables.extras(extra,U32.from_nat(start),Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),a))

def model(+c: O.Cells,+before: Nat,+after: Nat,+k: Nat,+i: Nat,+extra: Nat,+start: Nat,
  +shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  +budget: {Nat.is_lt(Nat.add(Stored.count(before,after),k),129n) == True{} : Bool},
  +ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k))))) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Array.get(U64,run(O.pack(c),before,after,k,extra,start),Spec.query(Nat.add(before,k),i)) ==
    (run(O.pack(c),before,after,k,extra,start),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}:
  +q = Spec.query(Nat.add(before,k),i)
  +valid = selected_valid(before,after,k,budget)
  bounds = Index.interval(Nat.add(before,k),i,valid,ib)
  +qb = Bounds.allocation(Nat.add(before,k),U32.from_nat(i),bounds)
  +lower = Index.lower(Nat.add(before,k),i,valid,ib)
  Pair.read_pair(run(O.pack(c),before,after,k,extra,start),q,Spec.expected(Nat.add(before,k),i),
    Equal.trans(U64,S.value(Array.get(U64,run(O.pack(c),before,after,k,extra,start),q)),
      S.value(Array.get(U64,Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),O.pack(c)),q)),Spec.expected(Nat.add(before,k),i),
      Extras.value(extra,Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),O.pack(c)),start,q,
        Equal.trans(O.Shape,O.shape(Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),O.pack(c))),O.shape(O.pack(c)),C.full(17n),
          Build.tables(Stored.count(before,after),F.key(k),P.prefix(k),O.pack(c)),shape),ebudget,
        qb,outside(q,lower)),
      Equal.cong(Array<U64> & U64,U64,r => S.value(r),
        Array.get(U64,Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),O.pack(c)),q),
        (Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),O.pack(c)),Spec.expected(Nat.add(before,k),i)),
        Stored.actual_model(c,before,after,k,i,shape,budget,ib))))

def lift(-a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,+i: Nat,+extra: Nat,+start: Nat,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_lt(Nat.add(Stored.count(before,after),k),129n) == True{} : Bool},
  +ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k))))) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,run(a,before,after,k,extra,start),Spec.query(Nat.add(before,k),i)) ==
    (run(a,before,after,k,extra,start),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Array.get(U64,run(_,before,after,k,extra,start),Spec.query(Nat.add(before,k),i)) ==
    (run(a,before,after,k,extra,start),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}
  %same : {Array.get(U64,run(O.pack(c),before,after,k,extra,start),Spec.query(Nat.add(before,k),i)) ==
    (run(_,before,after,k,extra,start),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}
  model(c,before,after,k,i,extra,start,Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,x => O.shape(x),O.pack(c),a,same),shape),budget,ib,ebudget)

def actual(a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,+i: Nat,+extra: Nat,+start: Nat,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_lt(Nat.add(Stored.count(before,after),k),129n) == True{} : Bool},
  +ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k))))) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Array.get(U64,run(a,before,after,k,extra,start),Spec.query(Nat.add(before,k),i)) ==
    (run(a,before,after,k,extra,start),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}:
  lift(a,before,after,k,i,extra,start,shape,budget,ib,ebudget,R.reify(a))

def initialized(+d: Nat,+seed: U64,+before: Nat,+after: Nat,+k: Nat,+i: Nat,+extra: Nat,+start: Nat,
  depth: {d == 17n : Nat},
  budget: {Nat.is_lt(Nat.add(Stored.count(before,after),k),129n) == True{} : Bool},
  +ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k))))) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Array.get(U64,run(Array.new(U64,d,seed),before,after,k,extra,start),Spec.query(Nat.add(before,k),i)) ==
    (run(Array.new(U64,d,seed),before,after,k,extra,start),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}:
  actual(Array.new(U64,d,seed),before,after,k,i,extra,start,
    Equal.trans(O.Shape,O.shape(Array.new(U64,d,seed)),C.full(d),C.full(17n),Complete.allocation(d,seed),
      Equal.cong(Nat,O.Shape,x => C.full(x),d,17n,depth)),budget,ib,ebudget)
