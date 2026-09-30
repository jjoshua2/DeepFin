# Compose stored-header correctness with the qualified final extras frame.
import Base
import ./Facts.bend as F
import ./Addresses.bend as A
import ./OneBlock.bend as One
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

def outside(+q: U32,e: {U32.is_lt(q,256) == True{} : Bool}) ->
  {ExtrasSpec.outside(q) == True{} : Bool}:
  Equal.cong(Bool,Bool,b => Bool.or(b,U32.is_le(512,q)),U32.is_lt(q,256),True{},e)

def run(a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,extra: Nat,start: Nat) -> Array<U64>:
  Tables.extras(extra,U32.from_nat(start),Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),a))

def model(+c: O.Cells,+before: Nat,+after: Nat,+k: Nat,+which: Bool,+extra: Nat,+start: Nat,
  +shape: {O.shape(O.pack(c)) == C.full(17n) : O.Shape},
  +budget: {Nat.is_lt(Nat.add(Stored.count(before,after),k),129n) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Array.get(U64,run(O.pack(c),before,after,k,extra,start),A.query(Nat.add(before,k),which)) ==
    (run(O.pack(c),before,after,k,extra,start),One.expected(Nat.add(before,k),which)) : Array<U64> & U64}:
  +q = A.query(Nat.add(before,k),which)
  +qb = A.query_bound(Nat.add(before,k),which,selected_valid(before,after,k,budget))
  Pair.read_pair(run(O.pack(c),before,after,k,extra,start),q,One.expected(Nat.add(before,k),which),
    Equal.trans(U64,S.value(Array.get(U64,run(O.pack(c),before,after,k,extra,start),q)),
      S.value(Array.get(U64,Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),O.pack(c)),q)),One.expected(Nat.add(before,k),which),
      Extras.value(extra,Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),O.pack(c)),start,q,
        Equal.trans(O.Shape,O.shape(Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),O.pack(c))),O.shape(O.pack(c)),C.full(17n),
          Build.tables(Stored.count(before,after),F.key(k),P.prefix(k),O.pack(c)),shape),ebudget,
        Order.lt_lt(q,256,131072,qb,{==}),outside(q,qb)),
      Equal.cong(Array<U64> & U64,U64,r => S.value(r),
        Array.get(U64,Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),O.pack(c)),q),
        (Tables.tables(Stored.count(before,after),F.key(k),P.prefix(k),O.pack(c)),One.expected(Nat.add(before,k),which)),
        Stored.actual_model(c,before,after,k,which,shape,budget))))

def lift(-a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,+which: Bool,+extra: Nat,+start: Nat,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_lt(Nat.add(Stored.count(before,after),k),129n) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool},
  cert: Exists(O.Cells,c => {O.pack(c) == a : Array<U64>})) ->
  {Array.get(U64,run(a,before,after,k,extra,start),A.query(Nat.add(before,k),which)) ==
    (run(a,before,after,k,extra,start),One.expected(Nat.add(before,k),which)) : Array<U64> & U64}:
  (+c,eq) = cert
  +same: {O.pack(c) == a : Array<U64>} = eq
  %same : {Array.get(U64,run(_,before,after,k,extra,start),A.query(Nat.add(before,k),which)) ==
    (run(a,before,after,k,extra,start),One.expected(Nat.add(before,k),which)) : Array<U64> & U64}
  %same : {Array.get(U64,run(O.pack(c),before,after,k,extra,start),A.query(Nat.add(before,k),which)) ==
    (run(_,before,after,k,extra,start),One.expected(Nat.add(before,k),which)) : Array<U64> & U64}
  model(c,before,after,k,which,extra,start,Equal.trans(O.Shape,O.shape(O.pack(c)),O.shape(a),C.full(17n),
    Equal.cong(Array<U64>,O.Shape,x => O.shape(x),O.pack(c),a,same),shape),budget,ebudget)

def actual(a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,+which: Bool,+extra: Nat,+start: Nat,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_lt(Nat.add(Stored.count(before,after),k),129n) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Array.get(U64,run(a,before,after,k,extra,start),A.query(Nat.add(before,k),which)) ==
    (run(a,before,after,k,extra,start),One.expected(Nat.add(before,k),which)) : Array<U64> & U64}:
  lift(a,before,after,k,which,extra,start,shape,budget,ebudget,R.reify(a))

def initialized(+d: Nat,+seed: U64,+before: Nat,+after: Nat,+k: Nat,+which: Bool,+extra: Nat,+start: Nat,
  depth: {d == 17n : Nat},
  budget: {Nat.is_lt(Nat.add(Stored.count(before,after),k),129n) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Array.get(U64,run(Array.new(U64,d,seed),before,after,k,extra,start),A.query(Nat.add(before,k),which)) ==
    (run(Array.new(U64,d,seed),before,after,k,extra,start),One.expected(Nat.add(before,k),which)) : Array<U64> & U64}:
  actual(Array.new(U64,d,seed),before,after,k,which,extra,start,
    Equal.trans(O.Shape,O.shape(Array.new(U64,d,seed)),C.full(d),C.full(17n),Complete.allocation(d,seed),
      Equal.cong(Nat,O.Shape,x => C.full(x),d,17n,depth)),budget,ebudget)
