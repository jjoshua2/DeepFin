import Base
import ./LAWS.bend as Laws
import ./PROOF.bend as Proof
import ./Spec.bend as Spec
import ../layout/Spec.bend as Layout
import ../prefix/Spec.bend as P
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S
import ../../Tables.bend as Tables

def selected(a: Array<U64>,+before: Nat,+after: Nat,+k: Nat,+i: Nat,
  shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_lt(Nat.add(Spec.count(before,after),k),129n) == True{} : Bool},
  ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k))))) == True{} : Bool}) ->
  {Array.get(U64,Spec.table(a,before,after,k),Spec.query(Nat.add(before,k),i)) ==
    (Spec.table(a,before,after,k),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}:
  Laws.stored_data_after_tables(a,before,after,k,i,shape,budget,ib)

def initialized(+d: Nat,+seed: U64,+before: Nat,+after: Nat,+k: Nat,+i: Nat,+extra: Nat,+start: Nat,
  depth: {d == 17n : Nat},budget: {Nat.is_lt(Nat.add(Spec.count(before,after),k),129n) == True{} : Bool},
  ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k))))) == True{} : Bool},
  ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}) ->
  {Array.get(U64,Spec.run(Array.new(U64,d,seed),before,after,k,extra,start),Spec.query(Nat.add(before,k),i)) ==
    (Spec.run(Array.new(U64,d,seed),before,after,k,extra,start),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}:
  Laws.stored_data_after_extras(d,seed,before,after,k,i,extra,start,depth,budget,ib,ebudget)

def earlier(a: Array<U64>,+n: Nat,+k: Nat,+q: U32,shape: {O.shape(a) == C.full(17n) : O.Shape},
  budget: {Nat.is_lt(Nat.add(n,k),129n) == True{} : Bool},lower: {U32.is_le(512,q) == True{} : Bool},
  qb: {U32.is_lt(q,131072) == True{} : Bool},before: {U32.is_lt(q,P.prefix(k)) == True{} : Bool}) ->
  {Array.get(U64,Tables.tables(n,U32.from_nat(k),P.prefix(k),a),q) ==
    (Tables.tables(n,U32.from_nat(k),P.prefix(k),a),S.value(Array.get(U64,a,q))) : Array<U64> & U64}:
  Laws.later_tables_preserve_data(a,n,k,q,shape,budget,lower,qb,before)

# Satisfiable full ranges, positive suffix, first and last relative positions.
def first_block_budget() -> {Nat.is_lt(Nat.add(Spec.count(0n,127n),0n),129n) == True{} : Bool}: {==}
def last_block_budget() -> {Nat.is_lt(Nat.add(Spec.count(127n,0n),0n),129n) == True{} : Bool}: {==}
def extras_budget() -> {Nat.is_le(Nat.add(64n,0n),64n) == True{} : Bool}: {==}
def first_interior() -> {Nat.is_lt(0n,U32.to_nat(Layout.size(0))) == True{} : Bool}: {==}
def last_interior() -> {Nat.is_lt(63n,U32.to_nat(Layout.size(127))) == True{} : Bool}: {==}
def earlier_data() -> {U32.is_lt(512,P.prefix(1n)) == True{} : Bool}: {==}
def excluded_endpoint() -> {Nat.is_lt(4096n,U32.to_nat(Layout.size(0))) == False{} : Bool}: {==}
