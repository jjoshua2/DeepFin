# Computed data through actual later blocks and extras; independent rays remain open.
import Base
import ./Spec.bend as Spec
import ../../Tables.bend as Tables
import ../prefix/Spec.bend as P
import ../layout/Spec.bend as Layout
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S

law later_tables_preserve_data:
  for a: Array<U64>
  for +n: Nat
  for +k: Nat
  for +q: U32
  for shape: {O.shape(a) == C.full(17n) : O.Shape}
  for budget: {Nat.is_lt(Nat.add(n,k),129n) == True{} : Bool}
  for lower: {U32.is_le(512,q) == True{} : Bool}
  for qb: {U32.is_lt(q,131072) == True{} : Bool}
  for before: {U32.is_lt(q,P.prefix(k)) == True{} : Bool}
  {Array.get(U64,Tables.tables(n,U32.from_nat(k),P.prefix(k),a),q) ==
    (Tables.tables(n,U32.from_nat(k),P.prefix(k),a),S.value(Array.get(U64,a,q))) : Array<U64> & U64}

law stored_data_after_tables:
  for a: Array<U64>
  for +before: Nat
  for +after: Nat
  for +k: Nat
  for +i: Nat
  for shape: {O.shape(a) == C.full(17n) : O.Shape}
  for budget: {Nat.is_lt(Nat.add(Spec.count(before,after),k),129n) == True{} : Bool}
  for ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k))))) == True{} : Bool}
  {Array.get(U64,Spec.table(a,before,after,k),Spec.query(Nat.add(before,k),i)) ==
    (Spec.table(a,before,after,k),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}

law stored_data_after_extras:
  for +d: Nat
  for +seed: U64
  for +before: Nat
  for +after: Nat
  for +k: Nat
  for +i: Nat
  for +extra: Nat
  for +start: Nat
  for depth: {d == 17n : Nat}
  for budget: {Nat.is_lt(Nat.add(Spec.count(before,after),k),129n) == True{} : Bool}
  for ib: {Nat.is_lt(i,U32.to_nat(Layout.size(U32.from_nat(Nat.add(before,k))))) == True{} : Bool}
  for ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}
  {Array.get(U64,Spec.run(Array.new(U64,d,seed),before,after,k,extra,start),Spec.query(Nat.add(before,k),i)) ==
    (Spec.run(Array.new(U64,d,seed),before,after,k,extra,start),Spec.expected(Nat.add(before,k),i)) : Array<U64> & U64}
