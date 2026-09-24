# Stored values, not just preservation of arbitrary incoming headers.
import Base
import ../../Tables.bend as Tables
import ./Spec.bend as Spec
import ../prefix/Spec.bend as P
import ../complete/Spec.bend as C
import ../storage/Observe.bend as O

law own_block_header:
  for a: Array<U64>
  for +k: Nat
  for +which: Bool
  for valid: {Nat.is_lt(k,128n) == True{} : Bool}
  for shape: {O.shape(a) == C.full(17n) : O.Shape}
  {Array.get(U64,Tables.tables(1n,Spec.key(k),P.prefix(k),a),Spec.query(k,which)) ==
    (Tables.tables(1n,Spec.key(k),P.prefix(k),a),Spec.expected(k,which)) : Array<U64> & U64}

law stored_header_after_tables:
  for a: Array<U64>
  for +before: Nat
  for +after: Nat
  for +k: Nat
  for +which: Bool
  for shape: {O.shape(a) == C.full(17n) : O.Shape}
  for budget: {Nat.is_lt(Nat.add(Spec.count(before,after),k),129n) == True{} : Bool}
  {Array.get(U64,Tables.tables(Spec.count(before,after),Spec.key(k),P.prefix(k),a),Spec.query(Nat.add(before,k),which)) ==
    (Tables.tables(Spec.count(before,after),Spec.key(k),P.prefix(k),a),Spec.expected(Nat.add(before,k),which)) : Array<U64> & U64}

law stored_header_after_extras:
  for +d: Nat
  for +seed: U64
  for +before: Nat
  for +after: Nat
  for +k: Nat
  for +which: Bool
  for +extra: Nat
  for +start: Nat
  for depth: {d == 17n : Nat}
  for budget: {Nat.is_lt(Nat.add(Spec.count(before,after),k),129n) == True{} : Bool}
  for ebudget: {Nat.is_le(Nat.add(extra,start),64n) == True{} : Bool}
  {Array.get(U64,Spec.run(Array.new(U64,d,seed),before,after,k,extra,start),Spec.query(Nat.add(before,k),which)) ==
    (Spec.run(Array.new(U64,d,seed),before,after,k,extra,start),Spec.expected(Nat.add(before,k),which)) : Array<U64> & U64}
