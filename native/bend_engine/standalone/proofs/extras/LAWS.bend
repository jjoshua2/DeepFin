# Final-phase preservation; neither stored-header values nor ray geometry are assumed proved.
import Base
import ./Spec.bend as Spec
import ../../Tables.bend as Tables
import ../complete/Spec.bend as Complete
import ../storage/Observe.bend as O
import ../separation/Spec.bend as S

law bounded_extras_read:
  for a: Array<U64>
  for +n: Nat
  for +k: Nat
  for +q: U32
  for shape: {O.shape(a) == Complete.full(17n) : O.Shape}
  for budget: {Nat.is_le(Nat.add(n,k),64n) == True{} : Bool}
  for qb: {U32.is_lt(q,131072) == True{} : Bool}
  for outside: {Spec.outside(q) == True{} : Bool}
  {Array.get(U64,Tables.extras(n,U32.from_nat(k),a),q) ==
    (Tables.extras(n,U32.from_nat(k),a),S.value(Array.get(U64,a,q))) : Array<U64> & U64}

law table_pipeline_extras_read:
  for +d: Nat
  for +n: Nat
  for +key: U32
  for +at: U32
  for +seed: U64
  for +extra: Nat
  for +k: Nat
  for +q: U32
  for depth: {d == 17n : Nat}
  for budget: {Nat.is_le(Nat.add(extra,k),64n) == True{} : Bool}
  for qb: {U32.is_lt(q,131072) == True{} : Bool}
  for outside: {Spec.outside(q) == True{} : Bool}
  {Array.get(U64,Tables.extras(extra,U32.from_nat(k),Tables.tables(n,key,at,Array.new(U64,d,seed))),q) ==
    (Tables.extras(extra,U32.from_nat(k),Tables.tables(n,key,at,Array.new(U64,d,seed))),
      S.value(Array.get(U64,Tables.tables(n,key,at,Array.new(U64,d,seed)),q))) : Array<U64> & U64}
